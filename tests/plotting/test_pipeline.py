"""
Tests for PlottingPipeline.

Tests cover:
- Bundle creation with/without collapse
- Scaler handling (OHE vs collapsed space)
- Beta scaling application
- Feature name reconstruction
- Integration with service objects
- Error handling
"""

from itertools import combinations

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from prism.plotting.pipeline import PlottingPipeline
from prism.preprocessing import NoScaler, PRiSMScaler


def generate_all_feature_names(univariate_names):
    """Helper: Generate all_feature_names (univariate + bivariate pairs) like production."""
    all_names = list(univariate_names)
    # Add all bivariate pairs in canonical order
    for i, j in combinations(range(len(univariate_names)), 2):
        pair_name = f"{univariate_names[i]} : {univariate_names[j]}"
        all_names.append(pair_name)
    return all_names


class TestPlottingPipelineWithoutCollapse:
    """Test PlottingPipeline without one-hot group collapse."""

    @pytest.fixture
    def mock_lasso_results(self):
        """Mock LassoResultsManager without collapse."""

        class MockLassoResults:
            def __init__(self):
                self.univariate_feature_names = ['age', 'bmi', 'glucose', 'bp', 'cholesterol']
                self.all_feature_names = generate_all_feature_names(self.univariate_feature_names)

            def get_selected_univariate_indices(self):
                return [0, 2, 4]  # age, glucose, cholesterol

            def get_selected_bivariate_index_pairs(self):
                return [(0, 2), (2, 4)]  # age-glucose, glucose-cholesterol

            def get_selected_beta(self):
                # FULL format: 5 univariate + 10 bivariate pairs = 15 total
                # Univariate betas: [age, bmi, glucose, bp, cholesterol]
                # Selected: age=1.5, glucose=2.5, cholesterol=0.3 (others=0)
                # Bivariate pairs in canonical order:
                # (0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)
                # Selected: (0,2)=0.5, (2,4)=0.8 (others=0)
                beta = np.zeros(15)
                beta[0] = 1.5  # age
                beta[2] = 2.5  # glucose
                beta[4] = 0.3  # cholesterol
                beta[5 + 1] = 0.5  # (0,2): age-glucose, position 1 in bivariate section
                beta[5 + 8] = 0.8  # (2,4): glucose-cholesterol, position 8 in bivariate section
                return beta

        return MockLassoResults()

    @pytest.fixture
    def simple_model(self):
        """Simple mock model that returns constant predictions."""

        class SimpleModel:
            def predict_proba(self, x, device='cpu'):
                # Return constant predictions
                batch_size = x.shape[0] if hasattr(x, 'shape') else len(x)
                return torch.ones(batch_size, 1, device=device) * 0.5

            def __call__(self, x):
                return self.predict_proba(x)

        return SimpleModel()

    @pytest.fixture
    def test_data(self):
        """Create test data (5 features, 100 samples)."""
        np.random.seed(42)
        data = np.column_stack(
            [
                np.random.uniform(18, 90, 100),  # age
                np.random.uniform(15, 40, 100),  # bmi
                np.random.uniform(70, 200, 100),  # glucose
                np.random.uniform(80, 180, 100),  # bp
                np.random.uniform(100, 300, 100),  # cholesterol
            ]
        )
        return torch.from_numpy(data).float()

    @pytest.fixture
    def pipeline_simple(self, mock_lasso_results):
        """Pipeline without collapse."""
        return PlottingPipeline(
            lasso_results=mock_lasso_results,
            group_manager=None,
            label_manager=None,
        )

    def test_initialization(self, pipeline_simple, mock_lasso_results):
        """Test pipeline initialization."""
        assert pipeline_simple.lasso_results is mock_lasso_results
        assert pipeline_simple.group_manager is None
        assert pipeline_simple.label_manager is None

    def test_prepare_bundle_basic(self, pipeline_simple, simple_model, test_data):
        """Test basic bundle preparation."""
        bundle = pipeline_simple.prepare_plotting_bundle(
            x=test_data,
            model=simple_model,
            feature_names=['age', 'bmi', 'glucose', 'bp', 'cholesterol'],
        )

        # Check bundle structure
        assert bundle.n_univariate == 3
        assert bundle.n_bivariate == 2
        assert bundle.has_services is True

        # Check service objects exist
        assert bundle.index_mapper is not None
        assert bundle.metadata_registry is not None

        # Check index mapper properties
        assert bundle.index_mapper.n_original == 5
        assert bundle.index_mapper.n_collapsed == 5
        assert bundle.index_mapper.n_dense == 3
        assert bundle.index_mapper.is_collapse_mode is False

    def test_prepare_bundle_with_scaler(self, pipeline_simple, simple_model, test_data):
        """Test bundle preparation with scaler."""
        # Create and fit scaler
        sklearn_scaler = StandardScaler().fit(test_data.numpy())
        scaler = PRiSMScaler(scaler=sklearn_scaler)

        bundle = pipeline_simple.prepare_plotting_bundle(
            x=test_data,
            model=simple_model,
            scaler=scaler,
            feature_names=['age', 'bmi', 'glucose', 'bp', 'cholesterol'],
        )

        # Scaler should be passed through unchanged (no collapse)
        assert bundle.scaler is scaler

    def test_beta_scaling(self, pipeline_simple, simple_model, test_data):
        """Test beta scaling application."""
        bundle = pipeline_simple.prepare_plotting_bundle(
            x=test_data,
            model=simple_model,
            feature_names=['age', 'bmi', 'glucose', 'bp', 'cholesterol'],
        )

        # Get original responses (store by dense index)
        original_responses = [info.response.copy() for info in bundle.univariate_features()]

        # Apply beta scaling
        bundle = pipeline_simple.apply_beta_scaling(bundle)

        # Check that responses were scaled correctly
        # Beta values: [1.5, 2.0, 2.5, 0.5, 0.3]
        # Selected indices: [0, 2, 4] -> dense [0, 1, 2]
        # So betas for selected are: [1.5, 2.5, 0.3] (but we use dense order)
        beta = pipeline_simple.lasso_results.get_selected_beta()

        for dense_idx, info in enumerate(bundle.univariate_features()):
            expected = original_responses[dense_idx] * beta[dense_idx]
            np.testing.assert_array_almost_equal(info.response, expected)

    def test_feature_name_inference(self, pipeline_simple, simple_model, test_data):
        """Test feature name inference when not provided."""
        # Don't provide feature_names - should reconstruct
        bundle = pipeline_simple.prepare_plotting_bundle(
            x=test_data,
            model=simple_model,
        )

        # Should have reconstructed from LASSO results
        # Check univariate portion (first 5 names)
        expected_univariate = ['age', 'bmi', 'glucose', 'bp', 'cholesterol']
        assert bundle.all_feature_names[:5] == expected_univariate
        # all_feature_names should also include bivariate pairs
        assert len(bundle.all_feature_names) == 15  # 5 univariate + 10 bivariate pairs


class TestPlottingPipelineWithCollapse:
    """Test PlottingPipeline with one-hot group collapse."""

    @pytest.fixture
    def mock_group_manager(self):
        """Mock OneHotGroupManager."""

        class MockGroupManager:
            def __init__(self):
                self.groups_dict = {
                    'diagn': ['diagn_CAD', 'diagn_Valve', 'diagn_Other'],
                    'blood_type': ['blood_A', 'blood_B', 'blood_O'],
                }

            def is_categorical_group(self, name):
                return name in self.groups_dict

            def to_indices(self, feature_names):
                """Convert groups to index ranges as list of tuples."""
                result = []
                for group_name, members in self.groups_dict.items():
                    indices = []
                    for member in members:
                        if member in feature_names:
                            indices.append(feature_names.index(member))
                    if indices:
                        result.append(tuple(indices))
                return result

            def create_collapsed_scaler(self, scaler, feature_names):
                # Mock: just return a new scaler with correct dimensions
                # In real implementation, this would actually collapse
                collapsed_data = np.random.uniform(0, 1, (100, 4))
                sklearn_scaler = StandardScaler().fit(collapsed_data)
                return PRiSMScaler(scaler=sklearn_scaler)

        return MockGroupManager()

    @pytest.fixture
    def mock_lasso_results_collapsed(self):
        """Mock LassoResultsManager with collapse."""

        class MockLassoResults:
            def __init__(self):
                self.univariate_feature_names = ['age', 'bmi', 'diagn', 'blood_type']
                self.all_feature_names = generate_all_feature_names(self.univariate_feature_names)

            def get_selected_univariate_indices(self):
                return [2, 3]  # diagn, blood_type

            def get_selected_bivariate_index_pairs(self):
                return [(2, 3)]  # diagn-blood_type

            def get_selected_beta(self):
                # FULL format: 4 univariate + 6 bivariate pairs = 10 total
                # Pairs: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
                # Selected: diagn=1.2, blood_type=0.8, (2,3)=0.5
                beta = np.zeros(10)
                beta[2] = 1.2  # diagn
                beta[3] = 0.8  # blood_type
                beta[4 + 5] = 0.5  # (2,3): diagn-blood_type, position 5 in bivariate
                return beta

        return MockLassoResults()

    @pytest.fixture
    def test_data_ohe(self):
        """Create test data in OHE space (8 features)."""
        np.random.seed(42)
        data = np.column_stack(
            [
                np.random.uniform(18, 90, 100),  # age
                np.random.uniform(15, 40, 100),  # bmi
                np.random.choice([0, 1], 100),  # diagn_CAD
                np.random.choice([0, 1], 100),  # diagn_Valve
                np.random.choice([0, 1], 100),  # diagn_Other
                np.random.choice([0, 1], 100),  # blood_A
                np.random.choice([0, 1], 100),  # blood_B
                np.random.choice([0, 1], 100),  # blood_O
            ]
        )
        return torch.from_numpy(data).float()

    @pytest.fixture
    def simple_model(self):
        """Simple mock model."""

        class SimpleModel:
            def predict_proba(self, x, device='cpu'):
                batch_size = x.shape[0] if hasattr(x, 'shape') else len(x)
                return torch.ones(batch_size, 1, device=device) * 0.5

            def __call__(self, x):
                return self.predict_proba(x)

        return SimpleModel()

    @pytest.fixture
    def pipeline_collapse(self, mock_lasso_results_collapsed, mock_group_manager):
        """Pipeline with collapse."""
        return PlottingPipeline(
            lasso_results=mock_lasso_results_collapsed,
            group_manager=mock_group_manager,
            label_manager=None,
        )

    def test_collapse_mode_detection(self, pipeline_collapse, simple_model, test_data_ohe):
        """Test that collapse mode is detected correctly."""
        bundle = pipeline_collapse.prepare_plotting_bundle(
            x=test_data_ohe,
            model=simple_model,
            scaler=NoScaler(),
            feature_names=[
                'age',
                'bmi',
                'diagn_CAD',
                'diagn_Valve',
                'diagn_Other',
                'blood_A',
                'blood_B',
                'blood_O',
            ],
        )

        # Check collapse mode is active
        assert bundle.index_mapper.is_collapse_mode is True
        assert bundle.index_mapper.n_original == 8
        assert bundle.index_mapper.n_collapsed == 4

    def test_scaler_collapse(self, pipeline_collapse, simple_model, test_data_ohe):
        """Test automatic scaler collapse."""
        # Create scaler in OHE space (8 features)
        sklearn_scaler = StandardScaler().fit(test_data_ohe.numpy())
        scaler_ohe = PRiSMScaler(scaler=sklearn_scaler)

        bundle = pipeline_collapse.prepare_plotting_bundle(
            x=test_data_ohe,
            model=simple_model,
            scaler=scaler_ohe,
            feature_names=[
                'age',
                'bmi',
                'diagn_CAD',
                'diagn_Valve',
                'diagn_Other',
                'blood_A',
                'blood_B',
                'blood_O',
            ],
        )

        # Scaler should have been collapsed
        assert bundle.scaler is not scaler_ohe  # Different scaler object
        # Should pass validation (collapsed scaler has 4 features)
        assert bundle.has_services is True

    def test_feature_name_reconstruction(self, pipeline_collapse, mock_group_manager):
        """Test feature name reconstruction from collapsed names."""
        # Test the internal method
        original_names = pipeline_collapse._reconstruct_feature_names()

        # Should expand groups
        expected = [
            'age',
            'bmi',
            'diagn_CAD',
            'diagn_Valve',
            'diagn_Other',
            'blood_A',
            'blood_B',
            'blood_O',
        ]
        assert original_names == expected


class TestPlottingPipelineWithLabels:
    """Test PlottingPipeline with FeatureLabelManager."""

    @pytest.fixture
    def mock_label_manager(self):
        """Mock FeatureLabelManager."""

        class MockLabelManager:
            def get_label(self, name):
                labels = {
                    'age': 'Age (years)',
                    'bmi': 'BMI (kg/m²)',
                    'glucose': 'Glucose\n(mg/dL)',
                }
                return labels.get(name, name)

        return MockLabelManager()

    @pytest.fixture
    def mock_lasso_results(self):
        """Mock LassoResultsManager."""

        class MockLassoResults:
            def __init__(self):
                self.univariate_feature_names = ['age', 'bmi', 'glucose']
                self.all_feature_names = generate_all_feature_names(self.univariate_feature_names)

            def get_selected_univariate_indices(self):
                return [0, 2]

            def get_selected_bivariate_index_pairs(self):
                return []

            def get_selected_beta(self):
                # FULL format: 3 univariate + 3 bivariate pairs = 6 total
                # Selected: age=1.0, glucose=1.5
                beta = np.zeros(6)
                beta[0] = 1.0  # age
                beta[2] = 1.5  # glucose
                return beta

        return MockLassoResults()

    @pytest.fixture
    def simple_model(self):
        """Simple mock model."""

        class SimpleModel:
            def predict_proba(self, x, device='cpu'):
                batch_size = x.shape[0] if hasattr(x, 'shape') else len(x)
                return torch.ones(batch_size, 1, device=device) * 0.5

            def __call__(self, x):
                return self.predict_proba(x)

        return SimpleModel()

    @pytest.fixture
    def test_data(self):
        """Create test data."""
        np.random.seed(42)
        data = np.random.uniform(0, 1, (100, 3))
        return torch.from_numpy(data).float()

    def test_labels_in_bundle(
        self, mock_lasso_results, mock_label_manager, simple_model, test_data
    ):
        """Test that labels are correctly propagated to bundle."""
        pipeline = PlottingPipeline(
            lasso_results=mock_lasso_results,
            group_manager=None,
            label_manager=mock_label_manager,
        )

        bundle = pipeline.prepare_plotting_bundle(
            x=test_data,
            model=simple_model,
            feature_names=['age', 'bmi', 'glucose'],
        )

        # Check labels
        features = bundle.univariate_features()
        assert features[0].label == 'Age (years)'
        assert features[1].label == 'Glucose\n(mg/dL)'


class TestPlottingPipelineScalerHandling:
    """Test scaler handling edge cases."""

    @pytest.fixture
    def mock_lasso_results(self):
        """Mock LassoResultsManager."""

        class MockLassoResults:
            def __init__(self):
                self.univariate_feature_names = ['a', 'b', 'c']
                self.all_feature_names = generate_all_feature_names(self.univariate_feature_names)

            def get_selected_univariate_indices(self):
                return [0, 1]

            def get_selected_bivariate_index_pairs(self):
                return []

            def get_selected_beta(self):
                # FULL format: 3 univariate + 3 bivariate pairs = 6 total
                # Selected: a=1.0, b=1.0
                beta = np.zeros(6)
                beta[0] = 1.0  # a
                beta[1] = 1.0  # b
                return beta

        return MockLassoResults()

    @pytest.fixture
    def simple_model(self):
        """Simple mock model."""

        class SimpleModel:
            def predict_proba(self, x, device='cpu'):
                batch_size = x.shape[0] if hasattr(x, 'shape') else len(x)
                return torch.ones(batch_size, 1, device=device) * 0.5

            def __call__(self, x):
                return self.predict_proba(x)

        return SimpleModel()

    @pytest.fixture
    def test_data(self):
        """Create test data."""
        np.random.seed(42)
        data = np.random.uniform(0, 1, (100, 3))
        return torch.from_numpy(data).float()

    def test_no_scaler(self, mock_lasso_results, simple_model, test_data):
        """Test pipeline with no scaler."""
        pipeline = PlottingPipeline(
            lasso_results=mock_lasso_results,
            group_manager=None,
            label_manager=None,
        )

        bundle = pipeline.prepare_plotting_bundle(
            x=test_data,
            model=simple_model,
            scaler=None,
            feature_names=['a', 'b', 'c'],
        )

        assert bundle.scaler is None

    def test_scaler_passthrough_no_collapse(self, mock_lasso_results, simple_model, test_data):
        """Test that scaler is passed through when no collapse."""
        sklearn_scaler = StandardScaler().fit(test_data.numpy())
        scaler = PRiSMScaler(scaler=sklearn_scaler)

        pipeline = PlottingPipeline(
            lasso_results=mock_lasso_results,
            group_manager=None,
            label_manager=None,
        )

        bundle = pipeline.prepare_plotting_bundle(
            x=test_data,
            model=simple_model,
            scaler=scaler,
            feature_names=['a', 'b', 'c'],
        )

        # Scaler should be unchanged
        assert bundle.scaler is scaler


class TestPlottingPipelineBetaScaling:
    """Test beta scaling edge cases."""

    @pytest.fixture
    def mock_lasso_results(self):
        """Mock LassoResultsManager."""

        class MockLassoResults:
            def __init__(self):
                self.univariate_feature_names = ['a', 'b', 'c']
                self.all_feature_names = generate_all_feature_names(self.univariate_feature_names)

            def get_selected_univariate_indices(self):
                return [0, 1, 2]

            def get_selected_bivariate_index_pairs(self):
                return [(0, 1), (1, 2)]

            def get_selected_beta(self):
                # FULL format: 3 univariate + 3 bivariate pairs = 6 total
                # Pairs: (0,1), (0,2), (1,2)
                # Selected: a=2.0, b=3.0, c=1.5, (0,1)=0.5, (1,2)=0.8
                beta = np.zeros(6)
                beta[0] = 2.0  # a
                beta[1] = 3.0  # b
                beta[2] = 1.5  # c
                beta[3 + 0] = 0.5  # (0,1): a-b, position 0 in bivariate
                beta[3 + 2] = 0.8  # (1,2): b-c, position 2 in bivariate
                return beta

        return MockLassoResults()

    @pytest.fixture
    def simple_model(self):
        """Simple mock model."""

        class SimpleModel:
            def predict_proba(self, x, device='cpu'):
                batch_size = x.shape[0] if hasattr(x, 'shape') else len(x)
                return torch.ones(batch_size, 1, device=device) * 0.5

            def __call__(self, x):
                return self.predict_proba(x)

        return SimpleModel()

    @pytest.fixture
    def test_data(self):
        """Create test data."""
        np.random.seed(42)
        data = np.random.uniform(0, 1, (100, 3))
        return torch.from_numpy(data).float()

    def test_beta_scaling_univariate(self, mock_lasso_results, simple_model, test_data):
        """Test beta scaling for univariate features."""
        pipeline = PlottingPipeline(
            lasso_results=mock_lasso_results,
            group_manager=None,
            label_manager=None,
        )

        bundle = pipeline.prepare_plotting_bundle(
            x=test_data,
            model=simple_model,
            feature_names=['a', 'b', 'c'],
        )

        # Store original values (by dense index)
        original_responses = [info.response.copy() for info in bundle.univariate_features()]

        # Apply beta scaling
        bundle = pipeline.apply_beta_scaling(bundle)

        # Check that scaling occurred correctly
        # Beta = [2.0, 3.0, 1.5, 0.5, 0.8]
        # First 3 are for univariate features
        beta = mock_lasso_results.get_selected_beta()
        for dense_idx, info in enumerate(bundle.univariate_features()):
            expected = original_responses[dense_idx] * beta[dense_idx]
            np.testing.assert_array_almost_equal(info.response, expected)

    def test_beta_scaling_bivariate(self, mock_lasso_results, simple_model, test_data):
        """Test beta scaling for bivariate features."""
        pipeline = PlottingPipeline(
            lasso_results=mock_lasso_results,
            group_manager=None,
            label_manager=None,
        )

        bundle = pipeline.prepare_plotting_bundle(
            x=test_data,
            model=simple_model,
            feature_names=['a', 'b', 'c'],
        )

        # Store original values (by dense index)
        original_responses = [info.response.copy() for info in bundle.bivariate_pairs()]

        # Apply beta scaling
        bundle = pipeline.apply_beta_scaling(bundle)

        # Check that bivariate responses were scaled correctly
        # Beta = [2.0, 3.0, 1.5, 0.5, 0.8]
        # First 3 are univariate, next 2 are bivariate
        beta = mock_lasso_results.get_selected_beta()
        n_univariate = bundle.n_univariate
        for dense_pair_idx, info in enumerate(bundle.bivariate_pairs()):
            beta_idx = n_univariate + dense_pair_idx
            expected = original_responses[dense_pair_idx] * beta[beta_idx]
            np.testing.assert_array_almost_equal(info.response, expected)
