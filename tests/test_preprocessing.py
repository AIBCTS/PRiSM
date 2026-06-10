"""Tests for preprocessing module."""

import numpy as np
import pandas as pd
import pytest

from prism.data_utilities import convert_to_categorical, enforce_binary_target_encoding
from prism.preprocessing import (
    MedianStdScaler,
    NoScaler,
    OneHotGroupManager,
    PRiSMScaler,
    collapse_onehot_features,
)


class TestNoScaler:
    """Tests for NoScaler (pass-through scaler)."""

    def test_fit_transform_passthrough(self):
        """Test that NoScaler passes data through unchanged."""
        scaler = NoScaler()
        data = np.array([[1, 2], [3, 4], [5, 6]])

        result = scaler.fit_transform(data)

        np.testing.assert_array_equal(result, data)

    def test_transform_passthrough(self):
        """Test that transform passes data through unchanged."""
        scaler = NoScaler()
        train_data = np.array([[1, 2], [3, 4]])
        test_data = np.array([[5, 6], [7, 8]])

        scaler.fit(train_data)
        result = scaler.transform(test_data)

        np.testing.assert_array_equal(result, test_data)

    def test_inverse_transform_passthrough(self):
        """Test that inverse_transform passes data through unchanged."""
        scaler = NoScaler()
        data = np.array([[1, 2], [3, 4]])

        scaler.fit(data)
        transformed = scaler.transform(data)
        result = scaler.inverse_transform(transformed)

        np.testing.assert_array_equal(result, data)

    def test_with_dataframe(self):
        """Test NoScaler with pandas DataFrame (converts to numpy)."""
        scaler = NoScaler()
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        result = scaler.fit_transform(df)

        # Scaler converts to numpy array
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, df.values)


class TestMedianStdScaler:
    """Tests for MedianStdScaler."""

    def test_fit_transform_basic(self):
        """Test basic fit_transform functionality."""
        scaler = MedianStdScaler()
        data = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)

        result = scaler.fit_transform(data)

        # Check that median is approximately 0 and std is approximately 1
        assert result.mean(axis=0)[0] == pytest.approx(0.0, abs=0.5)
        assert result.std(axis=0)[0] == pytest.approx(1.0, abs=0.3)

    def test_inverse_transform_roundtrip(self):
        """Test that inverse_transform recovers original data."""
        scaler = MedianStdScaler()
        data = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)

        transformed = scaler.fit_transform(data)
        recovered = scaler.inverse_transform(transformed)

        np.testing.assert_array_almost_equal(recovered, data, decimal=10)

    def test_transform_uses_training_statistics(self):
        """Test that transform uses statistics from fit."""
        scaler = MedianStdScaler()
        train_data = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        test_data = np.array([[10, 20], [30, 40]], dtype=float)

        scaler.fit(train_data)
        result = scaler.transform(test_data)

        # Result should be scaled using train statistics
        # Values should be large (since test data is much larger than train)
        assert np.abs(result).mean() > 1.0

    def test_with_single_feature(self):
        """Test scaler with single feature."""
        scaler = MedianStdScaler()
        data = np.array([[1], [2], [3], [4], [5]], dtype=float)

        result = scaler.fit_transform(data)

        # Should be properly scaled
        assert result.shape == data.shape


class TestPRiSMScaler:
    """Tests for PRiSMScaler."""

    def test_fit_transform_basic(self):
        """Test basic fit_transform functionality."""
        scaler = PRiSMScaler()
        data = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)

        result = scaler.fit_transform(data)

        # Result should be scaled (different from input)
        assert not np.array_equal(result, data)
        # Result should have same shape
        assert result.shape == data.shape

    def test_inverse_transform_roundtrip(self):
        """Test that inverse_transform recovers original data."""
        scaler = PRiSMScaler()
        data = np.array([[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]], dtype=float)

        transformed = scaler.fit_transform(data)
        recovered = scaler.inverse_transform(transformed)

        np.testing.assert_array_almost_equal(recovered, data, decimal=6)

    def test_fit_then_transform_separate(self):
        """Test calling fit and transform separately."""
        scaler = PRiSMScaler()
        data = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)

        scaler.fit(data)
        result = scaler.transform(data)

        # Should produce same result as fit_transform
        expected = scaler.fit_transform(data)
        np.testing.assert_array_almost_equal(result, expected, decimal=10)

    def test_with_constant_feature(self):
        """Test scaler with constant feature (zero variance)."""
        scaler = PRiSMScaler()
        # First column is constant
        data = np.array([[1, 2], [1, 4], [1, 6]], dtype=float)

        result = scaler.fit_transform(data)

        # Should handle constant feature without errors
        assert result.shape == data.shape
        # Constant feature should remain constant (no division by zero)
        assert np.isfinite(result).all()

    def test_with_dataframe(self):
        """Test PRiSMScaler with pandas DataFrame (converts to numpy)."""
        scaler = PRiSMScaler()
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})

        result = scaler.fit_transform(df)

        # Scaler converts to numpy array
        assert isinstance(result, np.ndarray)
        assert result.shape == df.shape

    def test_preserves_column_names_in_attribute(self):
        """Test that column names are stored in feature_names_ attribute."""
        scaler = PRiSMScaler()
        df = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6], "feature3": [7, 8, 9]})

        scaler.fit(df)

        # Feature names should be stored
        assert scaler.feature_names_ is not None
        assert list(scaler.feature_names_) == ["feature1", "feature2", "feature3"]


class TestEnforceBinaryTargetEncoding:
    """Tests for enforce_binary_target_encoding function."""

    def test_already_binary_integers(self):
        """Test with already binary (0, 1) integers."""
        df = pd.DataFrame({"target": [0, 1, 0, 1, 0], "feature": [1, 2, 3, 4, 5]})

        result = enforce_binary_target_encoding(df, "target")

        # Should remain unchanged
        assert result["target"].tolist() == [0, 1, 0, 1, 0]
        assert result["target"].dtype in [np.int32, np.int64]

    def test_binary_floats_converted(self):
        """Test conversion of binary floats (0.0, 1.0) to integers."""
        df = pd.DataFrame({"target": [0.0, 1.0, 0.0, 1.0, 0.0], "feature": [1, 2, 3, 4, 5]})

        result = enforce_binary_target_encoding(df, "target")

        # Should convert to integers
        assert result["target"].tolist() == [0, 1, 0, 1, 0]
        assert result["target"].dtype in [np.int32, np.int64]

    def test_string_labels_converted(self):
        """Test conversion of string labels to binary."""
        df = pd.DataFrame(
            {"target": ["Yes", "No", "Yes", "No", "Yes"], "feature": [1, 2, 3, 4, 5]}
        )

        result = enforce_binary_target_encoding(df, "target")

        # Should be converted to 0 and 1
        assert set(result["target"].unique()) == {0, 1}
        assert result["target"].dtype in [np.int32, np.int64]

    def test_non_binary_warns_and_returns_unchanged(self):
        """Test that non-binary target warns and returns unchanged."""
        df = pd.DataFrame({"target": [0, 1, 2, 1, 0], "feature": [1, 2, 3, 4, 5]})  # 3 classes

        # Should return dataframe unchanged (with warning)
        result = enforce_binary_target_encoding(df, "target")

        # Should have same values as input
        pd.testing.assert_frame_equal(result, df)

    def test_preserves_other_columns(self):
        """Test that other columns are not modified."""
        df = pd.DataFrame(
            {
                "target": [0, 1, 0, 1],
                "feature1": [1.5, 2.5, 3.5, 4.5],
                "feature2": ["A", "B", "C", "D"],
            }
        )

        result = enforce_binary_target_encoding(df, "target")

        # Other columns should be unchanged
        pd.testing.assert_series_equal(result["feature1"], df["feature1"])
        pd.testing.assert_series_equal(result["feature2"], df["feature2"])


class TestConvertToCategorical:
    """Tests for convert_to_categorical function."""

    def test_convert_low_cardinality_numeric(self):
        """Test conversion of low-cardinality numeric to categorical."""
        n = 100
        df = pd.DataFrame(
            {
                "low_card": [1, 2, 3] * (n // 3) + [1] * (n % 3),  # 3 unique values
                "high_card": list(range(n)),  # 100 unique values
            }
        )

        result = convert_to_categorical(df, categorical_threshold=10, convert_numeric=True)

        # low_card should be converted to categorical
        assert isinstance(result["low_card"].dtype, pd.CategoricalDtype)
        # high_card should remain numeric
        assert not isinstance(result["high_card"].dtype, pd.CategoricalDtype)

    def test_no_conversion_when_disabled(self):
        """Test that numeric conversion is disabled when convert_numeric=False."""
        df = pd.DataFrame(
            {
                "low_card": [1, 2, 3, 1, 2, 3],  # 3 unique values
            }
        )

        result = convert_to_categorical(df, categorical_threshold=10, convert_numeric=False)

        # Should not convert numeric columns
        assert result["low_card"].dtype != "category"

    def test_string_columns_converted(self):
        """Test that string columns are converted to categorical."""
        df = pd.DataFrame(
            {
                "string_col": ["A", "B", "C", "A", "B"],
                "numeric_col": [1, 2, 3, 4, 5],
            }
        )

        result = convert_to_categorical(df, categorical_threshold=10, convert_numeric=False)

        # String column should be categorical
        assert result["string_col"].dtype.name == "category"


@pytest.mark.integration
class TestPreprocessingIntegration:
    """Integration tests for preprocessing components."""

    def test_scaler_chaining(self):
        """Test using multiple scalers in sequence."""
        data = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)

        # Chain PRiSMScaler and MedianStdScaler
        scaler1 = PRiSMScaler()
        scaler2 = MedianStdScaler()

        intermediate = scaler1.fit_transform(data)
        result = scaler2.fit_transform(intermediate)

        # Should produce valid output
        assert result.shape == data.shape
        assert np.isfinite(result).all()

    def test_roundtrip_with_different_scalers(self):
        """Test roundtrip transformations with different scalers."""
        data = np.array([[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]], dtype=float)

        for ScalerClass in [NoScaler, MedianStdScaler, PRiSMScaler]:
            scaler = ScalerClass()
            transformed = scaler.fit_transform(data)
            recovered = scaler.inverse_transform(transformed)

            np.testing.assert_array_almost_equal(
                recovered, data, decimal=6, err_msg=f"Failed for {ScalerClass.__name__}"
            )


class TestOneHotGroupManager:
    """Tests for OneHotGroupManager class."""

    @pytest.fixture
    def sample_groups(self):
        """Create sample group structure for testing."""
        return {
            'diagn': ['diagn_CAD', 'diagn_Congenital', 'diagn_Graftfailure'],
            'recethcat': ['recethcat_White', 'recethcat_Black'],
        }

    @pytest.fixture
    def sample_reference_columns(self):
        """Create sample reference columns for testing."""
        return {'diagn': 'diagn_Cardiomyopathy', 'recethcat': 'recethcat_Hispanic'}

    @pytest.fixture
    def sample_feature_names(self):
        """Create sample feature names for testing."""
        return [
            'age',
            'bmi',
            'diagn_CAD',
            'diagn_Congenital',
            'diagn_Graftfailure',
            'recethcat_White',
            'recethcat_Black',
            'score',
        ]

    def test_init(self, sample_groups, sample_reference_columns):
        """Test OneHotGroupManager initialization."""
        manager = OneHotGroupManager(sample_groups, sample_reference_columns)

        assert manager.groups_dict == sample_groups
        assert manager.reference_columns == sample_reference_columns

    def test_init_no_reference(self, sample_groups):
        """Test OneHotGroupManager initialization without reference columns."""
        manager = OneHotGroupManager(sample_groups)

        assert manager.groups_dict == sample_groups
        assert manager.reference_columns == {}

    def test_to_indices(self, sample_groups, sample_reference_columns, sample_feature_names):
        """Test conversion from feature names to indices."""
        manager = OneHotGroupManager(sample_groups, sample_reference_columns)
        indices = manager.to_indices(sample_feature_names)

        # Should return two groups
        assert len(indices) == 2

        # First group (diagn): indices 2, 3, 4
        assert (2, 3, 4) in indices

        # Second group (recethcat): indices 5, 6
        assert (5, 6) in indices

    def test_to_indices_missing_features(self, sample_groups):
        """Test to_indices when some features are missing."""
        manager = OneHotGroupManager(sample_groups)
        feature_names = ['age', 'diagn_CAD', 'diagn_Congenital']  # Missing diagn_Graftfailure

        indices = manager.to_indices(feature_names)

        # Should still return diagn group (has 2 members)
        assert len(indices) == 1
        assert indices[0] == (1, 2)

    def test_to_indices_too_few_members(self, sample_groups):
        """Test to_indices when group has fewer than 2 members after filtering."""
        manager = OneHotGroupManager(sample_groups)
        feature_names = ['age', 'diagn_CAD', 'recethcat_White']  # Only 1 member per group

        indices = manager.to_indices(feature_names)

        # Should return None (no valid groups)
        assert indices is None

    def test_from_preprocessing_metadata(self, sample_groups, sample_reference_columns):
        """Test loading from preprocessing metadata."""
        metadata = {'onehot_groups': sample_groups, 'reference_columns': sample_reference_columns}
        manager = OneHotGroupManager.from_preprocessing_metadata(metadata)

        assert manager.groups_dict == sample_groups
        assert manager.reference_columns == sample_reference_columns


class TestCollapseOnehotFeatures:
    """Tests for collapse_onehot_features utility function."""

    @pytest.fixture
    def sample_manager(self):
        """Create sample manager for testing."""
        groups = {'diagn': ['diagn_CAD', 'diagn_Congenital'], 'race': ['race_White', 'race_Black']}
        references = {'diagn': 'diagn_Cardiomyopathy', 'race': 'race_Hispanic'}
        return OneHotGroupManager(groups, references)

    def test_collapse_basic(self, sample_manager):
        """Test basic collapsing of one-hot features."""
        # One-hot encoded data
        X = np.array(
            [
                [25.0, 1, 0, 1, 0, 100.0],  # CAD (1), White (1)
                [30.0, 0, 1, 0, 1, 120.0],  # Congenital (2), Black (2)
                [35.0, 0, 0, 0, 0, 110.0],  # Reference (0), Reference (0)
            ]
        )
        feature_names = [
            'age',
            'diagn_CAD',
            'diagn_Congenital',
            'race_White',
            'race_Black',
            'score',
        ]

        X_collapsed, collapsed_names = collapse_onehot_features(X, sample_manager, feature_names)

        # Check shape: 6 features -> 4 features (age, diagn, race, score)
        assert X_collapsed.shape == (3, 4)

        # Check collapsed names
        assert 'diagn' in collapsed_names
        assert 'race' in collapsed_names
        assert 'age' in collapsed_names
        assert 'score' in collapsed_names

        # Check categorical values
        # First row: diagn_CAD active (category 1), race_White active (category 1)
        diagn_idx = collapsed_names.index('diagn')
        race_idx = collapsed_names.index('race')
        assert X_collapsed[0, diagn_idx] == 1
        assert X_collapsed[0, race_idx] == 1

        # Second row: diagn_Congenital active (category 2), race_Black active (category 2)
        assert X_collapsed[1, diagn_idx] == 2
        assert X_collapsed[1, race_idx] == 2

        # Third row: both reference (category 0)
        assert X_collapsed[2, diagn_idx] == 0
        assert X_collapsed[2, race_idx] == 0

    def test_collapse_with_non_grouped_features(self, sample_manager):
        """Test that non-grouped features are copied unchanged."""
        X = np.array(
            [
                [25.0, 1, 0, 1, 0, 100.0],
                [30.0, 0, 1, 0, 1, 120.0],
            ]
        )
        feature_names = [
            'age',
            'diagn_CAD',
            'diagn_Congenital',
            'race_White',
            'race_Black',
            'score',
        ]

        X_collapsed, collapsed_names = collapse_onehot_features(X, sample_manager, feature_names)

        # Check that non-grouped features are preserved
        age_idx = collapsed_names.index('age')
        score_idx = collapsed_names.index('score')

        np.testing.assert_array_equal(X_collapsed[:, age_idx], [25.0, 30.0])
        np.testing.assert_array_equal(X_collapsed[:, score_idx], [100.0, 120.0])

    def test_collapse_with_pandas_dataframe(self, sample_manager):
        """Test collapse with pandas DataFrame input."""
        X = pd.DataFrame(
            {
                'age': [25.0, 30.0],
                'diagn_CAD': [1, 0],
                'diagn_Congenital': [0, 1],
                'race_White': [1, 0],
                'race_Black': [0, 1],
                'score': [100.0, 120.0],
            }
        )
        feature_names = X.columns.tolist()

        X_collapsed, collapsed_names = collapse_onehot_features(X, sample_manager, feature_names)

        # Should return numpy array
        assert isinstance(X_collapsed, np.ndarray)
        assert X_collapsed.shape == (2, 4)


class TestCollapseIntegration:
    """Integration tests for collapse functionality with partial responses."""

    def test_reference_values_present_in_collapsed_data(self):
        """Test that reference state (0 values) appears in collapsed categorical columns."""
        # Create data with reference states (all zeros in one-hot group)
        X = np.array(
            [
                [25.0, 1, 0, 1, 0, 100.0],  # CAD=1, White=1
                [30.0, 0, 1, 0, 1, 120.0],  # Congenital=2, Black=2
                [35.0, 0, 0, 0, 0, 110.0],  # Reference=0, Reference=0
            ]
        )
        feature_names = [
            'age',
            'diagn_CAD',
            'diagn_Congenital',
            'race_White',
            'race_Black',
            'score',
        ]

        groups = {'diagn': ['diagn_CAD', 'diagn_Congenital'], 'race': ['race_White', 'race_Black']}
        references = {'diagn': 'diagn_Cardiomyopathy', 'race': 'race_Hispanic'}
        manager = OneHotGroupManager(groups, references)

        X_collapsed, collapsed_names = collapse_onehot_features(X, manager, feature_names)

        # Check that reference values (0) are present
        diagn_idx = collapsed_names.index('diagn')
        race_idx = collapsed_names.index('race')

        assert 0 in X_collapsed[:, diagn_idx], "Reference value 0 should be present in diagn"
        assert 0 in X_collapsed[:, race_idx], "Reference value 0 should be present in race"

        # Verify the reference row has 0 for both categorical variables
        assert X_collapsed[2, diagn_idx] == 0, "Third row should have diagn=0 (reference)"
        assert X_collapsed[2, race_idx] == 0, "Third row should have race=0 (reference)"

    def test_categorical_values_range(self):
        """Test that collapsed categorical columns have correct value range."""
        X = np.array(
            [
                [1, 0, 0, 0],  # Cat 1
                [0, 1, 0, 0],  # Cat 2
                [0, 0, 1, 0],  # Cat 3
                [0, 0, 0, 1],  # Cat 4
                [0, 0, 0, 0],  # Reference (Cat 0)
            ]
        )
        feature_names = ['var_A', 'var_B', 'var_C', 'var_D']

        groups = {'var': ['var_A', 'var_B', 'var_C', 'var_D']}
        references = {'var': 'var_Reference'}
        manager = OneHotGroupManager(groups, references)

        X_collapsed, collapsed_names = collapse_onehot_features(X, manager, feature_names)

        var_idx = collapsed_names.index('var')
        unique_values = np.unique(X_collapsed[:, var_idx])

        # Should have values 0 (reference) through 4 (4 active categories)
        expected = np.array([0, 1, 2, 3, 4])
        np.testing.assert_array_equal(unique_values, expected)

    def test_collapse_with_partial_responses_integration(self):
        """Integration test: collapse with partial responses."""
        import torch

        from prism.partial_responses import partial_responses

        # Create simple data
        X = torch.tensor(
            [
                [25.0, 1, 0, 1, 0, 100.0],
                [30.0, 0, 1, 0, 1, 120.0],
                [35.0, 0, 0, 0, 0, 110.0],
                [40.0, 1, 0, 0, 0, 130.0],
            ],
            dtype=torch.float32,
        )

        feature_names = [
            'age',
            'diagn_CAD',
            'diagn_Congenital',
            'race_White',
            'race_Black',
            'score',
        ]

        groups = {'diagn': ['diagn_CAD', 'diagn_Congenital'], 'race': ['race_White', 'race_Black']}
        references = {'diagn': 'diagn_Cardiomyopathy', 'race': 'race_Hispanic'}
        manager = OneHotGroupManager(groups, references)

        # Simple mock model
        class SimpleMockModel:
            def predict_proba(self, x, device=None):
                return torch.sigmoid(x.mean(dim=1, keepdim=True))

        model = SimpleMockModel()

        # Calculate partial responses with collapsing
        pr = partial_responses(
            X,
            model,
            x_train=X,
            method='lebesgue',
            device='cpu',
            batch_size=2,
            group_manager=manager,
            feature_names=feature_names,
            scaler=NoScaler(),
        )

        # Expected dimensions:
        # Univariate: 4 collapsed (diagn, race, age, score)
        # Bivariate: 6 collapsed (from 4 collapsed features: 4*3/2)
        expected_univariate = 4
        expected_bivariate = expected_univariate * (expected_univariate - 1) // 2  # 6
        expected_total = expected_univariate + expected_bivariate  # 10

        assert pr.shape == (
            4,
            expected_total,
        ), f"Expected shape (4, {expected_total}), got {pr.shape}"

    def test_manager_from_metadata_roundtrip(self):
        """Test loading OneHotGroupManager from metadata structure."""
        # Simulate metadata structure as saved to JSON
        metadata = {
            'onehot_group_manager': {
                '_type': 'OneHotGroupManager',
                'groups_dict': {
                    'diagn': ['diagn_CAD', 'diagn_Congenital'],
                    'race': ['race_White', 'race_Black'],
                },
                'reference_columns': {'diagn': 'diagn_Cardiomyopathy', 'race': 'race_Hispanic'},
            }
        }

        # Load from metadata
        manager = OneHotGroupManager.from_preprocessing_metadata(metadata)

        assert manager.groups_dict == metadata['onehot_group_manager']['groups_dict']
        assert manager.reference_columns == metadata['onehot_group_manager']['reference_columns']

        # Verify it works for collapsing
        X = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 0, 0]])
        feature_names = ['diagn_CAD', 'diagn_Congenital', 'race_White', 'race_Black']

        X_collapsed, collapsed_names = collapse_onehot_features(X, manager, feature_names)

        assert X_collapsed.shape == (3, 2)
        assert collapsed_names == ['diagn', 'race']
        assert 0 in X_collapsed[:, 0]  # Reference present in diagn
        assert 0 in X_collapsed[:, 1]  # Reference present in race

    def test_bivariate_collapse_dimensions(self):
        """Test that bivariate partial responses are collapsed correctly."""
        import torch

        from prism.partial_responses import partial_responses

        # Create sample data with one-hot groups
        # 2 regular features + 2 one-hot groups (2 members each) = 6 features total
        X = torch.tensor(
            [
                [25.0, 100.0, 1, 0, 1, 0],  # age=25, score=100, diagn=CAD, race=White
                [30.0, 120.0, 0, 1, 0, 1],  # age=30, score=120, diagn=Congenital, race=Black
                [35.0, 110.0, 0, 0, 0, 0],  # age=35, score=110, diagn=ref, race=ref
            ],
            dtype=torch.float32,
        )

        feature_names = [
            'age',
            'score',
            'diagn_CAD',
            'diagn_Congenital',
            'race_White',
            'race_Black',
        ]

        groups = {'diagn': ['diagn_CAD', 'diagn_Congenital'], 'race': ['race_White', 'race_Black']}
        references = {'diagn': 'diagn_Cardiomyopathy', 'race': 'race_Hispanic'}
        manager = OneHotGroupManager(groups, references)

        # Simple mock model
        class SimpleMockModel:
            def predict_proba(self, x, device=None):
                return torch.sigmoid(x.mean(dim=1, keepdim=True))

        model = SimpleMockModel()

        # Calculate partial responses with collapsing
        pr = partial_responses(
            X,
            model,
            x_train=X,
            method='lebesgue',
            device='cpu',
            batch_size=10,
            group_manager=manager,
            feature_names=feature_names,
            scaler=NoScaler(),
        )

        # Expected dimensions:
        # Original: 6 features → 6 univariate + 6*5/2=15 bivariate = 21 total
        # Collapsed: 4 features (age, score, diagn, race) → 4 univariate + 4*3/2=6 bivariate = 10 total
        n_univariate_collapsed = 4  # age, score, diagn, race
        n_bivariate_collapsed = n_univariate_collapsed * (n_univariate_collapsed - 1) // 2  # 6
        expected_total = n_univariate_collapsed + n_bivariate_collapsed  # 10

        assert pr.shape == (
            3,
            expected_total,
        ), f"Expected shape (3, {expected_total}), got {pr.shape}"

        # Verify dimension reduction
        n_original_univariate = 6
        n_original_bivariate = 6 * 5 // 2  # 15
        original_total = n_original_univariate + n_original_bivariate  # 21
        reduction = 100 * (1 - expected_total / original_total)

        assert reduction > 0, "Collapsing should reduce dimensions"
        assert (
            pr.shape[1] == expected_total
        ), f"Total dimensions should be {expected_total}, got {pr.shape[1]}"

    def test_encoding_order_consistency(self):
        """
        CRITICAL: Test that collapse_onehot_features and partial_responses use
        the same category integer encoding order.

        Both must use groups_dict order explicitly:
        - Category 1 = groups_dict[group_name][0]
        - Category 2 = groups_dict[group_name][1]
        - etc.

        This test verifies that when feature_names are in a different order
        than groups_dict, both functions still produce consistent encodings.
        """
        import torch

        from prism.partial_responses import PartialResponseCalculator

        # Create groups_dict with specific order
        groups = {'diagn': ['diagn_CAD', 'diagn_Congenital', 'diagn_Valve']}  # Specific order
        references = {'diagn': 'diagn_Cardiomyopathy'}
        manager = OneHotGroupManager(groups, references)

        # Create feature_names in DIFFERENT order (alphabetical)
        feature_names = sorted(['diagn_CAD', 'diagn_Congenital', 'diagn_Valve'])
        # feature_names = ['diagn_CAD', 'diagn_Congenital', 'diagn_Valve'] - alphabetically same
        # But let's make them explicitly different:
        feature_names = ['diagn_Valve', 'diagn_CAD', 'diagn_Congenital']  # Different order!

        # Create test data: each row has one active category
        # Row 0: diagn_Valve active
        # Row 1: diagn_CAD active
        # Row 2: diagn_Congenital active
        X = np.array(
            [
                [1, 0, 0],  # diagn_Valve (index 0 in feature_names)
                [0, 1, 0],  # diagn_CAD (index 1 in feature_names)
                [0, 0, 1],  # diagn_Congenital (index 2 in feature_names)
                [0, 0, 0],  # Reference
            ]
        )

        # Collapse using collapse_onehot_features
        X_collapsed, collapsed_names = collapse_onehot_features(X, manager, feature_names)

        # The encoding should follow groups_dict order, NOT feature_names order:
        # groups_dict['diagn'] = ['diagn_CAD', 'diagn_Congenital', 'diagn_Valve']
        # So: diagn_CAD -> 1, diagn_Congenital -> 2, diagn_Valve -> 3

        # Row 0: diagn_Valve active -> should be category 3
        # Row 1: diagn_CAD active -> should be category 1
        # Row 2: diagn_Congenital active -> should be category 2
        # Row 3: Reference -> should be category 0

        expected_categories = [3, 1, 2, 0]  # Based on groups_dict order

        actual_categories = X_collapsed[:, 0].astype(int).tolist()

        assert (
            actual_categories == expected_categories
        ), f"collapse_onehot_features encoding mismatch: expected {expected_categories}, got {actual_categories}"

        # Now verify PartialResponseCalculator._create_collapsed_mapping uses same order
        class DummyModel:
            def predict_proba(self, x, device=None):
                return torch.zeros((x.shape[0], 1))

        calc = PartialResponseCalculator(
            DummyModel(),
            input_dim=len(feature_names),
            x_train=torch.tensor(X, dtype=torch.float32),
            group_manager=manager,
            feature_names=feature_names,
            device='cpu',
            scaler=NoScaler(),
        )

        # Check the index_mapping - it should be in groups_dict order
        diagn_collapsed_idx = collapsed_names.index('diagn')
        mapped_indices = calc.index_mapping[diagn_collapsed_idx]

        # mapped_indices should be in groups_dict order:
        # [feature_names.index('diagn_CAD'), feature_names.index('diagn_Congenital'), feature_names.index('diagn_Valve')]
        # = [1, 2, 0]
        expected_mapped_indices = [feature_names.index(f) for f in groups['diagn']]

        assert (
            list(mapped_indices) == expected_mapped_indices
        ), f"_create_collapsed_mapping order mismatch: expected {expected_mapped_indices}, got {list(mapped_indices)}"

    def test_create_collapsed_scaler_basic(self):
        """Test creating a collapsed scaler from original scaler."""
        # Create sample data with one-hot groups
        X = np.array(
            [
                [25.0, 100.0, 1, 0, 1, 0],
                [30.0, 120.0, 0, 1, 0, 1],
                [35.0, 110.0, 0, 0, 0, 0],
            ]
        )
        feature_names = [
            'age',
            'score',
            'diagn_CAD',
            'diagn_Congenital',
            'race_White',
            'race_Black',
        ]

        # Create and fit original scaler (6 features)
        scaler = PRiSMScaler()
        scaler.fit(X)

        # Create OneHotGroupManager
        groups = {'diagn': ['diagn_CAD', 'diagn_Congenital'], 'race': ['race_White', 'race_Black']}
        references = {'diagn': 'diagn_Cardiomyopathy', 'race': 'race_Hispanic'}
        manager = OneHotGroupManager(groups, references)

        # Create collapsed scaler
        collapsed_scaler = manager.create_collapsed_scaler(scaler, feature_names)

        # Get collapsed feature names to understand ordering
        _, collapsed_names = collapse_onehot_features(X, manager, feature_names)

        # Verify scaler has correct dimensions
        assert collapsed_scaler.scaler.median_.shape[0] == len(
            collapsed_names
        ), f"Should have {len(collapsed_names)} collapsed features"
        assert collapsed_scaler.scaler.std_.shape[0] == len(
            collapsed_names
        ), f"Should have {len(collapsed_names)} collapsed features"

        # Verify continuous features have correct scaling parameters
        age_collapsed_idx = collapsed_names.index('age')
        age_original_idx = feature_names.index('age')
        assert np.isclose(
            collapsed_scaler.scaler.median_[age_collapsed_idx],
            scaler.scaler.median_[age_original_idx],
        ), "Age median should match original"
        assert np.isclose(
            collapsed_scaler.scaler.std_[age_collapsed_idx], scaler.scaler.std_[age_original_idx]
        ), "Age std should match original"

        # Verify categorical features use identity scaling
        # With sd_scale=2.0 (default), we need std=0.5 so inverse_transform gives identity:
        # inverse_transform(x) = x * std * sd_scale = x * 0.5 * 2.0 = x
        diagn_idx = collapsed_names.index('diagn')
        expected_cat_std = 1.0 / scaler.scaler.sd_scale  # 1/2 = 0.5 for default
        assert (
            collapsed_scaler.scaler.median_[diagn_idx] == 0.0
        ), "Categorical feature should have median=0"
        assert np.isclose(
            collapsed_scaler.scaler.std_[diagn_idx], expected_cat_std
        ), f"Categorical feature should have std={expected_cat_std} for identity transform"

    def test_create_collapsed_scaler_with_noscaler(self):
        """Test creating collapsed scaler when original is NoScaler."""
        feature_names = ['age', 'score', 'diagn_CAD', 'diagn_Congenital']

        # Create NoScaler
        scaler = PRiSMScaler(NoScaler())

        # Create OneHotGroupManager
        groups = {'diagn': ['diagn_CAD', 'diagn_Congenital']}
        references = {'diagn': 'diagn_Cardiomyopathy'}
        manager = OneHotGroupManager(groups, references)

        # Create collapsed scaler
        collapsed_scaler = manager.create_collapsed_scaler(scaler, feature_names)

        # Should return a NoScaler
        assert isinstance(
            collapsed_scaler.scaler, NoScaler
        ), "Should return NoScaler when original is NoScaler"

    def test_create_collapsed_scaler_dimensions_match(self):
        """Test that collapsed scaler dimensions match collapsed feature count."""
        # Create sample data with proper one-hot encoding
        # Structure: f1, f2, cat1_A, cat1_B, cat1_C, cat2_X, cat2_Y, f3
        n_samples = 10
        X = np.zeros((n_samples, 8))
        X[:, 0] = np.random.randn(n_samples)  # f1: continuous
        X[:, 1] = np.random.randn(n_samples)  # f2: continuous
        # cat1: one-hot (reference would be all zeros = cat1_D)
        for i in range(n_samples):
            if i % 4 < 3:  # 75% have a category set
                X[i, 2 + (i % 3)] = 1  # Set one of cat1_A, cat1_B, cat1_C
        # cat2: one-hot (reference would be all zeros = cat2_Z)
        for i in range(n_samples):
            if i % 3 < 2:  # 67% have a category set
                X[i, 5 + (i % 2)] = 1  # Set one of cat2_X, cat2_Y
        X[:, 7] = np.random.randn(n_samples)  # f3: continuous

        feature_names = ['f1', 'f2', 'cat1_A', 'cat1_B', 'cat1_C', 'cat2_X', 'cat2_Y', 'f3']

        # Create and fit original scaler (8 features)
        scaler = PRiSMScaler()
        scaler.fit(X)

        # Create OneHotGroupManager
        groups = {'cat1': ['cat1_A', 'cat1_B', 'cat1_C'], 'cat2': ['cat2_X', 'cat2_Y']}
        references = {'cat1': 'cat1_D', 'cat2': 'cat2_Z'}
        manager = OneHotGroupManager(groups, references)

        # Get collapsed feature count (8 → 5: f1, f2, cat1, cat2, f3)
        _, collapsed_names = collapse_onehot_features(X, manager, feature_names)

        # Create collapsed scaler
        collapsed_scaler = manager.create_collapsed_scaler(scaler, feature_names)

        # Verify dimensions match
        assert collapsed_scaler.scaler.median_.shape[0] == len(
            collapsed_names
        ), f"Scaler should have {len(collapsed_names)} features"
        assert collapsed_scaler.scaler.std_.shape[0] == len(
            collapsed_names
        ), f"Scaler should have {len(collapsed_names)} features"


class TestDetectReferenceColumns:
    """Tests for detect_reference_columns using encoding_metadata."""

    def test_detect_reference_columns_uses_encoding_metadata(self):
        """Verify detect_reference_columns uses encoding_metadata correctly."""
        from prism.preprocessing import detect_reference_columns

        df = pd.DataFrame(
            {
                'diagn_A': [1, 0, 0],
                'diagn_B': [0, 1, 0],
                'diagn_C': [0, 0, 1],
            }
        )
        encoding_metadata = {
            'diagn': {
                'encoding_type': 'one-hot',
                'created_columns': ['diagn_A', 'diagn_B', 'diagn_C'],
            }
        }

        groups, refs = detect_reference_columns(df, encoding_metadata)

        assert 'diagn' in groups
        assert len(groups['diagn']) == 3
        assert refs['diagn'] == 'diagn_A'  # alphabetically first

    def test_detect_reference_columns_most_common_strategy(self):
        """Verify most_common reference column strategy works."""
        from prism.preprocessing import detect_reference_columns

        df = pd.DataFrame(
            {
                'cat_A': [1, 1, 1, 0, 0],  # 3 occurrences (most common)
                'cat_B': [0, 0, 0, 1, 0],  # 1 occurrence
                'cat_C': [0, 0, 0, 0, 1],  # 1 occurrence
            }
        )
        encoding_metadata = {
            'cat': {
                'encoding_type': 'one-hot',
                'created_columns': ['cat_A', 'cat_B', 'cat_C'],
            }
        }

        groups, refs = detect_reference_columns(
            df, encoding_metadata, reference_column_strategy='most_common'
        )

        assert refs['cat'] == 'cat_A'  # most common

    def test_detect_reference_columns_manual_strategy(self):
        """Verify manual reference column strategy works."""
        from prism.preprocessing import detect_reference_columns

        df = pd.DataFrame(
            {
                'cat_A': [1, 0, 0],
                'cat_B': [0, 1, 0],
                'cat_C': [0, 0, 1],
            }
        )
        encoding_metadata = {
            'cat': {
                'encoding_type': 'one-hot',
                'created_columns': ['cat_A', 'cat_B', 'cat_C'],
            }
        }

        groups, refs = detect_reference_columns(
            df,
            encoding_metadata,
            reference_column_strategy='manual',
            manual_reference_columns={'cat': 'cat_B'},
        )

        assert refs['cat'] == 'cat_B'

    def test_detect_reference_columns_skips(self):
        """Verify ordinal encoded columns are not treated as one-hot groups."""
        from prism.preprocessing import detect_reference_columns

        df = pd.DataFrame(
            {
                'severity': [0, 1, 2],
                'diagn_A': [1, 0, 0],
                'diagn_B': [0, 1, 1],
            }
        )
        encoding_metadata = {
            'severity': {
                'encoding_type': 'ordinal',
                'created_columns': ['severity'],
            },
            'diagn': {
                'encoding_type': 'one-hot',
                'created_columns': ['diagn_A', 'diagn_B'],
            },
        }

        groups, refs = detect_reference_columns(df, encoding_metadata)

        # Only diagn should be detected, not severity
        assert 'diagn' in groups
        assert 'severity' not in groups

    def test_detect_reference_columns_no_false_positives_with_shared_prefix(self):
        """Verify features with shared prefix are NOT incorrectly grouped.

        This is the key test that validates the fix for the DAT_* issue.
        When using encoding_metadata, continuous features like DAT_AGE, DAT_TBILI
        should NOT be grouped together as a one-hot group.
        """
        from prism.preprocessing import detect_reference_columns

        # Simulate a dataframe with features that share prefix but are NOT one-hot
        df = pd.DataFrame(
            {
                'DAT_AGE': [25, 30, 35, 40],
                'DAT_TBILI': [1.0, 1.2, 0.9, 1.1],
                'DAT_ISCHTIME': [3.5, 4.0, 3.2, 5.1],
                'diagn_A': [1, 0, 0, 1],
                'diagn_B': [0, 1, 1, 0],
            }
        )

        # Encoding metadata only contains the actual categorical column
        encoding_metadata = {
            'diagn': {
                'encoding_type': 'one-hot',
                'created_columns': ['diagn_A', 'diagn_B'],
            }
            # Note: DAT_AGE, DAT_TBILI, DAT_ISCHTIME are NOT in encoding_metadata
            # because they are continuous features, not categorical
        }

        groups, refs = detect_reference_columns(df, encoding_metadata)

        # Should have 'diagn' group but NOT 'DAT' group
        assert 'diagn' in groups
        assert 'DAT' not in groups
        assert len(groups) == 1  # Only one group

    def test_detect_reference_columns_empty_metadata(self):
        """Verify empty encoding_metadata returns empty groups."""
        from prism.preprocessing import detect_reference_columns

        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        encoding_metadata = {}

        groups, refs = detect_reference_columns(df, encoding_metadata)

        assert groups == {}
        assert refs == {}

    def test_detect_reference_columns_min_group_size(self):
        """Verify min_group_size parameter is respected."""
        from prism.preprocessing import detect_reference_columns

        df = pd.DataFrame(
            {
                'single_A': [1, 0],  # Only one column, should not form a group
                'pair_A': [1, 0],
                'pair_B': [0, 1],
            }
        )
        encoding_metadata = {
            'single': {
                'encoding_type': 'one-hot',
                'created_columns': ['single_A'],  # Only 1 column
            },
            'pair': {
                'encoding_type': 'one-hot',
                'created_columns': ['pair_A', 'pair_B'],  # 2 columns
            },
        }

        groups, refs = detect_reference_columns(df, encoding_metadata, min_group_size=2)

        # 'single' should not be included (only 1 column)
        # 'pair' should be included (2 columns)
        assert 'single' not in groups
        assert 'pair' in groups


class TestDataProvenanceInMetadata:
    """Tests for data provenance in preprocessing metadata."""

    def test_metadata_includes_input_hash(self, tmp_path):
        """Test that preprocessing metadata includes input file hash."""
        import pandas as pd

        from prism.preprocessing import preprocess_data

        # Create test CSV file
        test_file = tmp_path / "test_input.csv"
        df = pd.DataFrame(
            {'numeric_col': [1, 2, 3, 4, 5], 'category_col': ['A', 'B', 'A', 'B', 'A']}
        )
        df.to_csv(test_file, index=False)

        # Preprocess with input file path
        processed_df, metadata = preprocess_data(df, input_file_path=test_file, random_state=42)

        # Verify data_provenance structure
        assert 'data_provenance' in metadata
        assert 'hash_algorithm' in metadata['data_provenance']
        assert metadata['data_provenance']['hash_algorithm'] == 'sha256'

        # Verify input hash info
        assert 'input' in metadata['data_provenance']
        input_info = metadata['data_provenance']['input']
        assert input_info is not None
        assert 'hash' in input_info
        assert 'size_bytes' in input_info
        assert input_info['hash'] is not None
        assert input_info['size_bytes'] > 0

    def test_metadata_without_input_path(self):
        """Test that preprocessing works without input file path (backward compatibility)."""
        import pandas as pd

        from prism.preprocessing import preprocess_data

        df = pd.DataFrame(
            {'numeric_col': [1, 2, 3, 4, 5], 'category_col': ['A', 'B', 'A', 'B', 'A']}
        )

        # Preprocess without input file path
        processed_df, metadata = preprocess_data(df, random_state=42)

        # Should still work, but input hash is None
        assert 'data_provenance' in metadata
        assert metadata['data_provenance']['input'] is None

    def test_hash_matches_expected_value(self, tmp_path):
        """Test that computed hash matches expected value."""
        import hashlib

        import pandas as pd

        from prism.preprocessing import preprocess_data

        # Create test CSV file
        test_file = tmp_path / "test_input.csv"
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        df.to_csv(test_file, index=False)

        # Compute expected hash manually
        with open(test_file, 'rb') as f:
            expected_hash = hashlib.sha256(f.read()).hexdigest()

        # Preprocess with input file path
        processed_df, metadata = preprocess_data(df, input_file_path=test_file, random_state=42)

        # Verify hash matches
        assert metadata['data_provenance']['input']['hash'] == expected_hash
