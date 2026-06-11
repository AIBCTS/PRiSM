"""
Tests for JSON export functionality.
"""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from prism.plotting.json_export import (
    JSON_SCHEMA_VERSION,
    _compute_histogram_stats,
    _convert_to_json_serializable,
    save_nomogram_json,
)
from prism.plotting_data import FeatureInfo, FeaturePairInfo, PlottingDataBundle


class TestConvertToJsonSerializable:
    """Test the _convert_to_json_serializable helper function."""

    def test_numpy_array_to_list(self):
        """Test numpy arrays are converted to lists."""
        arr = np.array([1, 2, 3])
        result = _convert_to_json_serializable(arr)
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_numpy_2d_array_to_nested_list(self):
        """Test 2D numpy arrays become nested lists."""
        arr = np.array([[1, 2], [3, 4]])
        result = _convert_to_json_serializable(arr)
        assert result == [[1, 2], [3, 4]]

    def test_numpy_integer_to_int(self):
        """Test numpy integers become Python ints."""
        val = np.int64(42)
        result = _convert_to_json_serializable(val)
        assert result == 42
        assert isinstance(result, int)

    def test_numpy_float_to_float(self):
        """Test numpy floats become Python floats."""
        val = np.float64(3.14)
        result = _convert_to_json_serializable(val)
        assert result == pytest.approx(3.14)
        assert isinstance(result, float)

    def test_numpy_bool_to_bool(self):
        """Test numpy bools become Python bools."""
        val = np.bool_(True)
        result = _convert_to_json_serializable(val)
        assert result is True
        assert isinstance(result, bool)

    def test_nested_dict_conversion(self):
        """Test nested dicts with numpy values are converted."""
        data = {
            'arr': np.array([1, 2, 3]),
            'nested': {
                'val': np.int64(5),
                'float': np.float32(2.5),
            },
        }
        result = _convert_to_json_serializable(data)
        assert result == {
            'arr': [1, 2, 3],
            'nested': {
                'val': 5,
                'float': pytest.approx(2.5),
            },
        }

    def test_list_of_numpy_values(self):
        """Test lists containing numpy values are converted."""
        data = [np.int64(1), np.float64(2.0), np.array([3, 4])]
        result = _convert_to_json_serializable(data)
        assert result == [1, 2.0, [3, 4]]

    def test_primitive_types_unchanged(self):
        """Test primitive types pass through unchanged."""
        assert _convert_to_json_serializable(42) == 42
        assert _convert_to_json_serializable(3.14) == 3.14
        assert _convert_to_json_serializable("hello") == "hello"
        assert _convert_to_json_serializable(None) is None


class TestComputeHistogramStats:
    """Test the _compute_histogram_stats helper function."""

    def test_continuous_histogram(self):
        """Test histogram computation for continuous data."""
        data = np.linspace(0, 100, 1000)
        result = _compute_histogram_stats(data, is_categorical=False, n_bins=10)

        assert 'bin_edges' in result
        assert 'counts' in result
        assert len(result['bin_edges']) == 11  # n_bins + 1 edges
        assert len(result['counts']) == 10
        assert sum(result['counts']) == 1000

    def test_categorical_histogram(self):
        """Test histogram computation for categorical data."""
        data = np.array([0, 0, 0, 1, 1, 2])
        result = _compute_histogram_stats(data, is_categorical=True)

        assert 'categories' in result
        assert 'counts' in result
        assert result['categories'] == [0, 1, 2]
        assert result['counts'] == [3, 2, 1]

    def test_empty_data(self):
        """Test handling of empty data."""
        data = np.array([])

        # Continuous
        result = _compute_histogram_stats(data, is_categorical=False)
        assert result == {'bin_edges': [], 'counts': []}

        # Categorical
        result = _compute_histogram_stats(data, is_categorical=True)
        assert result == {'categories': [], 'counts': []}

    def test_nan_values_filtered(self):
        """Test that NaN values are filtered out."""
        data = np.array([1.0, 2.0, np.nan, 3.0, np.nan])
        result = _compute_histogram_stats(data, is_categorical=False, n_bins=3)

        assert sum(result['counts']) == 3  # Only non-NaN values counted

    def test_custom_bin_count(self):
        """Test custom number of bins."""
        data = np.linspace(0, 100, 100)
        result = _compute_histogram_stats(data, is_categorical=False, n_bins=5)

        assert len(result['bin_edges']) == 6  # 5 bins + 1
        assert len(result['counts']) == 5


class TestSaveNomogramJson:
    """Test the save_nomogram_json function."""

    @pytest.fixture
    def mock_lasso_results(self):
        """Create a mock LassoResultsManager."""
        mock = MagicMock()
        mock.base_model_name = "test_model"

        # Mock the selected model with intercept
        selected_model = MagicMock()
        selected_model.intercept_ = np.array([-0.5])
        mock.get_selected_model.return_value = selected_model

        # Mock lambda values
        mock.selected_lambda_index = 42
        # Create a lambdas array where index 42 has value 0.001
        lambdas = np.zeros(100)
        lambdas[42] = 0.001
        mock.lambdas = lambdas

        # Mock beta coefficients: 2 univariate + 1 bivariate = 3 total
        # Beta vector layout: [univ_0, univ_1, biv_(0,1)]
        mock.get_selected_beta.return_value = np.array([0.3, -0.2, 0.1])

        # n_univ must match the number of univariate features so that
        # the bivariate beta index is computed correctly:
        #   beta_idx = n_univ + pair_offset
        # For pair (0, 1) with n_univ=2: offset = 0*2 - 0*1//2 + (1-0-1) = 0
        #   => beta_idx = 2 + 0 = 2  =>  beta_vector[2] = 0.1
        mock.n_univ = 2

        return mock

    @pytest.fixture
    def mock_bundle(self):
        """Create a mock PlottingDataBundle with test data."""
        bundle = PlottingDataBundle(
            all_feature_names=['age', 'income', 'age_income'],
            n_steps=50,
            categorical_threshold=15,
        )

        # Add x_data for histogram computation
        bundle.x_data = np.column_stack(
            [
                np.random.uniform(20, 80, 100),  # age
                np.random.uniform(30000, 100000, 100),  # income
            ]
        )

        # Add univariate features
        bundle._univariate_info = [
            FeatureInfo(
                index=0,
                name='age',
                label='Age (years)',
                is_categorical=False,
                response=np.linspace(-0.5, 0.5, 50),
                x_values=np.linspace(20, 80, 50),
            ),
            FeatureInfo(
                index=1,
                name='income',
                label='Income ($)',
                is_categorical=False,
                response=np.linspace(-0.3, 0.3, 50),
                x_values=np.linspace(30000, 100000, 50),
            ),
        ]

        # Add bivariate pairs
        bundle._bivariate_info = [
            FeaturePairInfo(
                indices=(0, 1),
                names=('age', 'income'),
                labels=('Age (years)', 'Income ($)'),
                is_categorical=(False, False),
                response=np.linspace(-0.1, 0.1, 100),
                x_values=np.column_stack(
                    [
                        np.repeat(np.linspace(20, 80, 10), 10),
                        np.tile(np.linspace(30000, 100000, 10), 10),
                    ]
                ),
                skipped=False,
            ),
        ]

        return bundle

    def test_save_creates_valid_json(self, mock_bundle, mock_lasso_results, tmp_path):
        """Test that save_nomogram_json creates a valid JSON file."""
        file_path = tmp_path / "test_nomogram.json"

        result_path = save_nomogram_json(
            bundle=mock_bundle,
            lasso_results=mock_lasso_results,
            file_path=file_path,
            comment="Test comment",
            method="dirac",
        )

        assert result_path.exists()
        assert result_path.suffix == '.json'

        # Verify it's valid JSON
        with open(result_path, 'r') as f:
            data = json.load(f)

        assert isinstance(data, dict)

    def test_json_schema_version(self, mock_bundle, mock_lasso_results, tmp_path):
        """Test that JSON includes schema version."""
        file_path = tmp_path / "test_nomogram.json"

        save_nomogram_json(
            bundle=mock_bundle,
            lasso_results=mock_lasso_results,
            file_path=file_path,
        )

        with open(file_path, 'r') as f:
            data = json.load(f)

        assert data['version'] == JSON_SCHEMA_VERSION

    def test_metadata_section(self, mock_bundle, mock_lasso_results, tmp_path):
        """Test that metadata section is complete."""
        file_path = tmp_path / "test_nomogram.json"

        save_nomogram_json(
            bundle=mock_bundle,
            lasso_results=mock_lasso_results,
            file_path=file_path,
            comment="Test comment",
            method="lebesgue",
        )

        with open(file_path, 'r') as f:
            data = json.load(f)

        metadata = data['metadata']
        assert 'generated_at' in metadata
        assert metadata['base_model'] == 'test_model'
        assert metadata['method'] == 'lebesgue'
        assert metadata['n_steps'] == 50
        assert metadata['categorical_threshold'] == 15
        assert metadata['comment'] == 'Test comment'

    def test_model_section(self, mock_bundle, mock_lasso_results, tmp_path):
        """Test that model section contains intercept and lambda info."""
        file_path = tmp_path / "test_nomogram.json"

        save_nomogram_json(
            bundle=mock_bundle,
            lasso_results=mock_lasso_results,
            file_path=file_path,
        )

        with open(file_path, 'r') as f:
            data = json.load(f)

        model = data['model']
        assert model['intercept'] == pytest.approx(-0.5)
        assert model['selected_lambda'] == pytest.approx(0.001)
        assert model['selected_lambda_index'] == 42

    def test_univariate_section(self, mock_bundle, mock_lasso_results, tmp_path):
        """Test that univariate features are saved correctly."""
        file_path = tmp_path / "test_nomogram.json"

        save_nomogram_json(
            bundle=mock_bundle,
            lasso_results=mock_lasso_results,
            file_path=file_path,
        )

        with open(file_path, 'r') as f:
            data = json.load(f)

        univariate = data['univariate']
        assert 'age' in univariate
        assert 'income' in univariate

        # Check age feature structure
        age = univariate['age']
        assert age['index'] == 0
        assert age['name'] == 'age'
        assert age['label'] == 'Age (years)'
        assert age['is_categorical'] is False
        assert len(age['x_values']) == 50
        assert len(age['response']) == 50
        assert age['beta'] == pytest.approx(0.3)
        assert 'histogram' in age
        assert 'bin_edges' in age['histogram']
        assert 'counts' in age['histogram']

    def test_bivariate_section(self, mock_bundle, mock_lasso_results, tmp_path):
        """Test that bivariate features are saved correctly."""
        file_path = tmp_path / "test_nomogram.json"

        save_nomogram_json(
            bundle=mock_bundle,
            lasso_results=mock_lasso_results,
            file_path=file_path,
        )

        with open(file_path, 'r') as f:
            data = json.load(f)

        bivariate = data['bivariate']
        assert 'age__income' in bivariate

        pair = bivariate['age__income']
        assert pair['indices'] == [0, 1]
        assert pair['names'] == ['age', 'income']
        assert pair['labels'] == ['Age (years)', 'Income ($)']
        assert pair['is_categorical'] == [False, False]
        assert len(pair['x_values_1']) == 100
        assert len(pair['x_values_2']) == 100
        assert len(pair['response']) == 100
        assert pair['skipped'] is False

    def test_category_labels_included(self, mock_lasso_results, tmp_path):
        """Test that category labels are included when provided."""
        bundle = PlottingDataBundle(
            all_feature_names=['gender'],
            n_steps=10,
            categorical_threshold=15,
        )
        bundle.x_data = np.array([[0], [1], [0], [1], [0]])
        bundle._univariate_info = [
            FeatureInfo(
                index=0,
                name='gender',
                label='Gender',
                is_categorical=True,
                response=np.array([0.2, -0.2]),
                x_values=np.array([0, 1]),
            ),
        ]
        bundle._bivariate_info = []

        file_path = tmp_path / "test_nomogram.json"

        save_nomogram_json(
            bundle=bundle,
            lasso_results=mock_lasso_results,
            file_path=file_path,
            category_labels={'gender': {0: 'Male', 1: 'Female'}},
        )

        with open(file_path, 'r') as f:
            data = json.load(f)

        gender = data['univariate']['gender']
        assert 'category_labels' in gender
        assert gender['category_labels'] == {'0': 'Male', '1': 'Female'}

    def test_skipped_bivariate_excluded(self, mock_lasso_results, tmp_path):
        """Test that skipped bivariate pairs are excluded."""
        bundle = PlottingDataBundle(
            all_feature_names=['a', 'b'],
            n_steps=10,
            categorical_threshold=15,
        )
        bundle.x_data = np.random.rand(100, 2)
        bundle._univariate_info = [
            FeatureInfo(
                index=0,
                name='a',
                label='A',
                is_categorical=False,
                response=np.ones(10),
                x_values=np.linspace(0, 1, 10),
            ),
        ]
        bundle._bivariate_info = [
            FeaturePairInfo(
                indices=(0, 1),
                names=('a', 'b'),
                labels=('A', 'B'),
                is_categorical=(False, False),
                response=None,
                x_values=None,
                skipped=True,
            ),
        ]

        file_path = tmp_path / "test_nomogram.json"

        save_nomogram_json(
            bundle=bundle,
            lasso_results=mock_lasso_results,
            file_path=file_path,
        )

        with open(file_path, 'r') as f:
            data = json.load(f)

        # Skipped pairs should not appear
        assert len(data['bivariate']) == 0

    def test_auto_path_generation(self, mock_bundle, mock_lasso_results, tmp_path):
        """Test auto-generated path includes model name."""
        with patch('prism.plotting.json_export.MODELS_DIR', str(tmp_path / 'models')):
            result_path = save_nomogram_json(
                bundle=mock_bundle,
                lasso_results=mock_lasso_results,
                comment="Test",
            )

            assert result_path.exists()
            assert 'test_model' in str(result_path)
            assert result_path.suffix == '.json'

    def test_json_is_human_readable(self, mock_bundle, mock_lasso_results, tmp_path):
        """Test that JSON output is formatted with indentation."""
        file_path = tmp_path / "test_nomogram.json"

        save_nomogram_json(
            bundle=mock_bundle,
            lasso_results=mock_lasso_results,
            file_path=file_path,
        )

        with open(file_path, 'r') as f:
            content = f.read()

        # Check for indentation (human-readable formatting)
        assert '\n  ' in content  # At least some indentation

    def test_no_x_data_still_works(self, mock_lasso_results, tmp_path):
        """Test that function works when bundle has no x_data for histograms."""
        bundle = PlottingDataBundle(
            all_feature_names=['age'],
            n_steps=10,
            categorical_threshold=15,
        )
        bundle.x_data = None  # No histogram data
        bundle._univariate_info = [
            FeatureInfo(
                index=0,
                name='age',
                label='Age',
                is_categorical=False,
                response=np.linspace(-0.5, 0.5, 10),
                x_values=np.linspace(20, 80, 10),
            ),
        ]
        bundle._bivariate_info = []

        file_path = tmp_path / "test_nomogram.json"

        result_path = save_nomogram_json(
            bundle=bundle,
            lasso_results=mock_lasso_results,
            file_path=file_path,
        )

        assert result_path.exists()

        with open(file_path, 'r') as f:
            data = json.load(f)

        # Histogram should be empty but not cause an error
        assert data['univariate']['age']['histogram'] == {}

    def test_bivariate_beta_with_many_features(self, tmp_path):
        """Test that bivariate betas are looked up correctly when only a subset is selected.

        Regression test: previously the bivariate beta index was computed from
        the count of *selected* univariates and the position in the *selected*
        bivariate list, instead of using the total n_univ and the canonical
        combinatorial offset.  This caused wrong (often zero) beta values.
        """
        # Simulate 5 total univariate features, 2 selected univariates, 1 selected bivariate
        n_univ = 5
        # combinations(range(5), 2) has 10 pairs:
        # (0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)
        # We'll select pair (2, 4) -> position 8 in the full list
        # beta_idx = 5 + 8 = 13
        betas = np.zeros(n_univ + 10)  # 5 univ + 10 biv
        betas[1] = 0.5  # univariate feature 1
        betas[3] = -0.4  # univariate feature 3
        betas[13] = 0.77  # bivariate pair (2, 4) at offset 8

        mock_lasso = MagicMock()
        mock_lasso.base_model_name = "test"
        mock_lasso.n_univ = n_univ
        selected_model = MagicMock()
        selected_model.intercept_ = np.array([0.0])
        mock_lasso.get_selected_model.return_value = selected_model
        mock_lasso.selected_lambda_index = 0
        mock_lasso.lambdas = np.array([0.01])
        mock_lasso.get_selected_beta.return_value = betas

        bundle = PlottingDataBundle(
            all_feature_names=['f0', 'f1', 'f2', 'f3', 'f4'],
            n_steps=10,
            categorical_threshold=15,
        )
        bundle.x_data = np.random.rand(50, 5)
        # Only 2 selected univariates (indices 1 and 3)
        bundle._univariate_info = [
            FeatureInfo(
                index=1,
                name='f1',
                label='F1',
                is_categorical=False,
                response=np.ones(10),
                x_values=np.linspace(0, 1, 10),
            ),
            FeatureInfo(
                index=3,
                name='f3',
                label='F3',
                is_categorical=False,
                response=np.ones(10),
                x_values=np.linspace(0, 1, 10),
            ),
        ]
        # Only 1 selected bivariate pair (2, 4)
        bundle._bivariate_info = [
            FeaturePairInfo(
                indices=(2, 4),
                names=('f2', 'f4'),
                labels=('F2', 'F4'),
                is_categorical=(False, False),
                response=np.ones(25),
                x_values=np.column_stack(
                    [
                        np.repeat(np.linspace(0, 1, 5), 5),
                        np.tile(np.linspace(0, 1, 5), 5),
                    ]
                ),
                skipped=False,
            ),
        ]

        file_path = tmp_path / "test_biv_beta.json"
        save_nomogram_json(bundle=bundle, lasso_results=mock_lasso, file_path=file_path)

        with open(file_path, 'r') as f:
            data = json.load(f)

        # Verify univariate betas are correct
        assert data['univariate']['f1']['beta'] == pytest.approx(0.5)
        assert data['univariate']['f3']['beta'] == pytest.approx(-0.4)

        # The critical check: bivariate beta must be 0.77, not 0.0
        pair = data['bivariate']['f2__f4']
        assert pair['beta'] == pytest.approx(
            0.77
        ), f"Bivariate beta for (2,4) should be 0.77 but got {pair['beta']}"

    def test_beta_scaled_metadata_flag(self, mock_bundle, mock_lasso_results, tmp_path):
        """Test that beta_scaled flag is present in metadata."""
        file_path = tmp_path / "test_nomogram.json"

        save_nomogram_json(
            bundle=mock_bundle,
            lasso_results=mock_lasso_results,
            file_path=file_path,
        )

        with open(file_path, 'r') as f:
            data = json.load(f)

        assert data['metadata']['beta_scaled'] is True
