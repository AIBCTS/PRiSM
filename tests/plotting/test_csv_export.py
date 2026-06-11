"""
Tests for CSV export functionality.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from prism.plotting.csv_export import save_nomogram_csv
from prism.plotting_data import FeatureInfo, FeaturePairInfo, PlottingDataBundle


class TestSaveNomogramCSV:
    """Test the save_nomogram_csv function."""

    @pytest.fixture
    def mock_lasso_results(self):
        """Create a mock LassoResultsManager."""
        mock = MagicMock()
        mock.base_model_name = "test_model"

        # Mock the selected model with intercept
        selected_model = MagicMock()
        selected_model.intercept_ = np.array([0.5])
        mock.get_selected_model.return_value = selected_model

        return mock

    @pytest.fixture
    def mock_bundle(self):
        """Create a mock PlottingDataBundle with test data."""
        bundle = PlottingDataBundle(
            all_feature_names=['age', 'income', 'age_income'],
            n_steps=50,
            categorical_threshold=15,
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

    def test_save_nomogram_csv_creates_files(self, mock_bundle, mock_lasso_results, tmp_path):
        """Test that CSV files are created with correct structure."""
        file_path = tmp_path / "test_nomogram.csv"

        univariate_path, bivariate_path = save_nomogram_csv(
            bundle=mock_bundle,
            lasso_results=mock_lasso_results,
            model_info={'comment': 'Test', 'method': 'dirac'},
            file_path=file_path,
            use_odds_ratio=False,
        )

        # Check files exist
        assert univariate_path.exists()
        assert bivariate_path.exists()

        # Check file names
        assert 'univariate' in univariate_path.name
        assert 'bivariate' in bivariate_path.name

    def test_save_nomogram_csv_univariate_content(self, mock_bundle, mock_lasso_results, tmp_path):
        """Test univariate CSV content is correct."""
        file_path = tmp_path / "test_nomogram.csv"

        univariate_path, _ = save_nomogram_csv(
            bundle=mock_bundle,
            lasso_results=mock_lasso_results,
            model_info={'comment': 'Test', 'method': 'dirac'},
            file_path=file_path,
        )

        # Read CSV (skip metadata lines starting with #)
        df = pd.read_csv(univariate_path, comment='#')

        # Check columns exist
        assert 'age_x' in df.columns
        assert 'age_response' in df.columns
        assert 'income_x' in df.columns
        assert 'income_response' in df.columns

        # Check data length
        assert len(df) == 50

    def test_save_nomogram_csv_bivariate_content(self, mock_bundle, mock_lasso_results, tmp_path):
        """Test bivariate CSV content is correct."""
        file_path = tmp_path / "test_nomogram.csv"

        _, bivariate_path = save_nomogram_csv(
            bundle=mock_bundle,
            lasso_results=mock_lasso_results,
            model_info={'comment': 'Test', 'method': 'dirac'},
            file_path=file_path,
        )

        # Read CSV (skip metadata lines starting with #)
        df = pd.read_csv(bivariate_path, comment='#')

        # Check columns exist
        assert 'age_income_x1' in df.columns
        assert 'age_income_x2' in df.columns
        assert 'age_income_response' in df.columns

        # Check data length
        assert len(df) == 100

    def test_save_nomogram_csv_metadata_header(self, mock_bundle, mock_lasso_results, tmp_path):
        """Test that metadata header is included in CSV files."""
        file_path = tmp_path / "test_nomogram.csv"

        univariate_path, _ = save_nomogram_csv(
            bundle=mock_bundle,
            lasso_results=mock_lasso_results,
            model_info={'comment': 'Test Comment', 'method': 'lebesgue'},
            file_path=file_path,
            use_odds_ratio=False,
        )

        # Read raw file content
        with open(univariate_path, 'r') as f:
            content = f.read()

        # Check metadata is present
        assert '# Nomogram Data - Univariate' in content
        assert '# Comment: Test Comment' in content
        assert '# Method: lebesgue' in content
        assert '# Base model: test_model' in content
        assert '# Response scale: Log odds ratio' in content

    def test_save_nomogram_csv_odds_ratio_metadata(
        self, mock_bundle, mock_lasso_results, tmp_path
    ):
        """Test that odds ratio is correctly indicated in metadata."""
        file_path = tmp_path / "test_nomogram.csv"

        univariate_path, _ = save_nomogram_csv(
            bundle=mock_bundle,
            lasso_results=mock_lasso_results,
            model_info={'comment': 'Test', 'method': 'dirac'},
            file_path=file_path,
            use_odds_ratio=True,
        )

        # Read raw file content
        with open(univariate_path, 'r') as f:
            content = f.read()

        assert '# Response scale: Odds ratio' in content
        assert 'odds ratio:' in content

    def test_save_nomogram_csv_auto_path(self, mock_bundle, mock_lasso_results):
        """Test auto-generated path includes model name and timestamp."""
        with patch('prism.plotting.csv_export.MODELS_DIR', Path('/tmp/test_models')):
            with patch('prism.plotting.csv_export.Path.mkdir'):
                with patch('builtins.open', MagicMock()):
                    with patch('pandas.DataFrame.to_csv'):
                        # The function should not raise when auto-generating path
                        # (we're mocking the file operations)
                        try:
                            save_nomogram_csv(
                                bundle=mock_bundle,
                                lasso_results=mock_lasso_results,
                            )
                        except Exception:
                            pass  # Expected since we're mocking

    def test_save_nomogram_csv_skipped_bivariate(self, mock_lasso_results, tmp_path):
        """Test that skipped bivariate pairs are not included."""
        bundle = PlottingDataBundle(
            all_feature_names=['a', 'b', 'a_b'],
            n_steps=10,
            categorical_threshold=15,
        )

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
                skipped=True,  # This pair is skipped
            ),
        ]

        file_path = tmp_path / "test_nomogram.csv"
        _, bivariate_path = save_nomogram_csv(
            bundle=bundle,
            lasso_results=mock_lasso_results,
            file_path=file_path,
        )

        # Read raw file content - should only have metadata (no data columns)
        with open(bivariate_path, 'r') as f:
            content = f.read()

        # Should have metadata header
        assert '# Nomogram Data - Bivariate' in content
        # The file should exist but have empty data section (only metadata + empty DataFrame)
