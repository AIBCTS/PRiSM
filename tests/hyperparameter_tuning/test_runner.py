"""Tests for run_hyperparameter_tuning.py runner script."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prism.cli.tune import list_available_configs, load_preprocessed_data


class TestLoadPreprocessedData:
    """Test loading preprocessed data with config-specified target/id columns."""

    @pytest.fixture
    def temp_data_dir(self, tmp_path, monkeypatch):
        """Create temporary data directory with test CSV files."""
        # Patch INTERIM_DATA_DIR in the actual module where it's used
        import prism.cli.tune as runner

        monkeypatch.setattr(runner, 'INTERIM_DATA_DIR', tmp_path)
        return tmp_path

    def create_test_csv(self, filepath, target_col='target', id_col='id', n_rows=50):
        """Create a test CSV file."""
        np.random.seed(42)
        data = {
            id_col: [f'ID{i}' for i in range(n_rows)],
            target_col: np.random.randint(0, 2, n_rows),
            'feature1': np.random.randn(n_rows),
            'feature2': np.random.randn(n_rows),
            'feature3': np.random.randn(n_rows),
        }
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        return df

    def test_load_with_default_target(self, temp_data_dir):
        """Test loading data when target matches default candidates."""
        # Create files with default 'target' column
        self.create_test_csv(temp_data_dir / 'test_train.csv', target_col='target', id_col='id')
        self.create_test_csv(temp_data_dir / 'test_test.csv', target_col='target', id_col='id')

        X_train, y_train, X_test, y_test, target_col, id_col = load_preprocessed_data(
            'test', verbose=False
        )

        assert target_col == 'target'
        assert id_col == 'id'
        assert 'feature1' in X_train.columns
        assert 'target' not in X_train.columns
        assert 'id' not in X_train.columns

    def test_load_with_config_target_candidates(self, temp_data_dir):
        """Test loading data with config-specified target candidates."""
        # Create files with custom target column
        self.create_test_csv(
            temp_data_dir / 'custom_train.csv', target_col='MORTALITY_365D', id_col='TRR_ID_CODE'
        )
        self.create_test_csv(
            temp_data_dir / 'custom_test.csv', target_col='MORTALITY_365D', id_col='TRR_ID_CODE'
        )

        # Without config candidates, this would fail
        X_train, y_train, X_test, y_test, target_col, id_col = load_preprocessed_data(
            'custom',
            target_candidates=['MORTALITY_365D'],
            id_candidates=['TRR_ID_CODE'],
            verbose=False,
        )

        assert target_col == 'MORTALITY_365D'
        assert id_col == 'TRR_ID_CODE'
        assert len(y_train) > 0

    def test_load_case_insensitive_matching(self, temp_data_dir):
        """Test that column matching is case-insensitive."""
        # Create files with uppercase columns
        self.create_test_csv(
            temp_data_dir / 'upper_train.csv', target_col='MORTALITY_365D', id_col='TRR_ID_CODE'
        )
        self.create_test_csv(
            temp_data_dir / 'upper_test.csv', target_col='MORTALITY_365D', id_col='TRR_ID_CODE'
        )

        # Config specifies lowercase, should still match
        X_train, y_train, X_test, y_test, target_col, id_col = load_preprocessed_data(
            'upper',
            target_candidates=['mortality_365d'],  # lowercase
            id_candidates=['trr_id_code'],  # lowercase
            verbose=False,
        )

        # Should find the actual column names (uppercase)
        assert target_col == 'MORTALITY_365D'
        assert id_col == 'TRR_ID_CODE'

    def test_load_config_candidates_take_priority(self, temp_data_dir):
        """Test that config candidates are tried before defaults."""
        # Create files with both 'target' and 'MORTALITY_365D' columns
        np.random.seed(42)
        n_rows = 50
        data = {
            'id': [f'ID{i}' for i in range(n_rows)],
            'target': np.random.randint(0, 2, n_rows),  # Default target
            'MORTALITY_365D': np.random.randint(0, 2, n_rows),  # Config target
            'feature1': np.random.randn(n_rows),
        }
        df = pd.DataFrame(data)
        df.to_csv(temp_data_dir / 'priority_train.csv', index=False)
        df.to_csv(temp_data_dir / 'priority_test.csv', index=False)

        # Config candidate should be selected first
        X_train, y_train, X_test, y_test, target_col, id_col = load_preprocessed_data(
            'priority', target_candidates=['MORTALITY_365D'], verbose=False
        )

        assert target_col == 'MORTALITY_365D'

    def test_load_missing_target_raises_error(self, temp_data_dir):
        """Test that missing target column raises ValueError with helpful message."""
        # Create files with non-matching target column
        self.create_test_csv(
            temp_data_dir / 'bad_train.csv', target_col='some_random_column', id_col='id'
        )
        self.create_test_csv(
            temp_data_dir / 'bad_test.csv', target_col='some_random_column', id_col='id'
        )

        with pytest.raises(ValueError) as exc_info:
            load_preprocessed_data('bad', verbose=False)

        # Check error message includes useful info
        assert 'Could not auto-detect target column' in str(exc_info.value)
        assert 'Available columns' in str(exc_info.value)

    def test_load_missing_id_column_warning(self, temp_data_dir, capsys):
        """Test that missing ID column produces warning but still works."""
        # Create files without standard ID column
        np.random.seed(42)
        n_rows = 50
        data = {
            'target': np.random.randint(0, 2, n_rows),
            'feature1': np.random.randn(n_rows),
            'feature2': np.random.randn(n_rows),
        }
        df = pd.DataFrame(data)
        df.to_csv(temp_data_dir / 'noid_train.csv', index=False)
        df.to_csv(temp_data_dir / 'noid_test.csv', index=False)

        X_train, y_train, X_test, y_test, target_col, id_col = load_preprocessed_data(
            'noid', verbose=True
        )

        # Should still work, but ID should be None
        assert id_col is None
        assert 'feature1' in X_train.columns

        # Check warning was printed
        captured = capsys.readouterr()
        assert 'Warning' in captured.out or 'Could not detect ID' in captured.out

    def test_load_missing_train_file_raises_error(self, temp_data_dir):
        """Test that missing train file raises FileNotFoundError."""
        # Only create test file
        self.create_test_csv(temp_data_dir / 'missing_test.csv')

        with pytest.raises(FileNotFoundError) as exc_info:
            load_preprocessed_data('missing', verbose=False)

        assert 'training data not found' in str(exc_info.value).lower()

    def test_load_missing_test_file_raises_error(self, temp_data_dir):
        """Test that missing test file raises FileNotFoundError."""
        # Only create train file
        self.create_test_csv(temp_data_dir / 'missing2_train.csv')

        with pytest.raises(FileNotFoundError) as exc_info:
            load_preprocessed_data('missing2', verbose=False)

        assert 'test data not found' in str(exc_info.value).lower()


class TestListAvailableConfigs:
    """Test listing available config files."""

    @pytest.fixture
    def fixture_config_dir(self, tmp_path, monkeypatch):
        """Create a temporary config directory with test YAML files."""
        import prism.cli.tune as runner

        config_dir = tmp_path / "example_notebooks" / "config"
        config_dir.mkdir(parents=True)

        # Create fixture configs
        (config_dir / "alpha.yaml").write_text("dataset: alpha\n")
        (config_dir / "beta.yaml").write_text("dataset: beta\n")
        (config_dir / "gamma.yaml").write_text("dataset: gamma\n")

        monkeypatch.setattr(runner, 'PROJECT_ROOT', tmp_path)
        return config_dir

    def test_list_configs(self, fixture_config_dir):
        """Test that list_available_configs returns config names."""
        configs = list_available_configs()

        assert isinstance(configs, list)
        assert set(configs) == {'alpha', 'beta', 'gamma'}

    def test_configs_are_sorted(self, fixture_config_dir):
        """Test that config list is sorted."""
        configs = list_available_configs()
        assert configs == sorted(configs)

    def test_list_configs_includes_htx_example(self):
        """Test that the real config directory includes htx_example."""
        configs = list_available_configs()
        assert 'htx_example' in configs


class TestIntegration:
    """Integration tests using actual data (if available)."""

    @pytest.mark.skipif(
        not (
            Path(__file__).parent.parent.parent / 'data' / 'interim' / 'htx_example_train.csv'
        ).exists(),
        reason="htx_example preprocessed data not available",
    )
    def test_load_htx_example_data(self):
        """Test loading actual htx_example preprocessed data."""
        X_train, y_train, X_test, y_test, target_col, id_col = load_preprocessed_data(
            'htx_example',
            target_candidates=['event_oneyear'],
            id_candidates=['trr_id_code'],
            verbose=False,
        )

        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert len(y_train) > 0
        assert set(y_train.unique()).issubset({0, 1})
