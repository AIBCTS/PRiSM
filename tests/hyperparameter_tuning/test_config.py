"""Tests for hyperparameter tuning configuration."""

import pytest

from prism.hyperparameter_tuning.config import TuningConfig


class TestTuningConfig:
    """Test TuningConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TuningConfig()
        assert config.enabled is False
        assert config.n_trials == 20
        assert config.metric == 'test_auc'
        assert config.direction == 'maximize'
        assert config.pruning_enabled is True
        assert config.pruning_warmup_steps == 5
        assert config.n_jobs == 1
        assert config.params_file is None
        assert config.skip_saved_params is False
        assert config.custom_search_space is None

    def test_n_jobs_config(self):
        """Test n_jobs configuration."""
        # Default
        config = TuningConfig()
        assert config.n_jobs == 1

        # Custom value
        config = TuningConfig(n_jobs=4)
        assert config.n_jobs == 4

        # Use all CPUs
        config = TuningConfig(n_jobs=-1)
        assert config.n_jobs == -1

    def test_params_file_config(self):
        """Test params_file configuration."""
        # Default
        config = TuningConfig()
        assert config.params_file is None

        # With path specified
        config = TuningConfig(params_file='models/credit-g_mlp_best_params.json')
        assert config.params_file == 'models/credit-g_mlp_best_params.json'

        # With absolute path
        config = TuningConfig(params_file='/absolute/path/to/params.json')
        assert config.params_file == '/absolute/path/to/params.json'

    def test_custom_config(self):
        """Test custom configuration values."""
        config = TuningConfig(
            enabled=True,
            n_trials=30,
            metric='accuracy',
            direction='maximize',
            pruning_enabled=False,
            pruning_warmup_steps=10,
            custom_search_space={'lr': [0.001, 0.01]},
        )
        assert config.enabled is True
        assert config.n_trials == 30
        assert config.metric == 'accuracy'
        assert config.direction == 'maximize'
        assert config.pruning_enabled is False
        assert config.pruning_warmup_steps == 10
        assert config.custom_search_space == {'lr': [0.001, 0.01]}

    def test_invalid_direction(self):
        """Test that invalid direction raises ValueError."""
        with pytest.raises(ValueError, match="direction must be"):
            TuningConfig(direction='invalid')

    def test_invalid_n_trials(self):
        """Test that invalid n_trials raises ValueError."""
        with pytest.raises(ValueError, match="n_trials must be"):
            TuningConfig(n_trials=0)

        with pytest.raises(ValueError, match="n_trials must be"):
            TuningConfig(n_trials=-5)

    def test_invalid_pruning_warmup_steps(self):
        """Test that invalid pruning_warmup_steps raises ValueError."""
        with pytest.raises(ValueError, match="pruning_warmup_steps must be"):
            TuningConfig(pruning_warmup_steps=-1)

    def test_minimize_direction(self):
        """Test minimize direction is valid."""
        config = TuningConfig(direction='minimize', metric='brier')
        assert config.direction == 'minimize'
        assert config.metric == 'brier'

    def test_skip_saved_params_default(self):
        """Test skip_saved_params defaults to False."""
        config = TuningConfig()
        assert config.skip_saved_params is False

    def test_skip_saved_params_enabled(self):
        """Test skip_saved_params can be set to True."""
        config = TuningConfig(skip_saved_params=True)
        assert config.skip_saved_params is True
        assert config.params_file is None  # Should not have params_file

    def test_skip_saved_params_with_params_file_raises_error(self):
        """Test that skip_saved_params=True with params_file raises ValueError."""
        with pytest.raises(ValueError, match="Cannot specify both skip_saved_params"):
            TuningConfig(skip_saved_params=True, params_file='models/some_params.json')

    def test_skip_saved_params_false_with_params_file_ok(self):
        """Test that skip_saved_params=False with params_file is allowed."""
        config = TuningConfig(skip_saved_params=False, params_file='models/some_params.json')
        assert config.skip_saved_params is False
        assert config.params_file == 'models/some_params.json'
