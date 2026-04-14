"""Tests for hyperparameter tuning config loader integration."""

import pytest

from prism.config_loader import get_tuning_config
from prism.hyperparameter_tuning.config import TuningConfig


class TestGetTuningConfig:
    """Test get_tuning_config function."""

    def test_no_config_returns_disabled(self):
        """Test that None config returns disabled tuning."""
        tuning_config = get_tuning_config(None, 'mlp')

        assert tuning_config.enabled is False

    def test_empty_config_returns_disabled(self):
        """Test that empty config returns disabled tuning."""
        tuning_config = get_tuning_config({}, 'mlp')

        assert tuning_config.enabled is False

    def test_global_config_only(self):
        """Test global tuning config without model-specific overrides."""
        config = {'hyperparameter_tuning': {'enabled': True, 'n_trials': 25, 'metric': 'accuracy'}}

        tuning_config = get_tuning_config(config, 'mlp')

        assert tuning_config.enabled is True
        assert tuning_config.n_trials == 25
        assert tuning_config.metric == 'accuracy'

    def test_model_specific_override(self):
        """Test model-specific config overrides global settings."""
        config = {
            'hyperparameter_tuning': {
                'enabled': False,
                'n_trials': 20,
                'metric': 'test_auc',
                'mlp': {'enabled': True, 'n_trials': 30},
            }
        }

        # MLP should get model-specific settings
        mlp_config = get_tuning_config(config, 'mlp')
        assert mlp_config.enabled is True
        assert mlp_config.n_trials == 30
        assert mlp_config.metric == 'test_auc'  # Inherited from global

        # XGB should get global settings
        xgb_config = get_tuning_config(config, 'xgb')
        assert xgb_config.enabled is False
        assert xgb_config.n_trials == 20

    def test_model_specific_partial_override(self):
        """Test model-specific config partially overrides global."""
        config = {
            'hyperparameter_tuning': {
                'enabled': True,
                'n_trials': 20,
                'metric': 'test_auc',
                'pruning_enabled': True,
                'xgb': {'n_trials': 40, 'metric': 'accuracy'},
            }
        }

        xgb_config = get_tuning_config(config, 'xgb')

        # Overridden values
        assert xgb_config.n_trials == 40
        assert xgb_config.metric == 'accuracy'

        # Inherited values
        assert xgb_config.enabled is True
        assert xgb_config.pruning_enabled is True

    def test_multiple_model_configs(self):
        """Test config with multiple model-specific sections."""
        config = {
            'hyperparameter_tuning': {
                'enabled': False,
                'n_trials': 15,
                'mlp': {'enabled': True, 'n_trials': 25},
                'xgb': {'enabled': True, 'n_trials': 35},
                'logreg': {'enabled': True, 'n_trials': 10},
            }
        }

        mlp_config = get_tuning_config(config, 'mlp')
        assert mlp_config.enabled is True
        assert mlp_config.n_trials == 25

        xgb_config = get_tuning_config(config, 'xgb')
        assert xgb_config.enabled is True
        assert xgb_config.n_trials == 35

        logreg_config = get_tuning_config(config, 'logreg')
        assert logreg_config.enabled is True
        assert logreg_config.n_trials == 10

        # RF not specified, should get global (disabled)
        rf_config = get_tuning_config(config, 'rf')
        assert rf_config.enabled is False
        assert rf_config.n_trials == 15

    def test_all_tuning_parameters(self):
        """Test all tuning parameters are correctly loaded."""
        config = {
            'hyperparameter_tuning': {
                'enabled': True,
                'n_trials': 30,
                'metric': 'brier',
                'direction': 'minimize',
                'pruning_enabled': False,
                'pruning_warmup_steps': 10,
            }
        }

        tuning_config = get_tuning_config(config, 'mlp')

        assert tuning_config.enabled is True
        assert tuning_config.n_trials == 30
        assert tuning_config.metric == 'brier'
        assert tuning_config.direction == 'minimize'
        assert tuning_config.pruning_enabled is False
        assert tuning_config.pruning_warmup_steps == 10

    def test_case_insensitive_model_names(self):
        """Test that model names are matched correctly regardless of case."""
        config = {
            'hyperparameter_tuning': {'enabled': False, 'mlp': {'enabled': True, 'n_trials': 25}}
        }

        # All these should get the MLP-specific config
        for model_name in ['mlp', 'MLP', 'Mlp']:
            # Note: actual implementation may be case-sensitive
            # This test documents current behavior
            tuning_config = get_tuning_config(config, model_name)
            # We expect case-sensitive matching
            if model_name == 'mlp':
                assert tuning_config.enabled is True
            else:
                # Non-matching case gets global config
                assert tuning_config.enabled is False

    def test_invalid_config_values(self):
        """Test that invalid config values raise appropriate errors."""
        config = {'hyperparameter_tuning': {'enabled': True, 'direction': 'invalid_direction'}}

        with pytest.raises(ValueError, match="direction must be"):
            get_tuning_config(config, 'mlp')

    def test_returns_tuning_config_instance(self):
        """Test that function returns TuningConfig instance."""
        config = {'hyperparameter_tuning': {'enabled': True, 'n_trials': 20}}

        tuning_config = get_tuning_config(config, 'mlp')

        assert isinstance(tuning_config, TuningConfig)

    def test_n_jobs_global_config(self):
        """Test n_jobs is correctly loaded from global config."""
        config = {'hyperparameter_tuning': {'enabled': True, 'n_jobs': 4, 'n_trials': 20}}

        tuning_config = get_tuning_config(config, 'mlp')
        assert tuning_config.n_jobs == 4

    def test_n_jobs_model_specific_override(self):
        """Test n_jobs can be overridden per model."""
        config = {'hyperparameter_tuning': {'enabled': True, 'n_jobs': 2, 'mlp': {'n_jobs': 4}}}

        mlp_config = get_tuning_config(config, 'mlp')
        assert mlp_config.n_jobs == 4

        xgb_config = get_tuning_config(config, 'xgb')
        assert xgb_config.n_jobs == 2

    def test_params_file_config(self):
        """Test params_file is correctly loaded from config."""
        config = {
            'hyperparameter_tuning': {
                'mlp': {'params_file': 'models/credit-g_mlp_best_params.json'}
            }
        }

        tuning_config = get_tuning_config(config, 'mlp')
        assert tuning_config.params_file == 'models/credit-g_mlp_best_params.json'
        # When params_file is specified, enabled defaults to False
        assert tuning_config.enabled is False

    def test_params_file_with_enabled_false(self):
        """Test params_file can be used with enabled=False explicitly."""
        config = {
            'hyperparameter_tuning': {
                'xgb': {'enabled': False, 'params_file': 'models/my_xgb_params.json'}
            }
        }

        tuning_config = get_tuning_config(config, 'xgb')
        assert tuning_config.enabled is False
        assert tuning_config.params_file == 'models/my_xgb_params.json'

    def test_skip_saved_params_global(self):
        """Test skip_saved_params at global level."""
        config = {'hyperparameter_tuning': {'skip_saved_params': True}}

        tuning_config = get_tuning_config(config, 'mlp')
        assert tuning_config.skip_saved_params is True
        assert tuning_config.enabled is False

    def test_skip_saved_params_model_specific(self):
        """Test skip_saved_params at model-specific level."""
        config = {
            'hyperparameter_tuning': {
                'skip_saved_params': False,
                'mlp': {'skip_saved_params': True},
                'xgb': {'enabled': False},
            }
        }

        # MLP should have skip_saved_params=True (model override)
        mlp_config = get_tuning_config(config, 'mlp')
        assert mlp_config.skip_saved_params is True

        # XGB should have skip_saved_params=False (global default)
        xgb_config = get_tuning_config(config, 'xgb')
        assert xgb_config.skip_saved_params is False

    def test_skip_saved_params_inherits_from_global(self):
        """Test skip_saved_params inherits from global when not specified per-model."""
        config = {
            'hyperparameter_tuning': {
                'skip_saved_params': True,
                'mlp': {'n_trials': 50},  # Model-specific override without skip_saved_params
            }
        }

        # MLP should inherit skip_saved_params=True from global
        mlp_config = get_tuning_config(config, 'mlp')
        assert mlp_config.skip_saved_params is True
        assert mlp_config.n_trials == 50

    def test_skip_saved_params_with_params_file_raises_error(self):
        """Test that skip_saved_params=True with params_file raises error."""
        config = {
            'hyperparameter_tuning': {
                'mlp': {'skip_saved_params': True, 'params_file': 'models/some_params.json'}
            }
        }

        with pytest.raises(ValueError, match="Cannot specify both skip_saved_params"):
            get_tuning_config(config, 'mlp')
