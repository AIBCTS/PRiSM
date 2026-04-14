"""Tests for hyperparameter search spaces.

These tests verify that search spaces include the default values from DEFAULT_PARAMS,
ensuring that tuning can find parameters at least as good as the current configuration.
"""

import optuna
import pytest

from prism.hyperparameter_tuning.search_spaces import (
    DEFAULT_PARAMS,
    SEARCH_SPACE_REGISTRY,
    get_default_params,
    get_logreg_search_space,
    get_mlp_search_space,
    get_prn_search_space,
    get_rf_search_space,
    get_search_space,
    get_xgb_search_space,
)


def _get_search_space_bounds(search_space_fn):
    """Get actual bounds for each parameter from Optuna distribution objects.

    Uses Optuna's distribution metadata (low/high/choices) rather than sampling,
    which guarantees we get the exact declared bounds.

    Only returns bounds for tuned parameters (those registered via trial.suggest_*).
    Fixed parameters included in the returned dict are not captured here.

    Args:
        search_space_fn: Function that takes an Optuna trial and returns params dict

    Returns:
        Dict mapping param names to (min_value, max_value, all_values) tuples.
        For categorical params, all_values contains the choices;
        for numeric params, all_values is an empty set.
    """
    study = optuna.create_study()
    trial = study.ask()
    # Call the function to register distributions with the trial
    search_space_fn(trial)

    bounds = {}
    for name, dist in trial.distributions.items():
        if hasattr(dist, 'choices'):
            # CategoricalDistribution
            values = set(dist.choices)
            bounds[name] = (min(values), max(values), values)
        else:
            # IntDistribution or FloatDistribution
            bounds[name] = (dist.low, dist.high, set())

    return bounds


def _check_default_in_range(default_value, min_val, max_val, all_values, param_name):
    """Check if a default value is within the search space range.

    For categorical parameters, checks if value is in the set of possible values.
    For numeric parameters, checks if value is within [min, max] range.
    """
    # For categorical parameters (non-empty set of choices), check exact membership
    if all_values:
        return default_value in all_values

    # For continuous/integer parameters, check range
    return min_val <= default_value <= max_val


def _get_fixed_params(search_space_fn):
    """Get fixed (non-tuned) parameters from a search space function.

    Returns keys present in the returned dict but not registered as Optuna distributions.
    """
    study = optuna.create_study()
    trial = study.ask()
    params = search_space_fn(trial)
    tuned_keys = set(trial.distributions.keys())
    return {k: v for k, v in params.items() if k not in tuned_keys}


class TestSearchSpaces:
    """Test search space functions."""

    def test_mlp_search_space_returns_expected_keys(self):
        """Test MLP search space returns correct parameter keys."""
        study = optuna.create_study()
        trial = study.ask()
        params = get_mlp_search_space(trial)

        assert 'n_hidden' in params
        assert 'lr' in params
        assert 'weight_decay' in params
        assert 'patience' in params
        assert 'batch_size' in params

    def test_mlp_search_space_includes_defaults(self):
        """Test MLP search space ranges include the known-good defaults."""
        bounds = _get_search_space_bounds(get_mlp_search_space)

        for param, default in DEFAULT_PARAMS['mlp'].items():
            assert param in bounds, f"MLP default param '{param}' not in tuned search space"
            min_val, max_val, all_values = bounds[param]
            assert _check_default_in_range(
                default, min_val, max_val, all_values, param
            ), f"MLP default {param}={default} not in search space range [{min_val}, {max_val}]"

    def test_logreg_search_space_returns_expected_keys(self):
        """Test Logistic Regression search space returns correct parameter keys."""
        study = optuna.create_study()
        trial = study.ask()
        params = get_logreg_search_space(trial)

        assert 'lr' in params
        assert 'weight_decay' in params
        assert 'patience' in params

    def test_logreg_search_space_includes_defaults(self):
        """Test Logistic Regression search space ranges include the known-good defaults."""
        bounds = _get_search_space_bounds(get_logreg_search_space)

        for param, default in DEFAULT_PARAMS['logreg'].items():
            assert param in bounds, f"LogReg default param '{param}' not in tuned search space"
            min_val, max_val, all_values = bounds[param]
            assert _check_default_in_range(
                default, min_val, max_val, all_values, param
            ), f"LogReg default {param}={default} not in search space range [{min_val}, {max_val}]"

    def test_xgb_search_space_returns_expected_keys(self):
        """Test XGBoost search space returns correct parameter keys."""
        study = optuna.create_study()
        trial = study.ask()
        params = get_xgb_search_space(trial)

        assert 'max_depth' in params
        assert 'learning_rate' in params
        assert 'subsample' in params
        assert 'colsample_bytree' in params
        assert 'min_child_weight' in params
        assert 'n_estimators' in params
        assert 'gamma' in params
        assert 'reg_alpha' in params
        assert 'reg_lambda' in params
        assert 'early_stopping_rounds' in params

    def test_xgb_search_space_includes_defaults(self):
        """Test XGBoost search space ranges include the known-good defaults."""
        bounds = _get_search_space_bounds(get_xgb_search_space)

        for param, default in DEFAULT_PARAMS['xgb'].items():
            assert param in bounds, f"XGB default param '{param}' not in tuned search space"
            min_val, max_val, all_values = bounds[param]
            assert _check_default_in_range(
                default, min_val, max_val, all_values, param
            ), f"XGB default {param}={default} not in search space range [{min_val}, {max_val}]"

    def test_xgb_early_stopping_rounds_is_fixed(self):
        """Test that early_stopping_rounds is a fixed constant, not tuned."""
        fixed = _get_fixed_params(get_xgb_search_space)
        assert 'early_stopping_rounds' in fixed
        assert fixed['early_stopping_rounds'] == 10

    def test_rf_search_space_returns_expected_keys(self):
        """Test Random Forest search space returns correct parameter keys."""
        study = optuna.create_study()
        trial = study.ask()
        params = get_rf_search_space(trial)

        assert 'n_estimators' in params
        assert 'max_depth' in params
        assert 'min_child_weight' in params
        assert 'subsample' in params
        assert 'colsample_bynode' in params

    def test_rf_search_space_includes_defaults(self):
        """Test Random Forest search space ranges include the known-good defaults."""
        bounds = _get_search_space_bounds(get_rf_search_space)

        for param, default in DEFAULT_PARAMS['rf'].items():
            assert param in bounds, f"RF default param '{param}' not in tuned search space"
            min_val, max_val, all_values = bounds[param]
            assert _check_default_in_range(
                default, min_val, max_val, all_values, param
            ), f"RF default {param}={default} not in search space range [{min_val}, {max_val}]"

    def test_prn_search_space_returns_expected_keys(self):
        """Test PRN search space returns correct parameter keys.

        Note: subnet_nodes is NOT included because the mask shape depends on it
        and is created before tuning (during LASSO feature selection).
        """
        study = optuna.create_study()
        trial = study.ask()
        params = get_prn_search_space(trial)

        # subnet_nodes is NOT tuned (mask shape depends on it)
        assert 'subnet_nodes' not in params
        assert 'lr' in params
        assert 'weight_decay' in params
        assert 'patience' in params
        assert 'batch_size' in params

    def test_prn_search_space_includes_defaults(self):
        """Test PRN search space ranges include the known-good defaults."""
        bounds = _get_search_space_bounds(get_prn_search_space)

        for param, default in DEFAULT_PARAMS['prn'].items():
            assert param in bounds, f"PRN default param '{param}' not in tuned search space"
            min_val, max_val, all_values = bounds[param]
            assert _check_default_in_range(
                default, min_val, max_val, all_values, param
            ), f"PRN default {param}={default} not in search space range [{min_val}, {max_val}]"

    def test_get_search_space_registry(self):
        """Test search space registry contains all model types."""
        assert 'mlp' in SEARCH_SPACE_REGISTRY
        assert 'logreg' in SEARCH_SPACE_REGISTRY
        assert 'xgb' in SEARCH_SPACE_REGISTRY
        assert 'rf' in SEARCH_SPACE_REGISTRY
        assert 'prn' in SEARCH_SPACE_REGISTRY

    def test_get_search_space_mlp(self):
        """Test get_search_space wrapper for MLP."""
        study = optuna.create_study()
        trial = study.ask()
        params = get_search_space('mlp', trial)

        assert 'n_hidden' in params
        assert 'lr' in params

    def test_get_search_space_case_insensitive(self):
        """Test get_search_space is case insensitive."""
        study = optuna.create_study()
        trial = study.ask()

        params_lower = get_search_space('mlp', trial)
        trial2 = study.ask()
        params_upper = get_search_space('MLP', trial2)

        # Both should have same keys
        assert set(params_lower.keys()) == set(params_upper.keys())

    def test_get_search_space_invalid_model(self):
        """Test get_search_space raises error for invalid model."""
        study = optuna.create_study()
        trial = study.ask()

        with pytest.raises(ValueError, match="Unsupported model type"):
            get_search_space('invalid_model', trial)


class TestDefaultParams:
    """Test default parameters registry."""

    def test_default_params_exist_for_all_model_types(self):
        """Test that DEFAULT_PARAMS has entries for all registered model types."""
        for model_type in SEARCH_SPACE_REGISTRY:
            assert (
                model_type in DEFAULT_PARAMS
            ), f"Missing DEFAULT_PARAMS entry for model type: {model_type}"

    def test_get_default_params_returns_dict(self):
        """Test that get_default_params returns a dict for all known model types."""
        for model_type in DEFAULT_PARAMS:
            result = get_default_params(model_type)
            assert isinstance(result, dict), f"Expected dict for {model_type}, got {type(result)}"
            assert len(result) > 0, f"Empty defaults for {model_type}"

    def test_get_default_params_returns_none_for_unknown(self):
        """Test that get_default_params returns None for unknown model types."""
        assert get_default_params('unknown_model') is None

    def test_get_default_params_case_insensitive(self):
        """Test that get_default_params is case insensitive."""
        assert get_default_params('MLP') == get_default_params('mlp')
        assert get_default_params('XGB') == get_default_params('xgb')

    def test_default_params_are_tunable(self):
        """Test that all DEFAULT_PARAMS keys correspond to tuned search space params.

        DEFAULT_PARAMS are enqueued as Optuna trials, so every key must be
        a parameter registered via trial.suggest_*. Fixed (non-tuned) params
        should not be in DEFAULT_PARAMS.
        """
        for model_type, defaults in DEFAULT_PARAMS.items():
            bounds = _get_search_space_bounds(SEARCH_SPACE_REGISTRY[model_type])
            for key in defaults:
                assert key in bounds, (
                    f"{model_type}: DEFAULT_PARAMS key '{key}' is not a tuned "
                    f"search space parameter (not in trial.distributions)"
                )

    def test_enqueue_defaults_accepted_by_optuna(self):
        """Test that default params can be enqueued into an Optuna study."""
        for model_type, defaults in DEFAULT_PARAMS.items():
            study = optuna.create_study()
            # This should not raise
            study.enqueue_trial(defaults)

            # Run one trial to verify the enqueued params are used
            def objective(trial):
                _ = get_search_space(model_type, trial)
                return 0.5  # Dummy value

            study.optimize(objective, n_trials=1, show_progress_bar=False)
            assert len(study.trials) == 1
            # Verify the trial used our enqueued params
            for key, value in defaults.items():
                assert (
                    study.trials[0].params[key] == value
                ), f"{model_type}: enqueued {key}={value} but got {study.trials[0].params[key]}"
