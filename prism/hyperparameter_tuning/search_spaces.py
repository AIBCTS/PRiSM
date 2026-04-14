"""Default hyperparameter search spaces for different model types.

Search spaces are designed to include the known-working defaults from the
training notebooks (example_notebooks/modelling/ and prism_analysis.py).
This ensures tuning can find parameters at least as good as the current
configuration while exploring potentially better alternatives.

Default values in docstrings are sourced from:
- MLP: example_notebooks/modelling/train_mlp.py
- LogReg: example_notebooks/modelling/train_logreg.py
- XGB: example_notebooks/modelling/train_xgb.py
- RF: example_notebooks/modelling/train_rf.py
- PRN: example_notebooks/prism_analysis.py
"""

from typing import Callable, Dict, Optional

import optuna


def get_mlp_search_space(trial: optuna.Trial) -> Dict:
    """Define search space for MLP hyperparameters.

    Current defaults (from train_mlp.py):
    - n_hidden: 30
    - lr: 0.0015
    - weight_decay: 5e-5
    - patience: 15
    - batch_size: 512

    Args:
        trial: Optuna trial object

    Returns:
        Dictionary of hyperparameters
    """
    return {
        'n_hidden': trial.suggest_int('n_hidden', 20, 40),
        'lr': trial.suggest_float('lr', 3e-4, 3e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-4, log=True),
        'patience': trial.suggest_int('patience', 8, 20),
        'batch_size': trial.suggest_categorical('batch_size', [512, 1024, 2048]),
    }


def get_logreg_search_space(trial: optuna.Trial) -> Dict:
    """Define search space for Logistic Regression hyperparameters.

    Current defaults (from train_logreg.py):
    - lr: 0.02
    - weight_decay: 1e-3
    - patience: 12

    Args:
        trial: Optuna trial object

    Returns:
        Dictionary of hyperparameters
    """
    return {
        'lr': trial.suggest_float('lr', 3e-3, 3e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 2e-2, log=True),
        'patience': trial.suggest_int('patience', 3, 15),
    }


def get_xgb_search_space(trial: optuna.Trial) -> Dict:
    """Define search space for XGBoost hyperparameters.

    Current defaults (from train_xgb.py):
    - max_depth: 4
    - learning_rate: 0.055
    - subsample: 0.75
    - colsample_bytree: 0.65
    - min_child_weight: 9
    - n_estimators: 280
    - early_stopping_rounds: 10

    Args:
        trial: Optuna trial object

    Returns:
        Dictionary of hyperparameters
    """
    return {
        'max_depth': trial.suggest_int('max_depth', 2, 6),
        'learning_rate': trial.suggest_float('learning_rate', 2e-2, 1e-1, log=True),
        'subsample': trial.suggest_float('subsample', 0.7, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.8),
        'min_child_weight': trial.suggest_int('min_child_weight', 5, 15),
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'early_stopping_rounds': 10,
        'gamma': trial.suggest_float('gamma', 0, 3.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-6, 5.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
    }


def get_rf_search_space(trial: optuna.Trial) -> Dict:
    """Define search space for Random Forest hyperparameters.

    Note: XGBRFClassifier uses colsample_bynode (per-node column sampling),
    which is analogous to sklearn RandomForest's max_features parameter.

    Current defaults (from train_rf.py):
    - n_estimators: 250
    - max_depth: 8
    - min_child_weight: 8
    - subsample: 0.7
    - colsample_bynode: 0.6

    Args:
        trial: Optuna trial object

    Returns:
        Dictionary of hyperparameters
    """
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'max_depth': trial.suggest_int('max_depth', 4, 8),
        'min_child_weight': trial.suggest_int('min_child_weight', 8, 20),
        'subsample': trial.suggest_float('subsample', 0.7, 0.9),
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.6, 0.8),
    }


def get_prn_search_space(trial: optuna.Trial) -> Dict:
    """Define search space for PRN (Partial Response Network) hyperparameters.

    Note: subnet_nodes is NOT tuned because the mask shape from LASSO depends on it.
    The mask is created with a fixed subnet_nodes value during LASSO feature selection.

    Current defaults (from prism_analysis.py):
    - lr: 0.001
    - weight_decay: 1e-5
    - patience: 100
    - batch_size: 256

    Args:
        trial: Optuna trial object

    Returns:
        Dictionary of hyperparameters
    """
    return {
        # Note: subnet_nodes is fixed at 5 (the default used in get_mask())
        # because the mask shape depends on it and is created before tuning
        'lr': trial.suggest_float('lr', 3e-4, 3e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-4, log=True),
        'patience': trial.suggest_int('patience', 30, 100),
        'batch_size': trial.suggest_categorical('batch_size', [128, 256, 512]),
    }


# Registry mapping model types to search space functions
SEARCH_SPACE_REGISTRY: Dict[str, Callable[[optuna.Trial], Dict]] = {
    'mlp': get_mlp_search_space,
    'logreg': get_logreg_search_space,
    'xgb': get_xgb_search_space,
    'rf': get_rf_search_space,
    'prn': get_prn_search_space,
}


def get_search_space(model_type: str, trial: optuna.Trial) -> Dict:
    """Get hyperparameter search space for a given model type.

    Args:
        model_type: Model type (e.g., 'mlp', 'xgb', 'logreg', 'rf', 'prn')
        trial: Optuna trial object

    Returns:
        Dictionary of hyperparameters sampled from the search space

    Raises:
        ValueError: If model_type is not supported
    """
    model_type_lower = model_type.lower()
    if model_type_lower not in SEARCH_SPACE_REGISTRY:
        raise ValueError(
            f"Unsupported model type: {model_type}. "
            f"Supported types: {list(SEARCH_SPACE_REGISTRY.keys())}"
        )
    return SEARCH_SPACE_REGISTRY[model_type_lower](trial)


# Known-good default parameters from training notebooks.
# These are enqueued as trial 0 so Optuna always evaluates the baseline config.
DEFAULT_PARAMS: Dict[str, Dict] = {
    'mlp': {
        'n_hidden': 30,
        'lr': 0.0015,
        'weight_decay': 5e-5,
        'patience': 15,
        'batch_size': 512,
    },
    'logreg': {
        'lr': 0.02,
        'weight_decay': 1e-3,
        'patience': 12,
    },
    'xgb': {
        'max_depth': 4,
        'learning_rate': 0.055,
        'subsample': 0.75,
        'colsample_bytree': 0.65,
        'min_child_weight': 9,
        'n_estimators': 280,
        'gamma': 1.8,
        'reg_alpha': 3e-5,
        'reg_lambda': 0.03,
    },
    'rf': {
        'n_estimators': 250,
        'max_depth': 8,
        'min_child_weight': 8,
        'subsample': 0.7,
        'colsample_bynode': 0.6,
    },
    'prn': {
        'lr': 0.001,
        'weight_decay': 1e-5,
        'patience': 100,
        'batch_size': 256,
    },
}


def get_default_params(model_type: str) -> Optional[Dict]:
    """Get known-good default parameters for a model type.

    These defaults are sourced from the training notebooks and are used
    as the first enqueued trial so Optuna always evaluates the baseline.

    Args:
        model_type: Model type (e.g., 'mlp', 'xgb', 'logreg', 'rf', 'prn')

    Returns:
        Dictionary of default parameters, or None if model type is not found
    """
    return DEFAULT_PARAMS.get(model_type.lower())
