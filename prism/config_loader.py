"""Configuration loading utilities for PRiSM.

This module provides centralized configuration loading functionality used across
preprocessing, training, and analysis notebooks. It consolidates functions that
were previously duplicated across multiple notebooks.

Functions:
    load_config: Load YAML config and extract dataset name (for pipeline runners)
    load_config_if_exists: Load YAML configuration file for a dataset
    detect_target_and_id_columns: Auto-detect target and ID columns from DataFrame
    load_label_file: Load variable label mappings from CSV files
    parse_lasso_lambda_config: Parse LASSO lambda selection configuration
    apply_lasso_lambda_selection: Apply LASSO lambda selection based on configuration
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

import pandas as pd
import yaml

from prism.config import PROJ_ROOT

if TYPE_CHECKING:
    from prism.lasso import LassoResultsManager

logger = logging.getLogger(__name__)

# Import TuningConfig for hyperparameter tuning support
try:
    from prism.hyperparameter_tuning import TuningConfig
except ImportError:
    # Hyperparameter tuning module not yet available or optuna not installed
    TuningConfig = None


# =============================================================================
# LASSO LAMBDA SELECTION CONFIGURATION
# =============================================================================

VALID_LASSO_METHODS = (
    'max_test_auc',
    'min_test_auc',
    'non_inferiority',
    'by_features',
    'by_index',
)
LassoMethodType = Literal[
    'max_test_auc', 'min_test_auc', 'non_inferiority', 'by_features', 'by_index'
]


class LassoConfigurationError(ValueError):
    """Raised when LASSO configuration is invalid or missing."""

    pass


@dataclass
class LassoLambdaConfig:
    """Configuration for a single LASSO lambda selection.

    Attributes
    ----------
    method : str
        The selection method: 'max_test_auc', 'min_test_auc', 'non_inferiority',
        'by_features', or 'by_index'
    beta_threshold : float
        Threshold for LASSO beta coefficients to consider a feature as selected (default: 0.1)
    target_ratio : float
        For 'max_test_auc': fraction of max AUC to target (default: 0.99)
    min_auc : float or None
        For 'min_test_auc': minimum AUROC required
    ni_level : float or None
        For 'non_inferiority': margin for acceptable AUC loss (default: 0.1 = 10%).
        Threshold = max_auc - ni_level * (max_auc - 0.5)
    target_features : int or None
        For 'by_features': number of features to select
    lambda_index : int or None
        For 'by_index': direct lambda index selection
    """

    method: LassoMethodType
    beta_threshold: float = 0.1
    target_ratio: float = 0.99  # for max_test_auc
    min_auc: float = None  # for min_test_auc
    ni_level: float = None  # for non_inferiority
    target_features: int = None  # for by_features
    lambda_index: int = None  # for by_index

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization (excludes None values).

        Returns
        -------
        Dict[str, Any]
            Dictionary representation with only non-None values
        """
        from dataclasses import asdict

        return {k: v for k, v in asdict(self).items() if v is not None}


# Default LASSO lambda selection configuration
DEFAULT_LASSO_LAMBDA_CONFIG = LassoLambdaConfig(
    method='max_test_auc',
    beta_threshold=0.1,
    target_ratio=0.998,
)


def get_lasso_lambda_config(
    config: Optional[Dict],
    stage: Literal['blackbox', 'prn'],
    model: Optional[str] = None,
) -> LassoLambdaConfig:
    """Get LASSO lambda selection configuration with sensible defaults.

    This is a user-friendly wrapper around parse_lasso_lambda_config that
    returns default configuration when no config is provided, making it
    suitable for interactive notebook use.

    Lookup order (first match wins):
    1. Model-specific: lasso_lambda_selection.{model}.{stage}
    2. Generic: lasso_lambda_selection.{stage}
    3. Default: max_test_auc with target_ratio=0.998

    Parameters
    ----------
    config : dict or None
        Full configuration dictionary from YAML, or None for defaults
    stage : str
        Either 'blackbox' or 'prn'
    model : str, optional
        Model name (e.g., 'mlp', 'xgb') for model-specific configuration

    Returns
    -------
    LassoLambdaConfig
        Configuration for the specified stage, using defaults if not configured

    Examples
    --------
    >>> # With no config, returns default (max_test_auc with target_ratio=0.998)
    >>> lasso_config = get_lasso_lambda_config(None, 'blackbox')
    >>> lasso_config.method
    'max_test_auc'

    >>> # With model-specific config
    >>> config = {'lasso_lambda_selection': {'mlp': {'blackbox': {...}}}}
    >>> lasso_config = get_lasso_lambda_config(config, 'blackbox', model='mlp')

    >>> # Falls back to generic if model-specific not found
    >>> config = {'lasso_lambda_selection': {'blackbox': {...}}}
    >>> lasso_config = get_lasso_lambda_config(config, 'blackbox', model='xgb')
    """
    # Check if config has lasso_lambda_selection
    if config is None or 'lasso_lambda_selection' not in config:
        logger.info(
            f"No lasso_lambda_selection config for {stage}, using default: "
            f"max_test_auc with target_ratio=0.998"
        )
        return DEFAULT_LASSO_LAMBDA_CONFIG

    lasso_config = config.get('lasso_lambda_selection', {})

    # Try model-specific configuration first
    if model and model in lasso_config:
        model_config = lasso_config[model]
        if isinstance(model_config, dict) and stage in model_config:
            logger.info(f"Using model-specific lasso config: {model}.{stage}")
            return parse_lasso_lambda_config(config, stage, model=model)

    # Fall back to generic configuration
    if stage in lasso_config:
        # Check it's actually a stage config, not a model name
        stage_config = lasso_config[stage]
        if isinstance(stage_config, dict) and 'method' in stage_config:
            logger.info(f"Using generic lasso config: {stage}")
            return parse_lasso_lambda_config(config, stage)

    logger.info(
        f"No lasso_lambda_selection.{stage} config, using default: "
        f"max_test_auc with target_ratio=0.998"
    )
    return DEFAULT_LASSO_LAMBDA_CONFIG


def parse_lasso_lambda_config(
    config: Dict,
    stage: Literal['blackbox', 'prn'],
    model: Optional[str] = None,
) -> LassoLambdaConfig:
    """Parse LASSO lambda selection configuration for a specific stage.

    Parameters
    ----------
    config : dict
        Full configuration dictionary from YAML
    stage : str
        Either 'blackbox' or 'prn'
    model : str, optional
        Model name (e.g., 'mlp', 'xgb') for model-specific configuration.
        If provided, looks up lasso_lambda_selection.{model}.{stage}

    Returns
    -------
    LassoLambdaConfig
        Validated configuration for the specified stage

    Raises
    ------
    LassoConfigurationError
        If configuration is missing or invalid

    Examples
    --------
    >>> # Generic config
    >>> config = {
    ...     'lasso_lambda_selection': {
    ...         'blackbox': {'method': 'max_test_auc', 'target_ratio': 0.998},
    ...         'prn': {'method': 'by_features', 'target_features': 10}
    ...     }
    ... }
    >>> blackbox_config = parse_lasso_lambda_config(config, 'blackbox')
    >>> blackbox_config.method
    'max_test_auc'

    >>> # Model-specific config
    >>> config = {
    ...     'lasso_lambda_selection': {
    ...         'mlp': {
    ...             'blackbox': {'method': 'by_features', 'target_features': 15}
    ...         }
    ...     }
    ... }
    >>> mlp_config = parse_lasso_lambda_config(config, 'blackbox', model='mlp')
    """
    # Check for top-level key
    lasso_config = config.get('lasso_lambda_selection') if config else None
    if lasso_config is None:
        raise LassoConfigurationError(
            "LASSO lambda selection not configured.\n"
            "Add 'lasso_lambda_selection' with both 'blackbox' and 'prn' settings.\n\n"
            "Example:\n"
            "  lasso_lambda_selection:\n"
            "    blackbox:\n"
            "      method: 'max_test_auc'\n"
            "      target_ratio: 0.998\n"
            "    prn:\n"
            "      method: 'max_test_auc'\n"
            "      target_ratio: 0.998\n"
        )

    # Get stage-specific config (model-specific or generic)
    if model and model in lasso_config:
        model_config = lasso_config[model]
        stage_config = model_config.get(stage) if isinstance(model_config, dict) else None
        config_path = f"{model}.{stage}"
    else:
        stage_config = lasso_config.get(stage)
        config_path = stage

    if stage_config is None:
        raise LassoConfigurationError(
            f"LASSO lambda selection for '{config_path}' not configured.\n"
            f"Both 'blackbox' and 'prn' must be explicitly specified in lasso_lambda_selection."
        )

    # Validate method
    method = stage_config.get('method')
    if method is None:
        raise LassoConfigurationError(
            f"Missing 'method' in lasso_lambda_selection.{config_path}.\n"
            f"Valid methods: {', '.join(VALID_LASSO_METHODS)}"
        )

    if method not in VALID_LASSO_METHODS:
        raise LassoConfigurationError(
            f"Invalid LASSO selection method '{method}' for {config_path}.\n"
            f"Valid methods: {', '.join(VALID_LASSO_METHODS)}"
        )

    # Extract common parameters (support both 'beta_threshold' and legacy 'threshold')
    beta_threshold = stage_config.get('beta_threshold', stage_config.get('threshold', 0.1))

    # Validate method-specific required parameters
    if method == 'by_features':
        target_features = stage_config.get('target_features')
        if target_features is None:
            raise LassoConfigurationError(
                f"LASSO method 'by_features' requires 'target_features' parameter "
                f"for {stage} configuration."
            )
        if not isinstance(target_features, int) or target_features <= 0:
            raise LassoConfigurationError(
                f"'target_features' must be a positive integer for {stage}, "
                f"got: {target_features}"
            )
        return LassoLambdaConfig(
            method=method,
            beta_threshold=beta_threshold,
            target_features=target_features,
        )

    elif method == 'by_index':
        lambda_index = stage_config.get('lambda_index')
        if lambda_index is None:
            raise LassoConfigurationError(
                f"LASSO method 'by_index' requires 'lambda_index' parameter "
                f"for {stage} configuration."
            )
        if not isinstance(lambda_index, int) or lambda_index < 0:
            raise LassoConfigurationError(
                f"'lambda_index' must be a non-negative integer for {stage}, "
                f"got: {lambda_index}"
            )
        return LassoLambdaConfig(
            method=method,
            beta_threshold=beta_threshold,
            lambda_index=lambda_index,
        )

    elif method == 'min_test_auc':
        min_auc = stage_config.get('min_auc')
        if min_auc is None:
            raise LassoConfigurationError(
                f"LASSO method 'min_test_auc' requires 'min_auc' parameter "
                f"for {stage} configuration."
            )
        if not isinstance(min_auc, (int, float)) or not 0 < min_auc <= 1:
            raise LassoConfigurationError(
                f"'min_auc' must be a float between 0 and 1 for {stage}, " f"got: {min_auc}"
            )
        return LassoLambdaConfig(
            method=method,
            beta_threshold=beta_threshold,
            min_auc=float(min_auc),
        )

    elif method == 'non_inferiority':
        ni_level = stage_config.get('ni_level', 0.1)  # Default to 10%
        if not isinstance(ni_level, (int, float)) or not 0 < ni_level <= 1:
            raise LassoConfigurationError(
                f"'ni_level' must be a float in range (0, 1] for {stage}, " f"got: {ni_level}"
            )
        return LassoLambdaConfig(
            method=method,
            beta_threshold=beta_threshold,
            ni_level=float(ni_level),
        )

    else:  # max_test_auc
        target_ratio = stage_config.get('target_ratio', 0.99)
        if not isinstance(target_ratio, (int, float)) or not 0 < target_ratio <= 1:
            raise LassoConfigurationError(
                f"'target_ratio' must be a float between 0 and 1 for {stage}, "
                f"got: {target_ratio}"
            )
        return LassoLambdaConfig(
            method=method,
            beta_threshold=beta_threshold,
            target_ratio=float(target_ratio),
        )


def apply_lasso_lambda_selection(
    lasso_results: 'LassoResultsManager',
    lasso_config: LassoLambdaConfig,
    reference_auc: float = None,
) -> int:
    """Apply LASSO lambda selection based on configuration.

    Parameters
    ----------
    lasso_results : LassoResultsManager
        The LASSO results object to configure
    lasso_config : LassoLambdaConfig
        Validated configuration from parse_lasso_lambda_config
    reference_auc : float, optional
        Reference AUC for non-inferiority comparison (e.g., original blackbox
        model's test AUROC). Only used when method is 'non_inferiority'.
        When None, the non-inferiority method falls back to max LASSO test AUC.

    Returns
    -------
    int
        The selected lambda index

    Examples
    --------
    >>> lasso_config = parse_lasso_lambda_config(config, 'blackbox')
    >>> selected_idx = apply_lasso_lambda_selection(
    ...     lasso_results, lasso_config, reference_auc=0.85)
    """
    if lasso_config.method == 'max_test_auc':
        return lasso_results.select_lambda_max_test_auc(
            threshold=lasso_config.beta_threshold,
            target_ratio=lasso_config.target_ratio,
        )
    elif lasso_config.method == 'min_test_auc':
        return lasso_results.select_lambda_min_test_auc(
            threshold=lasso_config.beta_threshold,
            min_auc=lasso_config.min_auc,
        )
    elif lasso_config.method == 'non_inferiority':
        return lasso_results.select_lambda_non_inferiority(
            threshold=lasso_config.beta_threshold,
            ni_level=lasso_config.ni_level,
            reference_auc=reference_auc,
        )
    elif lasso_config.method == 'by_features':
        return lasso_results.select_lambda_by_features(
            target_features=lasso_config.target_features,
            threshold=lasso_config.beta_threshold,
        )
    else:  # by_index
        return lasso_results.select_lambda(
            lambda_index=lasso_config.lambda_index,
            threshold=lasso_config.beta_threshold,
        )


# =============================================================================
# HYPERPARAMETER TUNING CONFIGURATION
# =============================================================================


def get_tuning_config(config: Optional[Dict], model_type: str) -> 'TuningConfig':
    """Get hyperparameter tuning configuration with model-specific override support.

    Lookup order (first match wins):
    1. Model-specific: hyperparameter_tuning.{model_type}
    2. Global: hyperparameter_tuning (top-level settings)
    3. Default: TuningConfig(enabled=False)

    Parameters
    ----------
    config : dict or None
        Full configuration dictionary from YAML, or None for defaults
    model_type : str
        Model name (e.g., 'mlp', 'xgb', 'logreg', 'rf', 'prn')

    Returns
    -------
    TuningConfig
        Configuration for hyperparameter tuning with model-specific overrides applied

    Examples
    --------
    >>> # With no config, returns default (disabled)
    >>> tuning_config = get_tuning_config(None, 'mlp')
    >>> tuning_config.enabled
    False

    >>> # With global config only
    >>> config = {'hyperparameter_tuning': {'enabled': True, 'n_trials': 20}}
    >>> tuning_config = get_tuning_config(config, 'mlp')
    >>> tuning_config.n_trials
    20

    >>> # With model-specific override
    >>> config = {
    ...     'hyperparameter_tuning': {
    ...         'enabled': True,
    ...         'n_trials': 20,
    ...         'mlp': {'enabled': True, 'n_trials': 25}
    ...     }
    ... }
    >>> tuning_config = get_tuning_config(config, 'mlp')
    >>> tuning_config.n_trials
    25
    """
    if TuningConfig is None:
        # Hyperparameter tuning not available
        logger.warning(
            "Hyperparameter tuning module not available. "
            "Install optuna to enable hyperparameter tuning."
        )

        # Return a simple object with enabled=False
        class DisabledTuning:
            enabled = False

        return DisabledTuning()

    # If no config, return default (disabled)
    if config is None or 'hyperparameter_tuning' not in config:
        logger.info("No hyperparameter_tuning config, using default (disabled)")
        return TuningConfig(enabled=False)

    tuning_section = config.get('hyperparameter_tuning', {})

    # Start with global settings
    global_settings = {
        key: value
        for key, value in tuning_section.items()
        if not isinstance(value, dict)  # Exclude model-specific sections
    }

    # Check for model-specific override
    if model_type in tuning_section and isinstance(tuning_section[model_type], dict):
        logger.info(f"Using model-specific tuning config for {model_type}")
        # Merge global settings with model-specific overrides
        model_settings = {**global_settings, **tuning_section[model_type]}
        return TuningConfig(**model_settings)

    # Use global settings only
    if global_settings:
        logger.info(f"Using global tuning config for {model_type}")
        return TuningConfig(**global_settings)

    # Empty tuning section
    return TuningConfig(enabled=False)


# =============================================================================
# CONFIG LOADING FOR PIPELINE RUNNERS
# =============================================================================


def load_config(
    config_name: str,
    config_dir: Optional[Union[str, Path]] = None,
) -> Tuple[Dict, str]:
    """Load configuration from YAML file and extract dataset name.

    This is the primary function for pipeline runners (run_prism_pipeline.py,
    run_prism_parallel.py) that need to process config names as arguments.

    Parameters
    ----------
    config_name : str
        The config name to load (e.g., 'htx_example')
        Corresponds to example_notebooks/config/{config_name}.yaml
    config_dir : str or Path, optional
        Directory containing config files. If None, uses example_notebooks/config

    Returns
    -------
    tuple of (dict, str)
        (config_dict, dataset_name) where:
        - config_dict: Full configuration dictionary from YAML
        - dataset_name: The dataset name from config['dataset'] or inferred from config_name

    Raises
    ------
    FileNotFoundError
        If the config file does not exist
    ValueError
        If the config file is invalid or missing required fields

    Examples
    --------
    >>> config, dataset = load_config('htx_example')
    >>> print(f"Dataset: {dataset}, Models: {config.get('models', ['mlp'])}")
    Dataset: htx_example, Models: ['mlp10']
    """
    if config_dir is None:
        config_dir = PROJ_ROOT / "example_notebooks" / "config"
    else:
        config_dir = Path(config_dir)

    config_file = config_dir / f"{config_name}.yaml"

    if not config_file.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_file}\n"
            f"Create a config file at example_notebooks/config/{config_name}.yaml "
            f"or use PRISM_DATASET for quick mode without a config file."
        )

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in config file {config_file}: {e}")

    # Extract dataset name: explicit 'dataset' field or infer from config name
    dataset_name = config.get('dataset')
    if not dataset_name:
        # Fallback: use config name (backward compatibility)
        # This allows old configs without 'dataset:' field to still work
        dataset_name = config_name
        logger.info(
            f"No 'dataset:' field in {config_name}.yaml, using config name as dataset: {dataset_name}"
        )

    logger.info(f"Loaded config '{config_name}' for dataset '{dataset_name}'")
    return config, dataset_name


def get_models_from_config(config: Dict, default_models: List[str] = None) -> List[str]:
    """Extract models list from config, with fallback to defaults.

    Parameters
    ----------
    config : dict
        Configuration dictionary (from load_config or load_config_if_exists)
    default_models : list of str, optional
        Default models if not specified in config. Defaults to ['mlp'].

    Returns
    -------
    list of str
        List of model names to train/analyze

    Examples
    --------
    >>> config, _ = load_config('htx_example')
    >>> models = get_models_from_config(config)
    >>> print(models)
    ['mlp10']
    """
    if default_models is None:
        default_models = ['mlp']

    models = config.get('models')
    if models:
        return models if isinstance(models, list) else [models]
    return default_models


# =============================================================================
# DEFAULT CANDIDATE LISTS
# =============================================================================
# These are the default candidates used for auto-detection across all notebooks.
# They can be overridden by config files or explicitly passed to functions.


DEFAULT_TARGET_CANDIDATES = [
    'var1',
    'event_oneyear',
    'target',
    'outcome',
    'y',
    'label',
    'Deceased_30D',
]

DEFAULT_ID_CANDIDATES = [
    'trr_id_code',
    'TRR_ID_CODE',
    'id',
    'patient_id',
    'subject_id',
    'var_id',
]

DEFAULT_LABEL_FILE_CANDIDATES = [
    # Dataset-specific files (highest priority)
    '{dataset_prefix}_variable_labels.csv',  # Most specific naming
    '{dataset_prefix}_variable_label.csv',
    '{dataset_prefix}_labels.csv',
    '{dataset_prefix}_label.csv',
    # Generic fallback files (lower priority)
    'data_label.csv',  # Generic in notebooks dir
    'data_labels.csv',
    'generic_variable_labels.csv',  # Template
    'data/raw/data_label.csv',  # Generic in raw data
    'data/raw/data_labels.csv',
]


# =============================================================================
# CONFIGURATION LOADING
# =============================================================================


def load_config_if_exists(
    dataset_prefix: str,
    config_dir: Optional[Union[str, Path]] = None,
) -> Dict:
    """Load configuration from YAML file if it exists.

    Looks for a YAML file named '{dataset_prefix}.yaml' in the config directory.
    This is used to load dataset-specific settings for preprocessing, training,
    and analysis notebooks.

    Parameters
    ----------
    dataset_prefix : str
        The dataset prefix to look for in config files (e.g., 'htx_example')
    config_dir : str or Path, optional
        Directory containing config files. If None, uses example_notebooks/config

    Returns
    -------
    dict
        Configuration dictionary with keys like 'integer_encoding', 'selected_features',
        'target_candidates', 'id_candidates', 'splitting_method', etc.
        Returns empty dict if no config file found.

    Examples
    --------
    >>> config = load_config_if_exists('htx_example')
    >>> target_candidates = config.get('target_candidates', DEFAULT_TARGET_CANDIDATES)
    """
    if config_dir is None:
        # Use project base directory to find config directory
        config_dir = PROJ_ROOT / "example_notebooks" / "config"

        if not config_dir.exists():
            logger.warning(f"Config directory not found at {config_dir}, skipping config loading")
            return {}
    else:
        config_dir = Path(config_dir)

    config_file = config_dir / f"{dataset_prefix}.yaml"

    if not config_file.exists():
        logger.info(f"No config file found for '{dataset_prefix}' at {config_file}")
        return {}

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        logger.info(f"Loaded config from {config_file}")
        return config
    except Exception as e:
        logger.warning(f"Failed to load config from {config_file}: {e}")
        return {}


# =============================================================================
# COLUMN DETECTION
# =============================================================================


def detect_target_and_id_columns(
    df: pd.DataFrame,
    target_candidates: Optional[List[str]] = None,
    id_candidates: Optional[List[str]] = None,
    verbose: bool = True,
) -> Tuple[Optional[str], Optional[str]]:
    """Automatically detect target and ID columns from a DataFrame.

    Searches for columns matching the candidate lists in order. For target columns,
    if no candidate is found, falls back to looking for binary (0/1) columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to search for target and ID columns
    target_candidates : list of str, optional
        List of potential target column names to search for in order.
        If None, uses DEFAULT_TARGET_CANDIDATES.
    id_candidates : list of str, optional
        List of potential ID column names to search for in order.
        If None, uses DEFAULT_ID_CANDIDATES.
    verbose : bool, default True
        Whether to print detection results

    Returns
    -------
    tuple of (str or None, str or None)
        (target_column, id_column) - None if not found

    Examples
    --------
    >>> target_col, id_col = detect_target_and_id_columns(train_df)
    >>> print(f"Target: {target_col}, ID: {id_col}")
    """
    if target_candidates is None:
        target_candidates = DEFAULT_TARGET_CANDIDATES
    if id_candidates is None:
        id_candidates = DEFAULT_ID_CANDIDATES

    if verbose:
        print("Detecting target and ID columns...")
        print(f"  Available columns: {list(df.columns)}")
        print(f"  Target candidates: {target_candidates}")
        print(f"  ID candidates: {id_candidates}")

    # Detect target column
    target_column = None
    for candidate in target_candidates:
        if candidate in df.columns:
            target_column = candidate
            if verbose:
                print(f"  Found target column: {candidate}")
            break

    # Fallback: look for binary columns
    if target_column is None:
        if verbose:
            print(f"  WARNING: No target column found from candidates: {target_candidates}")
        binary_cols = []
        for col in df.columns:
            unique_vals = sorted(df[col].dropna().unique())
            if len(unique_vals) == 2 and set(unique_vals) == {0, 1}:
                binary_cols.append(col)

        if binary_cols:
            target_column = binary_cols[0]
            if verbose:
                print(f"  Found binary column as fallback target: {target_column}")
        else:
            if verbose:
                print("  No binary columns found for fallback")

    # Detect ID column
    id_column = None
    for candidate in id_candidates:
        if candidate in df.columns:
            id_column = candidate
            if verbose:
                print(f"  Found ID column: {candidate}")
            break

    if id_column is None and verbose:
        print(f"  WARNING: No ID column found from candidates: {id_candidates}")
        print("  Will create synthetic IDs if needed")

    return target_column, id_column


# =============================================================================
# LABEL FILE LOADING
# =============================================================================


def load_label_file(
    label_file_candidates: Optional[List[str]] = None,
    data_dir: Optional[Union[str, Path]] = None,
    dataset_prefix: Optional[str] = None,
    verbose: bool = False,
) -> Optional[Dict[str, str]]:
    """Load variable label mapping from CSV file.

    Searches through candidate label file paths in priority order and loads the first
    one found. Dataset-specific files take precedence over generic templates. If
    multiple files exist, emits a warning about shadowing.

    Supports multiple CSV formats:
    - Template format with 'processed_name' and 'user_label' columns
    - Simple two-column format (name, label)
    - Original HTx format with escaped newlines

    Parameters
    ----------
    label_file_candidates : list of str, optional
        List of potential label file paths to search.
        If None, uses DEFAULT_LABEL_FILE_CANDIDATES.
        Can contain '{dataset_prefix}' placeholder which will be replaced.
    data_dir : str or Path, optional
        Base directory for relative paths. If None, uses example_notebooks directory.
    dataset_prefix : str, optional
        Dataset prefix to substitute in '{dataset_prefix}' placeholders.
    verbose : bool, default False
        Whether to print detailed search progress (for interactive debugging).
        When True, prints candidates checked, files found, and which was selected.

    Returns
    -------
    dict or None
        Dictionary mapping processed variable names to user-friendly labels.
        Returns None if no label file found.

    Examples
    --------
    >>> labels = load_label_file(dataset_prefix='htx_example')
    >>> if labels:
    ...     display_name = labels.get('age_ordinal', 'age_ordinal')

    >>> # Interactive debugging
    >>> labels = load_label_file(
    ...     dataset_prefix='htx_example',
    ...     verbose=True
    ... )
    Scanning for label files...
      Candidates to check: 7
      Dataset prefix: htx_example
      Found 2 existing file(s):
        1. htx_example_labels.csv - SELECTED
        2. generic_variable_labels.csv - (shadowed)
    """
    if label_file_candidates is None:
        label_file_candidates = DEFAULT_LABEL_FILE_CANDIDATES.copy()

    # Substitute dataset_prefix placeholder
    if dataset_prefix:
        label_file_candidates = [
            f.replace('{dataset_prefix}', dataset_prefix) for f in label_file_candidates
        ]

    if data_dir is None:
        data_dir = PROJ_ROOT / "example_notebooks"
    else:
        data_dir = Path(data_dir)

    # =========================================================================
    # Phase 1: Scan all candidates and collect existing files
    # =========================================================================
    logger.debug(f"Scanning {len(label_file_candidates)} label file candidates")
    found_files = []  # List of (candidate_name, resolved_path) tuples

    for label_file in label_file_candidates:
        # Handle absolute paths and relative paths
        if label_file.startswith('data/'):
            # For data/ paths, try multiple possible locations
            possible_paths = [
                PROJ_ROOT / "example_notebooks" / label_file,
                PROJ_ROOT / label_file,
                data_dir / label_file,
            ]
            label_path = None
            for path in possible_paths:
                if path.exists():
                    label_path = path
                    break
            if label_path is None:
                continue
        else:
            label_path = data_dir / label_file

        if label_path.exists():
            found_files.append((label_file, label_path))

    # =========================================================================
    # Phase 2: Categorize and warn about multiple files
    # =========================================================================
    if verbose:
        print("Scanning for label files...")
        print(f"  Candidates to check: {len(label_file_candidates)}")
        print(f"  Dataset prefix: {dataset_prefix or 'None'}")
        print(f"  Base directory: {data_dir}")

    if len(found_files) == 0:
        if verbose:
            print("  No label files found")
        logger.info("No label file found, will use original variable names")
        return None

    if verbose:
        print(f"  Found {len(found_files)} existing file(s):")
        for i, (candidate, path) in enumerate(found_files, 1):
            marker = "SELECTED" if i == 1 else "(shadowed)"
            print(f"    {i}. {path.name} - {marker}")

    # Categorize files as dataset-specific vs generic
    if len(found_files) > 1:
        dataset_specific = [
            f
            for f in found_files
            if dataset_prefix and dataset_prefix in f[0]  # candidate name contains prefix
        ]
        generic = [f for f in found_files if f not in dataset_specific]

        # Emit appropriate log message
        if len(dataset_specific) > 1:
            logger.warning(
                f"Multiple dataset-specific label files found for '{dataset_prefix}'. "
                f"Using first by precedence: {found_files[0][1].name}"
            )
        elif len(dataset_specific) >= 1 and len(generic) >= 1:
            # Dataset-specific shadowing generic (expected, good)
            logger.info(
                f"Using dataset-specific label file '{found_files[0][1].name}' "
                f"(shadowing {len(generic)} generic file(s))"
            )
        else:  # Multiple generic files
            logger.warning(
                f"Multiple generic label files found. "
                f"Using first by precedence: {found_files[0][1].name}"
            )

        # Always log full list at DEBUG level
        logger.debug(f"All found files: {[str(p) for _, p in found_files]}")

    # Select first file by precedence
    selected_candidate, selected_path = found_files[0]
    logger.info(f"Found label file: {selected_path}")

    # =========================================================================
    # Phase 3: Load selected file
    # =========================================================================
    try:
        # Try to detect format by reading with headers first
        label_df = pd.read_csv(selected_path)

        # Check if this is template format (has required columns)
        if 'processed_name' in label_df.columns and 'user_label' in label_df.columns:
            # Template format with user_label column
            variable_labels = {}
            for _, row in label_df.iterrows():
                processed_name = row['processed_name']
                user_label = row['user_label']
                # Skip placeholder labels that start with [USER:
                if isinstance(user_label, str) and not user_label.startswith('[USER:'):
                    variable_labels[processed_name] = user_label
                else:
                    variable_labels[processed_name] = processed_name
            logger.info(
                f"Loaded {len(variable_labels)} labels from {selected_candidate} (template format)"
            )
            return variable_labels

        # Not template format - try simple two-column formats
        # Re-read without headers
        label_df = pd.read_csv(selected_path, header=None)

        if len(label_df.columns) >= 2:
            variable_labels = dict(zip(label_df.iloc[:, 0], label_df.iloc[:, 1]))
            # Fix escaped newlines in labels (for original htx format)
            for key, value in variable_labels.items():
                if isinstance(value, str):
                    variable_labels[key] = value.replace('\\n', '\n')
            logger.info(
                f"Loaded {len(variable_labels)} labels from {selected_candidate} (simple format)"
            )
            return variable_labels

    except Exception as e:
        logger.warning(f"Error loading {selected_candidate}: {e}")
        return None

    logger.info("No label file found, will use original variable names")
    return None
