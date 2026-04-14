"""Notebook utilities for PRiSM analysis notebooks.

This module provides convenience functions for loading data, models, and metadata
in interactive notebooks. It consolidates common loading patterns to keep notebooks
focused on analysis workflow.

Functions:
    validate_dataset_configured: Validate dataset is configured, raise helpful error if not
    load_train_test_val_data: Load latest train/test/val data files by timestamp
    load_model_checkpoint: Load trained model from checkpoint with unified format handling
    load_preprocessing_metadata: Load preprocessing metadata including OneHotGroupManager
    get_analysis_params: Extract analysis parameters from config with sensible defaults
"""

import json
import logging
import os
import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn

from prism.model_persistence import save_partial_responses
from prism.preprocessing import OneHotGroupManager, PRiSMScaler

if TYPE_CHECKING:
    from prism.config_loader import LassoLambdaConfig

logger = logging.getLogger(__name__)


def validate_dataset_configured(
    dataset_prefix: Optional[str],
    config_name: Optional[str] = None,
) -> None:
    """Validate dataset is configured, raising helpful error if not.

    This replaces the ~15 lines of boilerplate validation code in each notebook.

    Parameters
    ----------
    dataset_prefix : str or None
        The dataset prefix from config (e.g., 'htx_example', 'my_dataset')
    config_name : str or None
        The config name if using YAML config mode

    Raises
    ------
    ValueError
        If dataset_prefix is None, with helpful instructions for configuration
    """
    if dataset_prefix is None:
        raise ValueError(
            "No dataset configured!\n\n"
            "Please configure your dataset in the .env file:\n"
            "  1. Copy .env.example to .env\n"
            "  2. Set PRISM_CONFIG=your_config (loads example_notebooks/config/your_config.yaml)\n"
            "     OR set PRISM_DATASET=your_dataset (quick mode, no YAML needed)\n\n"
            "Example .env contents:\n"
            "  PRISM_CONFIG=htx_example"
        )

    print(f"Dataset: {dataset_prefix}")
    if config_name:
        print(f"Config: {config_name}.yaml")


def load_train_test_val_data(
    model_prefix: str,
    dataset_prefix: str,
    processed_data_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Load latest train/test/val data files by timestamp.

    Finds files matching pattern: {dataset}_{model}_*_{train,test,val}.csv
    Groups by timestamp and returns the most recent set.

    Parameters
    ----------
    model_prefix : str
        Model type prefix (e.g., 'mlp', 'xgb', 'logreg', 'mlp10')
    dataset_prefix : str
        Dataset prefix (e.g., 'htx_example', 'my_dataset')
    processed_data_dir : Path
        Directory containing processed data files

    Returns
    -------
    tuple
        (train_df, test_df, val_df, timestamp)
        - train_df: Training data DataFrame
        - test_df: Test data DataFrame
        - val_df: Validation data DataFrame
        - timestamp: The timestamp string from the loaded files

    Raises
    ------
    FileNotFoundError
        If no matching data files are found or required splits are missing
    """
    # Handle special case for mlp10, where mlp data should be used
    data_model_prefix = 'mlp' if model_prefix == 'mlp10' else model_prefix
    dataset_filenames = f'{dataset_prefix}_{data_model_prefix}_*'
    dataset_files = list(processed_data_dir.glob(dataset_filenames))

    if not dataset_files:
        raise FileNotFoundError(
            f"No data files matching {dataset_filenames} found in {processed_data_dir}"
        )

    # Group files by timestamp
    file_groups: Dict[str, List[Path]] = {}
    for file_path in dataset_files:
        # Extract timestamp from filename
        parts = file_path.stem.split('_')
        if len(parts) >= 3:
            timestamp = '_'.join(parts[2:-1])  # Extract timestamp portion
            if timestamp not in file_groups:
                file_groups[timestamp] = []
            file_groups[timestamp].append(file_path)

    # Get the latest timestamp
    latest_timestamp = sorted(file_groups.keys())[-1] if file_groups else None

    if not latest_timestamp:
        raise FileNotFoundError("Could not identify timestamp in data filenames")

    print(f"Loading data files with timestamp: {latest_timestamp}")

    # Find train, test, and validation files for this timestamp
    train_file = next(
        (f for f in file_groups[latest_timestamp] if f.name.endswith('_train.csv')), None
    )
    test_file = next(
        (f for f in file_groups[latest_timestamp] if f.name.endswith('_test.csv')), None
    )
    val_file = next(
        (f for f in file_groups[latest_timestamp] if f.name.endswith('_val.csv')), None
    )

    if train_file is None or test_file is None or val_file is None:
        raise FileNotFoundError("Could not find all required data files (train, test, val)")

    # Load the datasets
    print(f"Loading train data from: {train_file}")
    train_df = pd.read_csv(train_file, comment='#')

    print(f"Loading test data from: {test_file}")
    test_df = pd.read_csv(test_file, comment='#')

    print(f"Loading validation data from: {val_file}")
    val_df = pd.read_csv(val_file, comment='#')

    return train_df, test_df, val_df, latest_timestamp


def load_model_checkpoint(
    model_prefix: str,
    dataset_prefix: str,
    models_dir: Path,
    device: torch.device,
    X_train: pd.DataFrame,
    random_seed: int = 257,
) -> Tuple[nn.Module, Optional[PRiSMScaler], Dict[str, Any], List[str]]:
    """Load trained model from checkpoint with unified format handling.

    Handles 3 checkpoint formats:
    1. Direct model serialization (nn.Module)
    2. Checkpoint dict with 'model' key (full model)
    3. Checkpoint dict with state_dict (requires reconstruction)

    Parameters
    ----------
    model_prefix : str
        Model type prefix (e.g., 'mlp', 'xgb', 'logreg', 'mlp10')
    dataset_prefix : str
        Dataset prefix (e.g., 'htx_example', 'my_dataset')
    models_dir : Path
        Directory containing model files
    device : torch.device
        Device to load model onto
    X_train : pd.DataFrame
        Training features (used for input_dim inference if not in checkpoint)
    random_seed : int
        Random seed fallback if not in checkpoint (default: 257)

    Returns
    -------
    tuple
        (model, scaler, hyperparameters, feature_names)
        - model: Loaded model (nn.Module or wrapped sklearn model)
        - scaler: PRiSMScaler or None if not in checkpoint
        - hyperparameters: Dict of model hyperparameters
        - feature_names: List of feature names

    Raises
    ------
    FileNotFoundError
        If no matching model files are found
    ValueError
        If checkpoint format is unknown or missing required keys
    """
    # Handle special case for mlp10, where mlp model should be used
    load_model_prefix = 'mlp' if model_prefix == 'mlp10' else model_prefix
    load_model_name = f"{dataset_prefix}_{load_model_prefix}"

    # Set base model directory
    if model_prefix == 'mlp10':
        # MODELS_DIR is currently: .../mlp10/models/
        # We need to look in: .../mlp/models/
        model_dir = models_dir.parent.parent / 'mlp' / 'models'
    else:
        model_dir = models_dir

    # Search for model files - first directly, then in subdirectory
    model_filenames = f'{load_model_name}_model_*.pt'
    model_files = list(model_dir.glob(model_filenames))

    if not model_files:
        # Try subdirectory (how training notebooks save models: models/{prefix}_{model}/)
        subdir = model_dir / load_model_name
        if subdir.exists():
            logger.info(f"Model not found in {model_dir}, checking subdirectory {subdir}")
            model_dir = subdir
            model_files = list(model_dir.glob(model_filenames))

    if not model_files:
        raise FileNotFoundError(
            f"No model files found in {model_dir} directory matching {model_filenames}"
        )

    # Get the most recent model file
    latest_model_path = max(model_files, key=os.path.getctime)
    print(f"Loading model from: {latest_model_path}")

    # Load the checkpoint
    checkpoint = torch.load(latest_model_path, map_location=device, weights_only=False)

    # Initialize scaler from checkpoint if present
    scaler = _extract_scaler_from_checkpoint(checkpoint)

    # Handle different checkpoint formats
    if isinstance(checkpoint, torch.nn.Module):
        # Case 1: Direct model serialization (full Python model)
        model, hyperparameters, feature_names = _load_direct_model(
            checkpoint, X_train, random_seed
        )
    elif isinstance(checkpoint, dict) and 'model' in checkpoint:
        # Case 2: Checkpoint contains full model object
        model, scaler, hyperparameters, feature_names = _load_model_from_dict(
            checkpoint, X_train, random_seed, scaler
        )
    elif isinstance(checkpoint, dict):
        # Case 3: Checkpoint contains model state dict and requires reinstantiation
        model, scaler, hyperparameters, feature_names = _load_model_from_state_dict(
            checkpoint, X_train, device, random_seed, scaler
        )
    else:
        raise ValueError(f"Unknown checkpoint format: {type(checkpoint)}")

    # Set model to evaluation mode only if it's a PyTorch model
    if isinstance(model, torch.nn.Module):
        model.eval()
        model = model.to(device)
        print(f"PyTorch model moved to device: {device}")

    # Sync XGBoost/RF model device to match user's selected device
    _sync_xgb_device(model, device)

    # Print model information
    model_class = type(model)
    print(f"\nModel class: {model_class.__name__}")
    print("\nModel hyperparameters:")
    for param, value in hyperparameters.items():
        print(f"  {param}: {value}")

    print(f"\nNumber of features: {len(feature_names)}")
    print(f"Feature names: {feature_names[:5]}... (showing first 5)")

    return model, scaler, hyperparameters, feature_names


def _extract_scaler_from_checkpoint(checkpoint: Any) -> Optional[PRiSMScaler]:
    """Extract and validate scaler from checkpoint."""
    if isinstance(checkpoint, dict) and 'scaler' in checkpoint:
        loaded_scaler = checkpoint['scaler']
        if isinstance(loaded_scaler, PRiSMScaler):
            print("Successfully loaded PRiSMScaler from checkpoint")
            return loaded_scaler
        else:
            print(
                f"Warning: Found non-PRiSMScaler in checkpoint "
                f"({type(loaded_scaler).__name__}), wrapping it in PRiSMScaler"
            )
            return PRiSMScaler(scaler=loaded_scaler)
    else:
        warnings.warn("No scaler found in checkpoint. Setting scaler = None.", UserWarning)
        return None


def _load_direct_model(
    model: nn.Module,
    X_train: pd.DataFrame,
    random_seed: int,
) -> Tuple[nn.Module, Dict[str, Any], List[str]]:
    """Load directly serialized PyTorch model (Case 1)."""
    logger.info("Case 1: Loading directly serialized PyTorch model")
    model_class = type(model)
    print(f"Successfully loaded a direct PyTorch model of type: {model_class.__name__}")

    # Check if the model is a wrapper
    if hasattr(model, 'model'):
        underlying_model = model.model
        underlying_model_class = type(underlying_model)
        print(f"This is a wrapper model containing: {underlying_model_class.__name__}")

    # Extract hyperparameters from model if possible
    hyperparameters: Dict[str, Any] = {}
    if hasattr(model, 'input_dim'):
        hyperparameters['input_dim'] = model.input_dim
    else:
        hyperparameters['input_dim'] = X_train.shape[1]
        print(
            f"Input dimension not found in model, using X_train shape: {hyperparameters['input_dim']}"
        )

    # Extract random seed if available
    if hasattr(model, 'random_seed'):
        hyperparameters['random_seed'] = model.random_seed
    elif hasattr(model, 'seed'):
        hyperparameters['seed'] = model.seed
    else:
        hyperparameters['random_seed'] = random_seed
        print(f"Random seed not found in model, using notebook random_seed: {random_seed}")

    # Extract feature names if available or use X_train columns
    if hasattr(model, 'feature_names') and model.feature_names is not None:
        feature_names = list(model.feature_names)
    else:
        feature_names = list(X_train.columns)

    return model, hyperparameters, feature_names


def _load_model_from_dict(
    checkpoint: Dict[str, Any],
    X_train: pd.DataFrame,
    random_seed: int,
    scaler: Optional[PRiSMScaler],
) -> Tuple[nn.Module, Optional[PRiSMScaler], Dict[str, Any], List[str]]:
    """Load model from checkpoint dictionary with full model (Case 2)."""
    logger.info("Case 2: Loading model from checkpoint dictionary with full model")
    model = checkpoint['model']
    model_class = type(model)
    print(f"Successfully loaded model from checkpoint dictionary: {model_class.__name__}")

    # Re-extract scaler from this specific checkpoint format
    if 'scaler' in checkpoint:
        loaded_scaler = checkpoint['scaler']
        if isinstance(loaded_scaler, PRiSMScaler):
            scaler = loaded_scaler
            print("Successfully loaded PRiSMScaler from checkpoint")
        else:
            print(
                f"Warning: Found non-PRiSMScaler in checkpoint "
                f"({type(loaded_scaler).__name__}), wrapping it in PRiSMScaler"
            )
            scaler = PRiSMScaler(scaler=loaded_scaler)
    else:
        warnings.warn(
            "No scaler found in checkpoint. Setting scaler = PRiSMScaler(scaler=None).",
            UserWarning,
        )
        scaler = PRiSMScaler(scaler=None)

    # Check if this is a wrapper model and display underlying model info
    if hasattr(model, 'model'):
        underlying_model = model.model
        underlying_model_class = type(underlying_model)
        print(f"This is a wrapper model containing: {underlying_model_class.__name__}")
    elif 'underlying_model_class' in checkpoint:
        underlying_model_class = checkpoint['underlying_model_class']
        print(f"This is a wrapper model containing: {underlying_model_class.__name__}")
        if 'underlying_model' in checkpoint:
            print(f"Underlying model parameters: {checkpoint['underlying_model']}")

    # Extract hyperparameters from model config if available
    hyperparameters = checkpoint.get('model_config', {})
    if not hyperparameters and hasattr(model, 'input_dim'):
        hyperparameters['input_dim'] = model.input_dim
    if 'input_dim' not in hyperparameters and 'input_size' in hyperparameters:
        hyperparameters['input_dim'] = hyperparameters['input_size']
    if 'input_dim' not in hyperparameters:
        hyperparameters['input_dim'] = X_train.shape[1]
        print(
            f"Input dimension not found in model config, using X_train shape: {hyperparameters['input_dim']}"
        )

    # Extract random seed
    if 'random_seed' in hyperparameters:
        pass
    elif 'seed' in hyperparameters:
        hyperparameters['random_seed'] = hyperparameters['seed']
    elif hasattr(model, 'random_seed'):
        hyperparameters['random_seed'] = model.random_seed
    elif hasattr(model, 'seed'):
        hyperparameters['random_seed'] = model.seed
    else:
        hyperparameters['random_seed'] = random_seed
        print(f"Random seed not found in model config, using notebook random_seed: {random_seed}")

    # Extract feature names
    feature_names = checkpoint.get('feature_names', list(X_train.columns))

    return model, scaler, hyperparameters, feature_names


def _load_model_from_state_dict(
    checkpoint: Dict[str, Any],
    X_train: pd.DataFrame,
    device: torch.device,
    random_seed: int,
    scaler: Optional[PRiSMScaler],
) -> Tuple[nn.Module, Optional[PRiSMScaler], Dict[str, Any], List[str]]:
    """Load model by reconstructing from state dict (Case 3)."""
    logger.info("Case 3: Reconstructing model from state dict and metadata")

    # Extract model information
    model_class = checkpoint.get('model_class')
    model_state_dict = checkpoint.get('model_state_dict')
    hyperparameters = checkpoint.get('hyperparameters', {})
    feature_names = checkpoint.get('feature_names', list(X_train.columns))

    # Re-extract scaler from checkpoint
    if 'scaler' in checkpoint:
        loaded_scaler = checkpoint['scaler']
        if isinstance(loaded_scaler, PRiSMScaler):
            scaler = loaded_scaler
            print("Successfully loaded PRiSMScaler from checkpoint")
        else:
            print(
                f"Warning: Found non-PRiSMScaler in checkpoint "
                f"({type(loaded_scaler).__name__}), wrapping it in PRiSMScaler"
            )
            scaler = PRiSMScaler(scaler=loaded_scaler)
    else:
        warnings.warn(
            "No scaler found in checkpoint. Setting scaler = PRiSMScaler(scaler=None).",
            UserWarning,
        )
        scaler = PRiSMScaler(scaler=None)

    # Check if we have enough information to reconstruct the model
    if not model_class or not model_state_dict:
        raise ValueError(
            "Checkpoint does not contain required 'model_class' or 'model_state_dict'"
        )

    # Check if this is a wrapper model
    if 'underlying_model_class' in checkpoint:
        underlying_model_class = checkpoint['underlying_model_class']
        print(f"This is a wrapper model containing: {underlying_model_class.__name__}")

    # If 'input_dim' not included in hyperparameters, calculate from X
    if 'input_dim' not in hyperparameters:
        hyperparameters['input_dim'] = X_train.shape[1]
        print(
            f"Input dimension not found in hyperparameters, using X_train shape: {hyperparameters['input_dim']}"
        )

    # Check for either of 'random_seed' or 'seed' in hyperparameters
    random_seed_param = 'random_seed' if 'random_seed' in hyperparameters else 'seed'
    if random_seed_param in hyperparameters:
        model_random_seed = hyperparameters[random_seed_param]
        print(f"Using model random_seed from model hyperparameters: {model_random_seed}")
    else:
        model_random_seed = random_seed
        print(f"Using model random_seed from notebook: {random_seed}")

    # Recreate the model
    model = model_class(
        input_features=hyperparameters['input_dim'], random_seed=model_random_seed
    ).to(device)
    model.load_state_dict(model_state_dict)
    print(f"Successfully reconstructed model from state dict: {model_class.__name__}")

    return model, scaler, hyperparameters, feature_names


def _sync_xgb_device(model: Any, device: torch.device) -> None:
    """Sync XGBoost/RF model device to match user's selected device."""
    from prism.device_tools import to_xgb_device

    # Check if wrapped in SklearnWrapper
    inner_model = model.model if hasattr(model, 'model') else model

    if hasattr(inner_model, 'set_params') and hasattr(inner_model, 'get_params'):
        try:
            params = inner_model.get_params()
            # Check if XGBoost-like (has n_estimators and objective)
            if 'n_estimators' in params and 'objective' in params:
                xgb_device = to_xgb_device(device)
                inner_model.set_params(device=xgb_device)
                print(
                    f"XGBoost model device set to: {xgb_device} (matching user device: {device})"
                )

                # Re-detect GPU capability on the wrapper after device change
                if hasattr(model, 'refresh_gpu_detection'):
                    gpu_enabled = model.refresh_gpu_detection()
                    if gpu_enabled:
                        from prism.wrapper import CUPY_AVAILABLE

                        if CUPY_AVAILABLE:
                            print("GPU-accelerated inference enabled (cupy available)")
                        else:
                            print(
                                "GPU model detected but cupy not installed -- using CPU fallback"
                            )
        except Exception:
            pass  # Not XGBoost, skip


def load_preprocessing_metadata(
    dataset_prefix: str,
    processed_data_dir: Path,
) -> Tuple[Optional[OneHotGroupManager], Optional[Dict[str, Any]]]:
    """Load preprocessing metadata including OneHotGroupManager.

    Finds latest metadata file with exact prefix matching to avoid
    confusing similar dataset names (e.g., htx_unos vs htx_unos_top40).

    Parameters
    ----------
    dataset_prefix : str
        Dataset prefix (e.g., 'htx_example', 'my_dataset')
    processed_data_dir : Path
        Directory containing processed data files

    Returns
    -------
    tuple
        (group_manager, metadata_dict)
        - group_manager: OneHotGroupManager or None if not in metadata
        - metadata_dict: Full preprocessing metadata dict or None if file not found
    """
    # Find the latest preprocessing metadata file
    # Use exact prefix matching to avoid confusing similar dataset names
    metadata_pattern = f"preprocessing_metadata_{dataset_prefix}_*.json"
    all_metadata_files = list(Path(processed_data_dir).glob(metadata_pattern))

    # Filter for exact prefix match: prefix must be followed by underscore and timestamp
    # Pattern: preprocessing_metadata_{DATASET_PREFIX}_YYYYMMDD_HHMMSS.json
    exact_match_pattern = re.compile(
        rf"^preprocessing_metadata_{re.escape(dataset_prefix)}_\d{{8}}_\d{{6}}\.json$"
    )
    metadata_files = [f for f in all_metadata_files if exact_match_pattern.match(f.name)]

    if not metadata_files and all_metadata_files:
        # Warn about skipped files that didn't match exactly
        skipped = [f.name for f in all_metadata_files if f not in metadata_files]
        print(
            f"Found {len(all_metadata_files)} files matching glob, but none with exact prefix match."
        )
        print(f"Skipped (wrong prefix): {skipped}")

    if not metadata_files:
        # Fallback: try any preprocessing metadata (old naming without dataset prefix)
        print(
            f"No dataset-specific metadata found ({metadata_pattern}), trying generic pattern..."
        )
        metadata_files = list(Path(processed_data_dir).glob("preprocessing_metadata_*.json"))

    if metadata_files:
        # Sort by modification time to get the most recent file (not alphabetically!)
        metadata_path = max(metadata_files, key=lambda p: p.stat().st_mtime)
        print(f"Loading preprocessing metadata from: {metadata_path.name}")

        with open(metadata_path, 'r') as f:
            preprocessing_metadata = json.load(f)

        # Load OneHotGroupManager
        if (
            'onehot_group_manager' in preprocessing_metadata
            and preprocessing_metadata['onehot_group_manager'] is not None
        ):
            group_manager = OneHotGroupManager.from_preprocessing_metadata(preprocessing_metadata)
            print(f"Loaded OneHotGroupManager with {len(group_manager.groups_dict)} groups")
            return group_manager, preprocessing_metadata
        else:
            print("No OneHotGroupManager found in metadata - collapse disabled")
            return None, preprocessing_metadata
    else:
        print("No preprocessing metadata found - collapse disabled")
        return None, None


def get_analysis_params(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract analysis parameters from config with sensible defaults.

    Parameters
    ----------
    config : dict or None
        Configuration dictionary from YAML file, or None for defaults

    Returns
    -------
    dict
        Dictionary with keys:
        - random_seed: int (default: 257)
        - partial_response_method: str (default: 'lebesgue')
        - trim_quantile: float or None (default: None)
        - save_nomogram_json: bool (default: True)
        - save_figs: bool (default: True)
        - batch_size_scaler: int (default: 1)
    """
    defaults = {
        'random_seed': 257,
        'partial_response_method': 'lebesgue',
        'trim_quantile': None,
        'save_nomogram_json': True,
        'save_figs': True,
        'batch_size_scaler': 1,
    }

    if config is None:
        return defaults

    return {k: config.get(k, v) for k, v in defaults.items()}


# =============================================================================
# TERM COUNTING UTILITIES
# =============================================================================


def get_term_counts(
    lasso_results: Any,
    threshold: float = 0.1,
) -> Dict[str, int]:
    """Count selected univariate and bivariate terms from LASSO results.

    Uses the same threshold-based selection as the nomogram export pipeline,
    so counts match what appears in the nomogram JSON files.

    Parameters
    ----------
    lasso_results : LassoResultsManager
        LASSO results with a selected lambda
    threshold : float
        Coefficient magnitude threshold for feature selection (default: 0.1)

    Returns
    -------
    dict
        {'n_univ': int, 'n_biv': int, 'n_total': int}
    """
    n_univ = len(lasso_results.get_selected_univariate_indices(threshold=threshold))
    n_biv = len(lasso_results.get_selected_bivariate_indices(threshold=threshold))
    return {
        'n_univ': n_univ,
        'n_biv': n_biv,
        'n_total': n_univ + n_biv,
    }


def extract_performance_data(
    results_dict: Dict[str, Any],
    lasso_results_blackbox: Any,
    lasso_results_prn: Any,
    n_blackbox_features: int,
    threshold: float = 0.1,
) -> Tuple[Dict[tuple, Dict[str, Any]], List[Dict[str, Any]]]:
    """Extract model performance data with correct term counts per stage.

    Term counts match the nomogram JSON output and count_pipeline_terms.py:
    - Blackbox: original input features (collapsed, pre-encoding)
    - Blackbox-Nomogram: LASSO-selected terms (univariate + bivariate)
    - PRN: same as Blackbox-Nomogram (PRN receives exactly those terms)
    - PRN-Nomogram: second LASSO-selected terms after PRN re-decomposition

    Parameters
    ----------
    results_dict : dict
        Dictionary mapping (stage, split) keys to performance result dicts.
        Each result dict must have 'auroc' and 'auroc_ci' keys.
        Stages: 'Blackbox', 'Blackbox-Nomogram', 'PRN', 'PRN-Nomogram'
        Splits: 'Train', 'Test', 'Val'
    lasso_results_blackbox : LassoResultsManager
        Blackbox LASSO results (with selected lambda)
    lasso_results_prn : LassoResultsManager
        PRN LASSO results (with selected lambda)
    n_blackbox_features : int
        Number of original input features for the blackbox model
        (collapsed feature count, excluding target/ID)
    threshold : float
        Beta coefficient threshold for LASSO selection (default: 0.1)

    Returns
    -------
    tuple
        (performance_data, csv_data)
        - performance_data: dict mapping (model, split) -> {auroc, auroc_ci, terms}
          where terms is {n_univ, n_biv, n_total}
        - csv_data: list of dicts suitable for DataFrame/CSV export
    """
    # Compute term counts per stage
    bb_nom_terms = get_term_counts(lasso_results_blackbox, threshold=threshold)
    prn_nom_terms = get_term_counts(lasso_results_prn, threshold=threshold)

    blackbox_terms = {'n_univ': n_blackbox_features, 'n_biv': 0, 'n_total': n_blackbox_features}
    # PRN receives exactly the terms selected by blackbox LASSO
    prn_terms = bb_nom_terms.copy()

    stage_terms = {
        'Blackbox': blackbox_terms,
        'Blackbox-Nomogram': bb_nom_terms,
        'PRN': prn_terms,
        'PRN-Nomogram': prn_nom_terms,
    }

    model_order = ['Blackbox', 'Blackbox-Nomogram', 'PRN', 'PRN-Nomogram']
    dataset_order = ['Train', 'Test', 'Val']

    performance_data = {}
    csv_data = []

    for dataset in dataset_order:
        for model in model_order:
            key = (model, dataset)
            terms = stage_terms[model]

            if key in results_dict:
                data = results_dict[key]
                performance_data[key] = {
                    'auroc': data['auroc'],
                    'auroc_ci': data['auroc_ci'],
                    'terms': terms,
                }
                ci_lower, ci_upper = data['auroc_ci']
                csv_data.append(
                    {
                        'Dataset': dataset,
                        'Model': model,
                        'n_terms': terms['n_total'],
                        'n_univ': terms['n_univ'],
                        'n_biv': terms['n_biv'],
                        'AUROC': data['auroc'],
                        'CI_Lower': ci_lower,
                        'CI_Upper': ci_upper,
                    }
                )
            else:
                csv_data.append(
                    {
                        'Dataset': dataset,
                        'Model': model,
                        'n_terms': 'N/A',
                        'n_univ': 'N/A',
                        'n_biv': 'N/A',
                        'AUROC': 'N/A',
                        'CI_Lower': 'N/A',
                        'CI_Upper': 'N/A',
                    }
                )

    return performance_data, csv_data


# =============================================================================
# PRN CACHING UTILITIES
# =============================================================================


def load_cached_partial_responses(
    model_identifier: str,
    stage: str,
    models_dir: Path,
    expected_train_count: int,
    expected_test_count: int,
    expected_val_count: Optional[int] = None,
) -> Optional[Dict[str, torch.Tensor]]:
    """Load cached partial responses from disk with validation.

    Parameters
    ----------
    model_identifier : str
        Model identifier (e.g., 'htx_example_mlp')
    stage : str
        Either 'blackbox' or 'prn'
    models_dir : Path
        Directory containing model outputs
    expected_train_count : int
        Expected number of training samples
    expected_test_count : int
        Expected number of test samples
    expected_val_count : int, optional
        Expected number of validation samples (only checked if provided and in cache)

    Returns
    -------
    Optional[Dict[str, torch.Tensor]]
        Dictionary with 'train', 'test', and optionally 'val' tensors,
        or None if not found or validation fails
    """
    pr_dir = models_dir / 'partial_responses'

    if not pr_dir.exists():
        logger.info(f"Partial responses directory not found: {pr_dir}")
        return None

    # Find cached files
    pr_pattern = f"{stage}_{model_identifier}_*_partial_responses.pt"
    pr_files = list(pr_dir.glob(pr_pattern))

    if not pr_files:
        logger.info(f"No cached {stage} partial responses found (pattern: {pr_pattern})")
        return None

    # Get most recent file
    latest_pr_file = max(pr_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Found cached {stage} partial responses: {latest_pr_file.name}")

    try:
        cached_pr = torch.load(latest_pr_file, weights_only=False)

        # Validate model identifier
        if cached_pr.get('model_identifier') != model_identifier:
            logger.warning(
                f"Model mismatch: cached='{cached_pr.get('model_identifier')}', "
                f"current='{model_identifier}'"
            )
            return None

        # Validate train sample count
        if cached_pr['train'].shape[0] != expected_train_count:
            logger.warning(
                f"Train count mismatch: cached={cached_pr['train'].shape[0]}, "
                f"expected={expected_train_count}"
            )
            return None

        # Validate test sample count
        if cached_pr['test'].shape[0] != expected_test_count:
            logger.warning(
                f"Test count mismatch: cached={cached_pr['test'].shape[0]}, "
                f"expected={expected_test_count}"
            )
            return None

        # Validate val sample count if provided
        if expected_val_count is not None and 'val' in cached_pr:
            if cached_pr['val'].shape[0] != expected_val_count:
                logger.warning(
                    f"Val count mismatch: cached={cached_pr['val'].shape[0]}, "
                    f"expected={expected_val_count}"
                )
                # Val mismatch is not fatal - can recalculate just val

        # Return validated cache - keep on CPU for sklearn compatibility
        result = {
            'train': cached_pr['train'].cpu(),
            'test': cached_pr['test'].cpu(),
        }

        if 'val' in cached_pr and cached_pr['val'] is not None:
            result['val'] = cached_pr['val'].cpu()

        logger.info(f"[OK] Loaded cached {stage} partial responses")
        logger.info(f"  Shapes: train={result['train'].shape}, test={result['test'].shape}")

        return result

    except Exception as e:
        logger.warning(f"Error loading cached partial responses: {e}")
        return None


def validate_lasso_lambda_config(
    cached_config: Dict[str, Any],
    current_config: 'LassoLambdaConfig',
    stage: str,
    strict: bool = True,
) -> None:
    """Validate that cached LASSO lambda config matches current config.

    Parameters
    ----------
    cached_config : Dict[str, Any]
        Lambda selection config stored in the cached LASSO file
    current_config : LassoLambdaConfig
        Current lambda selection config from YAML
    stage : str
        Stage name for error messages ('blackbox' or 'prn')
    strict : bool
        If True, raises error on mismatch. If False, logs warning only.

    Raises
    ------
    LassoConfigurationError
        If strict=True and configs don't match
    """
    from prism.config_loader import LassoConfigurationError

    if cached_config is None:
        if strict:
            raise LassoConfigurationError(
                f"Cached {stage} LASSO results have no lambda_selection_config. "
                f"Cannot validate configuration match. "
                f"Re-run the source pipeline or set force_recalculate_{stage}_lasso=true."
            )
        else:
            logger.warning(f"Cached {stage} LASSO has no config - cannot validate")
            return

    # Convert current config to dict for comparison
    current_dict = current_config.to_dict()

    # Compare relevant fields
    mismatches = []
    compare_fields = [
        'method',
        'beta_threshold',
        'target_ratio',
        'min_auc',
        'ni_level',
        'target_features',
        'lambda_index',
    ]

    for field in compare_fields:
        cached_val = cached_config.get(field)
        current_val = current_dict.get(field)

        # Skip None values (only compare if set)
        if cached_val is None and current_val is None:
            continue

        if cached_val != current_val:
            mismatches.append(f"  {field}: cached={cached_val}, current={current_val}")

    if mismatches:
        mismatch_str = "\n".join(mismatches)
        msg = (
            f"Lambda selection config mismatch for {stage}:\n"
            f"{mismatch_str}\n\n"
            f"The PRN mask depends on the blackbox LASSO selection.\n"
            f"Using cached PRN with different blackbox settings may produce "
            f"incorrect results.\n\n"
            f"Options:\n"
            f"  1. Update your config to match the cached settings\n"
            f"  2. Set force_recalculate_{stage}_lasso=true to recalculate\n"
            f"  3. Set load_cached_prn=false to retrain PRN from scratch"
        )

        if strict:
            raise LassoConfigurationError(msg)
        else:
            logger.warning(msg)


def load_cached_lasso_results(
    model_identifier: str,
    stage: str,
    models_dir: Path,
    current_lambda_config: Optional['LassoLambdaConfig'] = None,
    validate_lambda_config: bool = True,
) -> Optional[Tuple[Any, Dict[str, Any]]]:
    """Load cached LASSO results from disk with optional config validation.

    Parameters
    ----------
    model_identifier : str
        Model identifier (e.g., 'htx_example_mlp')
    stage : str
        Either 'blackbox' or 'prn'
    models_dir : Path
        Directory containing model outputs
    current_lambda_config : LassoLambdaConfig, optional
        Current lambda selection config for validation
    validate_lambda_config : bool
        If True and current_lambda_config provided, validates config match.
        Raises LassoConfigurationError on mismatch.

    Returns
    -------
    Optional[Tuple[LassoResultsManager, Dict[str, Any]]]
        (lasso_results, metadata_dict) or None if not found
    """
    lasso_dir = models_dir / 'lasso_results'

    if not lasso_dir.exists():
        logger.info(f"LASSO results directory not found: {lasso_dir}")
        return None

    # Find cached files
    lasso_pattern = f"{stage}_{model_identifier}_*_lasso.pt"
    lasso_files = list(lasso_dir.glob(lasso_pattern))

    if not lasso_files:
        logger.info(f"No cached {stage} LASSO results found (pattern: {lasso_pattern})")
        return None

    # Get most recent file
    latest_lasso_file = max(lasso_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Found cached {stage} LASSO results: {latest_lasso_file.name}")

    try:
        cached_lasso = torch.load(latest_lasso_file, weights_only=False)

        # Validate model identifier
        if cached_lasso.get('model_identifier') != model_identifier:
            logger.warning(
                f"Model mismatch: cached='{cached_lasso.get('model_identifier')}', "
                f"current='{model_identifier}'"
            )
            return None

        # Extract LASSO results (handle different key names for blackbox vs prn)
        if stage == 'blackbox':
            lasso_results = cached_lasso.get('lasso_results')
        else:  # prn
            lasso_results = cached_lasso.get('lasso_results_prn') or cached_lasso.get(
                'lasso_results'
            )

        if lasso_results is None:
            logger.warning("No lasso_results found in cached file")
            return None

        # Validate lambda config if requested
        if validate_lambda_config and current_lambda_config is not None:
            cached_lambda_config = cached_lasso.get('lambda_selection_config')
            validate_lasso_lambda_config(
                cached_lambda_config, current_lambda_config, stage, strict=True
            )

        logger.info(f"[OK] Loaded cached {stage} LASSO results")
        # Use getattr for backwards compatibility with older cached results
        n_features = getattr(lasso_results, 'n_features', 'unknown')
        logger.info(
            f"  Lambda sweep: {len(lasso_results.lambdas)} values, " f"{n_features} features"
        )

        return lasso_results, cached_lasso

    except Exception as e:
        # Re-raise LassoConfigurationError without wrapping
        from prism.config_loader import LassoConfigurationError

        if isinstance(e, LassoConfigurationError):
            raise
        logger.warning(f"Error loading cached LASSO results: {e}")
        return None


def load_cached_prn_model(
    model_identifier: str,
    model_prefix: str,
    models_dir: Path,
    device: torch.device,
) -> Optional[Tuple[nn.Module, Dict[str, Any]]]:
    """Load cached PRN model from disk.

    Parameters
    ----------
    model_identifier : str
        Model identifier (e.g., 'htx_example_mlp')
    model_prefix : str
        Model prefix (e.g., 'mlp', 'xgb')
    models_dir : Path
        Directory containing model outputs
    device : torch.device
        Device to load model onto

    Returns
    -------
    Optional[Tuple[nn.Module, Dict[str, Any]]]
        (model, checkpoint_dict) or None if not found
    """
    from prism.hyperparameter_tuning import load_tuned_model

    # PRN model identifier follows pattern: {dataset}_{model_prefix}_prn
    prn_model_identifier = f"{model_identifier}_prn"

    try:
        checkpoint = load_tuned_model('prn', prn_model_identifier, models_dir)

        if checkpoint is not None:
            model = checkpoint['model']

            # Move to device if it's a PyTorch model
            if isinstance(model, nn.Module):
                model = model.to(device)
                model.eval()
                logger.info(f"Loaded cached PRN model, moved to device: {device}")

            return model, checkpoint

    except Exception as e:
        logger.warning(f"Error loading cached PRN model: {e}")

    return None


# =============================================================================
# INCREMENTAL SAVE UTILITIES
# =============================================================================


def save_partial_responses_artifact(
    pr_train: torch.Tensor,
    pr_test: torch.Tensor,
    pr_val: Optional[torch.Tensor] = None,
    *,
    feature_names: List[str],
    model_identifier: str,
    partial_response_method: str,
    models_dir: Path,
    stage: str,
    hyperparameters: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save partial responses incrementally (train+test, optionally val).

    Parameters
    ----------
    pr_train, pr_test : torch.Tensor
        Partial response tensors for train and test splits.
    pr_val : torch.Tensor, optional
        Partial response tensor for validation split.
    feature_names : list of str
        Feature names associated with the partial responses.
    model_identifier : str
        Model identifier (e.g., 'htx_example_mlp').
    partial_response_method : str
        Method used ('lebesgue' or 'dirac').
    models_dir : Path
        Base models directory.
    stage : str
        Either 'blackbox' or 'prn'.
    hyperparameters : dict, optional
        Model hyperparameters to include in metadata.

    Returns
    -------
    Path
        Path to the saved file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{stage}_{model_identifier}_{timestamp}"

    pr_dict = {
        'train': pr_train,
        'test': pr_test,
        'feature_names': feature_names,
        'model_identifier': model_identifier,
        'partial_response_method': partial_response_method,
        'hyperparameters': hyperparameters or {},
        'timestamp': timestamp,
    }
    if pr_val is not None:
        pr_dict['val'] = pr_val

    pr_save_dir = models_dir / 'partial_responses'
    pr_save_dir.mkdir(exist_ok=True)

    save_partial_responses(pr_dict, pr_save_dir, base_filename)
    save_path = pr_save_dir / f"{base_filename}_partial_responses.pt"
    print(f"[SAVE] {stage} partial responses -> {save_path.name}")
    return save_path


def save_lasso_artifact(
    lasso_results: Any,
    *,
    feature_names: List[str],
    model_identifier: str,
    partial_response_method: str,
    models_dir: Path,
    stage: str,
    lambda_config: Any = None,
    hyperparameters: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save LASSO results incrementally.

    Parameters
    ----------
    lasso_results : LassoResultsManager
        LASSO results object.
    feature_names : list of str
        Feature names.
    model_identifier : str
        Model identifier.
    partial_response_method : str
        Method used.
    models_dir : Path
        Base models directory.
    stage : str
        Either 'blackbox' or 'prn'.
    lambda_config : LassoLambdaConfig, optional
        Lambda selection config for reproducibility.
    hyperparameters : dict, optional
        Model hyperparameters.

    Returns
    -------
    Path
        Path to the saved file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{stage}_{model_identifier}_lasso_{timestamp}"

    # Use correct key name based on stage
    results_key = 'lasso_results' if stage == 'blackbox' else 'lasso_results_prn'
    hp_key = 'hyperparameters' if stage == 'blackbox' else 'prn_hyperparameters'

    lasso_dict = {
        results_key: lasso_results,
        'feature_names': feature_names,
        'model_identifier': model_identifier,
        'partial_response_method': partial_response_method,
        hp_key: hyperparameters or {},
        'timestamp': timestamp,
        'lambda_selection_config': lambda_config.to_dict() if lambda_config is not None else None,
    }

    lasso_save_dir = models_dir / 'lasso_results'
    lasso_save_dir.mkdir(exist_ok=True)

    save_path = lasso_save_dir / f"{base_filename}_lasso.pt"
    torch.save(lasso_dict, save_path)
    print(f"[SAVE] {stage} LASSO results -> {save_path.name}")
    return save_path


def save_shapley_artifact(
    shapley_results: Any,
    *,
    feature_names: List[str],
    model_identifier: str,
    partial_response_method: str,
    models_dir: Path,
    stage: str,
    hyperparameters: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save Shapley-like values incrementally.

    Parameters
    ----------
    shapley_results : dict
        Shapley-like extraction results.
    feature_names : list of str
        Feature names.
    model_identifier : str
        Model identifier.
    partial_response_method : str
        Method used.
    models_dir : Path
        Base models directory.
    stage : str
        Either 'blackbox' or 'prn'.
    hyperparameters : dict, optional
        Model hyperparameters.

    Returns
    -------
    Path
        Path to the saved file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{stage}_{model_identifier}_shapley_{timestamp}"

    # Use correct key name based on stage
    results_key = 'shapley_results' if stage == 'blackbox' else 'shapley_results_prn'
    hp_key = 'hyperparameters' if stage == 'blackbox' else 'prn_hyperparameters'

    shapley_dict = {
        results_key: shapley_results,
        'feature_names': feature_names,
        'model_identifier': model_identifier,
        'partial_response_method': partial_response_method,
        hp_key: hyperparameters or {},
        'timestamp': timestamp,
    }

    shapley_save_dir = models_dir / 'shapley_results'
    shapley_save_dir.mkdir(exist_ok=True)

    save_path = shapley_save_dir / f"{base_filename}_shapley.pt"
    torch.save(shapley_dict, save_path)
    print(f"[SAVE] {stage} Shapley results -> {save_path.name}")
    return save_path


def plot_training_history(
    history: Optional[Dict[str, Any]],
    title_prefix: str = '',
    figsize: Tuple[int, int] = (10, 10),
    x_label: str = 'Epoch',
) -> bool:
    """Plot training history (loss and AUROC) from a training run.

    Handles two formats:
    - PyTorch models (MLP/LogReg/PRN): dict with train_loss, test_loss, train_auroc,
      test_auroc, best_epoch, early_stop_epoch
    - XGBoost: dict with test_loss and optional best_iteration

    Args:
        history: Training history dict, or None if unavailable.
        title_prefix: Optional prefix for plot titles (e.g., 'PRN').
        figsize: Figure size for the plot.
        x_label: Label for the x-axis (e.g., 'Epoch' or 'Boosting Round').

    Returns:
        True if a plot was created, False otherwise.
    """
    import matplotlib.pyplot as plt

    if not history or not history.get('test_loss'):
        print("No training history available to plot.")
        return False

    has_auroc = bool(history.get('train_auroc'))
    has_train_loss = bool(history.get('train_loss'))
    prefix = f"{title_prefix} " if title_prefix else ''

    if has_auroc:
        # Loss + AUROC (PyTorch models: MLP, LogReg, PRN)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        epochs = range(len(history['train_loss']))

        ax1.plot(epochs, history['train_loss'], label='Train Loss', markersize=2)
        ax1.plot(epochs, history['test_loss'], label='Test Loss', markersize=2)
        if history.get('best_epoch') is not None:
            ax1.axvline(
                x=history['best_epoch'],
                color='g',
                linestyle='--',
                label=f"Best {x_label} ({history['best_epoch']})",
            )
        if history.get('early_stop_epoch') is not None:
            ax1.axvline(
                x=history['early_stop_epoch'],
                color='r',
                linestyle='--',
                label=f"Early Stop ({history['early_stop_epoch']})",
            )
        ax1.set_ylabel('Loss')
        ax1.set_title(f'{prefix}Training and Test Loss')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)

        ax2.plot(epochs, history['train_auroc'], label='Train AUROC', markersize=2)
        ax2.plot(epochs, history['test_auroc'], label='Test AUROC', markersize=2)
        if history.get('best_epoch') is not None:
            ax2.axvline(x=history['best_epoch'], color='g', linestyle='--')
        if history.get('early_stop_epoch') is not None:
            ax2.axvline(x=history['early_stop_epoch'], color='r', linestyle='--')
        ax2.set_xlabel(x_label)
        ax2.set_ylabel('AUROC')
        ax2.set_title(f'{prefix}Training and Test AUROC')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()
    elif has_train_loss:
        # Train + test loss only, no AUROC (XGBoost)
        plt.figure(figsize=(10, 6))
        epochs = range(len(history['train_loss']))
        plt.plot(epochs, history['train_loss'], label='Train Loss')
        plt.plot(epochs, history['test_loss'], label='Test Loss')

        if history.get('best_epoch') is not None:
            plt.axvline(
                x=history['best_epoch'],
                color='g',
                linestyle='--',
                label=f"Best {x_label} ({history['best_epoch']})",
            )

        plt.title(f'{prefix}Training and Test Loss')
        plt.xlabel(x_label)
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    else:
        # Test loss only (legacy)
        plt.figure(figsize=(10, 6))
        logloss = history['test_loss']
        epochs = range(1, len(logloss) + 1)
        plt.plot(epochs, logloss, 'r-', label='Test Loss')

        best_iter = history.get('best_iteration')
        if best_iter is not None:
            plt.axvline(
                x=best_iter, color='g', linestyle='--', label=f'Best Iteration: {best_iter}'
            )
            if best_iter - 1 < len(logloss):
                plt.plot(best_iter, logloss[best_iter - 1], 'go', markersize=8)

        plt.title(f'{prefix}Loss During Training')
        plt.xlabel(x_label)
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return True


def get_training_history(
    model: Any,
    checkpoint: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Extract training history from a model or checkpoint.

    Checks (in order):
    1. checkpoint['training_history'] (saved from tuning)
    2. model.training_history_ (attached by train_model)
    3. model.evals_result() (XGBoost models after direct training)

    Args:
        model: The trained model.
        checkpoint: Optional tuned model checkpoint dict.

    Returns:
        Training history dict, or None if unavailable.
    """
    # From checkpoint (tuned model)
    if checkpoint is not None:
        history = checkpoint.get('training_history')
        if history:
            return history

    # From model attribute (PyTorch models via train_model)
    if hasattr(model, 'training_history_'):
        return model.training_history_

    # From XGBoost evals_result (direct training)
    if hasattr(model, 'evals_result') and callable(model.evals_result):
        try:
            evals = model.evals_result()
            if evals:
                best_iter = getattr(model, 'best_iteration', None)
                # Two eval sets: validation_0=train, validation_1=test
                if 'validation_1' in evals:
                    train_evals = evals.get('validation_0', {})
                    test_evals = evals.get('validation_1', {})
                    return {
                        'train_loss': train_evals.get('logloss', []),
                        'test_loss': test_evals.get('logloss', []),
                        'best_epoch': best_iter,
                    }
                # Single eval set: validation_0=test (legacy)
                elif 'validation_0' in evals:
                    test_evals = evals['validation_0']
                    return {
                        'test_loss': test_evals.get('logloss', []),
                        'best_iteration': best_iter,
                    }
        except Exception:
            pass

    return None
