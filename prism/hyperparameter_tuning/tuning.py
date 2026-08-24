"""Main orchestration functions for hyperparameter tuning.

This module provides functions to run Optuna hyperparameter tuning,
save and load best parameters, and manage the tuning workflow.

GPU Parallelization:
    When using n_jobs > 1 with GPU, this module automatically enables
    multiprocessing mode (instead of threading) for safe CUDA parallelization.
    Each worker process runs trials sequentially with proper GPU memory management.
"""

import json
import logging
import multiprocessing
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import optuna
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from prism.device_tools import cleanup_gpu_memory
from prism.hyperparameter_tuning.config import TuningConfig
from prism.hyperparameter_tuning.objectives import get_objective
from prism.hyperparameter_tuning.search_spaces import get_default_params

# Setup logging
logger = logging.getLogger(__name__)


# Get a spawn context for CUDA-safe multiprocessing
# Using get_context('spawn') instead of set_start_method() ensures we always
# use spawn regardless of the global default, which is critical on Linux
# where the default is 'fork' (incompatible with CUDA)
def _get_spawn_context():
    """Get a spawn multiprocessing context for CUDA-safe process creation.

    On Linux, the default multiprocessing method is 'fork', which is incompatible
    with CUDA because it copies the parent's CUDA context. Using spawn ensures
    each worker process starts fresh without inheriting CUDA state.

    Using get_context('spawn') instead of set_start_method('spawn') avoids
    conflicts when other code has already set a different start method.

    Returns:
        multiprocessing context using 'spawn' method
    """
    return multiprocessing.get_context('spawn')


def _convert_data_for_multiprocessing(data):
    """Convert data to CPU numpy for safe multiprocessing pickling.

    GPU tensors cannot be pickled across processes, so we convert to numpy.

    Args:
        data: Input data (tensor, numpy array, or pandas DataFrame/Series)

    Returns:
        numpy array on CPU
    """
    import pandas as pd

    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    elif isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        return data.values
    elif isinstance(data, np.ndarray):
        return data
    else:
        return np.array(data)


def _optuna_worker_process(
    study_name: str,
    storage_url: str,
    worker_id: int,
    n_trials_worker: int,
    device: str,
    model_type: str,
    objective_kwargs: Dict,
    random_seed: int,
):
    """Worker process for Optuna multiprocessing.

    Each worker loads the shared study from storage and runs trials sequentially.
    GPU memory is cleared at start and end of the worker process.

    Args:
        study_name: Name of the Optuna study
        storage_url: SQLite storage URL for the shared study
        worker_id: Worker process ID (for logging)
        n_trials_worker: Number of trials for this worker to run
        device: Device to use for training
        model_type: Model type for objective function
        objective_kwargs: Kwargs for objective function
        random_seed: Random seed for reproducibility
    """
    # Initialize random seeds in worker process for reproducibility
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    # Clear GPU cache at start of worker (full cleanup: gc + sync + empty for all backends)
    cleanup_gpu_memory(torch.device(device))

    try:
        # Load the shared study with a seeded sampler for reproducibility
        # (load_study creates an unseeded TPESampler by default)
        sampler = TPESampler(seed=random_seed)
        study = optuna.load_study(study_name=study_name, storage=storage_url, sampler=sampler)

        # Update kwargs with device
        # For XGBoost/RF, use CPU since data is numpy (converted for pickling)
        # and tree-based models don't benefit much from GPU
        worker_device = device
        if model_type.lower() in ('xgb', 'rf'):
            worker_device = 'cpu'
        objective_kwargs['device'] = worker_device

        # Create objective
        objective = get_objective(model_type, **objective_kwargs)

        # Run trials sequentially in this worker
        study.optimize(
            objective,
            n_trials=n_trials_worker,
            n_jobs=1,  # Sequential within worker
            show_progress_bar=False,
        )

        logger.debug(f"Worker {worker_id} completed {n_trials_worker} trials")

    finally:
        # Clear GPU cache at end of worker (full cleanup for all backends)
        cleanup_gpu_memory(torch.device(device))


def _cleanup_temp_directory(tmpdir: str, max_retries: int = 5, retry_delay: float = 0.5):
    """Clean up temporary directory with retry logic for Windows file locks.

    On Windows, SQLite database files may still be locked briefly after
    connections are closed. This function retries cleanup with delays.

    Args:
        tmpdir: Path to temporary directory to remove
        max_retries: Maximum number of cleanup attempts
        retry_delay: Delay in seconds between retries
    """
    import shutil

    for attempt in range(max_retries):
        try:
            shutil.rmtree(tmpdir)
            logger.debug(f"Cleaned up temp directory: {tmpdir}")
            return
        except PermissionError as e:
            if attempt < max_retries - 1:
                logger.debug(
                    f"Temp cleanup attempt {attempt + 1}/{max_retries} failed, "
                    f"retrying in {retry_delay}s: {e}"
                )
                time.sleep(retry_delay)
            else:
                # Final attempt failed - log warning but don't crash
                logger.warning(
                    f"Could not clean up temp directory {tmpdir}: {e}. "
                    f"The directory will be cleaned up by the OS later."
                )
        except Exception as e:
            logger.warning(f"Unexpected error cleaning up temp directory: {e}")
            break


def _run_parallel_optimization(
    model_type: str,
    tuning_config: TuningConfig,
    objective_kwargs: Dict,
    device: str,
    random_seed: int,
    n_hidden: Optional[int] = None,
    mask: Optional[Any] = None,
) -> Tuple[Dict, optuna.Study, Any]:
    """Run Optuna optimization using multiprocessing for parallel GPU trials.

    This is primarily for PyTorch models (MLP, LogReg, PRN) which have CUDA
    threading issues when using n_jobs > 1. XGBoost and RF manage their own
    parallelization and typically don't need this.

    Creates a SQLite-backed study that can be shared across worker processes.
    Each worker runs trials sequentially with proper GPU memory management.

    Args:
        model_type: Model type for objective function
        tuning_config: TuningConfig instance
        objective_kwargs: Kwargs for objective function (without device)
        device: Device to use for training
        random_seed: Random seed for reproducibility
        n_hidden: Fixed number of hidden units for MLP (optional)
        mask: Mask for MaskedMLP or PRN (optional)

    Returns:
        Tuple of (best_params dict, optuna.Study object, best_model or None)
    """
    # Get spawn context for CUDA-safe multiprocessing
    # This is critical on Linux where default is 'fork' (incompatible with CUDA)
    mp_ctx = _get_spawn_context()

    n_workers = tuning_config.n_jobs
    n_trials = tuning_config.n_trials

    # Create temporary directory manually (not as context manager)
    # This allows us to control cleanup timing for Windows compatibility
    tmpdir = tempfile.mkdtemp(prefix="optuna_mp_")
    storage_url = f"sqlite:///{tmpdir}/optuna_study.db"
    study_name = f"tuning_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    try:
        # Create the study with SQLite storage
        sampler = TPESampler(seed=random_seed)
        pruner = None
        if tuning_config.pruning_enabled:
            pruner = MedianPruner(
                n_startup_trials=tuning_config.pruning_warmup_steps,
                n_warmup_steps=tuning_config.pruning_warmup_steps,
            )

        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            direction=tuning_config.direction,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )

        # Enqueue known-good defaults as trial 0
        defaults = get_default_params(model_type)
        if defaults:
            study.enqueue_trial(defaults)
            logger.info(f"Enqueued default parameters as trial 0 for {model_type}")

        # Distribute trials across workers
        trials_per_worker = n_trials // n_workers
        extra_trials = n_trials % n_workers

        # Cap XGB/RF threads per worker to avoid oversubscription
        if model_type.lower() in ('xgb', 'rf') and 'nthread' not in objective_kwargs:
            cpu_count = os.cpu_count() or 1
            objective_kwargs['nthread'] = max(1, cpu_count // n_workers)
            logger.info(
                f"  nthread per worker: {objective_kwargs['nthread']} "
                f"({cpu_count} CPUs / {n_workers} workers)"
            )

        # Convert data to numpy for pickling (GPU tensors can't be pickled)
        mp_kwargs = objective_kwargs.copy()
        mp_kwargs['X_train'] = _convert_data_for_multiprocessing(objective_kwargs['X_train'])
        mp_kwargs['y_train'] = _convert_data_for_multiprocessing(objective_kwargs['y_train'])
        mp_kwargs['X_test'] = _convert_data_for_multiprocessing(objective_kwargs['X_test'])
        mp_kwargs['y_test'] = _convert_data_for_multiprocessing(objective_kwargs['y_test'])

        # Start worker processes using spawn context
        # Using mp_ctx.Process ensures spawn method even if global default is fork
        processes = []
        for i in range(n_workers):
            # Distribute extra trials to first workers
            worker_trials = trials_per_worker + (1 if i < extra_trials else 0)
            if worker_trials == 0:
                continue

            p = mp_ctx.Process(
                target=_optuna_worker_process,
                args=(
                    study_name,
                    storage_url,
                    i,
                    worker_trials,
                    device,
                    model_type,
                    mp_kwargs,
                    random_seed,  # Same seed for all workers; Optuna sampler handles diversity
                ),
            )
            processes.append(p)
            p.start()
            logger.debug(f"Started worker {i} with {worker_trials} trials")

        # Wait for all workers to complete
        for p in processes:
            p.join()

        # Reload study to get all results
        study = optuna.load_study(study_name=study_name, storage=storage_url)

        # Extract results before closing storage
        best_params = study.best_params.copy()
        best_value = study.best_value
        direction = tuning_config.direction

        # Close the storage connection explicitly to release file locks
        # This is critical for Windows where file locks prevent deletion
        if hasattr(study, '_storage') and study._storage is not None:
            storage = study._storage
            if hasattr(storage, '_engine'):
                storage._engine.dispose()
            elif hasattr(storage, 'close'):
                storage.close()

        # Also delete the study reference to help release resources
        del study

        # Force garbage collection to release any remaining references
        import gc

        gc.collect()

        logger.info(f"Best {tuning_config.metric}: {best_value:.4f}")
        logger.info(f"Best parameters: {best_params}")

        # Re-train best model to get the actual model object
        # (multiprocessing can't return model objects easily)
        # For XGBoost/RF, use CPU since data is numpy and tree-based models
        # don't benefit much from GPU for a single training run
        final_device = device
        if model_type.lower() in ('xgb', 'rf'):
            final_device = 'cpu'
            logger.debug(f"Using CPU for final {model_type} retraining (numpy data)")
        objective_kwargs['device'] = final_device
        final_objective = get_objective(model_type, **objective_kwargs)

        # Run one trial with the best params to get the model
        # Create a study that suggests the best params (in-memory, no SQLite)
        final_study = optuna.create_study(direction=direction)

        # Add the best trial to suggest its params
        final_study.enqueue_trial(best_params)
        final_study.optimize(final_objective, n_trials=1, show_progress_bar=False)

        best_model = getattr(final_objective, 'best_model', None)

        return best_params, final_study, best_model

    finally:
        # Clean up temp directory with retry logic for Windows
        _cleanup_temp_directory(tmpdir)


def run_hyperparameter_tuning(
    model_type: str,
    X_train,
    y_train,
    X_test,
    y_test,
    tuning_config: TuningConfig,
    random_seed: int,
    device: str = 'cpu',
    n_hidden: Optional[int] = None,
    mask: Optional[Any] = None,
) -> Tuple[Dict, optuna.Study, Any]:
    """Run Optuna hyperparameter tuning for a given model type.

    Args:
        model_type: Model type ('mlp', 'logreg', 'xgb', 'rf', 'prn')
        X_train: Training features
        y_train: Training labels
        X_test: Test features for evaluation
        y_test: Test labels for evaluation
        tuning_config: TuningConfig instance with tuning settings
        random_seed: Random seed for reproducibility
        device: PyTorch device ('cuda', 'mps', or 'cpu')
        n_hidden: Fixed number of hidden units for MLP (optional)
        mask: Mask for MaskedMLP or PRN (optional)

    Returns:
        Tuple of (best_params dict, optuna.Study object, best_model or None)
        The best_model is the trained model from the best trial, ready for use.

    Raises:
        ValueError: If model_type is not supported

    Note:
        For PyTorch models (MLP, LogReg, PRN): When n_jobs > 1 and device is
        'cuda' or 'mps', multiprocessing mode is automatically enabled for safe
        GPU parallelization. Each worker process runs trials sequentially with
        proper GPU memory management.

        For XGBoost and RF: Standard threading is used as these models manage
        their own parallelization and don't have CUDA threading issues.
    """
    n_jobs = getattr(tuning_config, 'n_jobs', 1)
    use_multiprocessing = getattr(tuning_config, 'use_multiprocessing', False)

    # Auto-enable multiprocessing for safe GPU parallelization
    # Only for PyTorch models (MLP, LogReg, PRN) which have CUDA threading issues
    # XGBoost and RF manage their own parallelization and don't need this
    is_gpu_device = device in ('cuda', 'mps')
    is_pytorch_model = model_type.lower() in ('mlp', 'logreg', 'prn')
    if n_jobs > 1 and is_gpu_device and is_pytorch_model and not use_multiprocessing:
        use_multiprocessing = True
        logger.info("Auto-enabling multiprocessing for safe GPU parallelization (PyTorch model)")

    logger.info(f"Starting hyperparameter tuning for {model_type}")
    logger.info(f"  n_trials: {tuning_config.n_trials}")
    logger.info(f"  metric: {tuning_config.metric}")
    logger.info(f"  direction: {tuning_config.direction}")
    logger.info(f"  device: {device}")
    logger.info(f"  n_jobs: {n_jobs}")
    if use_multiprocessing:
        logger.info("  mode: multiprocessing (GPU-safe)")

    # Prepare objective function kwargs (without device for multiprocessing)
    objective_kwargs = {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'tuning_config': tuning_config,
        'random_seed': random_seed,
    }

    # Add model-specific kwargs
    if model_type.lower() == 'mlp':
        objective_kwargs['n_hidden'] = n_hidden
        objective_kwargs['mask'] = mask
    elif model_type.lower() == 'prn':
        objective_kwargs['mask'] = mask

    # Use multiprocessing for GPU parallelization
    if use_multiprocessing and n_jobs > 1:
        return _run_parallel_optimization(
            model_type=model_type,
            tuning_config=tuning_config,
            objective_kwargs=objective_kwargs,
            device=device,
            random_seed=random_seed,
            n_hidden=n_hidden,
            mask=mask,
        )

    # Standard single-process or threaded optimization
    # Add device to kwargs for single-process mode
    objective_kwargs['device'] = device

    # Cap XGB/RF threads when running parallel threaded trials
    if n_jobs > 1 and model_type.lower() in ('xgb', 'rf') and 'nthread' not in objective_kwargs:
        cpu_count = os.cpu_count() or 1
        objective_kwargs['nthread'] = max(1, cpu_count // n_jobs)
        logger.info(
            f"  nthread per trial: {objective_kwargs['nthread']} "
            f"({cpu_count} CPUs / {n_jobs} jobs)"
        )

    # Create Optuna sampler
    sampler = TPESampler(seed=random_seed)

    # Create pruner if enabled
    if tuning_config.pruning_enabled:
        pruner = MedianPruner(
            n_startup_trials=tuning_config.pruning_warmup_steps,
            n_warmup_steps=tuning_config.pruning_warmup_steps,
        )
    else:
        pruner = None

    # Create study
    study = optuna.create_study(direction=tuning_config.direction, sampler=sampler, pruner=pruner)

    # Enqueue known-good defaults as trial 0
    defaults = get_default_params(model_type)
    if defaults:
        study.enqueue_trial(defaults)
        logger.info(f"Enqueued default parameters as trial 0 for {model_type}")

    # Create objective function
    objective = get_objective(model_type, **objective_kwargs)

    # Run optimization
    logger.info(f"Running {tuning_config.n_trials} trials (n_jobs={n_jobs})...")
    study.optimize(
        objective, n_trials=tuning_config.n_trials, n_jobs=n_jobs, show_progress_bar=True
    )

    # Get best parameters
    best_params = study.best_params
    best_value = study.best_value

    logger.info(f"Best {tuning_config.metric}: {best_value:.4f}")
    logger.info(f"Best parameters: {best_params}")

    # Get best model from objective (if tracking was enabled)
    best_model = getattr(objective, 'best_model', None)
    if best_model is not None:
        logger.info("Best model tracked from tuning trials")
    else:
        logger.info("No best model tracked (objective may not support model tracking)")

    return best_params, study, best_model


def save_best_params(
    best_params: Dict,
    model_type: str,
    dataset_prefix: str,
    output_dir: Path,
    study: Optional[optuna.Study] = None,
    random_seed: Optional[int] = None,
    tuning_config: Optional[Any] = None,
) -> Path:
    """Save best hyperparameters to JSON file.

    Args:
        best_params: Dictionary of best hyperparameters
        model_type: Model type ('mlp', 'logreg', 'xgb', 'rf', 'prn')
        dataset_prefix: Dataset name prefix
        output_dir: Directory to save parameters file
        study: Optional Optuna study object (for saving study statistics)
        random_seed: Optional random seed used for tuning (for reproducibility)
        tuning_config: Optional TuningConfig object (for reproducibility)

    Returns:
        Path to saved parameters file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{dataset_prefix}_{model_type}_best_params.json"
    filepath = output_dir / filename

    # Prepare data to save
    data = {'model_type': model_type, 'dataset': dataset_prefix, 'best_params': best_params}

    # Add study statistics if available
    if study is not None:
        data['study_stats'] = {
            'best_value': float(study.best_value),
            'n_trials': len(study.trials),
            'best_trial': study.best_trial.number,
        }

    # Add reproducibility info
    if random_seed is not None:
        data['random_seed'] = random_seed

    if tuning_config is not None:
        data['tuning_config'] = {
            'n_trials': tuning_config.n_trials,
            'metric': tuning_config.metric,
            'direction': tuning_config.direction,
            'pruning_enabled': tuning_config.pruning_enabled,
            'n_jobs': getattr(tuning_config, 'n_jobs', 1),
        }

    # Save to JSON
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved best parameters to {filepath}")

    return filepath


def save_best_model(
    model: Any,
    model_type: str,
    dataset_prefix: str,
    output_dir: Path,
    scaler: Optional[Any] = None,
    hyperparameters: Optional[Dict] = None,
    feature_names: Optional[list] = None,
    mask: Optional[Any] = None,
    test_metrics: Optional[Dict] = None,
) -> Path:
    """Save best model from hyperparameter tuning.

    Saves the best model in a checkpoint format compatible with training notebooks.
    The model can be loaded later to skip retraining when the tuned model exists.

    Args:
        model: The trained model to save
        model_type: Model type ('mlp', 'logreg', 'xgb', 'rf', 'prn')
        dataset_prefix: Dataset name prefix (e.g., 'htx_example_mlp' or 'htx_example_mlp_prn')
        output_dir: Directory to save the model file
        scaler: Optional scaler used for data preprocessing
        hyperparameters: Optional dict of hyperparameters used to train the model
        feature_names: Optional list of feature names
        mask: Optional mask for PRN models
        test_metrics: Optional dict of test metrics from tuning

    Returns:
        Path to saved model file
    """
    output_dir = Path(output_dir)

    # Create model subdirectory (e.g., models/htx_example_mlp/)
    model_subdir = output_dir / dataset_prefix
    model_subdir.mkdir(parents=True, exist_ok=True)

    filename = f"{dataset_prefix}_model_tuned.pt"
    filepath = model_subdir / filename

    # Prepare save dict in format compatible with training notebooks
    save_dict = {
        'model': model,
        'scaler': scaler,
        'hyperparameters': hyperparameters or {},
        'feature_names': feature_names,
        'tuned': True,  # Flag to indicate this is from tuning
        'model_type': model_type,
    }

    # Add model state dict and class for PyTorch models
    if hasattr(model, 'state_dict'):
        save_dict['model_state_dict'] = model.state_dict()
        save_dict['model_class'] = type(model)

    # Add mask for PRN models
    if mask is not None:
        save_dict['mask'] = mask

    # Add test metrics if available
    if test_metrics is not None:
        save_dict['test_metrics'] = test_metrics

    # Add training history if available (attached by train_model or XGB objective)
    if hasattr(model, 'training_history_'):
        save_dict['training_history'] = model.training_history_

    # Save using torch
    torch.save(save_dict, filepath)

    logger.info(f"Saved best tuned model to {filepath}")

    return filepath


def load_tuned_model(model_type: str, dataset_prefix: str, models_dir: Path) -> Optional[Dict]:
    """Load a tuned model if it exists.

    Args:
        model_type: Model type ('mlp', 'logreg', 'xgb', 'rf', 'prn')
        dataset_prefix: Dataset name prefix
        models_dir: Base models directory

    Returns:
        Checkpoint dict with model and metadata, or None if not found
    """
    models_dir = Path(models_dir)

    # Check in model subdirectory
    model_subdir = models_dir / dataset_prefix
    filename = f"{dataset_prefix}_model_tuned.pt"
    filepath = model_subdir / filename

    if not filepath.exists():
        logger.info(f"No tuned model found at {filepath}")
        return None

    logger.info(f"Loading tuned model from {filepath}")
    checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)

    # Verify it's a tuned model
    if not checkpoint.get('tuned', False):
        logger.warning(f"Model at {filepath} is not marked as tuned")

    return checkpoint


def load_best_params(model_type: str, dataset_prefix: str, params_dir: Path) -> Optional[Dict]:
    """Load best hyperparameters from JSON file.

    Args:
        model_type: Model type ('mlp', 'logreg', 'xgb', 'rf', 'prn')
        dataset_prefix: Dataset name prefix
        params_dir: Directory containing parameters file

    Returns:
        Dictionary of best hyperparameters, or None if file doesn't exist
    """
    params_dir = Path(params_dir)
    filename = f"{dataset_prefix}_{model_type}_best_params.json"
    filepath = params_dir / filename

    if not filepath.exists():
        logger.info(f"No saved parameters found at {filepath}")
        return None

    # Load from JSON
    with open(filepath, 'r') as f:
        data = json.load(f)

    logger.info(f"Loaded best parameters from {filepath}")
    logger.info(f"  Parameters: {data['best_params']}")

    if 'study_stats' in data:
        logger.info(f"  Best metric: {data['study_stats']['best_value']:.4f}")
        logger.info(f"  Number of trials: {data['study_stats']['n_trials']}")

    return data['best_params']


def load_params_from_file(params_file: str) -> Optional[Dict]:
    """Load hyperparameters from a specific JSON file path.

    This function is used when a params_file is explicitly specified in the
    config, allowing users to explicitly choose which tuned parameters to use.

    Args:
        params_file: Path to the best_params.json file. Can be absolute or
                     relative to the project root.

    Returns:
        Dictionary of best hyperparameters, or None if file doesn't exist

    Raises:
        FileNotFoundError: If the specified file doesn't exist
        ValueError: If the file format is invalid
    """
    from prism.config import PROJ_ROOT

    filepath = Path(params_file)

    # If relative path, resolve from project root
    if not filepath.is_absolute():
        filepath = PROJ_ROOT / filepath

    if not filepath.exists():
        raise FileNotFoundError(
            f"Specified params_file not found: {filepath}\n"
            f"Please check that the path is correct and the file exists."
        )

    # Load from JSON
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in params file {filepath}: {e}")

    if 'best_params' not in data:
        raise ValueError(
            f"Invalid params file format: {filepath}\n"
            f"Expected 'best_params' key in JSON, found: {list(data.keys())}"
        )

    logger.info(f"Loaded parameters from specified file: {filepath}")
    logger.info(f"  Parameters: {data['best_params']}")

    if 'study_stats' in data:
        logger.info(f"  Best metric: {data['study_stats']['best_value']:.4f}")
        logger.info(f"  From tuning with {data['study_stats']['n_trials']} trials")

    return data['best_params']


def print_tuning_summary(study: optuna.Study, model_type: str):
    """Print a summary of the hyperparameter tuning results.

    Args:
        study: Completed Optuna study
        model_type: Model type name
    """
    print(f"\n{'='*60}")
    print(f"Hyperparameter Tuning Summary: {model_type.upper()}")
    print(f"{'='*60}")
    print(f"Number of trials: {len(study.trials)}")
    print(f"Best value: {study.best_value:.4f}")
    print(f"Best trial: #{study.best_trial.number}")
    print("\nBest parameters:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")
    print(f"{'='*60}\n")
