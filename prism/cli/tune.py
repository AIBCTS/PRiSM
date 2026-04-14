#!/usr/bin/env python3
"""
PRiSM Hyperparameter Tuning Runner

Runs Optuna-based hyperparameter tuning for specified models using preprocessed data.
Saves best parameters to JSON files for use in training notebooks.

Prerequisites:
- Preprocessing must be completed first (run preprocessing.ipynb or run_prism_pipeline.py)
- Preprocessed data files must exist in data/interim/

Config files are located in example_notebooks/config/ and should include
hyperparameter_tuning settings for each model.

Output:
- Best parameters saved to models/{dataset}_{model}_best_params.json
- Optuna study summaries printed to console

Examples
--------
    # Run tuning for all models in config
    python run_hyperparameter_tuning.py htx_example

    # Run tuning for specific models
    python run_hyperparameter_tuning.py htx_example --models mlp xgb

    # Override number of trials
    python run_hyperparameter_tuning.py htx_example --trials 30

    # List available configs
    python run_hyperparameter_tuning.py --list-configs
"""

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

# Project paths -- CLI operates on user's working directory
PROJECT_ROOT = Path.cwd()

from prism.config import INTERIM_DATA_DIR, MODELS_DIR  # noqa: E402
from prism.config_loader import (  # noqa: E402
    get_models_from_config,
    get_tuning_config,
    load_config,
)
from prism.device_tools import get_device  # noqa: E402
from prism.hyperparameter_tuning import (  # noqa: E402
    print_tuning_summary,
    run_hyperparameter_tuning,
    save_best_params,
)


def list_available_configs() -> List[str]:
    """List all available config files in example_notebooks/config/."""
    config_dir = PROJECT_ROOT / "example_notebooks" / "config"
    if not config_dir.exists():
        print(f"Config directory not found: {config_dir}")
        return []

    configs = sorted([f.stem for f in config_dir.glob("*.yaml") if not f.name.startswith('.')])
    return configs


def load_preprocessed_data(
    dataset: str,
    target_candidates: List[str] = None,
    id_candidates: List[str] = None,
    verbose: bool = True,
):
    """Load preprocessed train and test data.

    Args:
        dataset: Dataset prefix (e.g., 'htx_example')
        target_candidates: List of possible target column names (from config)
        id_candidates: List of possible ID column names (from config)
        verbose: Whether to print loading info

    Returns:
        Tuple of (X_train, y_train, X_test, y_test, target_column, id_column)
    """
    train_file = INTERIM_DATA_DIR / f"{dataset}_train.csv"
    test_file = INTERIM_DATA_DIR / f"{dataset}_test.csv"

    if not train_file.exists():
        raise FileNotFoundError(
            f"Preprocessed training data not found: {train_file}\n"
            f"Run preprocessing first: preprocessing.ipynb or run_prism_pipeline.py"
        )

    if not test_file.exists():
        raise FileNotFoundError(
            f"Preprocessed test data not found: {test_file}\n"
            f"Run preprocessing first: preprocessing.ipynb or run_prism_pipeline.py"
        )

    if verbose:
        print(f"Loading preprocessed data from: {INTERIM_DATA_DIR}")
        print(f"  - {train_file.name}")
        print(f"  - {test_file.name}")

    # Load data (skip comment lines)
    data_train = pd.read_csv(train_file, comment='#')
    data_test = pd.read_csv(test_file, comment='#')

    if verbose:
        print(f"  Training shape: {data_train.shape}")
        print(f"  Test shape: {data_test.shape}")

    # Detect target column - use config candidates first, then defaults
    default_target_candidates = ['var1', 'event_oneyear', 'target', 'outcome', 'y', 'label']
    if target_candidates:
        # Config-provided candidates take priority
        all_target_candidates = list(target_candidates) + default_target_candidates
    else:
        all_target_candidates = default_target_candidates

    # Case-insensitive search for target
    target_column = None
    columns_lower = {col.lower(): col for col in data_train.columns}
    for candidate in all_target_candidates:
        if candidate.lower() in columns_lower:
            target_column = columns_lower[candidate.lower()]
            break

    # Detect ID column - use config candidates first, then defaults
    default_id_candidates = ['trr_id_code', 'id', 'patient_id', 'subject_id']
    if id_candidates:
        all_id_candidates = list(id_candidates) + default_id_candidates
    else:
        all_id_candidates = default_id_candidates

    # Case-insensitive search for ID
    id_column = None
    for candidate in all_id_candidates:
        if candidate.lower() in columns_lower:
            id_column = columns_lower[candidate.lower()]
            break

    if target_column is None:
        raise ValueError(
            f"Could not auto-detect target column in {train_file}\n"
            f"Tried: {all_target_candidates}\n"
            f"Available columns: {list(data_train.columns)}"
        )

    if id_column is None:
        if verbose:
            print("  Warning: Could not detect ID column, will include all features")
        drop_cols = [target_column]
    else:
        drop_cols = [target_column, id_column]

    if verbose:
        print(f"  Target column: {target_column}")
        print(f"  ID column: {id_column}")

    # Prepare feature matrices
    X_train = data_train.drop(drop_cols, axis=1)
    y_train = data_train[target_column]
    X_test = data_test.drop(drop_cols, axis=1)
    y_test = data_test[target_column]

    return X_train, y_train, X_test, y_test, target_column, id_column


def run_tuning_for_model(
    model_type: str,
    dataset: str,
    X_train,
    y_train,
    X_test,
    y_test,
    config: dict,
    random_seed: int,
    device: str,
    n_trials_override: int = None,
):
    """Run hyperparameter tuning for a single model.

    Args:
        model_type: Model name ('mlp', 'xgb', 'logreg', 'rf')
        dataset: Dataset prefix
        X_train, y_train: Training data
        X_test, y_test: Test data
        config: Configuration dictionary
        random_seed: Random seed
        device: PyTorch device
        n_trials_override: Override n_trials from config

    Returns:
        Best parameters dict or None if tuning disabled/failed
    """
    print(f"\n{'='*70}")
    print(f"HYPERPARAMETER TUNING: {model_type.upper()}")
    print(f"{'='*70}")

    # Get tuning config for this model
    tuning_config = get_tuning_config(config, model_type)

    # Override n_trials if specified
    if n_trials_override is not None:
        tuning_config.n_trials = n_trials_override
        print(f"Overriding n_trials to {n_trials_override}")

    if not tuning_config.enabled and n_trials_override is None:
        print(f"Hyperparameter tuning is disabled for {model_type}")
        print("Set hyperparameter_tuning.{model}.enabled: true in config or use --trials")
        print(f"{'='*70}")
        return None

    # Enable tuning if trials override is set
    if n_trials_override is not None:
        tuning_config.enabled = True

    print(f"Trials: {tuning_config.n_trials}")
    print(f"Metric: {tuning_config.metric}")
    print(f"Direction: {tuning_config.direction}")
    print(f"N_jobs: {getattr(tuning_config, 'n_jobs', 1)}")
    print()

    try:
        # Run hyperparameter tuning
        best_params, study, _best_model = run_hyperparameter_tuning(
            model_type=model_type,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            tuning_config=tuning_config,
            random_seed=random_seed,
            device=device,
        )

        # Print summary
        print_tuning_summary(study, model_type)

        # Save best parameters (with reproducibility info)
        save_best_params(
            best_params,
            model_type,
            dataset,
            MODELS_DIR,
            study,
            random_seed=random_seed,
            tuning_config=tuning_config,
        )

        return best_params

    except Exception as e:
        print(f"ERROR: Tuning failed for {model_type}: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Run hyperparameter tuning for PRiSM models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument('configs', nargs='*', help='Config names to run (e.g., htx_example)')

    parser.add_argument(
        '--models',
        nargs='+',
        default=None,
        help='Specific models to tune (e.g., --models mlp xgb)',
    )

    parser.add_argument(
        '--trials',
        type=int,
        default=None,
        help='Override number of trials (default: use config setting)',
    )

    parser.add_argument(
        '--list-configs', action='store_true', help='List available config files and exit'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (default: use config yaml value or 257)',
    )

    args = parser.parse_args()

    # List configs mode
    if args.list_configs:
        configs = list_available_configs()
        if configs:
            print("Available configurations:")
            for config in configs:
                print(f"  - {config}")
        else:
            print("No configuration files found")
        return 0

    # Validate arguments
    if not args.configs:
        parser.print_help()
        print("\nError: No config specified")
        print("Use --list-configs to see available configs")
        return 1

    # Get device
    device = get_device()
    print(f"Using device: {device}")

    # Process each config
    all_results = {}

    for config_name in args.configs:
        print(f"\n{'#'*70}")
        print(f"# CONFIG: {config_name}")
        print(f"{'#'*70}\n")

        try:
            # Load config
            config, dataset = load_config(config_name)
            print(f"Dataset: {dataset}")

            # Determine random seed (command line > config yaml > default)
            if args.seed is not None:
                random_seed = args.seed
                print(f"Random seed (from --seed): {random_seed}")
            elif config.get('random_seed') is not None:
                random_seed = config.get('random_seed')
                print(f"Random seed (from config yaml): {random_seed}")
            else:
                random_seed = 257
                print(f"Random seed (default): {random_seed}")

            # Set random seed for this config
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random_seed)

            # Get models from config or use command line override
            if args.models:
                models = args.models
                print(f"Models (from command line): {models}")
            else:
                models = get_models_from_config(config, default_models=['mlp'])
                print(f"Models (from config): {models}")

            # Load preprocessed data using config-specified target/id candidates
            target_candidates = config.get('target_candidates', [])
            id_candidates = config.get('id_candidates', [])

            X_train, y_train, X_test, y_test, target_col, id_col = load_preprocessed_data(
                dataset, target_candidates=target_candidates, id_candidates=id_candidates
            )

            # Run tuning for each model
            config_results = {}
            for model_type in models:
                result = run_tuning_for_model(
                    model_type=model_type,
                    dataset=dataset,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    config=config,
                    random_seed=random_seed,
                    device=device,
                    n_trials_override=args.trials,
                )
                config_results[model_type] = result

            all_results[config_name] = config_results

        except Exception as e:
            print(f"ERROR: Failed to process config {config_name}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Print final summary
    print(f"\n{'#'*70}")
    print("# TUNING COMPLETE")
    print(f"{'#'*70}\n")

    for config_name, results in all_results.items():
        print(f"{config_name}:")
        for model_type, params in results.items():
            if params:
                print(
                    f"  {model_type}: Parameters saved to models/{dataset}_{model_type}_best_params.json"
                )
            else:
                print(f"  {model_type}: Tuning disabled or failed")

    print("\nBest parameters have been saved. Run training notebooks to use them.")

    return 0


if __name__ == '__main__':
    sys.exit(main())
