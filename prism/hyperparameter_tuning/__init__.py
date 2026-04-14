"""Hyperparameter tuning module for PRiSM pipeline.

This module provides Optuna-based hyperparameter tuning for MLP, LogReg,
XGBoost, Random Forest, and PRN models.

Public API:
    - TuningConfig: Configuration dataclass for tuning settings
    - run_hyperparameter_tuning: Main function to run Optuna tuning
    - save_best_params: Save best parameters to JSON
    - load_best_params: Load best parameters from JSON (by model/dataset)
    - load_params_from_file: Load parameters from a specific file path
    - save_best_model: Save best tuned model to checkpoint
    - load_tuned_model: Load tuned model if it exists
    - print_tuning_summary: Print tuning results summary
"""

from prism.hyperparameter_tuning.config import TuningConfig
from prism.hyperparameter_tuning.tuning import (
    load_best_params,
    load_params_from_file,
    load_tuned_model,
    print_tuning_summary,
    run_hyperparameter_tuning,
    save_best_model,
    save_best_params,
)

__all__ = [
    'TuningConfig',
    'run_hyperparameter_tuning',
    'save_best_params',
    'load_best_params',
    'load_params_from_file',
    'save_best_model',
    'load_tuned_model',
    'print_tuning_summary',
]
