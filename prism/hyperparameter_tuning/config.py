"""Configuration for hyperparameter tuning."""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class TuningConfig:
    """Configuration for hyperparameter tuning with Optuna.

    Attributes:
        enabled: Whether hyperparameter tuning is enabled
        n_trials: Number of Optuna trials to run
        metric: Metric to optimize (e.g., 'test_auc', 'val_auc')
        direction: Optimization direction ('maximize' or 'minimize')
        pruning_enabled: Whether to enable Optuna pruning for early stopping
        pruning_warmup_steps: Number of warmup steps before pruning starts
        n_jobs: Number of parallel jobs for tuning (1=sequential, -1=all CPUs).
                Note: n_jobs > 1 may cause issues with PyTorch models on GPU.
        params_file: Path to a previously saved best_params.json file to load.
                     When specified with enabled=False, loads params from this file
                     instead of using defaults. Supports absolute paths or paths
                     relative to the project root.
        skip_saved_params: When True, skip loading any saved parameters from
                           MODELS_DIR and strictly use hardcoded defaults.
                           Useful for "no_tuning" configs that should not pick up
                           previously saved tuning results. Cannot be used together
                           with params_file.
        custom_search_space: Optional custom search space dict (overrides defaults)
        use_multiprocessing: Whether to use multiprocessing for parallel GPU trials.
                             When True, uses multiprocessing instead of threading for
                             safe GPU parallelization with n_jobs > 1. Auto-enabled
                             when n_jobs > 1 and device != 'cpu'.
    """

    enabled: bool = False
    n_trials: int = 20
    metric: str = 'test_auc'
    direction: str = 'maximize'
    pruning_enabled: bool = True
    pruning_warmup_steps: int = 5
    n_jobs: int = 1  # Sequential by default (Optuna default); increase based on CPUs
    params_file: Optional[str] = None  # Path to load params from (when enabled=False)
    skip_saved_params: bool = False  # Skip loading saved params, use defaults only
    custom_search_space: Optional[Dict] = field(default=None)
    use_multiprocessing: bool = False  # Use multiprocessing for parallel GPU trials

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.direction not in ('maximize', 'minimize'):
            raise ValueError(f"direction must be 'maximize' or 'minimize', got {self.direction}")
        if self.n_trials < 1:
            raise ValueError(f"n_trials must be >= 1, got {self.n_trials}")
        if self.pruning_warmup_steps < 0:
            raise ValueError(f"pruning_warmup_steps must be >= 0, got {self.pruning_warmup_steps}")
        if self.skip_saved_params and self.params_file:
            raise ValueError(
                "Cannot specify both skip_saved_params=True and params_file. "
                "skip_saved_params forces use of defaults, while params_file "
                "loads specific parameters."
            )
