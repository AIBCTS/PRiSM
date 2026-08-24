"""
Configuration management for LASSO implementation.

This module provides structured configuration classes for LASSO computations.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class LassoConfig:
    """Configuration parameters for LASSO regression."""

    nlambda: int = 100
    max_iter: int = 10000
    min_lambda: float = 0.001
    max_lambda: float = 1000
    regression_tol: float = 1e-4
    batch_size: int = 10
    max_workers: int = -1
    real_time_plot: bool = True
    seed: int = 257
    max_features: Optional[int] = None  # Stop when this many features are selected
    feature_threshold: float = 0.1  # Threshold for considering a feature as selected
    base_model_name: Optional[str] = (
        None  # Name of the base model used for generating partial responses
    )

    def validate(self):
        """Validate configuration parameters."""
        if self.nlambda <= 0:
            raise ValueError("nlambda must be positive")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if self.min_lambda <= 0:
            raise ValueError("min_lambda must be positive")
        if self.max_lambda <= self.min_lambda:
            raise ValueError("max_lambda must be greater than min_lambda")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.regression_tol <= 0:
            raise ValueError("regression_tol must be positive")
        if self.feature_threshold <= 0:
            raise ValueError("feature_threshold must be positive")
        if self.max_features is not None and self.max_features <= 0:
            raise ValueError("max_features must be positive if specified")
