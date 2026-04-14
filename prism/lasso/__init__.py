"""
LASSO Implementation for PRiSM

This module implements LASSO regression for feature selection in the PRiSM
framework. It is used to perform LASSO on partial responses, to help identify
the most important features and feature interactions.

Example Usage:
-------------
>>> from prism.lasso import LassoRegression
>>> lasso = LassoRegression()
>>> results = lasso.fit(
...     partial_responses_train,
...     partial_responses_test,
...     y_train,
...     y_test,
...     feature_names=feature_names
... )
>>> results.select_lambda_max_test_auc()
>>> selected_features = results.get_selected_feature_names()
"""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from .computation import LassoComputation
from .config import LassoConfig
from .lasso_results import LassoResultsManager

__all__ = ["LassoRegression", "LassoResultsManager"]


class LassoRegression:
    """
    Main interface for LASSO regression in PRiSM.

    This class provides a high-level interface for performing LASSO regression
    on partial responses, managing configuration, computation, and results.

    The lambda values are processed in batches (controlled by batch_size) and each batch is processed in parallel (controlled by max_workers). A larger batch_size means more lambda values are processed together, while max_workers determines how many parallel jobs can run within each batch.
    """

    def __init__(
        self,
        nlambda: int = 100,
        max_iter: int = 10000,
        min_lambda: float = 0.001,
        max_lambda: float = 1000,
        max_workers: int = -1,
        regression_tol: float = 1e-4,
        batch_size: int = 10,
        real_time_plot: bool = True,
        seed: int = 257,
        max_features: Optional[int] = None,
        feature_threshold: float = 0.1,
        base_model_name: Optional[str] = None,
    ):
        """
        Initialize LassoRegression with configuration parameters.

        Parameters
        ----------
        nlambda : int, optional
            Number of lambda values to use (default is 100)
        max_iter : int, optional
            Maximum number of iterations for solver (default is 10000)
        min_lambda : float, optional
            Minimum lambda value (default is 0.001)
        max_lambda : float, optional
            Maximum lambda value (default is 1000)
        batch_size : int, optional
            Number of lambda values to process in each batch (default is 10).
            Larger values process more lambda values together but require more memory.
            Should be chosen considering max_workers and total number of lambda values
            for optimal throughput.
        max_workers : int, optional
            Number of parallel workers for processing lambda values within each batch (default is -1, using all cores). Each worker processes one lambda value at a time within the current batch.
        regression_tol : float, optional
            Tolerance for LASSO convergence (default is 1e-4)
        real_time_plot : bool, optional
            Whether to show real-time plots (default is True)
        seed : int, optional
            Random seed for reproducibility (default is 257)
        max_features : Optional[int], optional
            Maximum number of features to select before stopping (default is None, no limit)
        feature_threshold : float, optional
            Threshold for considering a feature as selected (default is 0.1)
        base_model_name : Optional[str], optional
            Name of the base model used for generating partial responses (default is None)
        """
        self.config = LassoConfig(
            nlambda=nlambda,
            max_iter=max_iter,
            min_lambda=min_lambda,
            max_lambda=max_lambda,
            max_workers=max_workers,
            regression_tol=regression_tol,
            batch_size=batch_size,
            real_time_plot=real_time_plot,
            seed=seed,
            max_features=max_features,
            feature_threshold=feature_threshold,
            base_model_name=base_model_name,
        )
        self.computation = LassoComputation(self.config)

    def fit(
        self,
        partial_responses_train: torch.Tensor,
        partial_responses_test: torch.Tensor,
        y_train: np.ndarray,
        y_test: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Tuple[LassoResultsManager, Optional[plt.Figure]]:
        """
        Perform LASSO regression across a range of regularization strengths.

        Parameters
        ----------
        partial_responses_train : torch.Tensor
            Partial responses for training data
        partial_responses_test : torch.Tensor
            Partial responses for test data
        y_train : np.ndarray
            Training target values
        y_test : np.ndarray
            Test target values
        feature_names : Optional[List[str]], optional
            Names of univariate features (default is None)

        Returns
        -------
        Tuple[LassoResultsManager, Optional[plt.Figure]]
            Object containing LASSO results and utility methods for analysis,
            and the final visualization figure if real_time_plot is enabled
        """

        # Perform computation
        results, final_figure = self.computation.fit(
            partial_responses_train,
            partial_responses_test,
            y_train,
            y_test,
            feature_names=feature_names,
        )

        return results, final_figure
