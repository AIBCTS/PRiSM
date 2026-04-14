"""
Core computation module for LASSO regression.

This module implements the main LASSO computation logic with efficient batching
and early stopping capabilities.
"""

import logging
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from .config import LassoConfig
from .lasso_results import LassoResultsManager
from .metrics import compute_metrics
from .visualization import LassoVisualizer

logger = logging.getLogger(__name__)


class LassoComputation:
    """Class handling core LASSO computation logic."""

    def __init__(self, config: LassoConfig):
        """
        Initialize LassoComputation.

        Parameters
        ----------
        config : LassoConfig
            Configuration object containing LASSO parameters
        """
        self.config = config
        config.validate()

        # Log configuration as a single atomic message
        config_msg = (
            "Initializing LASSO computation with parameters:\n"
            f"  - max_iter: {config.max_iter:,d}\n"
            f"  - regression_tol: {config.regression_tol:.2e}\n"
            f"  - lambda range: {config.min_lambda:.2e} to {config.max_lambda:.2e}\n"
            f"  - nlambda: {config.nlambda}\n"
            f"  - batch_size: {config.batch_size}\n"
            f"  - feature_threshold: {config.feature_threshold}"
        )
        if config.max_features:
            config_msg += f"\n  - max_features: {config.max_features}"
        if config.base_model_name:
            config_msg += f"\n  - base_model_name: {config.base_model_name}"

        logger.info(config_msg)

        self.visualizer = LassoVisualizer(base_model_name=config.base_model_name)
        self._setup_lambda_values()

    def _setup_lambda_values(self) -> None:
        """Setup lambda values for the LASSO path."""
        self.lambda_values = np.logspace(
            np.log10(self.config.max_lambda),
            np.log10(self.config.min_lambda),
            self.config.nlambda,
        )
        logger.debug(
            f"LASSO path: {self.config.nlambda} values from {self.config.max_lambda:.4g} to {self.config.min_lambda:.4g}"
        )

    def _fit_single_lambda(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lambda_val: float,
        prev_coef: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, LogisticRegression, Dict[str, Any]]:
        """
        Fit a LASSO model for a single lambda value.

        Parameters
        ----------
        X : np.ndarray
            Input features
        y : np.ndarray
            Target values
        lambda_val : float
            Regularization strength
        prev_coef : Optional[np.ndarray]
            Previous coefficients for warm start

        Returns
        -------
        Tuple[np.ndarray, LogisticRegression, Dict[str, Any]]
            Fitted coefficients, model, and fit information
        """
        C = 1 / lambda_val
        model = LogisticRegression(
            C=C,
            penalty="l1",
            solver="saga",
            max_iter=self.config.max_iter,
            random_state=self.config.seed,
            warm_start=True,
            tol=self.config.regression_tol,
        )

        if prev_coef is not None:
            model.classes_ = np.array([0, 1])
            model.coef_ = prev_coef.reshape(1, -1)
            model.intercept_ = np.array([0.0])

        convergence_warning = False
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.fit(X, y)
            if any(issubclass(warn.category, ConvergenceWarning) for warn in w):
                convergence_warning = True
                n_iter = model.n_iter_[0]
                hit_max_iter = n_iter >= self.config.max_iter

                # Use a single log message
                warning_msg = (
                    f"Convergence warning for lambda={lambda_val:.4g}:"
                    f" {n_iter:,d} iterations"
                    f" ({(n_iter / self.config.max_iter) * 100:.1f}% of max_iter)"
                    f"{' - HIT MAX_ITER' if hit_max_iter else ''}"
                )
                logger.warning(warning_msg)

        # Return fit info for aggregation in main thread
        fit_info = {
            "lambda": lambda_val,
            "n_iter": model.n_iter_[0],
            "converged": not convergence_warning,
            "hit_max_iter": model.n_iter_[0] >= self.config.max_iter,
        }

        return model.coef_[0], model, fit_info

    def _count_selected_features(self, beta: np.ndarray, threshold: float = None) -> int:
        """
        Count the number of features with absolute coefficient value above threshold.

        Parameters
        ----------
        beta : np.ndarray
            Coefficient values
        threshold : float, optional
            Threshold for considering a feature as selected (default uses config value)

        Returns
        -------
        int
            Number of selected features
        """
        if threshold is None:
            threshold = self.config.feature_threshold
        return np.sum(np.abs(beta) > threshold)

    def _check_feature_limit(self, beta: np.ndarray) -> bool:
        """
        Check if the number of selected features has reached the maximum limit.

        Parameters
        ----------
        beta : np.ndarray
            Current coefficients

        Returns
        -------
        bool
            True if the feature limit has been reached
        """
        if self.config.max_features is None:
            return False  # No limit set, continue fitting

        num_selected = self._count_selected_features(beta)
        return num_selected >= self.config.max_features

    def fit(
        self,
        partial_responses_train: torch.Tensor,
        partial_responses_test: torch.Tensor,
        y_train: np.ndarray,
        y_test: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Tuple[LassoResultsManager, Optional[plt.Figure]]:
        """
        Fit LASSO models across lambda path.

        Parameters
        ----------
        partial_responses_train : torch.Tensor
            Training partial responses
        partial_responses_test : torch.Tensor
            Test partial responses
        y_train : np.ndarray
            Training labels
        y_test : np.ndarray
            Test labels
        feature_names : Optional[List[str]], default=None
            Names of the univariate features. If None, generic names will be generated.

        Returns
        -------
        Tuple[LassoResultsManager, Optional[plt.Figure]]
            Results manager object containing fitted models, coefficients and metrics,
            and the final visualization figure if real_time_plot is enabled
        """
        start_time = time.time()

        logger.info(
            f"Starting LASSO with {partial_responses_train.shape[0]} samples, {partial_responses_train.shape[1]} features"
        )

        # Convert inputs to numpy arrays
        X_train = partial_responses_train.cpu().numpy()
        X_test = partial_responses_test.cpu().numpy()
        y_train = y_train.to_numpy() if hasattr(y_train, "to_numpy") else y_train
        y_test = y_test.to_numpy() if hasattr(y_test, "to_numpy") else y_test

        num_features = X_train.shape[1]
        n_univ = int((np.sqrt(1 + 8 * num_features) - 1) / 2)

        # Use provided feature names or generate generic ones
        if feature_names is not None:
            if len(feature_names) != n_univ:
                raise ValueError(
                    f"Length of feature_names ({len(feature_names)}) must match number of univariate features ({n_univ})"
                )
            univariate_feature_names = feature_names
        else:
            univariate_feature_names = [f"Feature_{i+1}" for i in range(n_univ)]

        # Initialize arrays for coefficient storage
        betas = np.zeros((num_features, len(self.lambda_values)))
        models = []
        train_losses = np.zeros(len(self.lambda_values))
        test_losses = np.zeros(len(self.lambda_values))
        train_aucs = np.zeros(len(self.lambda_values))
        test_aucs = np.zeros(len(self.lambda_values))
        beta_counts_univ = np.zeros(len(self.lambda_values), dtype=int)
        beta_counts_biv = np.zeros(len(self.lambda_values), dtype=int)

        # Initialize lambda_values to use self.lambda_values by default
        lambda_values = self.lambda_values
        final_figure = None

        # Process lambda values in batches
        total_batches = (len(self.lambda_values) - 1) // self.config.batch_size + 1
        for batch_idx, batch_start in enumerate(
            range(0, len(self.lambda_values), self.config.batch_size)
        ):
            batch_end = min(batch_start + self.config.batch_size, len(self.lambda_values))
            batch_end = min(batch_start + self.config.batch_size, len(self.lambda_values))
            batch_lambdas = self.lambda_values[batch_start:batch_end]

            # Parallel processing of batch
            batch_results = Parallel(n_jobs=self.config.max_workers)(
                delayed(self._fit_single_lambda)(
                    X_train,
                    y_train,
                    lambda_val,
                    betas[:, batch_start - 1] if batch_start > 0 else None,
                )
                for lambda_val in batch_lambdas
            )

            # Track if we've hit the feature limit
            reached_feature_limit = False

            # Process batch results
            for i, (beta, model, fit_info) in enumerate(batch_results):
                idx = batch_start + i
                betas[:, idx] = beta
                models.append(model)

                # Compute metrics
                train_metrics, test_metrics = compute_metrics(
                    model, X_train, X_test, y_train, y_test
                )

                # Store metrics
                train_losses[idx] = train_metrics["loss"]
                test_losses[idx] = test_metrics["loss"]
                train_aucs[idx] = train_metrics["auc"]
                test_aucs[idx] = test_metrics["auc"]

                # Process beta counts and feature limits
                n_univ_features = np.sum(np.abs(beta[:n_univ]) > self.config.feature_threshold)
                n_biv_features = np.sum(np.abs(beta[n_univ:]) > self.config.feature_threshold)
                beta_counts_univ[idx] = n_univ_features
                beta_counts_biv[idx] = n_biv_features
                n_total_features = n_univ_features + n_biv_features

                # Check feature limit and log atomically
                if self.config.max_features and n_total_features >= self.config.max_features:
                    reached_feature_limit = True
                    feature_msg = (
                        f"Feature limit reached ({n_total_features}/{self.config.max_features} features) "
                        f"at lambda={fit_info['lambda']:.4g}:\n"
                        f"  - Univariate features: {n_univ_features}\n"
                        f"  - Bivariate features: {n_biv_features}\n"
                        f"The current batch (lambda values {batch_start} to {batch_end-1}) will complete before stopping."
                    )
                    logger.info(feature_msg)

            # Log progress after each batch
            if batch_end % self.config.batch_size == 0 or batch_end == len(self.lambda_values):
                n_max_iter_in_batch = sum(
                    1 for _, _, info in batch_results if info["hit_max_iter"]
                )

                batch_msg = (
                    f"LASSO progress: {batch_idx+1}/{total_batches} batches completed "
                    f"({batch_end}/{len(self.lambda_values)} lambda values)"
                )

                if n_max_iter_in_batch > 0:
                    batch_msg += f" - {n_max_iter_in_batch} hit max_iter in this batch"

                logger.debug(batch_msg)

            # Update visualization if requested
            if self.config.real_time_plot:
                metrics = {
                    "train_loss": train_losses,
                    "test_loss": test_losses,
                    "train_auc": train_aucs,
                    "test_auc": test_aucs,
                    "beta_counts_univ": beta_counts_univ,
                    "beta_counts_biv": beta_counts_biv,
                }
                is_final = reached_feature_limit or batch_end == len(self.lambda_values)
                final_figure = self.visualizer.display_progress(
                    self.lambda_values, metrics, batch_end, is_final=is_final, return_fig=is_final
                )

            # Stop if we've reached the feature limit
            if reached_feature_limit:
                # Final visualization with highlighting
                if not self.config.real_time_plot:
                    metrics = {
                        "train_loss": train_losses,
                        "test_loss": test_losses,
                        "train_auc": train_aucs,
                        "test_auc": test_aucs,
                        "beta_counts_univ": beta_counts_univ,
                        "beta_counts_biv": beta_counts_biv,
                    }
                    final_figure = self.visualizer.display_progress(
                        self.lambda_values, metrics, batch_end, is_final=True, return_fig=True
                    )
                # Trim results
                betas = betas[:, :batch_end]
                models = models[:batch_end]
                lambda_values = self.lambda_values[:batch_end]
                train_losses = train_losses[:batch_end]
                test_losses = test_losses[:batch_end]
                train_aucs = train_aucs[:batch_end]
                test_aucs = test_aucs[:batch_end]
                break

        # Create results manager
        results = LassoResultsManager(
            lambda_values,
            betas,
            models,
            univariate_feature_names,
            train_losses,
            test_losses,
            train_aucs,
            test_aucs,
            base_model_name=self.config.base_model_name,
        )

        # Compute summary statistics in main thread
        total_time = time.time() - start_time
        n_lambda = len(lambda_values)

        # Track which lambdas hit max_iter
        max_iter_indices = []
        n_max_iter = 0
        for i, model in enumerate(models):
            n_iter = model.n_iter_[0]
            if n_iter >= self.config.max_iter:
                n_max_iter += 1
                max_iter_indices.append(i)

        n_converged = n_lambda - n_max_iter

        # Log summary as a single atomic message
        summary_msg = (
            f"\nLASSO computation completed in {total_time:.2f}s\n"
            f"Training Summary:\n"
            f"  - Total lambda values processed: {n_lambda}\n"
            f"  - Converged normally: {n_converged} ({n_converged/n_lambda*100:.1f}%)\n"
            f"  - Hit max_iter: {n_max_iter} ({n_max_iter/n_lambda*100:.1f}%)"
        )

        if n_max_iter > 0:
            max_iter_lambda_values = sorted([lambda_values[i] for i in max_iter_indices])
            max_iter_lambdas = [f"{lambda_val:.4g}" for lambda_val in max_iter_lambda_values]
            max_iter_range = (
                f"(range: {max_iter_lambda_values[0]:.4g} to {max_iter_lambda_values[-1]:.4g})"
                if len(max_iter_lambda_values) > 1
                else ""
            )
            summary_msg += f"\n  - lambda values that hit max_iter: {', '.join(max_iter_lambdas)} {max_iter_range}"

        logger.info(summary_msg)

        # Notify user about log location
        # Get log file paths dynamically from the logging system
        log_files = []
        for handler in logging.getLogger().handlers + logger.handlers:
            if hasattr(handler, 'baseFilename'):
                log_files.append(handler.baseFilename)

        # Notify user about log location
        if log_files:
            print("For detailed logging information, check the log file(s):")
            for log_file in log_files:
                print(f"  - {log_file}")
        else:
            print("Logging is configured but no file handlers were found.")

        return results, final_figure
