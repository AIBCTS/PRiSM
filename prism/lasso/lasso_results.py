"""
Results management module for LASSO regression.

This module provides the LassoResultsManager class for analyzing and
visualizing LASSO regression results.
"""

import logging
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


class LassoResultsManager:
    """
    Manages and analyzes results from LASSO regression on partial responses.

    This class provides methods to select optimal lambda values, retrieve selected
    features and their coefficients, and visualize LASSO regression results.

    Parameters
    ----------
    lambdas : np.ndarray
        Array of lambda values used in LASSO regression.
    betas : np.ndarray
        Coefficient values for each feature across all lambda values.
    models : List[LogisticRegression]
        Fitted logistic regression models for each lambda.
    feature_names : List[str]
        Names of univariate features.
    train_losses, test_losses : np.ndarray
        Training and test loss values for each lambda.
    train_aucs, test_aucs : np.ndarray
        Training and test AUC scores for each lambda.
    base_model_name : str, optional
        Name of the base model used for generating partial responses.

    Attributes
    ----------
    selected_lambda_index : int or None
        Index of the selected lambda value.
    n_univ : int
        Number of univariate features.
    n_biv : int
        Number of bivariate feature combinations.
    num_features : int
        Total number of features (univariate + bivariate).
    all_feature_names : List[str]
        Names of all features, including bivariate combinations.
    base_model_name : str or None
        Name of the base model used for generating partial responses.

    Methods
    -------
    select_lambda(lambda_index)
        Select a specific lambda value by index.
    select_lambda_max_test_auc()
        Select the lambda that maximizes test AUC.
    get_selected_beta()
        Get the coefficient values for the selected lambda.
    get_selected_model()
        Get the logistic regression model for the selected lambda.
    plot_lambda_path()
        Plot the LASSO path showing coefficient values vs lambda.
    plot_beta_values()
        Plot the absolute beta coefficient values of selected features.
    get_mask()
        Generate a mask for selected features based on coefficient values.

    Examples
    --------
    >>> lasso_results = LassoResultsManager(lambdas, betas, models, feature_names,
    ...                                     train_losses, test_losses, train_aucs,
    ...                                     test_aucs)
    >>> lasso_results.select_lambda_max_test_auc()
    >>> selected_features = lasso_results.get_selected_feature_names()
    >>> lasso_results.plot_beta_values()
    """

    def __init__(
        self,
        lambdas: np.ndarray,
        betas: np.ndarray,
        models: List[LogisticRegression],
        feature_names: List[str],
        train_losses: np.ndarray,
        test_losses: np.ndarray,
        train_aucs: np.ndarray,
        test_aucs: np.ndarray,
        base_model_name: Optional[str] = None,
    ):
        self.lambdas = lambdas
        self.betas = betas
        self.models = models
        self.univariate_feature_names = feature_names
        self.train_losses = train_losses
        self.test_losses = test_losses
        self.train_aucs = train_aucs
        self.test_aucs = test_aucs
        self.selected_lambda_index = None
        self.base_model_name = base_model_name

        self.n_univ = len(feature_names)
        self.bivariate_inputs = list(combinations(range(self.n_univ), 2))
        self.n_biv = len(self.bivariate_inputs)
        self.num_features = self.n_univ + self.n_biv

        self._generate_all_feature_names()

    def _generate_all_feature_names(self):
        self.all_feature_names = self.univariate_feature_names.copy()
        for i, (f1, f2) in enumerate(self.bivariate_inputs):
            self.all_feature_names.append(
                f"{self.univariate_feature_names[f1]} : {self.univariate_feature_names[f2]}"
            )

    @property
    def univariate_feature_names_clean(self) -> List[str]:
        """Return univariate feature names with linebreaks replaced by spaces."""
        return [name.replace('\n', ' ') for name in self.univariate_feature_names]

    @property
    def all_feature_names_clean(self) -> List[str]:
        """Return all feature names with linebreaks replaced by spaces."""
        return [name.replace('\n', ' ') for name in self.all_feature_names]

    def _ensure_nonzero_features(self, index: int, threshold: float, console: Console) -> int:
        """
        Ensure the selected lambda has non-zero features.
        If not, warn and select the first lambda with non-zero features.
        If no such lambda exists, raise ValueError.
        """
        beta = self.betas[:, index]
        if np.sum(np.abs(beta) > threshold) > 0:
            return index

        logger.warning(
            "Selected lambda index %d has 0 features above threshold %.4f. "
            "Searching for first lambda with non-zero features.",
            index,
            threshold,
        )
        console.print(
            f"[bold yellow]Warning:[/bold yellow] Selected lambda (index {index}) has 0 features above threshold {threshold}. "
            "Searching for first lambda with non-zero features."
        )

        for i in range(len(self.lambdas)):
            if np.sum(np.abs(self.betas[:, i]) > threshold) > 0:
                logger.info("Fallback to lambda index %d with features.", i)
                console.print(f"[bold green]Fallback:[/bold green] Selected lambda index {i}.")
                return i

        error_msg = f"No lambda found with features above threshold {threshold}."
        logger.error(error_msg)
        raise ValueError(error_msg)

    def select_lambda(self, lambda_index: int, threshold: float = 0.1):
        """
        Select a specific lambda value by index.

        Parameters:
        lambda_index : int
            Index of the lambda value to select
        threshold : float, optional
            Threshold for considering a feature as selected (default is 0.1)

        Returns:
        int: The index of the selected lambda
        """
        console = Console()

        if lambda_index < 0 or lambda_index >= len(self.lambdas):
            raise ValueError("Invalid lambda index")

        lambda_index = self._ensure_nonzero_features(lambda_index, threshold, console)

        self.selected_lambda_index = lambda_index

        # Log the selection details
        logger.info("Manually selected lambda index: %d", lambda_index)
        logger.info("Selected lambda value: %.4f", self.lambdas[lambda_index])
        logger.info("Corresponding test AUC: %.4f", self.test_aucs[lambda_index])

        # Create a table for selection results
        table = Table(box=box.ROUNDED, title="Selected Lambda", title_style="bold blue")
        table.add_column("Metric", style="cyan", justify="right")
        table.add_column("Value", style="green")

        # Add basic info rows
        table.add_row("Lambda Index", f"{lambda_index}")
        table.add_row("Lambda Value", f"{self.lambdas[lambda_index]:.4f}")
        table.add_row("Test AUC", f"{self.test_aucs[lambda_index]:.4f}")
        table.add_row("Train AUC", f"{self.train_aucs[lambda_index]:.4f}")

        # Calculate feature counts for the selected model
        beta = self.betas[:, lambda_index]
        selected_features_count = np.sum(np.abs(beta) > threshold)
        univariate_count = np.sum(np.abs(beta[: self.n_univ]) > threshold)
        bivariate_count = np.sum(np.abs(beta[self.n_univ :]) > threshold)

        table.add_row("Total Features", f"{selected_features_count}")
        table.add_row("Univariate Features", f"{univariate_count}")
        table.add_row("Bivariate Features", f"{bivariate_count}")

        console.print(table)

        # Display selected features
        self._print_selected_features(threshold)

        return lambda_index

    def select_lambda_by_features(self, target_features: int, threshold: float = 0.1):
        """
        Select the lambda that reaches a specified number of features with the best test AUC.

        Parameters:
        target_features : int
            Target number of features to reach
        threshold : float, optional
            Threshold for considering a feature as selected (default is 0.1)

        Returns:
        int: The index of the selected lambda
        """
        console = Console()

        if target_features <= 0:
            raise ValueError("target_features must be positive")

        # Calculate feature counts for each lambda
        feature_counts = []
        for i in range(len(self.lambdas)):
            beta = self.betas[:, i]
            count = np.sum(np.abs(beta) > threshold)
            feature_counts.append(count)

        feature_counts = np.array(feature_counts)

        # Find all lambdas that reach the target number of features
        qualifying_indices = np.where(feature_counts == target_features)[0]

        if len(qualifying_indices) == 0:
            # If no lambda reaches the exact target, find the closest
            closest_count = feature_counts[np.argmin(np.abs(feature_counts - target_features))]
            qualifying_indices = np.where(feature_counts == closest_count)[0]
            logger.warning(
                "No lambda reaches exactly %d features. Using closest count: %d features.",
                target_features,
                closest_count,
            )

        # Among qualifying lambdas, select the one with the best test AUC
        if len(qualifying_indices) > 1:
            best_auc_among_qualifying = np.argmax(self.test_aucs[qualifying_indices])
            selected_index = qualifying_indices[best_auc_among_qualifying]
            logger.info(
                "Found %d lambdas with %d features. Selected lambda with best test AUC.",
                len(qualifying_indices),
                feature_counts[selected_index],
            )
        else:
            selected_index = qualifying_indices[0]

        selected_index = self._ensure_nonzero_features(selected_index, threshold, console)

        self.selected_lambda_index = selected_index

        # Log the selection details
        logger.info("Selected lambda index: %d", selected_index)
        logger.info("Selected lambda value: %.4f", self.lambdas[selected_index])
        logger.info(
            "Target features: %d, Actual features: %d",
            target_features,
            feature_counts[selected_index],
        )

        # Create a panel for model selection info
        selection_strategy = f"Lambda with {target_features} features and best test AUC"
        if feature_counts[selected_index] != target_features:
            selection_strategy = f"Lambda with closest to {target_features} features ({feature_counts[selected_index]}) and best test AUC"

        panel = Panel(
            f"Lambda Selection Strategy: {selection_strategy}",
            title="Model Selection",
            border_style="blue",
        )
        console.print(panel)

        # Create a table for results
        table = Table(box=box.ROUNDED, title="Selected Lambda Results", title_style="bold blue")
        table.add_column("Metric", style="cyan", justify="right")
        table.add_column("Value", style="green")

        # Add basic info rows
        table.add_row("Lambda Index", f"{selected_index}")
        table.add_row("Lambda Value", f"{self.lambdas[selected_index]:.4f}")
        table.add_row("Test AUC", f"{self.test_aucs[selected_index]:.4f}")
        table.add_row("Train AUC", f"{self.train_aucs[selected_index]:.4f}")

        # Calculate feature counts for the selected model
        beta = self.betas[:, selected_index]
        selected_features_count = np.sum(np.abs(beta) > threshold)
        univariate_count = np.sum(np.abs(beta[: self.n_univ]) > threshold)
        bivariate_count = np.sum(np.abs(beta[self.n_univ :]) > threshold)

        table.add_row("Target Features", f"{target_features}")
        table.add_row("Actual Features", f"{selected_features_count}")
        table.add_row("Univariate Features", f"{univariate_count}")
        table.add_row("Bivariate Features", f"{bivariate_count}")

        # Add info about multiple candidates if applicable
        if len(qualifying_indices) > 1:
            table.add_section()
            table.add_row("Candidates Found", f"{len(qualifying_indices)}")
            min_auc = np.min(self.test_aucs[qualifying_indices])
            max_auc = np.max(self.test_aucs[qualifying_indices])
            table.add_row("AUC Range", f"{min_auc:.4f} - {max_auc:.4f}")

        console.print(table)

        # Display selected features
        self._print_selected_features(threshold)

        return selected_index

    def select_lambda_max_test_auc(self, threshold: float = 0.1, target_ratio: float = 0.99):
        """
        Select the lambda that produces the highest test AUC or the first lambda that reaches
        a specified percentage of the maximum test AUC.

        Parameters:
        threshold : float, optional
            Threshold for considering a feature as selected (default is 0.1)
        target_ratio : float, optional
            The target ratio of the maximum test AUC to achieve (default is 0.99).
            A value of 1.0 will select the lambda with the maximum test AUC.
            Values < 1.0 will select the first lambda (moving from high to low regularization)
            that achieves at least this fraction of the maximum test AUC.

        Returns:
        int: The index of the selected lambda
        """
        console = Console()

        if target_ratio <= 0 or target_ratio > 1:
            logger.error("Invalid target_ratio: %f. Must be in range (0, 1]", target_ratio)
            raise ValueError("target_ratio must be in the range (0, 1]")

        max_auc = np.max(self.test_aucs)
        target_auc = max_auc * target_ratio

        # Find all lambdas that exceed the target AUC
        qualifying_indices = np.where(self.test_aucs >= target_auc)[0]

        if len(qualifying_indices) == 0:
            # This should not happen unless target_ratio > 1
            logger.warning("No lambdas meet the target ratio criteria. Selecting maximum AUC.")
            max_auc_index = np.argmax(self.test_aucs)
        else:
            # Get the index of the first lambda that exceeds the target
            # (remember lambdas are ordered from highest to lowest regularization)
            max_auc_index = qualifying_indices[0]
            logger.info(
                "Found %d lambda values meeting target ratio of %.3f",
                len(qualifying_indices),
                target_ratio,
            )

        max_auc_index = self._ensure_nonzero_features(max_auc_index, threshold, console)

        self.selected_lambda_index = max_auc_index

        # Log the selection details
        logger.info("Selected lambda index: %d", max_auc_index)
        logger.info("Selected lambda value: %.4f", self.lambdas[max_auc_index])
        logger.info("Corresponding test AUC: %.4f", self.test_aucs[max_auc_index])

        # Create a panel for model selection info
        panel = Panel(
            f"Lambda Selection Strategy: {'Maximum Test AUC' if target_ratio == 1.0 else f'Target Ratio {target_ratio:.1%} of Max AUC'}",
            title="Model Selection",
            border_style="blue",
        )
        console.print(panel)

        # Create a table for results
        table = Table(box=box.ROUNDED, title="Selected Lambda Results", title_style="bold blue")
        table.add_column("Metric", style="cyan", justify="right")
        table.add_column("Value", style="green")

        # Add basic info rows
        table.add_row("Lambda Index", f"{max_auc_index}")
        table.add_row("Lambda Value", f"{self.lambdas[max_auc_index]:.4f}")
        table.add_row("Test AUC", f"{self.test_aucs[max_auc_index]:.4f}")
        table.add_row("Train AUC", f"{self.train_aucs[max_auc_index]:.4f}")

        # Calculate feature counts for the selected model
        beta_max = self.betas[:, max_auc_index]
        selected_features_count = np.sum(np.abs(beta_max) > threshold)
        univariate_count = np.sum(np.abs(beta_max[: self.n_univ]) > threshold)
        bivariate_count = np.sum(np.abs(beta_max[self.n_univ :]) > threshold)

        table.add_row("Total Features", f"{selected_features_count}")
        table.add_row("Univariate Features", f"{univariate_count}")
        table.add_row("Bivariate Features", f"{bivariate_count}")

        # Add comparison info if using target ratio < 1.0
        if target_ratio < 1.0:
            max_possible_auc = np.max(self.test_aucs)
            max_possible_index = np.argmax(self.test_aucs)

            table.add_section()
            table.add_row("Maximum Possible AUC", f"{max_possible_auc:.4f}")
            table.add_row("Maximum AUC Lambda Index", f"{max_possible_index}")
            table.add_row(
                "Percent of Maximum AUC", f"{self.test_aucs[max_auc_index]/max_possible_auc:.1%}"
            )

            # Add feature comparison if the selected model is different from the max AUC model
            if max_auc_index != max_possible_index:
                beta_best = self.betas[:, max_possible_index]
                best_features_count = np.sum(np.abs(beta_best) > threshold)
                best_univariate_count = np.sum(np.abs(beta_best[: self.n_univ]) > threshold)
                best_bivariate_count = np.sum(np.abs(beta_best[self.n_univ :]) > threshold)

                table.add_section()
                table.add_row("Max AUC Model Features", f"{best_features_count}")
                table.add_row(
                    "Feature Reduction", f"{1 - selected_features_count/best_features_count:.1%}"
                )
                table.add_row(
                    "Univariate Feature Reduction",
                    (
                        f"{1 - univariate_count/best_univariate_count:.1%}"
                        if best_univariate_count > 0
                        else "N/A"
                    ),
                )
                table.add_row(
                    "Bivariate Feature Reduction",
                    (
                        f"{1 - bivariate_count/best_bivariate_count:.1%}"
                        if best_bivariate_count > 0
                        else "N/A"
                    ),
                )

        console.print(table)

        # Display selected features
        self._print_selected_features(threshold)

        return max_auc_index

    def select_lambda_non_inferiority(
        self,
        threshold: float = 0.1,
        ni_level: float = 0.1,
        reference_auc: Optional[float] = None,
    ) -> int:
        """
        Select the sparsest model meeting a dynamically computed AUC threshold.

        The non-inferiority threshold is computed as:
            threshold = ref_auc - ni_level * (ref_auc - 0.5)

        This accepts models that preserve at least (1 - ni_level) of the "useful AUC"
        above random chance (0.5). The method selects the first lambda (highest
        regularization = sparsest model) that meets this threshold.

        Parameters
        ----------
        threshold : float, optional
            Threshold for considering a feature as selected (default is 0.1)
        ni_level : float
            Non-inferiority margin, must be in range (0, 1]. Default is 0.1 (10%).
            - 0.1 = accept up to 10% loss of useful AUC
            - 0.2 = accept up to 20% loss of useful AUC
            - 1.0 = accept any model down to random chance (AUC=0.5)
        reference_auc : float, optional
            Reference AUC for the non-inferiority comparison (e.g., the original
            blackbox model's test AUROC). When provided, this is used instead of
            the max LASSO test AUC. When None (default), falls back to
            ``np.max(self.test_aucs)`` for backward compatibility.

        Returns
        -------
        int
            The index of the selected lambda

        Notes
        -----
        If no lambda achieves the non-inferiority threshold, falls back to selecting
        the lambda with maximum test AUC and issues a warning.

        Examples
        --------
        >>> # With reference_auc=0.85 and ni_level=0.1:
        >>> # useful_auc = 0.85 - 0.50 = 0.35
        >>> # allowed_loss = 0.1 * 0.35 = 0.035
        >>> # threshold = 0.85 - 0.035 = 0.815
        >>> lasso_results.select_lambda_non_inferiority(ni_level=0.1, reference_auc=0.85)

        >>> # Without reference_auc, falls back to max LASSO test AUC:
        >>> lasso_results.select_lambda_non_inferiority(ni_level=0.1)
        """
        console = Console()

        if ni_level <= 0 or ni_level > 1:
            raise ValueError(f"ni_level must be in range (0, 1], got {ni_level}")

        # Compute dynamic threshold
        if reference_auc is not None:
            ref_auc = reference_auc
            ref_source = "blackbox model"
        else:
            ref_auc = np.max(self.test_aucs)
            ref_source = "max LASSO test AUC"

        useful_auc = ref_auc - 0.5  # AUC above random chance
        allowed_loss = ni_level * useful_auc
        auc_threshold = ref_auc - allowed_loss

        # Log the calculation
        logger.info(
            "Non-inferiority threshold calculation: ref_auc=%.4f (%s), ni_level=%.2f, "
            "useful_auc=%.4f, allowed_loss=%.4f, threshold=%.4f",
            ref_auc,
            ref_source,
            ni_level,
            useful_auc,
            allowed_loss,
            auc_threshold,
        )

        # Find first lambda that meets threshold (highest regularization = sparsest)
        selected_index = None
        for i in range(len(self.lambdas)):
            if self.test_aucs[i] >= auc_threshold:
                selected_index = i
                break

        # Fallback to max test AUC if no lambda meets threshold
        if selected_index is None:
            logger.warning(
                "No lambda found with test AUC >= %.4f. "
                "Falling back to max test AUC selection.",
                auc_threshold,
            )
            console.print(
                f"[bold yellow]Warning:[/bold yellow] No lambda achieves non-inferiority "
                f"threshold={auc_threshold:.4f}. Falling back to max test AUC selection."
            )
            return self.select_lambda_max_test_auc(threshold=threshold)

        # Ensure selected lambda has non-zero features
        selected_index = self._ensure_nonzero_features(selected_index, threshold, console)

        self.selected_lambda_index = selected_index

        # Log the selection details
        logger.info(
            "Selected lambda index: %d (first to exceed threshold=%.4f)",
            selected_index,
            auc_threshold,
        )
        logger.info("Selected lambda value: %.4f", self.lambdas[selected_index])
        logger.info("Corresponding test AUC: %.4f", self.test_aucs[selected_index])

        # Create a panel for model selection info with threshold breakdown
        ni_percent = int(ni_level * 100)
        panel = Panel(
            f"Lambda Selection Strategy: Non-Inferiority (ni_level={ni_percent}%)\n"
            f"Reference AUC ({ref_source}): {ref_auc:.4f}\n"
            f"Dynamic Threshold: ref_auc ({ref_auc:.4f}) - {ni_percent}% * "
            f"useful_auc ({useful_auc:.4f}) = {auc_threshold:.4f}",
            title="Model Selection",
            border_style="blue",
        )
        console.print(panel)

        # Create a table for results
        table = Table(box=box.ROUNDED, title="Selected Lambda Results", title_style="bold blue")
        table.add_column("Metric", style="cyan", justify="right")
        table.add_column("Value", style="green")

        # Add basic info rows
        table.add_row("Lambda Index", f"{selected_index}")
        table.add_row("Lambda Value", f"{self.lambdas[selected_index]:.4f}")
        table.add_row("Test AUC", f"{self.test_aucs[selected_index]:.4f}")
        table.add_row("Train AUC", f"{self.train_aucs[selected_index]:.4f}")
        table.add_row("NI Threshold", f"{auc_threshold:.4f}")

        # Calculate feature counts for the selected model
        beta = self.betas[:, selected_index]
        selected_features_count = np.sum(np.abs(beta) > threshold)
        univariate_count = np.sum(np.abs(beta[: self.n_univ]) > threshold)
        bivariate_count = np.sum(np.abs(beta[self.n_univ :]) > threshold)

        table.add_row("Total Features", f"{selected_features_count}")
        table.add_row("Univariate Features", f"{univariate_count}")
        table.add_row("Bivariate Features", f"{bivariate_count}")

        # Show max AUC for comparison
        max_lasso_auc = np.max(self.test_aucs)
        max_auc_idx = np.argmax(self.test_aucs)
        if max_auc_idx != selected_index:
            table.add_section()
            table.add_row("Max LASSO Test AUC", f"{max_lasso_auc:.4f} (at index {max_auc_idx})")
            max_features = np.sum(np.abs(self.betas[:, max_auc_idx]) > threshold)
            table.add_row("Features at Max AUC", f"{max_features}")
            if max_features > 0:
                table.add_row(
                    "Feature Reduction", f"{1 - selected_features_count/max_features:.1%}"
                )

        console.print(table)

        # Display selected features
        self._print_selected_features(threshold)

        return selected_index

    def select_lambda_min_test_auc(self, threshold: float = 0.1, min_auc: float = 0.70) -> int:
        """
        Select the first lambda (highest regularization) that exceeds a minimum test AUC.

        This method finds the sparsest model (fewest features) that still achieves
        the specified minimum AUROC threshold. Useful when interpretability is
        prioritized over maximum performance.

        Parameters
        ----------
        threshold : float, optional
            Threshold for considering a feature as selected (default is 0.1)
        min_auc : float
            The minimum test AUC required. Must be between 0 and 1.

        Returns
        -------
        int
            The index of the selected lambda

        Notes
        -----
        If no lambda achieves the minimum AUC, falls back to selecting the lambda
        with maximum test AUC and issues a warning.

        Examples
        --------
        >>> lasso_results.select_lambda_min_test_auc(min_auc=0.75)
        Selected lambda index 45 with test AUC 0.7523 (first to exceed 0.75)
        """
        console = Console()

        if min_auc <= 0 or min_auc > 1:
            raise ValueError(f"min_auc must be between 0 and 1, got {min_auc}")

        # Find first lambda (from high regularization to low) that exceeds min_auc
        # Lambdas are typically ordered from high to low, so we iterate in order
        # to find the first (most regularized/sparsest) that meets the threshold
        selected_index = None
        for i in range(len(self.lambdas)):
            if self.test_aucs[i] >= min_auc:
                selected_index = i
                break

        # Fallback to max test AUC if no lambda exceeds the threshold
        if selected_index is None:
            max_auc = np.max(self.test_aucs)
            logger.warning(
                "No lambda found with test AUC >= %.4f. Maximum achieved: %.4f. "
                "Falling back to max test AUC selection.",
                min_auc,
                max_auc,
            )
            console.print(
                f"[bold yellow]Warning:[/bold yellow] No lambda achieves min_auc={min_auc:.4f}. "
                f"Max test AUC is {max_auc:.4f}. Falling back to max test AUC selection."
            )
            return self.select_lambda_max_test_auc(threshold=threshold)

        # Ensure selected lambda has non-zero features
        selected_index = self._ensure_nonzero_features(selected_index, threshold, console)

        self.selected_lambda_index = selected_index

        # Log the selection details
        logger.info(
            "Selected lambda index: %d (first to exceed min_auc=%.4f)", selected_index, min_auc
        )
        logger.info("Selected lambda value: %.4f", self.lambdas[selected_index])
        logger.info("Corresponding test AUC: %.4f", self.test_aucs[selected_index])

        # Create a panel for model selection info
        panel = Panel(
            f"Lambda Selection Strategy: First lambda with test AUC ≥ {min_auc:.4f}",
            title="Model Selection",
            border_style="blue",
        )
        console.print(panel)

        # Create a table for results
        table = Table(box=box.ROUNDED, title="Selected Lambda Results", title_style="bold blue")
        table.add_column("Metric", style="cyan", justify="right")
        table.add_column("Value", style="green")

        # Add basic info rows
        table.add_row("Lambda Index", f"{selected_index}")
        table.add_row("Lambda Value", f"{self.lambdas[selected_index]:.4f}")
        table.add_row("Test AUC", f"{self.test_aucs[selected_index]:.4f}")
        table.add_row("Train AUC", f"{self.train_aucs[selected_index]:.4f}")
        table.add_row("Min AUC Target", f"{min_auc:.4f}")

        # Calculate feature counts for the selected model
        beta = self.betas[:, selected_index]
        selected_features_count = np.sum(np.abs(beta) > threshold)
        univariate_count = np.sum(np.abs(beta[: self.n_univ]) > threshold)
        bivariate_count = np.sum(np.abs(beta[self.n_univ :]) > threshold)

        table.add_row("Total Features", f"{selected_features_count}")
        table.add_row("Univariate Features", f"{univariate_count}")
        table.add_row("Bivariate Features", f"{bivariate_count}")

        # Show max AUC for comparison
        max_auc = np.max(self.test_aucs)
        max_auc_idx = np.argmax(self.test_aucs)
        if max_auc_idx != selected_index:
            table.add_section()
            table.add_row("Max Test AUC", f"{max_auc:.4f} (at index {max_auc_idx})")
            max_features = np.sum(np.abs(self.betas[:, max_auc_idx]) > threshold)
            table.add_row("Features at Max AUC", f"{max_features}")

        console.print(table)

        # Display selected features
        self._print_selected_features(threshold)

        return selected_index

    def _print_selected_features(self, threshold: float = 0.1):
        """
        Print selected features and their beta values using Rich formatting.

        Parameters:
        threshold : float, optional
            Threshold for considering a feature as selected (default is 0.1)
        """
        console = Console()

        if self.selected_lambda_index is None:
            logger.warning("No lambda selected. Cannot print selected features.")
            console.print(
                "[bold yellow]Warning:[/bold yellow] No lambda selected. Select a lambda first."
            )
            return

        beta = self.get_selected_beta()
        selected_indices = np.where(np.abs(beta) > threshold)[0]

        if len(selected_indices) == 0:
            logger.warning("No features selected with |beta| > %f", threshold)
            console.print(
                f"[bold yellow]Note:[/bold yellow] No features have |beta| > {threshold}"
            )
            return

        logger.info("Found %d features with |beta| > %f", len(selected_indices), threshold)

        # Create a table for selected features
        table = Table(
            box=box.ROUNDED,
            title=f"Selected Features (|beta| > {threshold})",
            title_style="bold green",
        )

        # Add columns
        table.add_column("Coefficient (beta)", style="cyan", justify="right")
        table.add_column("Feature", style="green")
        table.add_column("Type", style="magenta", justify="center", width=12)

        # Add rows for each selected feature, sorted by absolute coefficient value
        for idx in sorted(selected_indices, key=lambda i: abs(beta[i]), reverse=True):
            feature_name = self.all_feature_names_clean[idx]
            beta_value = beta[idx]

            # Determine if this is a univariate or bivariate feature
            feature_type = "Univariate" if idx < self.n_univ else "Bivariate"

            # Add color coding based on coefficient sign
            beta_text = f"{beta_value:.4f}"
            if beta_value > 0:
                beta_text = f"[green]{beta_text}[/green]"
            else:
                beta_text = f"[red]{beta_text}[/red]"

            table.add_row(beta_text, feature_name, feature_type)

        # Summary row
        table.add_section()
        table.add_row(
            f"[bold]Total: {len(selected_indices)}[/bold]",
            f"[bold]Univariate: {sum(1 for idx in selected_indices if idx < self.n_univ)}[/bold]",
            f"[bold]Bivariate: {sum(1 for idx in selected_indices if idx >= self.n_univ)}[/bold]",
        )

        console.print(table)

    def get_selected_beta(self) -> np.ndarray:
        if self.selected_lambda_index is None:
            raise ValueError("No lambda selected")
        return self.betas[:, self.selected_lambda_index]

    def get_thresholded_model(self, threshold=0.1):
        if self.selected_lambda_index is None:
            raise ValueError("No lambda selected")

        model = self.models[self.selected_lambda_index]

        # Create a copy of the model
        thresholded_model = clone(model)

        # Copy the fitted attributes from the original model
        thresholded_model.classes_ = model.classes_.copy()
        thresholded_model.coef_ = model.coef_.copy()
        thresholded_model.intercept_ = model.intercept_.copy()

        # Zero out small coefficients
        mask = np.abs(thresholded_model.coef_[0]) <= threshold
        thresholded_model.coef_[0][mask] = 0

        return thresholded_model

    def get_selected_model(self, threshold=0.1) -> LogisticRegression:
        if self.selected_lambda_index is None:
            raise ValueError("No lambda selected")
        else:
            print(
                f"Logistic regression model for Lambda index {self.selected_lambda_index} ({self.lambdas[self.selected_lambda_index]:.4g}) selected. |beta| > {threshold} kept."
            )
        return self.get_thresholded_model(threshold=threshold)

    def get_selected_feature_indicies(self, threshold: float = 0.1) -> List[int]:
        beta = self.get_selected_beta()
        return np.where(np.abs(beta) > threshold)[0]

    def get_selected_feature_names(self, threshold: float = 0.1) -> List[str]:
        return [
            self.all_feature_names[i]
            for i in self.get_selected_feature_indicies(threshold=threshold)
        ]

    def get_selected_feature_names_clean(self, threshold: float = 0.1) -> List[str]:
        """Return selected feature names with linebreaks replaced by spaces."""
        return [
            self.all_feature_names_clean[i]
            for i in self.get_selected_feature_indicies(threshold=threshold)
        ]

    def get_selected_univariate_indices(self, threshold: float = 0.1) -> List[int]:
        beta = self.get_selected_beta()
        return [i for i in range(self.n_univ) if abs(beta[i]) > threshold]

    def get_selected_bivariate_indices(self, threshold: float = 0.1) -> List[int]:
        beta = self.get_selected_beta()
        return [i - self.n_univ for i in range(self.n_univ, len(beta)) if abs(beta[i]) > threshold]

    def get_selected_bivariate_index_pairs(self, threshold: float = 0.1) -> List[Tuple[int, int]]:
        return [self.bivariate_inputs[i] for i in self.get_selected_bivariate_indices()]

    def is_mixed_bivariate(
        self, feature1: int, feature2: int, x: np.ndarray, categorical_threshold: int
    ) -> bool:
        is_categorical1 = len(np.unique(x[:, feature1])) < categorical_threshold
        is_categorical2 = len(np.unique(x[:, feature2])) < categorical_threshold
        return is_categorical1 != is_categorical2

    def plot_lambda_path(self):
        fig, ax = plt.subplots(figsize=(8, 4))
        for i, name in enumerate(self.all_feature_names):
            ax.semilogx(self.lambdas, self.betas[i, :], label=name)
        ax.set_xlabel('Lambda')
        ax.set_ylabel('Beta coefficient value')
        title = 'LASSO path'
        if self.base_model_name:
            title += f' ({self.base_model_name})'
        ax.set_title(title)
        ax.invert_xaxis()
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)
        return fig

    def plot_beta_values(self):
        """Plot the absolute beta coefficient values, excluding features with |β| <= 1e-3.
        Features with near-zero coefficients (|β| <= 1e-3) are excluded from the plot
        to focus on features with meaningful contributions.

        This plot shows the magnitude of the beta coefficients in the LASSO model,
        indicating the strength of association between each feature and the target.
        """
        beta = self.get_selected_beta()
        # Filter out features with near-zero coefficients (|β| <= 1e-3)
        non_zero_features = [
            (name, abs_beta)
            for name, abs_beta in zip(self.all_feature_names_clean, np.abs(beta))
            if abs_beta > 1e-3
        ]
        if not non_zero_features:
            print("No features with |β| > 1e-3")
            return
        beta_values = non_zero_features
        beta_values.sort(key=lambda x: x[1], reverse=True)

        features, abs_betas = zip(*beta_values)

        plt.figure(figsize=(6, 6))
        plt.bar(features, abs_betas)
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Features')
        plt.ylabel('Absolute Beta coefficient value')
        title = 'Beta Coefficient Values'
        if self.base_model_name:
            title += f' ({self.base_model_name})'
        plt.title(title)
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)

    def plot_lasso_loss_path(self):
        plt.figure(figsize=(8, 4))
        plt.semilogx(self.lambdas, self.train_losses, label='Train loss')
        plt.semilogx(self.lambdas, self.test_losses, label='Test loss')
        plt.xlabel('Lambda')
        plt.ylabel('Log loss')
        title = 'LASSO path'
        if self.base_model_name:
            title += f' ({self.base_model_name})'
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show(block=False)
        plt.pause(0.1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'lambdas': self.lambdas,
            'betas': self.betas,
            'univariate_feature_names': self.univariate_feature_names,
            'bivariate_inputs': self.bivariate_inputs,
            'selected_lambda_index': self.selected_lambda_index,
            'base_model_name': self.base_model_name,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LassoResultsManager':
        manager = cls(
            data['lambdas'],
            data['betas'],
            [],  # models are not serialized
            data['univariate_feature_names'],
            data['bivariate_inputs'],
        )
        manager.selected_lambda_index = data['selected_lambda_index']
        manager.base_model_name = data.get('base_model_name', None)
        return manager

    def get_mask(
        self,
        threshold: float = 0.1,
        subnet_nodes: int = 5,
        bivariate_only_if_univariate: bool = False,
        include_bivariate_as_univariate: bool = False,
        onehot_groups: Optional[List[Tuple]] = None,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate a mask for selected features based on the selected beta.

        Parameters:
        -----------
        threshold : float, optional
            Threshold for considering a feature as selected (default is 0.1)
        subnet_nodes : int, optional
            Number of subnet nodes for each feature (default is 5)
        bivariate_only_if_univariate : bool, optional
            If True, include bivariate features only if both univariate features are selected (default is False)
        include_bivariate_as_univariate : bool, optional
            If True, include univariate features of selected bivariate features (default is False)
        onehot_groups : Optional[List[Tuple]], optional
            Groups of column indices representing one-hot encoded categorical features.
            Each tuple contains indices of columns that form a one-hot encoded group.
            If any feature from a group is selected, the entire group will be included.
        verbose : bool, optional
            If True, print selected feature names and show heatmap of the mask (default is True)

        Returns:
        --------
        Tuple[np.ndarray, int]
            Mask array for selected features and total number of selected features
        """
        beta = self.get_selected_beta()
        selected_indices = np.where(np.abs(beta) > threshold)[0]

        univ_selected = [idx for idx in selected_indices if idx < self.n_univ]

        # Handle one-hot groups
        if onehot_groups is not None:
            original_univ_selected = univ_selected.copy()
            groups_added = []

            for group_idx, group in enumerate(onehot_groups):
                # Check if any feature from this group is selected
                group_members_selected = [idx for idx in group if idx in univ_selected]

                if group_members_selected:
                    # Add all members of the group
                    for feature_idx in group:
                        if feature_idx not in univ_selected and feature_idx < self.n_univ:
                            univ_selected.append(feature_idx)

                    groups_added.append((group_idx, group, group_members_selected))

            # Log one-hot group additions
            if groups_added and verbose:
                logger.info("One-hot group processing:")
                for group_idx, group, originally_selected in groups_added:
                    group_names = [self.univariate_feature_names_clean[idx] for idx in group]
                    originally_selected_names = [
                        self.univariate_feature_names_clean[idx] for idx in originally_selected
                    ]
                    logger.info(
                        f"  Group {group_idx}: Originally selected {originally_selected_names}, added entire group {group_names}"
                    )

                added_count = len(univ_selected) - len(original_univ_selected)
                logger.info(
                    f"Added {added_count} additional features due to one-hot group constraints"
                )

        pr_names = []
        for idx in univ_selected:
            name = self.univariate_feature_names_clean[idx]
            pr_names.append(name)

        biv_selected_pairs = []
        for idx in selected_indices:
            if idx >= self.n_univ:
                first, second = self.bivariate_inputs[idx - self.n_univ]
                first_name = self.univariate_feature_names_clean[first]
                second_name = self.univariate_feature_names_clean[second]

                if bivariate_only_if_univariate:
                    if first in univ_selected and second in univ_selected:
                        biv_selected_pairs.append((first, second))
                        pr_names.append(f"{first_name} : {second_name}")
                elif include_bivariate_as_univariate:
                    univ_selected.extend(
                        feature for feature in [first, second] if feature not in univ_selected
                    )
                    biv_selected_pairs.append((first, second))
                    pr_names.append(f"{first_name} : {second_name}")
                else:
                    biv_selected_pairs.append((first, second))
                    pr_names.append(f"{first_name} : {second_name}")

        univ_selected = sorted(set(univ_selected))
        n_univ = len(univ_selected)
        n_biv = len(biv_selected_pairs)
        mask = np.zeros((self.n_univ, subnet_nodes * (n_univ + n_biv)))

        for i, idx in enumerate(univ_selected):
            mask[idx, i * subnet_nodes : (i + 1) * subnet_nodes] = 1

        biv_start = n_univ * subnet_nodes
        for i, (first, second) in enumerate(biv_selected_pairs):
            start_col = biv_start + i * subnet_nodes
            end_col = start_col + subnet_nodes
            mask[first, start_col:end_col] = 1
            mask[second, start_col:end_col] = 1

        if verbose:
            print("Selected features:", pr_names)
            # Dynamically scale figure height based on number of features
            fig_height = max(4, min(20, self.n_univ * 0.3))
            fig, ax = plt.subplots(figsize=(6, fig_height))
            heatmap = sns.heatmap(mask, ax=ax)
            heatmap.set_xlabel('subnet node index')
            heatmap.set_ylabel('input features')
            title = 'input mask'
            if self.base_model_name:
                title += f' ({self.base_model_name})'
            heatmap.set_title(title)
            # Explicitly set tick positions before labels to avoid mismatch
            ax.set_yticks(list(range(self.n_univ)))
            ax.set_yticklabels(self.univariate_feature_names_clean, rotation=0)
            plt.show(block=False)
        plt.pause(0.1)

        return mask, n_univ + n_biv

    def verify_prediction(
        self,
        phi: Union[np.ndarray, pd.DataFrame, torch.Tensor],
        y: Union[np.ndarray, pd.Series, torch.Tensor],
        sample_idx: Optional[int] = None,
        threshold: float = 0.1,
        match_tol: float = 1e-2,
        original_model_pred: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """
        Verify prediction calculation by comparing different approaches:
        1. Manual: Sum of weighted partial responses, for |β| > threshold
        logit(y) = β₀ + Σᵢ βᵢφᵢ(xᵢ) + Σᵢⱼ βᵢⱼφᵢⱼ(xᵢⱼ)
        2. LASSO: Direct prediction from the logistic regression model (all β)
        3. Original: Prediction from the original black box model (if provided)

        All comparisons are done in the logit space. For the original model predictions,
        which are probabilities from a sigmoid output layer, we convert back to logits
        using the inverse sigmoid (logit) function before comparison.

        Parameters
        ----------
        phi : Union[np.ndarray, pd.DataFrame, torch.Tensor]
            Partial responses φ(x). Shape should be (n_samples, n_features)
        y : Union[np.ndarray, pd.Series, torch.Tensor]
            Target values (0 or 1)
        sample_idx : Optional[int]
            Index of specific sample to analyze. If None, a random sample will be chosen.
        threshold : float
            Threshold for considering a feature as selected (default is 0.1)
        match_tol : float
            Tolerance for considering logit values as matching (default is 1e-3)
        original_model_pred : Optional[Union[np.ndarray, torch.Tensor]]
            Predictions from the original black box model (probabilities from sigmoid output)

        Returns
        -------
        Dict[str, Any]
            Dictionary containing the calculation details and verification results
        """

        if self.selected_lambda_index is None:
            raise ValueError("No lambda selected. Call select_lambda() first.")

        # Convert inputs to numpy arrays
        if isinstance(phi, torch.Tensor):
            phi = phi.cpu().numpy()
        elif isinstance(phi, pd.DataFrame):
            phi = phi.values

        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        elif isinstance(y, pd.Series):
            y = y.values

        if original_model_pred is not None:
            if isinstance(original_model_pred, torch.Tensor):
                original_model_pred = original_model_pred.cpu().numpy()

        # Get selected model and its components
        model = self.get_selected_model()
        beta = self.get_selected_beta()

        # Choose random sample if none specified
        if sample_idx is None:
            sample_idx = np.random.randint(0, len(phi))

        # Extract single sample
        phi_sample = phi[sample_idx]
        y_true = y[sample_idx]

        # Calculate all types of predictions

        # 1. Manual calculation of weighted sum (already in logit space)
        intercept = model.intercept_[0]
        contributions = []
        manual_sum = intercept

        # Process univariate terms
        n_univ = len(self.univariate_feature_names)
        for i in range(n_univ):
            if abs(beta[i]) > threshold:
                contrib = beta[i] * phi_sample[i]
                manual_sum += contrib
                contributions.append(
                    {
                        'type': 'univariate',
                        'feature': self.univariate_feature_names_clean[i],
                        'beta': beta[i],
                        'phi': phi_sample[i],
                        'contribution': contrib,
                    }
                )

        # Process bivariate terms
        for idx, (i, j) in enumerate(self.bivariate_inputs):
            beta_idx = n_univ + idx
            if abs(beta[beta_idx]) > threshold:
                contrib = beta[beta_idx] * phi_sample[beta_idx]
                manual_sum += contrib
                contributions.append(
                    {
                        'type': 'bivariate',
                        'feature': f"{self.univariate_feature_names_clean[i]} : {self.univariate_feature_names_clean[j]}",
                        'beta': beta[beta_idx],
                        'phi': phi_sample[beta_idx],
                        'contribution': contrib,
                    }
                )

        # Sort contributions by absolute magnitude
        contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)

        # 2. LASSO model direct prediction (convert probability to logit)
        lasso_prob = model.predict_proba(phi_sample.reshape(1, -1))[0, 1]
        # Clip probabilities to avoid log(0) or log(inf)
        lasso_prob = np.clip(lasso_prob, 1e-15, 1 - 1e-15)
        lasso_logit = np.log(lasso_prob / (1 - lasso_prob))

        # 3. Original model prediction (if provided)
        # Convert sigmoid output (probability) back to logit
        orig_logit = None
        orig_prob = None
        if original_model_pred is not None:
            orig_prob = original_model_pred[sample_idx]
            # Clip probabilities to avoid log(0) or log(inf)
            orig_prob = np.clip(orig_prob, 1e-15, 1 - 1e-15)
            orig_logit = np.log(orig_prob / (1 - orig_prob))

        # Calculate probability from manual sum using sigmoid
        manual_prob = 1 / (1 + np.exp(-manual_sum))

        # Create a console for rich output
        console = Console()

        # Create header
        header_text = Text("LASSO Logistic Regression Calculation Verification", style="bold blue")
        console.print(header_text)

        # Create info panel
        info_panel = Panel(
            f"Sample Index: {sample_idx}\nTrue y: {y_true}",
            title="Sample Information",
            border_style="cyan",
        )
        console.print(info_panel)

        # Create contributions table
        contrib_table = Table(
            box=box.ROUNDED,
            title=f"Partial Response Contributions (beta * phi), for |beta| > {threshold}",
            title_style="bold green",
            header_style="bold",
        )

        # Add columns
        contrib_table.add_column("Feature", style="cyan")
        contrib_table.add_column("Beta (beta)", justify="right", style="magenta")
        contrib_table.add_column("phi(x)", justify="right", style="yellow")
        contrib_table.add_column("beta*phi", justify="right", style="green")

        # Add rows for each contribution
        for contrib in contributions:
            contrib_table.add_row(
                contrib['feature'],
                f"{contrib['beta']:.4f}",
                f"{contrib['phi']:.4f}",
                f"{contrib['contribution']:.4f}",
            )

        # Add intercept and total
        contrib_table.add_row(
            "Intercept (beta_0)", f"{intercept:.4f}", "", f"{intercept:.4f}", style="bold"
        )
        contrib_table.add_row("Total", "", "", f"{manual_sum:.4f}", style="bold on grey85")

        console.print(contrib_table)

        # Create verification table
        verify_table = Table(
            box=box.ROUNDED,
            title="Verification Results (Logit Scale)",
            title_style="bold blue",
            header_style="bold",
        )

        # Add columns
        verify_table.add_column("Method", style="cyan")
        verify_table.add_column("Logit", justify="right", style="magenta")
        verify_table.add_column("Probability", justify="right", style="yellow")
        verify_table.add_column("vs Manual", justify="right", style="green")
        verify_table.add_column("Match?", justify="center")

        # Convert all values to Python floats for consistent printing
        manual_sum_float = float(manual_sum)
        manual_prob_float = float(manual_prob)
        lasso_logit_float = float(lasso_logit)
        lasso_prob_float = float(lasso_prob)

        # Add manual calculation row
        verify_table.add_row(
            "Manual Sum",
            f"{manual_sum_float:.4f}",
            f"{manual_prob_float:.4f}",
            "-",
            "[bold]ref[/bold]",
        )

        # Add LASSO model row
        lasso_logit_diff = float(abs(lasso_logit_float - manual_sum_float))
        lasso_match = (
            "[bold green]✓[/bold green]"
            if lasso_logit_diff < match_tol
            else "[bold red]✗[/bold red]"
        )
        verify_table.add_row(
            "LASSO Model",
            f"{lasso_logit_float:.4f}",
            f"{lasso_prob_float:.4f}",
            f"{lasso_logit_diff:.2e}",
            lasso_match,
        )

        # Add original model row (if provided)
        if original_model_pred is not None:
            orig_prob_float = float(orig_prob)
            orig_logit_float = float(orig_logit)
            orig_diff = float(abs(orig_logit_float - manual_sum_float))
            orig_match = (
                "[bold green]✓[/bold green]" if orig_diff < match_tol else "[bold red]✗[/bold red]"
            )
            verify_table.add_row(
                "Original Model",
                f"{orig_logit_float:.4f}",
                f"{orig_prob_float:.4f}",
                f"{orig_diff:.2e}",
                orig_match,
            )

        console.print(verify_table)

        # Print warnings for mismatches
        if lasso_logit_diff >= match_tol:
            console.print(
                f"\n[bold red]Warning: LASSO model logit mismatch exceeds tolerance of {match_tol:.1e}![/bold red]"
            )

        # Prepare verification results dictionary
        results = {
            'sample_idx': sample_idx,
            'y_true': y_true,
            'calculations': {
                'manual': {
                    'logit': manual_sum,
                    'probability': manual_prob,
                    'contributions': contributions,
                },
                'lasso': {
                    'logit': lasso_logit,
                    'probability': lasso_prob,
                    'difference': lasso_logit_diff,
                },
            },
            'verification': {'lasso_match': lasso_logit_diff < match_tol},
        }

        if original_model_pred is not None:
            results['calculations']['original'] = {
                'logit': orig_logit,
                'probability': orig_prob,
                'difference': orig_diff,
            }
            results['verification']['original_match'] = orig_diff < match_tol

        return results
