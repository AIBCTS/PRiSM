"""
Response plot utilities.

This module provides functions for visualizing partial responses from models.
The main entry point is `plot_partial_responses()` which uses the new
NomogramRenderer architecture.

"""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

from prism.lasso import LassoResultsManager
from prism.partial_responses import to_numpy
from prism.plotting import NomogramRenderer, PlotFormatter, PlottingPipeline
from prism.preprocessing import PRiSMScaler, build_ordinal_labels_dict

if TYPE_CHECKING:
    from prism.feature_labels import FeatureLabelManager
    from prism.preprocessing import OneHotGroupManager

logger = logging.getLogger(__name__)


def plot_partial_responses(
    lasso_results: LassoResultsManager,
    x: torch.Tensor,
    model: Any,
    scaler: Optional[PRiSMScaler] = None,
    n_steps: int = 50,
    method: str = "dirac",
    x_train: torch.Tensor = None,
    device: str = "cpu",
    categorical_threshold: int = 15,
    subtract_univariate: bool = True,
    subfig_size: float = 3.5,
    show_fig: bool = True,
    return_fig: bool = False,
    use_odds_ratio: bool = False,
    binary_labels: Optional[Dict[int, str]] = None,
    categorical_labels: Optional[Dict[str, Dict[float, str]]] = None,
    onehot_group_manager: Optional['OneHotGroupManager'] = None,
    label_manager: Optional['FeatureLabelManager'] = None,
    feature_names: Optional[List[str]] = None,
    trim_quantile: Optional[float] = None,
) -> Union[None, plt.Figure, Tuple[plt.Figure, Optional[plt.Figure]]]:
    """Generate a grid of subplots showing partial responses for the selected lambda.

    Calculates partial responses on synthetic grids by querying the model directly.
    Automatically handles collapsed one-hot groups when onehot_group_manager is provided.

    Note: Response plots show RAW partial responses (no beta scaling), unlike nomograms
    which scale responses by LASSO coefficients.

    Parameters
    ----------
    lasso_results : LassoResultsManager
        Manager containing feature selection results (in collapsed space if collapse enabled)
    x : torch.Tensor
        Input data tensor in one-hot encoded space (model input space)
    model : Any
        Model to query for generating responses (trained on one-hot features)
    scaler : Any, optional
        Original PRiSMScaler fit on one-hot features (auto-handled for collapse)
    n_steps : int, default=50
        Number of discretization steps for continuous features
    method : str, default="dirac"
        Method for partial response calculation ("dirac" or "lebesgue")
    x_train : torch.Tensor, optional
        Training data for reference (used by Lebesgue method)
    device : str, default="cpu"
        Device for tensor operations
    categorical_threshold : int, default=15
        Threshold for categorical feature detection
    subtract_univariate : bool, default=True
        Whether to subtract univariate effects from bivariate responses. When True (default),
        bivariate plots show only the interaction effect by subtracting out the univariate
        contributions. This isolates the true bivariate interaction, making the plots clearer
        and easier to interpret. Set to False to show the full bivariate response including
        univariate effects
    subfig_size : float, default=3.5
        Size of individual subplots
    show_fig : bool, default=True
        Whether to display the figure
    return_fig : bool, default=False
        Whether to return the figure object(s)
    use_odds_ratio : bool, default=False
        Whether to display odds ratios instead of log odds
    binary_labels : Dict[int, str], optional
        Labels for binary features
    categorical_labels : Dict[str, Dict[float, str]], optional
        Labels for categorical features
    onehot_group_manager : Optional[OneHotGroupManager], optional
        Manager for one-hot encoded groups (enables automatic collapse handling)
        When provided, automatically handles all collapse/expand operations
    label_manager : Optional[FeatureLabelManager], optional
        Manager for user-friendly feature labels
    feature_names : Optional[list], optional
        Original feature column names in the one-hot encoded space. Required when
        LASSO was initialized with labels instead of column names and x doesn't
        have a .columns attribute (e.g., for tensor inputs).
    trim_quantile : Optional[float], optional
        Fraction to trim from each tail when generating grids for continuous features.
        E.g., 0.01 uses the 1st to 99th percentile range. This limits plot axis ranges
        to exclude outliers while histograms may extend beyond the axis limits.

    Returns
    -------
    Union[None, plt.Figure, Tuple[plt.Figure, Optional[plt.Figure]]]
        - None if return_fig is False
        - (fig_responses, fig_heatmaps) tuple if return_fig is True
          - fig_responses: Main response plot grid (univariate + mixed bivariate)
          - fig_heatmaps: Non-mixed bivariate heatmaps (cat×cat, cont×cont), or None if none exist

    Notes
    -----
    The function automatically detects collapse mode when onehot_group_manager is provided.
    All index mapping, response collapsing, and scaler selection is handled internally.
    Users only need to pass the one-hot encoded data (x, model) and the group manager.

    Unlike nomogram(), this function does NOT apply beta scaling to responses. This shows
    the raw partial response values from the model, useful for understanding the model's
    behavior before LASSO regularization.
    """
    # Get feature names for pipeline
    if feature_names is not None:
        original_feature_names = feature_names
    elif hasattr(x, 'columns'):
        original_feature_names = list(x.columns)
    else:
        original_feature_names = None

    # Create PlottingPipeline (new architecture)
    pipeline = PlottingPipeline(
        lasso_results=lasso_results,
        group_manager=onehot_group_manager,
        label_manager=label_manager,
    )

    # Prepare plotting bundle (NO beta scaling for response plots)
    bundle = pipeline.prepare_plotting_bundle(
        x=x,
        model=model,
        scaler=scaler,
        n_steps=n_steps,
        method=method,
        x_train=x_train,
        device=device,
        categorical_threshold=categorical_threshold,
        subtract_univariate=subtract_univariate,
        feature_names=original_feature_names,
        trim_quantile=trim_quantile,
    )

    # Build categorical labels for proper category display
    # Start with one-hot group labels (if group_manager exists)
    combined_categorical_labels = (
        onehot_group_manager.build_categorical_labels_dict(label_manager)
        if onehot_group_manager
        else {}
    )

    # Add ordinal feature labels from metadata if available
    if hasattr(bundle, 'preprocessing_metadata') and bundle.preprocessing_metadata:
        ordinal_labels = build_ordinal_labels_dict(bundle.preprocessing_metadata)
        combined_categorical_labels.update(ordinal_labels)

    # Override with user-provided labels if given
    if categorical_labels:
        combined_categorical_labels.update(categorical_labels)

    # Create formatter and renderer
    formatter = PlotFormatter(
        use_odds_ratio=use_odds_ratio,
        binary_labels=binary_labels,
        categorical_labels=combined_categorical_labels,
    )
    renderer = NomogramRenderer(bundle, formatter, use_odds_ratio=use_odds_ratio)

    # Generate response plots using NomogramRenderer (univariate + mixed bivariate)
    fig_responses = renderer.render_response_plots(subfig_size=subfig_size)

    # Add title to response plots
    if isinstance(fig_responses, list):
        for f in fig_responses:
            f.suptitle(f"Partial responses for Selected Features ({method.title()})", y=1.01)
    else:
        fig_responses.suptitle(
            f"Partial responses for Selected Features ({method.title()})", y=1.01
        )

    # Generate non-mixed bivariate heatmaps (cat×cat, cont×cont) WITHOUT beta scaling
    fig_heatmaps = renderer.render_bivariate_heatmaps()

    # Show figures if requested
    if show_fig:
        # Show response plots
        if isinstance(fig_responses, list):
            for f in reversed(fig_responses):
                plt.figure(f.number)
                plt.show(block=False)
                plt.pause(0.1)
        else:
            plt.figure(fig_responses.number)
            plt.show(block=False)
            plt.pause(0.1)

        # Show heatmaps if they exist
        if fig_heatmaps is not None:
            if isinstance(fig_heatmaps, list):
                for f in reversed(fig_heatmaps):
                    plt.figure(f.number)
                    plt.show(block=False)
                    plt.pause(0.1)
            else:
                plt.figure(fig_heatmaps.number)
                plt.show(block=False)
                plt.pause(0.1)

    logger.info("Response plots generated using NomogramRenderer architecture (no beta scaling).")

    if return_fig:
        return (fig_responses, fig_heatmaps)
    else:
        # Close all figures
        if isinstance(fig_responses, list):
            for f in fig_responses:
                plt.close(f)
        else:
            plt.close(fig_responses)

        if fig_heatmaps is not None:
            if isinstance(fig_heatmaps, list):
                for f in fig_heatmaps:
                    plt.close(f)
            else:
                plt.close(fig_heatmaps)

        return None


def plot_raw_partial_responses(
    partial_responses,
    x,
    scaler=None,
    x0_median=None,
    x0_std=None,
    feature_names=None,
    n_steps: int = 15,
    categorical_threshold: int = 15,
    subfig_size: float = 3.5,
    show_fig=True,
    return_fig=True,
    use_odds_ratio=False,
):
    """
    Plot partial responses from raw partial response array.

    Parameters
    ----------
    partial_responses : np.ndarray
        Array of partial responses where first n_features columns are univariate
    x : array-like
        Input data used to determine categorical vs continuous features
    scaler : Any, optional
        PRiSMScaler or compatible scaler object for denormalization.
        If provided, takes precedence over x0_median/x0_std parameters.
    feature_names : List[str], optional
        Names of features. If None, will use Feature 1, Feature 2, etc.
    n_steps : int
        Number of steps used in discretization
    categorical_threshold : int
        Threshold for determining categorical features
    subfig_size : float
        Size of individual subplots
    show_fig : bool
        Whether to display the figure
    return_fig : bool
        Whether to return the figure object
    use_odds_ratio : bool
        Whether to convert log odds to odds ratios    Returns
    -------
    Optional[plt.Figure]
        The figure object if return_fig is True
    """
    # Handle scaler parameter - use scaler if provided, otherwise create from legacy parameters
    if scaler is not None:
        if not hasattr(scaler, 'inverse_transform'):
            raise ValueError("Scaler must have an 'inverse_transform' method")
        nomogram_scaler = scaler
    else:
        # Do not scale, use dummy scaler
        nomogram_scaler = PRiSMScaler(scaler=None)

    # Convert x to numpy if it's not already
    x = to_numpy(x)

    n_features = x.shape[1]

    # Extract univariate responses (first n_features columns)
    univariate_responses = partial_responses[:, :n_features]

    # If no feature names provided, create generic ones
    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(n_features)]
    else:
        # Replace newlines in provided feature names
        feature_names = [name.replace('\n', ' ') for name in feature_names]

    # Create subplot grid
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(n_cols * subfig_size, n_rows * subfig_size * 0.8))

    # For each feature
    for feature_idx in range(n_features):
        ax = plt.subplot(n_rows, n_cols, feature_idx + 1)
        feature_name = feature_names[feature_idx]

        # Get response values for this feature
        response = univariate_responses[:, feature_idx]

        # Get x values based on if it's categorical
        is_categorical = len(np.unique(x[:, feature_idx])) < categorical_threshold

        # Convert to odds ratio if requested
        if use_odds_ratio:
            response = np.exp(response)

        if is_categorical:
            # Get unique x values
            x_unique = np.unique(x[:, feature_idx])

            # Create dummy array with zeros except for the target feature for denormalization
            dummy_array = np.zeros((len(x_unique), n_features))
            for i, val in enumerate(x_unique):
                dummy_array[i, feature_idx] = val

            # Denormalize using scaler
            denormalized = nomogram_scaler.inverse_transform(dummy_array)
            x_denorm = denormalized[:, feature_idx]

            # For categorical, we'll use unique responses corresponding to unique x values
            unique_responses = np.array(
                [response[np.where(x[:, feature_idx] == val)[0][0]] for val in x_unique]
            )

            y_value = 0.5
            # Create scatter plot
            ax.scatter(
                unique_responses, np.full_like(unique_responses, y_value), marker="|", color='blue'
            )
            ax.plot(unique_responses, np.full_like(unique_responses, y_value), color='blue')

            # Add value annotations
            for i, value in enumerate(x_denorm):
                ax.annotate(
                    f"{value:.2g}",
                    (unique_responses[i], y_value),
                    xytext=(unique_responses[i], y_value + 0.003),
                    ha='center',
                    va='bottom',
                )

            ax.set_yticks([])
            ax.set_ylabel(feature_name)
            ax.set_xlabel('Odds ratio' if use_odds_ratio else 'Log odds ratio')

        else:
            # For continuous features, sort x and response by x values
            sort_idx = np.argsort(x[:, feature_idx])
            x_values = x[sort_idx, feature_idx]
            response_sorted = response[sort_idx]

            # Create dummy array with zeros except for the target feature for denormalization
            dummy_array = np.zeros((len(x_values), n_features))
            dummy_array[:, feature_idx] = x_values

            # Denormalize using scaler
            denormalized = nomogram_scaler.inverse_transform(dummy_array)
            x_denorm = denormalized[:, feature_idx]

            # Plot the main line
            ax.plot(x_denorm, response_sorted)

            # Add histogram of original data on twin axis
            hist_ax = ax.twinx()

            # Create dummy array for original data denormalization
            dummy_array_orig = np.zeros((len(x), n_features))
            dummy_array_orig[:, feature_idx] = x[:, feature_idx]

            # Denormalize original data
            denormalized_orig = nomogram_scaler.inverse_transform(dummy_array_orig)
            original_data = denormalized_orig[:, feature_idx]
            hist_ax.hist(original_data, bins=n_steps, alpha=0.3, color='gray')
            hist_ax.set_ylabel('Count', color='gray')
            hist_ax.tick_params(axis='y', labelcolor='gray')

            # Set x-ticks
            x_ticks = np.linspace(x_denorm.min(), x_denorm.max(), num=5)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels([f"{val:.2f}" for val in x_ticks])

            # Adjust y-axis for density
            ylim = hist_ax.get_ylim()
            hist_ax.set_ylim(0, ylim[1] * 1.2)

            ax.set_xlabel(feature_name)
            ax.set_ylabel('Odds ratio' if use_odds_ratio else 'Log odds ratio')

        # Add reference line
        if is_categorical:
            if use_odds_ratio:
                ax.axvline(x=1, color="black", alpha=0.3)
            else:
                ax.axvline(x=0, color="black", alpha=0.3)
        else:
            if use_odds_ratio:
                ax.axhline(y=1, color="black", alpha=0.3)
            else:
                ax.axhline(y=0, color="black", alpha=0.3)

    plt.tight_layout()

    if show_fig:
        plt.show(block=False)
        plt.pause(0.1)

    if return_fig:
        return fig
    else:
        plt.close(fig)
        return None
