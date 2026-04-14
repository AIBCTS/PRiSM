"""
Nomogram plotting.

This module provides the main nomogram() function for generating interpretable
visualizations of model responses. Uses the new NomogramRenderer architecture.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

from prism.lasso import LassoResultsManager
from prism.plotting import (
    NomogramRenderer,
    PlotFormatter,
    PlottingPipeline,
    load_nomogram_json,
    save_nomogram_csv,
    save_nomogram_json,
)
from prism.plotting.json_export import NomogramData
from prism.plotting_data import FeatureInfo, FeaturePairInfo, PlottingDataBundle
from prism.preprocessing import build_ordinal_labels_dict

if TYPE_CHECKING:
    from prism.feature_labels import FeatureLabelManager
    from prism.plotting.metadata import FeatureMetadata
    from prism.preprocessing import OneHotGroupManager, PRiSMScaler

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class NomogramResult:
    """Result container for nomogram() function.

    Attributes:
        fig_main: Main nomogram figure(s) with univariate and mixed bivariate features.
            Can be a single Figure or list of Figures if paginated.
        fig_bivariate: Non-mixed bivariate heatmap figure(s) (categorical×categorical,
            continuous×continuous). Can be a single Figure, list of Figures, or None
            if no non-mixed bivariate features exist.
        univariate_responses: List of response arrays for individual features.
        bivariate_responses: List of response arrays for feature interactions.
        x_univariate: List of x-value arrays for univariate responses.
        x_bivariate: List of x-value arrays for bivariate responses.
        selected_univariate_indices: List of indices for selected univariate features.
        selected_bivariate_pairs: List of (feature_i, feature_j) index tuples for
            selected bivariate interactions.
    """

    fig_main: Union[plt.Figure, List[plt.Figure], None]
    fig_bivariate: Union[plt.Figure, List[plt.Figure], None]
    univariate_responses: List[np.ndarray]
    bivariate_responses: List[np.ndarray]
    x_univariate: List[np.ndarray]
    x_bivariate: List[np.ndarray]
    selected_univariate_indices: List[int]
    selected_bivariate_pairs: List[Tuple[int, int]]


def nomogram(
    lasso_results: LassoResultsManager,
    x: torch.Tensor,
    *,
    # Data handling
    scaler: Optional['PRiSMScaler'] = None,
    x_train: torch.Tensor | None = None,
    feature_names: list[str] | None = None,
    onehot_group_manager: Optional['OneHotGroupManager'] = None,
    label_manager: Optional['FeatureLabelManager'] = None,
    # Computation settings
    model: Any,
    device: str = "cpu",
    method: str = "dirac",
    n_steps: int = 50,
    categorical_threshold: int = 15,
    trim_quantile: float | None = None,
    subtract_univariate: bool = True,
    # Display settings
    use_odds_ratio: bool = False,
    show_fig: bool = False,
    return_fig: bool = True,
    show_conversion_line: bool = False,
    two_column: bool = True,
    features_per_plot: int | None = None,
    legend_on_right: bool = False,
    surround_axes: bool = True,
    binary_labels: dict[int, str] | None = None,
    categorical_labels: dict[str, dict[float, str]] | None = None,
    # Export settings
    save_csv: bool = False,
    save_json: bool = False,
    csv_path: str | None = None,
    json_path: str | None = None,
    comment: str | None = None,
) -> NomogramResult:
    """Generate and display nomogram visualizations for interpreting model responses.

    Calculates partial responses on synthetic grids by querying the model directly.
    Automatically handles collapsed one-hot groups when onehot_group_manager is provided.

    Args:
        lasso_results: Manager containing feature selection results and metadata (in collapsed space if collapse enabled)
        x: Input data tensor in one-hot encoded space (model input space)

        # Data handling
        scaler: Original PRiSMScaler fit on one-hot features (auto-handled for collapse)
        x_train: Optional training data for certain response calculation methods
        feature_names: Optional list of original feature column names (one-hot encoded space).
            Required when LASSO was initialized with labels instead of column names
            and x doesn't have a .columns attribute (e.g., for tensor inputs).
        onehot_group_manager: Optional OneHotGroupManager for automatic collapse handling
            When provided, automatically handles all collapse/expand operations
        label_manager: Optional FeatureLabelManager for user-friendly feature labels

        # Computation settings
        model: Model to query for generating responses (trained on one-hot features)
        device: Computation device ("cpu", "cuda", or "mps", default: "cpu")
        method: Method for response calculation ("dirac" or "lebesgue")
        n_steps: Number of discretization steps for continuous features (default: 50)
        categorical_threshold: Maximum unique values for categorical treatment (default: 15)
        trim_quantile: Optional fraction to trim from each tail when generating grids
            for continuous features. E.g., 0.01 uses the 1st to 99th percentile range.
            This limits plot axis ranges to exclude outliers while histograms may extend
            beyond the axis limits.
        subtract_univariate: Whether to subtract univariate effects from bivariate responses (default: True).
            When True (default), bivariate plots show only the interaction effect by subtracting out
            the univariate contributions. This isolates the true bivariate interaction, making the
            plots clearer and easier to interpret. Set to False to show the full bivariate response
            including univariate effects

        # Display settings
        use_odds_ratio: Display odds ratios instead of log odds (default: False)
        show_fig: Whether to display generated figures when running outside of iPython (default: False)
        return_fig: Whether to return figure objects (default: True)
        show_conversion_line: If True, add a conversion line below the nomogram showing
            how to convert the sum of contributions to probability. The line includes
            markers with log odds (or total odds in odds ratio mode) above and
            corresponding probabilities below. Range is limited to practical probability
            values (0.01 to 0.99). Default: False.
        two_column: Enable two-column layout (default: True):
            - Features arranged in two columns per figure
            - Left column filled before right column
            - Even distribution if features_per_plot=None
            - Consistent axis scaling between columns
        features_per_plot: Controls feature distribution per page/column:
            - In single-column mode (two_column=False):
              Number of features per figure
            - In two-column mode (two_column=True):
              Number of features per column
            - If None:
              * Single-column: All features in one figure
              * Two-column: Features evenly split between columns
        legend_on_right: Place plot legends on right side (default: False)
        surround_axes: Show axes on both sides of plots (default: True)
        binary_labels: Optional Dict mapping of binary feature names to custom labels
        categorical_labels: Optional Dict mapping of categorical features names to value labels

        # Export settings
        save_csv: Save response data to CSV files (default: False)
        save_json: Save full nomogram data to JSON file (default: False).
            Note: Even if trim_quantile is set for plotting, the JSON export will
            contain the full range of data (untrimmed).
        csv_path: Optional custom path for saving CSV files.
        json_path: Optional custom path for saving JSON file.
        comment: Optional metadata comment for CSV and JSON output.
            values (0.01 to 0.99). Default: False.

    Returns:
        NomogramResult: A dataclass containing:
            - fig_main: Main nomogram figure(s) with univariate and mixed bivariate features
            - fig_bivariate: Non-mixed bivariate heatmap figure(s) or None
            - univariate_responses: List of response arrays for individual features
            - bivariate_responses: List of response arrays for feature interactions
            - x_univariate: List of x-value arrays for univariate responses
            - x_bivariate: List of x-value arrays for bivariate responses
            - selected_univariate_indices: List of indices for selected univariate features
            - selected_bivariate_pairs: List of index pairs for selected interactions

    Notes:
        - Generates visualizations for both univariate feature effects and bivariate
          interactions identified through LASSO feature selection
        - Supports flexible display options including single/two-column layouts,
          pagination, and consistent axis scaling
        - Can save underlying response data to CSV files for further analysis
        - Returns both figure objects and raw response data for additional processing
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

    # Prepare plotting bundle (without beta scaling)
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

    # Always apply beta scaling for nomogram
    bundle = pipeline.apply_beta_scaling(bundle)

    # Extract data from bundle for return values (backward compatibility)
    univariate_responses = [info.response for info in bundle.univariate_features()]
    bivariate_responses = [info.response for info in bundle.bivariate_pairs()]
    x_univariate = [info.x_values for info in bundle.univariate_features()]
    x_bivariate = [info.x_values for info in bundle.bivariate_pairs()]
    selected_univariate_indices = bundle.selected_univariate_indices
    selected_bivariate_pairs = [info.indices for info in bundle.bivariate_pairs()]

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

    # Extract intercept for conversion line if requested
    intercept = None
    if show_conversion_line:
        selected_model = lasso_results.get_selected_model()
        intercept = selected_model.intercept_[0]

    # Generate main nomogram (univariate + mixed bivariate)
    fig_main = renderer.render_nomogram(
        legend_on_right=legend_on_right,
        surround_axes=surround_axes,
        features_per_plot=features_per_plot,
        two_column=two_column,
        intercept=intercept,
        show_conversion_line=show_conversion_line,
    )

    # Show figures - handle both single figure and list of figures
    if show_fig and fig_main:
        if isinstance(fig_main, list):
            for fig in reversed(fig_main):
                plt.figure(fig.number)
                plt.show(block=False)
                plt.pause(0.1)
        else:
            plt.figure(fig_main.number)
            plt.show(block=False)
            plt.pause(0.1)

    # Generate separate plot for non-mixed bivariate responses (heatmaps/contours)
    fig_non_mixed = renderer.render_bivariate_heatmaps(features_per_plot=features_per_plot)

    if show_fig and fig_non_mixed:
        if isinstance(fig_non_mixed, list):
            for fig in reversed(fig_non_mixed):
                plt.figure(fig.number)
                plt.show(block=False)
                plt.pause(0.1)
        else:
            plt.figure(fig_non_mixed.number)
            plt.show(block=False)
            plt.pause(0.1)

    # Save data if requested (uses new PlottingDataBundle architecture)
    if save_csv:
        save_nomogram_csv(
            bundle=bundle,
            lasso_results=lasso_results,
            model_info={'comment': comment, 'method': method},
            file_path=csv_path,
            use_odds_ratio=use_odds_ratio,
        )

    # Save JSON data if requested
    if save_json:
        # Determine which bundle to use for JSON export
        # If trim_quantile was used for plotting, we need a new untrimmed bundle for JSON
        json_bundle = bundle
        if trim_quantile is not None:
            logger.info("Generating untrimmed bundle for JSON export...")
            # Create a new bundle with trim_quantile=None
            json_bundle = pipeline.prepare_plotting_bundle(
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
                trim_quantile=None,  # Force None for full range
            )

            # Always apply beta scaling for JSON export to match nomogram
            json_bundle = pipeline.apply_beta_scaling(json_bundle)

        save_nomogram_json(
            bundle=json_bundle,
            lasso_results=lasso_results,
            file_path=json_path,
            comment=comment,
            method=method,
            category_labels=combined_categorical_labels,
        )

    logger.info("Nomogram generation complete using NomogramRenderer architecture.")

    # Return results in NomogramResult dataclass
    if return_fig:
        return NomogramResult(
            fig_main=fig_main,
            fig_bivariate=fig_non_mixed,
            univariate_responses=univariate_responses,
            bivariate_responses=bivariate_responses,
            x_univariate=x_univariate,
            x_bivariate=x_bivariate,
            selected_univariate_indices=selected_univariate_indices,
            selected_bivariate_pairs=selected_bivariate_pairs,
        )
    else:
        return NomogramResult(
            fig_main=None,
            fig_bivariate=None,
            univariate_responses=univariate_responses,
            bivariate_responses=bivariate_responses,
            x_univariate=x_univariate,
            x_bivariate=x_bivariate,
            selected_univariate_indices=selected_univariate_indices,
            selected_bivariate_pairs=selected_bivariate_pairs,
        )


def display_nomograms_side_by_side(
    nomograms: List[Tuple[Optional[plt.Figure], Optional[plt.Figure]]],
    titles: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 100,
) -> plt.Figure:
    """Display multiple nomogram figure pairs side by side for comparison.

    Creates a composite figure showing nomograms from different models (e.g., blackbox
    vs PRN) arranged horizontally for easy visual comparison.

    Args:
        nomograms: List of (fig_main, fig_non_mixed) tuples from nomogram() calls.
            Each tuple contains the main nomogram figure and optional non-mixed
            bivariate heatmap figure. Example:
            [
                (nomogram_main_blackbox, nomogram_non_mixed_blackbox),
                (nomogram_main_prn, nomogram_non_mixed_prn),
            ]
        titles: Optional list of titles for each nomogram pair. If None,
            uses "Model 1", "Model 2", etc.
        figsize: Optional tuple (width, height) in inches. If None, auto-calculated
            based on number of nomograms.
        dpi: Resolution for the composite figure (default: 100).

    Returns:
        Composite matplotlib Figure with all nomograms arranged side by side.

    Example:
        >>> # Generate nomograms for two models
        >>> result_bb = nomogram(lasso_bb, x, model=model_bb)
        >>> result_prn = nomogram(lasso_prn, x, model=model_prn)
        >>>
        >>> # Display side by side
        >>> fig = display_nomograms_side_by_side(
        ...     nomograms=[(result_bb.fig_main, result_bb.fig_bivariate),
        ...                (result_prn.fig_main, result_prn.fig_bivariate)],
        ...     titles=["Blackbox Nomogram", "PRN Nomogram"]
        ... )
        >>> plt.show()
    """
    from io import BytesIO

    import matplotlib.image as mpimg

    n_models = len(nomograms)
    if n_models == 0:
        raise ValueError("At least one nomogram pair must be provided")

    # Default titles
    if titles is None:
        titles = [f"Model {i+1}" for i in range(n_models)]
    elif len(titles) != n_models:
        raise ValueError(
            f"Number of titles ({len(titles)}) must match number of nomograms ({n_models})"
        )

    # Collect all figures (main nomograms only for now - heatmaps shown separately)
    main_figs = []
    heatmap_figs = []
    for fig_main, fig_heatmap in nomograms:
        if fig_main is not None:
            # Handle paginated figures (list of figures)
            if isinstance(fig_main, list):
                main_figs.append(fig_main[0])  # Use first page for comparison
            else:
                main_figs.append(fig_main)
        if fig_heatmap is not None:
            if isinstance(fig_heatmap, list):
                heatmap_figs.append(fig_heatmap[0])
            else:
                heatmap_figs.append(fig_heatmap)

    if not main_figs:
        raise ValueError("No valid main nomogram figures found")

    # Convert figures to images for embedding
    def fig_to_image(fig):
        """Convert matplotlib figure to image array."""
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
        buf.seek(0)
        img = mpimg.imread(buf)
        buf.close()
        return img

    # Convert all main figures to images
    images = [fig_to_image(fig) for fig in main_figs]

    # Calculate composite figure size
    max_height = max(img.shape[0] for img in images)
    total_width = sum(img.shape[1] for img in images)

    if figsize is None:
        # Scale to reasonable size (assume ~100 dpi for display)
        figsize = (total_width / dpi, max_height / dpi)

    # Create composite figure
    fig, axes = plt.subplots(1, n_models, figsize=figsize)
    if n_models == 1:
        axes = [axes]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()

    # Show heatmaps in a separate figure if present
    if heatmap_figs:
        heatmap_images = [fig_to_image(fig) for fig in heatmap_figs]
        max_hm_height = max(img.shape[0] for img in heatmap_images)
        total_hm_width = sum(img.shape[1] for img in heatmap_images)
        hm_figsize = (total_hm_width / dpi, max_hm_height / dpi)

        fig_hm, axes_hm = plt.subplots(1, len(heatmap_figs), figsize=hm_figsize)
        if len(heatmap_figs) == 1:
            axes_hm = [axes_hm]

        for ax, img, title in zip(axes_hm, heatmap_images, titles[: len(heatmap_figs)]):
            ax.imshow(img)
            ax.set_title(f"{title} - Bivariate", fontsize=12, fontweight='bold')
            ax.axis('off')

        plt.tight_layout()

    return fig


class _JsonIndexMapper:
    """Minimal IndexMapper stub for JSON-loaded bundles.

    Provides just enough interface to satisfy NomogramRenderer requirements.
    """

    def __init__(self, selected_indices: List[int], all_feature_names: List[str]):
        self._selected_indices = selected_indices
        self._all_feature_names = all_feature_names

    def dense_to_collapsed(self, dense_idx: int) -> int:
        """Convert dense position to collapsed feature index."""
        return self._selected_indices[dense_idx]

    def collapsed_to_dense(self, collapsed_idx: int) -> Optional[int]:
        """Convert collapsed feature index to dense position."""
        try:
            return self._selected_indices.index(collapsed_idx)
        except ValueError:
            return None

    def collapsed_to_original(self, collapsed_idx: int) -> List[int]:
        """Return original indices (1:1 mapping for JSON data)."""
        return [collapsed_idx]

    @property
    def n_dense(self) -> int:
        return len(self._selected_indices)

    @property
    def n_collapsed(self) -> int:
        return len(self._all_feature_names)

    @property
    def n_original(self) -> int:
        return len(self._all_feature_names)

    @property
    def is_collapse_mode(self) -> bool:
        return False


class _JsonMetadataRegistry:
    """Minimal FeatureMetadataRegistry stub for JSON-loaded bundles.

    Provides just enough interface to satisfy NomogramRenderer requirements,
    specifically the get_by_collapsed() method used for is_categorical checks.
    """

    def __init__(self, univariate_info: List[FeatureInfo], all_feature_names: List[str]):
        from prism.plotting.metadata import FeatureMetadata

        self._metadata_by_collapsed: Dict[int, 'FeatureMetadata'] = {}

        # Build metadata entries from univariate info
        for dense_idx, info in enumerate(univariate_info):
            metadata = FeatureMetadata(
                dense_idx=dense_idx,
                collapsed_idx=info.index,
                original_indices=[info.index],
                column_name=info.name,
                user_label=info.label,
                is_categorical=info.is_categorical,
                is_collapsed_group=False,
                unique_values=np.unique(info.x_values) if info.x_values is not None else None,
                min_value=(
                    float(np.min(info.x_values))
                    if info.x_values is not None and not info.is_categorical
                    else None
                ),
                max_value=(
                    float(np.max(info.x_values))
                    if info.x_values is not None and not info.is_categorical
                    else None
                ),
            )
            self._metadata_by_collapsed[info.index] = metadata

        self._all_feature_names = all_feature_names

    def get_by_collapsed(self, collapsed_idx: int) -> Optional['FeatureMetadata']:
        """Get metadata by collapsed index."""
        return self._metadata_by_collapsed.get(collapsed_idx)

    def get_by_dense(self, dense_idx: int) -> Optional['FeatureMetadata']:
        """Get metadata by dense index."""
        for metadata in self._metadata_by_collapsed.values():
            if metadata.dense_idx == dense_idx:
                return metadata
        return None

    def get_by_name(self, column_name: str) -> Optional['FeatureMetadata']:
        """Get metadata by feature name."""
        for metadata in self._metadata_by_collapsed.values():
            if metadata.column_name == column_name:
                return metadata
        return None

    def __len__(self) -> int:
        return len(self._metadata_by_collapsed)

    def __iter__(self):
        return iter(self._metadata_by_collapsed.values())


def _bundle_from_nomogram_data(data: NomogramData) -> PlottingDataBundle:
    """Reconstruct PlottingDataBundle from loaded NomogramData.

    Creates FeatureInfo and FeaturePairInfo objects from the JSON structure.
    The bundle is suitable for rendering but does not include histogram data.
    Includes minimal stub services (IndexMapper, MetadataRegistry) to satisfy
    NomogramRenderer requirements.

    Parameters
    ----------
    data : NomogramData
        Loaded nomogram data from JSON file

    Returns
    -------
    PlottingDataBundle
        Bundle with univariate and bivariate info populated, plus stub services
    """
    # Build univariate info list
    univariate_info = []
    all_feature_names = []

    for feat_name, feat_data in data.univariate.items():
        info = FeatureInfo(
            index=feat_data['index'],
            name=feat_data['name'],
            label=feat_data['label'],
            is_categorical=feat_data['is_categorical'],
            response=np.array(feat_data['response']) if feat_data['response'] else None,
            x_values=np.array(feat_data['x_values']) if feat_data['x_values'] else None,
        )
        univariate_info.append(info)
        # Track feature names by index
        while len(all_feature_names) <= feat_data['index']:
            all_feature_names.append(f"feature_{len(all_feature_names)}")
        all_feature_names[feat_data['index']] = feat_data['name']

    # Sort univariate by index to maintain consistent ordering
    univariate_info.sort(key=lambda x: x.index)

    # Build bivariate info list
    bivariate_info = []

    for pair_key, pair_data in data.bivariate.items():
        # Reconstruct x_values as (n_points, 2) array
        x_vals_1 = pair_data.get('x_values_1', [])
        x_vals_2 = pair_data.get('x_values_2', [])

        if x_vals_1 and x_vals_2:
            x_values = np.column_stack([x_vals_1, x_vals_2])
        else:
            x_values = None

        info = FeaturePairInfo(
            indices=tuple(pair_data['indices']),
            names=tuple(pair_data['names']),
            labels=tuple(pair_data['labels']),
            is_categorical=tuple(pair_data['is_categorical']),
            response=np.array(pair_data['response']) if pair_data['response'] else None,
            x_values=x_values,
            skipped=pair_data.get('skipped', False),
        )
        bivariate_info.append(info)

        # Ensure feature names exist for bivariate indices
        for idx, name in zip(pair_data['indices'], pair_data['names']):
            while len(all_feature_names) <= idx:
                all_feature_names.append(f"feature_{len(all_feature_names)}")
            all_feature_names[idx] = name

    # Sort bivariate by indices for consistent ordering
    bivariate_info.sort(key=lambda x: x.indices)

    # Create minimal stub services for NomogramRenderer compatibility
    selected_indices = [info.index for info in univariate_info]
    index_mapper = _JsonIndexMapper(selected_indices, all_feature_names)
    metadata_registry = _JsonMetadataRegistry(univariate_info, all_feature_names)

    # Create bundle with stub services (no histogram data)
    bundle = PlottingDataBundle(
        all_feature_names=all_feature_names,
        _univariate_info=univariate_info,
        _bivariate_info=bivariate_info,
        scaler=None,  # Not needed - data already denormalized
        x_data=None,  # No histogram data
        n_steps=data.n_steps,
        categorical_threshold=data.categorical_threshold,
        index_mapper=index_mapper,
        metadata_registry=metadata_registry,
    )

    logger.debug(
        f"Reconstructed PlottingDataBundle from JSON: "
        f"{bundle.n_univariate} univariate, {bundle.n_bivariate} bivariate features"
    )

    return bundle


def nomogram_from_json(
    file_path: Union[str, Path],
    *,
    # Display settings
    use_odds_ratio: bool = False,
    show_fig: bool = False,
    return_fig: bool = True,
    show_conversion_line: bool = False,
    two_column: bool = True,
    features_per_plot: int | None = None,
    legend_on_right: bool = False,
    surround_axes: bool = True,
    binary_labels: dict[int, str] | None = None,
    categorical_labels: dict[str, dict[float, str]] | None = None,
) -> NomogramResult:
    """Generate nomogram from saved JSON file.

    Loads previously saved nomogram data and renders the nomogram plot.
    Does not require the original model, scaler, or LASSO results.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the JSON file containing saved nomogram data

    # Display settings
    use_odds_ratio : bool
        Display odds ratios instead of log odds (default: False)
    show_fig : bool
        Whether to display generated figures (default: False)
    return_fig : bool
        Whether to return figure objects (default: True)
    show_conversion_line : bool
        If True, add a conversion line showing log odds to probability mapping.
        Uses the intercept saved in the JSON file. Default: False.
    two_column : bool
        Enable two-column layout (default: True)
    features_per_plot : int | None
        Number of features per page/column (default: None for auto)
    legend_on_right : bool
        Place plot legends on right side (default: False)
    surround_axes : bool
        Show axes on both sides of plots (default: True)
    binary_labels : dict[int, str] | None
        Optional custom labels for binary features
    categorical_labels : dict[str, dict[float, str]] | None
        Optional custom labels for categorical features.
        Merged with labels from JSON (user overrides take precedence).

    Returns
    -------
    NomogramResult
        A dataclass containing:
        - fig_main: Main nomogram figure(s)
        - fig_bivariate: Non-mixed bivariate heatmap figure(s) or None
        - univariate_responses: List of response arrays
        - bivariate_responses: List of bivariate response arrays
        - x_univariate: List of x-value arrays
        - x_bivariate: List of bivariate x-value arrays
        - selected_univariate_indices: List of feature indices
        - selected_bivariate_pairs: List of feature pair indices

    Examples
    --------
    >>> # Load and render nomogram from saved JSON
    >>> result = nomogram_from_json(
    ...     "models/nomogram/htx_example_mlp/nomogram.json",
    ...     two_column=True,
    ...     show_conversion_line=True,
    ... )
    >>> result.fig_main.savefig("nomogram.png")

    >>> # Override category labels from JSON
    >>> result = nomogram_from_json(
    ...     "nomogram.json",
    ...     categorical_labels={"status": {0: "Bad", 1: "Good"}},
    ... )
    """
    # Load the JSON data
    data = load_nomogram_json(file_path)

    # Reconstruct the PlottingDataBundle
    bundle = _bundle_from_nomogram_data(data)

    # Build combined categorical labels
    # Start with labels from JSON
    combined_categorical_labels = data.get_category_labels()

    # Convert string keys back to float for compatibility
    for feat_name, label_dict in combined_categorical_labels.items():
        combined_categorical_labels[feat_name] = {float(k): v for k, v in label_dict.items()}

    # Override with user-provided labels
    if categorical_labels:
        combined_categorical_labels.update(categorical_labels)

    # Create formatter and renderer
    formatter = PlotFormatter(
        use_odds_ratio=use_odds_ratio,
        binary_labels=binary_labels,
        categorical_labels=combined_categorical_labels,
    )
    renderer = NomogramRenderer(bundle, formatter, use_odds_ratio=use_odds_ratio)

    # Get intercept for conversion line
    intercept = data.intercept if show_conversion_line else None

    # Generate main nomogram
    fig_main = renderer.render_nomogram(
        legend_on_right=legend_on_right,
        surround_axes=surround_axes,
        features_per_plot=features_per_plot,
        two_column=two_column,
        intercept=intercept,
        show_conversion_line=show_conversion_line,
    )

    # Show figures if requested
    if show_fig and fig_main:
        if isinstance(fig_main, list):
            for fig in reversed(fig_main):
                plt.figure(fig.number)
                plt.show(block=False)
                plt.pause(0.1)
        else:
            plt.figure(fig_main.number)
            plt.show(block=False)
            plt.pause(0.1)

    # Generate bivariate heatmaps
    fig_non_mixed = renderer.render_bivariate_heatmaps(features_per_plot=features_per_plot)

    if show_fig and fig_non_mixed:
        if isinstance(fig_non_mixed, list):
            for fig in reversed(fig_non_mixed):
                plt.figure(fig.number)
                plt.show(block=False)
                plt.pause(0.1)
        else:
            plt.figure(fig_non_mixed.number)
            plt.show(block=False)
            plt.pause(0.1)

    logger.info(f"Nomogram rendered from JSON: {file_path}")

    # Extract data for return values
    univariate_responses = [info.response for info in bundle.univariate_features()]
    bivariate_responses = [info.response for info in bundle.bivariate_pairs()]
    x_univariate = [info.x_values for info in bundle.univariate_features()]
    x_bivariate = [info.x_values for info in bundle.bivariate_pairs()]
    selected_univariate_indices = bundle.selected_univariate_indices
    selected_bivariate_pairs = [info.indices for info in bundle.bivariate_pairs()]

    # Return results
    if return_fig:
        return NomogramResult(
            fig_main=fig_main,
            fig_bivariate=fig_non_mixed,
            univariate_responses=univariate_responses,
            bivariate_responses=bivariate_responses,
            x_univariate=x_univariate,
            x_bivariate=x_bivariate,
            selected_univariate_indices=selected_univariate_indices,
            selected_bivariate_pairs=selected_bivariate_pairs,
        )
    else:
        return NomogramResult(
            fig_main=None,
            fig_bivariate=None,
            univariate_responses=univariate_responses,
            bivariate_responses=bivariate_responses,
            x_univariate=x_univariate,
            x_bivariate=x_bivariate,
            selected_univariate_indices=selected_univariate_indices,
            selected_bivariate_pairs=selected_bivariate_pairs,
        )
