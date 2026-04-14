"""
Nomogram rendering - Pure plotting without LASSO dependencies.

This module provides the NomogramRenderer class which generates nomogram visualizations
using the PlottingDataBundle service architecture. The renderer is completely decoupled
from LASSO results and operates purely on prepared plotting data.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from prism.plotting.formatter import (  # Range-adaptive formatters for heatmaps/contours; Label truncation
    PlotFormatter,
    calculate_required_precision,
    create_continuous_formatter,
    create_response_value_formatter,
    format_value_for_annotation,
    get_nice_ticks,
    truncate_label,
)
from prism.plotting_data import (
    BinaryFeatureGroup,
    FeatureInfo,
    FeaturePairInfo,
    PlottingDataBundle,
)

logger = logging.getLogger(__name__)


class _LayoutManager:
    """Handles figure layout creation and feature distribution across pages."""

    @staticmethod
    def create_single_column_figure(
        n_features: int, height_per_feature: float = 2.5
    ) -> Tuple[plt.Figure, gridspec.GridSpec]:
        """Create single-column figure with GridSpec.

        Args:
            n_features: Number of features to plot
            height_per_feature: Height allocated per feature in inches

        Returns:
            Tuple of (figure, gridspec)
        """
        fig_height = max(n_features * height_per_feature, 4)  # Minimum 4 inches
        fig = plt.figure(figsize=(10, fig_height))
        gs = gridspec.GridSpec(n_features, 1, figure=fig, hspace=0.3)
        return fig, gs

    @staticmethod
    def create_two_column_figure(
        n_features: int, height_per_feature: float = 2.5
    ) -> Tuple[plt.Figure, gridspec.GridSpec]:
        """Create two-column figure with GridSpec.

        Args:
            n_features: Total number of features across both columns
            height_per_feature: Height allocated per feature in inches

        Returns:
            Tuple of (figure, gridspec)
        """
        # Calculate rows needed (ceiling of features/2)
        n_rows = (n_features + 1) // 2
        fig_height = max(n_rows * height_per_feature, 4)
        fig = plt.figure(figsize=(18, fig_height))
        gs = gridspec.GridSpec(n_rows, 2, figure=fig, hspace=0.3, wspace=0.1)
        return fig, gs

    @staticmethod
    def create_single_column_figure_with_conversion(
        n_features: int, height_per_feature: float = 2.5, conversion_height: float = 1.2
    ) -> Tuple[plt.Figure, gridspec.GridSpec]:
        """Create single-column figure with GridSpec and conversion line row.

        Args:
            n_features: Number of features to plot
            height_per_feature: Height allocated per feature in inches
            conversion_height: Height of conversion line row in inches

        Returns:
            Tuple of (figure, gridspec)
        """
        fig_height = max(n_features * height_per_feature, 4) + conversion_height
        fig = plt.figure(figsize=(10, fig_height))
        height_ratios = [height_per_feature] * n_features + [conversion_height]
        gs = gridspec.GridSpec(
            n_features + 1, 1, figure=fig, hspace=0.3, height_ratios=height_ratios
        )
        return fig, gs

    @staticmethod
    def create_two_column_figure_with_conversion(
        n_features: int, height_per_feature: float = 2.5, conversion_height: float = 1.2
    ) -> Tuple[plt.Figure, gridspec.GridSpec]:
        """Create two-column figure with GridSpec and conversion line row.

        The conversion line spans both columns.

        Args:
            n_features: Total number of features across both columns
            height_per_feature: Height allocated per feature in inches
            conversion_height: Height of conversion line row in inches

        Returns:
            Tuple of (figure, gridspec)
        """
        n_rows = (n_features + 1) // 2
        fig_height = max(n_rows * height_per_feature, 4) + conversion_height
        fig = plt.figure(figsize=(18, fig_height))
        height_ratios = [height_per_feature] * n_rows + [conversion_height]
        gs = gridspec.GridSpec(
            n_rows + 1, 2, figure=fig, hspace=0.3, wspace=0.1, height_ratios=height_ratios
        )
        return fig, gs

    @staticmethod
    def distribute_features(
        features: List[Dict], features_per_plot: Optional[int], two_column: bool
    ) -> List[List[Dict]]:
        """Distribute features across pages.

        Args:
            features: List of feature dictionaries to distribute
            features_per_plot: Maximum features per page (or per column if two_column)
            two_column: Whether to use two-column layout

        Returns:
            List of feature groups for each page
        """
        if not two_column:
            # Single column mode
            if features_per_plot is None:
                return [features]
            return [
                features[i : i + features_per_plot]
                for i in range(0, len(features), features_per_plot)
            ]

        # Two-column mode
        if features_per_plot is None:
            # Auto-balance features between columns
            return [features]  # Single page with auto-balancing

        # Manual features per column
        features_per_page = features_per_plot * 2  # Both columns
        pages = []
        for i in range(0, len(features), features_per_page):
            pages.append(features[i : i + features_per_page])
        return pages


class NomogramRenderer:
    """Pure rendering class for nomogram visualizations.

    This class generates nomogram plots from a PlottingDataBundle with services.
    It has NO dependencies on LASSO results - all data preparation must be done
    via PlottingPipeline before creating the renderer.

    Data Contract:
    - All x-values in the bundle (x_univariate, x_bivariate, x_data) are ALREADY
      denormalized (ready for display). No inverse scaling is performed here.
    - This design was introduced to simplify the architecture and ensure that
      collapse_onehot_features() always receives unscaled data.

    Key features:
    - Univariate plots (categorical and continuous)
    - Mixed bivariate plots (cat x cont)
    - Heatmap plots (cat x cat)
    - Contour plots (cont x cont)
    - Response plots with histograms
    - Multi-column layouts and pagination

    Example:
        >>> from prism.plotting import PlottingPipeline, NomogramRenderer, PlotFormatter
        >>>
        >>> pipeline = PlottingPipeline(lasso_results, group_mgr, label_mgr)
        >>> bundle = pipeline.prepare_plotting_bundle(x, model, scaler=scaler)
        >>> bundle = pipeline.apply_beta_scaling(bundle)
        >>>
        >>> renderer = NomogramRenderer(bundle, PlotFormatter())
        >>> fig = renderer.render_nomogram()
    """

    def __init__(
        self, bundle: PlottingDataBundle, formatter: PlotFormatter, use_odds_ratio: bool = False
    ):
        """Initialize renderer with plotting bundle and formatter.

        Args:
            bundle: PlottingDataBundle with services (IndexMapper, FeatureMetadataRegistry).
                    Must have has_services=True.
            formatter: PlotFormatter for styling plots
            use_odds_ratio: If True, convert log odds to odds ratios (exp transformation)

        Raises:
            ValueError: If bundle does not have services enabled
        """
        # Validate bundle has services
        if not bundle.has_services:
            raise ValueError(
                "PlottingDataBundle must have services enabled.\n\n"
                "Use PlottingDataBundle.from_partial_responses_with_services() "
                "or PlottingPipeline.prepare_plotting_bundle()"
            )

        # Validate services are not None
        if bundle.index_mapper is None:
            raise ValueError("Bundle missing IndexMapper service")
        if bundle.metadata_registry is None:
            raise ValueError("Bundle missing FeatureMetadataRegistry service")

        # Store references
        self.bundle = bundle
        self.formatter = formatter
        self.use_odds_ratio = use_odds_ratio

        # Quick references to services
        self.metadata_registry = bundle.metadata_registry
        self.index_mapper = bundle.index_mapper

    def _format_category_label(
        self, feature_name: str, value: float, is_binary: bool = False
    ) -> str:
        """Format categorical value using label manager or default formatting.

        Args:
            feature_name: Name of the feature
            value: Denormalized value to format
            is_binary: Whether this is a binary feature

        Returns:
            Formatted label string
        """
        # Use formatter's label formatting
        return self.formatter.format_feature_label(feature_name, value, is_binary, precision=4)

    def _render_categorical(
        self,
        ax: plt.Axes,
        info: FeatureInfo,
        subplot_params: dict,
        x_range: Optional[Tuple[float, float]] = None,
    ):
        """Render categorical univariate feature.

        Args:
            ax: Matplotlib axes
            info: FeatureInfo for this feature
            subplot_params: Subplot formatting parameters
            x_range: Optional (min, max) range for x-axis (response values)
        """
        # x_values are already denormalized by PlottingPipeline
        x_denorm = info.x_values

        # Convert response to odds ratio if needed
        response = np.exp(info.response) if self.use_odds_ratio else info.response
        # Check if binary
        unique_values = np.unique(x_denorm)
        is_binary = (
            len(unique_values) == 2
            and any(np.isclose(unique_values, 0))
            and any(np.isclose(unique_values, 1))
        )

        # Format subplot and create categorical plot
        self.formatter.format_subplot(ax, **subplot_params)

        # IMPORTANT: Set axis limits BEFORE categorical plot formatting
        # so that overlap detection uses the final coordinate transform
        ax.set_ylim(0, 1)
        if x_range:
            ax.set_xlim(x_range)

        self.formatter.format_categorical_plot(
            ax=ax,
            response=response,
            y_value=0.5,  # Standard height
            line_color=None,  # Let formatter choose
            feature_name=info.name.replace('\n', ' '),
            feature_values=x_denorm,
            is_binary=is_binary,
        )

    def _render_continuous(
        self,
        ax: plt.Axes,
        info: FeatureInfo,
        subplot_params: dict,
        x_range: Optional[Tuple[float, float]] = None,
    ):
        """Render continuous univariate feature.

        Args:
            ax: Matplotlib axes
            info: FeatureInfo for this feature
            subplot_params: Subplot formatting parameters
            x_range: Optional (min, max) range for x-axis (response values)
        """
        # x_values are already denormalized by PlottingPipeline
        x_denorm = info.x_values

        # Convert response to odds ratio if needed
        response = np.exp(info.response) if self.use_odds_ratio else info.response

        # Format subplot and create continuous plot
        self.formatter.format_subplot(ax, **subplot_params)
        self.formatter.format_continuous_plot(
            ax=ax,
            response=response,
            data_range=x_denorm,
            feature_name=info.name.replace('\n', ' '),
        )

        # Apply shared x-axis range
        if x_range:
            ax.set_xlim(x_range)

    def _render_binary_group(
        self,
        ax: plt.Axes,
        group: BinaryFeatureGroup,
        subplot_params: dict,
        x_range: Optional[Tuple[float, float]] = None,
    ):
        """Render multiple binary features in a single subplot."""
        # Set up subplot
        self.formatter.format_subplot(ax, **subplot_params)
        ax.set_ylim(0, 1)  # Use full height for group

        if x_range:
            ax.set_xlim(x_range)

        # Suppress y-axis ticks (like other categorical plots)
        ax.set_yticks([])

        # Render each feature in the group
        for idx, feature_info in enumerate(group.features):
            y_pos = group.y_positions[idx]

            # Get response values (convert to odds ratio if needed)
            response = (
                np.exp(feature_info.response) if self.use_odds_ratio else feature_info.response
            )
            x_denorm = feature_info.x_values

            # Draw horizontal line and markers for this feature
            self._draw_binary_feature_line(ax, response, y_pos, feature_info)

            # Place variable name centered between the two points
            title_x = (response[0] + response[1]) / 2 if len(response) >= 2 else response[0]
            ax.text(
                title_x,
                y_pos + 0.04,  # Closer to line (reduced from 0.08)
                feature_info.label.replace('\n', ' ').replace('\r', ''),
                ha='center',
                va='bottom',
                fontsize=self.formatter.get_font_size('label'),  # Match y-axis labels
                fontweight='normal',  # Not bold
            )

            # Place category labels below the line (closer)
            unique_vals = np.unique(x_denorm)
            offset = 0.04  # Reduced from 0.08
            for val in unique_vals:
                # Find response value for this category
                mask = x_denorm == val
                if np.any(mask):
                    x_pos = response[mask][0]
                    label = self._format_category_label(feature_info.name, val, is_binary=True)

                    ax.text(
                        x_pos,
                        y_pos - offset,  # Closer to line
                        label,
                        ha='center',
                        va='top',
                        fontsize=self.formatter.get_font_size('tick'),
                    )

    def _draw_binary_feature_line(
        self, ax: plt.Axes, response: np.ndarray, y_pos: float, feature_info: FeatureInfo
    ):
        """Draw line and markers for a single binary feature."""
        # Use default color (color 1) for all lines
        (line,) = ax.plot(response, np.full_like(response, y_pos), color='C0')
        final_color = 'C0'

        # Add vertical markers
        marker_size = getattr(self.formatter, 'marker_size', 6)
        ax.scatter(
            response,
            np.full_like(response, y_pos),
            marker="|",
            color=final_color,
            s=marker_size * 4,
            linewidth=1,
            zorder=3,
        )

    def _render_univariate(
        self,
        ax: plt.Axes,
        info: FeatureInfo,
        subplot_params: dict,
        x_range: Optional[Tuple[float, float]] = None,
    ):
        """Render univariate feature (router to categorical or continuous).

        Args:
            ax: Matplotlib axes
            info: FeatureInfo for this feature
            subplot_params: Subplot formatting parameters
            x_range: Optional (min, max) range for x-axis
        """
        # Get metadata to determine type
        metadata = self.metadata_registry.get_by_collapsed(info.index)

        if metadata.is_categorical:
            self._render_categorical(ax, info, subplot_params, x_range)
        else:
            self._render_continuous(ax, info, subplot_params, x_range)

        # Set y-axis label (truncated for display)
        ax.set_ylabel(truncate_label(info.label), rotation=90, loc="center", labelpad=5)

    def _render_mixed(
        self,
        ax: plt.Axes,
        info: FeaturePairInfo,
        subplot_params: dict,
        legend_on_right: bool,
        x_range: Optional[Tuple[float, float]] = None,
    ):
        """Render mixed bivariate feature (one categorical, one continuous).

        Args:
            ax: Matplotlib axes
            info: FeaturePairInfo for this bivariate pair
            subplot_params: Subplot formatting parameters
            legend_on_right: Place legend on right side
            x_range: Optional (min, max) range for x-axis
        """
        # Format subplot first
        self.formatter.format_subplot(ax, **subplot_params)

        # Use info from FeaturePairInfo directly (no metadata lookup)
        i, j = info.indices
        name_i, name_j = info.names
        label_i, label_j = info.labels
        is_cat_i, is_cat_j = info.is_categorical

        # Determine which is categorical and which is continuous
        if is_cat_i:
            cat_name = name_i
            cat_label = label_i
            cont_label = label_j
        else:
            cat_name = name_j
            cat_label = label_j
            cont_label = label_i

        # Extract values from info.x_values (shape: (n_samples, 2))
        # Column 0 is always feature i, column 1 is always feature j
        # NOTE: x_values are already denormalized by PlottingPipeline
        cat_values_denorm = info.x_values[:, 0 if is_cat_i else 1]
        cont_values_denorm = info.x_values[:, 1 if is_cat_i else 0]

        # We still need the raw category values for grouping (before denorm these are same)
        cat_values_raw = cat_values_denorm

        # Convert response to odds ratio if needed
        response = np.exp(info.response) if self.use_odds_ratio else info.response

        # Group responses by category
        unique_cats = np.unique(cat_values_raw)
        response_data = {}

        for cat_val_raw in unique_cats:
            mask = cat_values_raw == cat_val_raw

            # Get denormalized category value for this group
            cat_val_denorm = cat_values_denorm[mask][0]  # All same value

            # Check if binary
            unique_cat_vals = np.unique(cat_values_denorm)
            is_binary = (
                len(unique_cat_vals) == 2
                and any(np.isclose(unique_cat_vals, 0))
                and any(np.isclose(unique_cat_vals, 1))
            )

            # Format label using the formatter
            label = self.formatter.format_feature_label(
                cat_name, cat_val_denorm, is_binary, precision=4
            )

            # Store response and continuous values for this category
            response_data[label] = (response[mask], cont_values_denorm[mask])

        # Generate mixed response plot using formatter
        # Note: set_xlabel=False because format_subplot already handles xlabel
        # positioning based on is_last parameter
        self.formatter.format_mixed_response_plot(
            ax=ax,
            response_data=response_data,
            cat_feature_name=cat_label.replace('\n', ' '),  # Remove newlines for legend
            cont_feature_name=cont_label,  # Keep newlines for y-axis label
            legend_on_right=legend_on_right,
            set_xlabel=False,
        )

        # Apply shared x-axis range
        if x_range:
            ax.set_xlim(x_range)

    def _render_heatmap(self, ax: plt.Axes, info: FeaturePairInfo):
        """Render categorical x categorical heatmap.

        Args:
            ax: Matplotlib axes
            info: FeaturePairInfo for this bivariate pair
        """
        # Apply default styling but without grid
        self.formatter.apply_defaults(ax)
        ax.grid(False)

        # Use info from FeaturePairInfo directly (no metadata lookup)
        i, j = info.indices
        name_i, name_j = info.names
        label_i, label_j = info.labels

        # Get unique values (already denormalized by PlottingPipeline)
        unique_x1 = np.unique(info.x_values[:, 0])
        unique_x2 = np.unique(info.x_values[:, 1])

        # Reshape to matrix
        response_matrix = info.response.reshape(len(unique_x1), len(unique_x2))

        # Convert to odds ratio if needed
        if self.use_odds_ratio:
            response_matrix = np.exp(response_matrix)

        # Create heatmap
        im = ax.imshow(response_matrix.T, cmap='viridis', aspect='auto', origin='lower')

        # x_values are already denormalized by PlottingPipeline
        denorm_x1 = unique_x1
        denorm_x2 = unique_x2

        # Format labels
        labels_x1 = [
            self.formatter.format_feature_label(name_i, val, len(unique_x1) == 2)
            for val in denorm_x1
        ]

        labels_x2 = [
            self.formatter.format_feature_label(name_j, val, len(unique_x2) == 2)
            for val in denorm_x2
        ]

        # Set ticks and labels
        ax.set_xticks(range(len(unique_x1)))
        ax.set_yticks(range(len(unique_x2)))
        ax.set_xticklabels(labels_x1, rotation=45, ha='right')
        ax.set_yticklabels(labels_x2)

        ax.set_xlabel(truncate_label(label_i.replace('\n', ' ')))
        ax.set_ylabel(truncate_label(label_j.replace('\n', ' ')))

        # Add colorbar with styling
        title = 'Odds ratio' if self.use_odds_ratio else 'Log odds ratio'
        cbar = plt.colorbar(im, ax=ax, label=title)

        # Apply styling to colorbar
        self.formatter.apply_defaults(cbar.ax)

        # Use adaptive formatter based on data range
        vmin, vmax = response_matrix.min(), response_matrix.max()
        cbar.ax.yaxis.set_major_formatter(create_response_value_formatter(vmin, vmax))
        cbar.ax.tick_params(axis='y', which='both')
        cbar.set_label(title, fontsize=self.formatter.get_font_size('label'))

        # Add text annotations for small grids (≤5 categories each)
        if len(unique_x1) <= 5 and len(unique_x2) <= 5:
            # Calculate precision for annotations based on data range
            precision = calculate_required_precision(vmin, vmax)

            for i_idx in range(len(unique_x1)):
                for j_idx in range(len(unique_x2)):
                    value = response_matrix[i_idx, j_idx]
                    ax.text(
                        i_idx,
                        j_idx,
                        format_value_for_annotation(value, precision),
                        ha="center",
                        va="center",
                        color="white",
                        bbox=dict(
                            boxstyle="round", facecolor="black", edgecolor="none", alpha=0.5
                        ),
                        fontsize=self.formatter.get_font_size('tick'),
                    )

    def _render_contour(self, ax: plt.Axes, info: FeaturePairInfo):
        """Render continuous x continuous contour plot.

        Args:
            ax: Matplotlib axes
            info: FeaturePairInfo for this bivariate pair
        """
        # Apply default styling
        self.formatter.apply_defaults(ax)

        # Use info from FeaturePairInfo directly (no metadata lookup)
        i, j = info.indices
        name_i, name_j = info.names
        label_i, label_j = info.labels

        # Infer grid size and reshape
        n_steps = int(np.sqrt(len(info.response)))

        # x_values are already denormalized by PlottingPipeline
        X = info.x_values[:, 0].reshape(n_steps, n_steps)
        Y = info.x_values[:, 1].reshape(n_steps, n_steps)
        Z = info.response.reshape(n_steps, n_steps)

        # Convert to odds ratio if needed
        if self.use_odds_ratio:
            Z = np.exp(Z)

        # Create contour plots
        contour_heatmap = ax.contourf(X, Y, Z, cmap='viridis', levels=20)
        contour_lines = ax.contour(
            X,
            Y,
            Z,
            colors='white',
            alpha=0.5,
            levels=10,
            linewidths=self.formatter.line_widths['grid'],
        )

        # Calculate precision for contour labels based on data range
        grid_z = Z  # Store reference for colorbar formatter below
        precision = calculate_required_precision(grid_z.min(), grid_z.max())

        ax.clabel(
            contour_lines,
            inline=True,
            fontsize=8,
            fmt=lambda x: format_value_for_annotation(x, precision),
        )

        # Set labels
        ax.set_xlabel(truncate_label(label_i.replace('\n', ' ')))
        ax.set_ylabel(truncate_label(label_j.replace('\n', ' ')))

        # Apply specialized formatters for axes
        ax.xaxis.set_major_formatter(create_continuous_formatter(name_i))
        ax.yaxis.set_major_formatter(create_continuous_formatter(name_j))

        # Add colorbar with styling
        title = 'Odds ratio' if self.use_odds_ratio else 'Log odds ratio'
        cbar = plt.colorbar(contour_heatmap, ax=ax, label=title)

        # Apply styling to colorbar
        self.formatter.apply_defaults(cbar.ax)

        # Use adaptive formatter based on data range
        vmin, vmax = grid_z.min(), grid_z.max()
        cbar.ax.yaxis.set_major_formatter(create_response_value_formatter(vmin, vmax))
        cbar.ax.tick_params(
            axis='y', which='major', length=self.formatter.tick_params['major']['length']
        )
        cbar.set_label(title, fontsize=self.formatter.get_font_size('label'))

    def _render_conversion_line(
        self,
        ax: plt.Axes,
        x_range: Tuple[float, float],
        intercept: float,
        n_ticks: int = 12,
    ) -> None:
        """Render conversion line showing sum-to-probability mapping.

        The conversion line helps users convert the sum of partial response
        contributions to a probability using the logistic function.

        Args:
            ax: Matplotlib axes (should span both columns in two-column layout)
            x_range: (min, max) range of summed response values (in odds ratio space if use_odds_ratio=True)
            intercept: LASSO intercept (beta_0) from the model
            n_ticks: Number of ticks on the conversion line (default: 12)
        """
        # Apply default styling
        self.formatter.apply_defaults(ax)
        ax.grid(False)

        # Convert x_range to log space if needed (x_range is in odds ratio space when use_odds_ratio=True)
        if self.use_odds_ratio:
            # x_range is in exp-space, convert to log-space for tick generation
            x_range_log = (np.log(x_range[0]), np.log(x_range[1]))
        else:
            x_range_log = x_range

        # Calculate practical probability limits
        # logit(0.01) = log(0.01/0.99) approx -4.595
        # logit(0.99) = log(0.99/0.01) approx 4.595
        LOGIT_001 = -4.595
        LOGIT_099 = 4.595

        # For practical probability range [0.01, 0.99]:
        # - min_sum = LOGIT_001 - intercept (gives P = 0.01)
        # - max_sum = LOGIT_099 - intercept (gives P = 0.99)
        practical_min_sum = LOGIT_001 - intercept
        practical_max_sum = LOGIT_099 - intercept

        # Clip to practical range, but also respect actual x_range (in log space)
        display_min = max(x_range_log[0], practical_min_sum)
        display_max = min(x_range_log[1], practical_max_sum)

        # If x_range is narrower than practical range, use x_range
        if display_min > display_max:
            display_min = x_range[0]
            display_max = x_range[1]

        # Generate nice tick positions first
        tick_positions = get_nice_ticks(display_min, display_max, n_ticks=n_ticks)

        # Set axis limits to span the actual tick range (so line extends to all ticks)
        ax.set_xlim(tick_positions.min(), tick_positions.max())
        ax.set_ylim(0, 1)

        # Hide all axes and spines (including bottom spine)
        ax.yaxis.set_visible(False)
        ax.xaxis.set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # Draw the horizontal conversion line at y=0.5
        ax.axhline(y=0.5, color='black', linewidth=1.5, zorder=2)

        # Calculate probabilities for each tick position
        total_log_odds = intercept + tick_positions
        probabilities = 1 / (1 + np.exp(-total_log_odds))

        # Handle odds ratio mode
        if self.use_odds_ratio:
            top_values = np.exp(tick_positions)
            top_label = "Total Odds Ratio"
        else:
            top_values = tick_positions
            top_label = "Log Odds Sum"

        # Add tick markers and labels
        for i, (pos, prob, top_val) in enumerate(zip(tick_positions, probabilities, top_values)):
            # Vertical tick mark
            ax.plot([pos, pos], [0.4, 0.6], 'k-', linewidth=1, zorder=3)

            # Top label (log odds sum or total odds ratio)
            if self.use_odds_ratio:
                # Format odds ratio values
                if top_val >= 10:
                    top_text = f"{top_val:.0f}"
                elif top_val >= 1:
                    top_text = f"{top_val:.1f}".rstrip('0').rstrip('.')
                else:
                    top_text = f"{top_val:.2f}".rstrip('0').rstrip('.')
            else:
                # Format log odds values
                top_text = f"{pos:.1f}".rstrip('0').rstrip('.')

            # Keep all labels centered to stay within line bounds
            ax.text(
                pos,
                0.62,
                top_text,
                ha='center',
                va='bottom',
                fontsize=self.formatter.get_font_size('tick'),
            )

            # Bottom label (probability)
            if prob >= 0.1:
                prob_text = f"{prob:.2f}".rstrip('0').rstrip('.')
            else:
                prob_text = f"{prob:.3f}".rstrip('0').rstrip('.')

            ax.text(
                pos,
                0.38,
                prob_text,
                ha='center',
                va='top',
                fontsize=self.formatter.get_font_size('tick'),
            )

        # Add axis labels at top and bottom
        ax.text(
            0.5,
            0.85,
            top_label,
            ha='center',
            va='bottom',
            transform=ax.transAxes,
            fontsize=self.formatter.get_font_size('label'),
            fontweight='bold',
        )
        ax.text(
            0.5,
            0.15,
            "Probability",
            ha='center',
            va='top',
            transform=ax.transAxes,
            fontsize=self.formatter.get_font_size('label'),
            fontweight='bold',
        )

        # Add baseline footnote
        baseline_prob = 1 / (1 + np.exp(-intercept))
        if self.use_odds_ratio:
            baseline_or = np.exp(intercept)
            if abs(intercept) >= 10:
                beta_text = f"{intercept:.1f}"
            elif abs(intercept) >= 1:
                beta_text = f"{intercept:.2f}"
            else:
                beta_text = f"{intercept:.3f}"

            if baseline_or >= 10:
                or_text = f"{baseline_or:.1f}"
            elif baseline_or >= 1:
                or_text = f"{baseline_or:.2f}"
            else:
                or_text = f"{baseline_or:.3f}"

            footnote = f"Baseline: log odds = {beta_text}, odds ratio = {or_text}, probability = {baseline_prob:.3f}"
        else:
            if abs(intercept) >= 10:
                beta_text = f"{intercept:.1f}"
            elif abs(intercept) >= 1:
                beta_text = f"{intercept:.2f}"
            else:
                beta_text = f"{intercept:.3f}"
            footnote = f"Baseline: log odds = {beta_text}, probability = {baseline_prob:.3f}"

        ax.text(
            0.0,
            0.02,
            footnote,
            ha='left',
            va='bottom',
            transform=ax.transAxes,
            fontsize=self.formatter.get_font_size('tick') * 0.9,
            style='italic',
            color='gray',
        )

    def _calculate_x_range(self, features_to_plot: List[Dict]) -> Optional[Tuple[float, float]]:
        """Calculate global x-axis range for response values (log odds or odds ratios).

        All subplots in a nomogram share the same x-axis scale to enable visual comparison.

        Args:
            features_to_plot: List of feature dictionaries

        Returns:
            (min, max) tuple for response values
        """
        response_min, response_max = None, None

        for feature_dict in features_to_plot:
            if feature_dict['type'] == 'binary_group':
                # Check all features in the group
                for info in feature_dict['group'].features:
                    response = np.exp(info.response) if self.use_odds_ratio else info.response
                    curr_min, curr_max = response.min(), response.max()

                    if response_min is None:
                        response_min = curr_min
                        response_max = curr_max
                    else:
                        response_min = min(response_min, curr_min)
                        response_max = max(response_max, curr_max)
            else:
                # Handle regular features
                info = feature_dict['info']

                # Get response values (convert to odds ratio if needed)
                response = np.exp(info.response) if self.use_odds_ratio else info.response

                curr_min, curr_max = response.min(), response.max()

                if response_min is None:
                    response_min = curr_min
                    response_max = curr_max
                else:
                    response_min = min(response_min, curr_min)
                    response_max = max(response_max, curr_max)

        if response_min is not None:
            # Add small margin (5%)
            margin = (response_max - response_min) * 0.05

            # Handle edge case: all responses are the same value
            if margin == 0:
                # Create a small range around the constant value
                if response_min == 0:
                    margin = 0.1  # Default margin for zero
                else:
                    margin = abs(response_min) * 0.1  # 10% of the value

            return (response_min - margin, response_max + margin)
        return None

    def _format_nomogram_x_axis(self, ax: plt.Axes, x_range: Tuple[float, float]) -> None:
        """Apply consistent x-axis formatting for nomogram plots.

        Sets adaptive tick formatting, increases tick count, and adds reference line.

        Args:
            ax: Matplotlib axes to format
            x_range: (min, max) tuple for x-axis range
        """
        # Apply adaptive response value formatter (matching response plots)
        ax.xaxis.set_major_formatter(create_response_value_formatter(x_range[0], x_range[1]))

        # Generate nice tick locations with more ticks (8-10 instead of default 5-6)
        # Use n_ticks=10 because MaxNLocator with steps=[1,2,2.5,5,10] produces discrete
        # jumps: nbins=8 gives ~6-7 ticks, nbins=10 gives ~10-12 ticks (filtered down to ~8-10)
        x_ticks = get_nice_ticks(x_range[0], x_range[1], n_ticks=10)
        ax.set_xticks(x_ticks)

        # Add black reference line at 0 (log odds) or 1 (odds ratio)
        ref_value = 1 if self.use_odds_ratio else 0

        # Only draw if reference value is within the axis range
        if x_range[0] <= ref_value <= x_range[1]:
            ax.axvline(
                x=ref_value,
                color='black',
                linewidth=self.formatter.line_widths['axis'],
                alpha=0.8,
                zorder=1,  # Behind data but above grid
            )

    def _get_column_axis_params(
        self, column: str, surround_axes: bool, is_single_column: bool = False
    ) -> Dict:
        """Get axis parameters for column positioning.

        Args:
            column: "left" or "right"
            surround_axes: Whether to show axes on both sides
            is_single_column: Whether this is a single-column layout

        Returns:
            Dictionary of axis formatting parameters
        """
        base_params = {
            'x_axis_position': 'both' if surround_axes else 'bottom',
            'x_ticks_position': 'both' if surround_axes else 'bottom',
            'x_tick_labels_position': 'both' if surround_axes else 'bottom',
            'x_label_position': 'bottom',
        }

        if column == "left":
            # For single-column with surround_axes, show both spines
            # For two-column, only show left spine
            return {
                **base_params,
                'y_axis_position': 'both' if (surround_axes and is_single_column) else 'left',
                'y_label_position': 'left',
            }
        else:  # right column (only used in two-column layout)
            return {
                **base_params,
                'y_axis_position': 'right' if surround_axes else 'left',
                'y_label_position': 'right' if surround_axes else 'left',
            }

    def _is_binary_feature(self, info: FeatureInfo) -> bool:
        """Check if a feature is binary (exactly 2 unique values)."""
        if not info.is_categorical or info.x_values is None:
            return False

        unique_values = np.unique(info.x_values)
        return len(unique_values) == 2

    def _group_binary_features(self, features: List[Dict]) -> List[Dict]:
        """Group binary categorical features (up to 3 per group)."""
        binary_features = []
        non_binary_features = []

        # Separate binary from non-binary
        for feature_dict in features:
            if feature_dict['type'] == 'univariate' and self._is_binary_feature(
                feature_dict['info']
            ):
                binary_features.append(feature_dict)
            else:
                non_binary_features.append(feature_dict)

        # Group binary features (up to 3 per group)
        grouped_features = []
        for i in range(0, len(binary_features), 3):
            group_features = binary_features[i : i + 3]

            if len(group_features) == 1:
                # Keep single binary features as-is for backward compatibility
                grouped_features.append(group_features[0])
            else:
                # Create a grouped feature entry (y_positions will be calculated dynamically)
                group = BinaryFeatureGroup(features=[f['info'] for f in group_features])
                grouped_features.append({'type': 'binary_group', 'group': group})

        # Add non-binary features
        grouped_features.extend(non_binary_features)

        return grouped_features

    def _render_single_column_page(
        self,
        features: List[Dict],
        x_range: Optional[Tuple[float, float]],
        legend_on_right: bool,
        surround_axes: bool,
        intercept: Optional[float] = None,
        show_conversion_line: bool = False,
    ) -> plt.Figure:
        """Render a single page with single-column layout.

        Args:
            features: List of feature dictionaries for this page
            x_range: Global x-range for continuous features
            legend_on_right: Place legends on right side
            surround_axes: Show axes on both sides
            intercept: Optional LASSO intercept (beta_0) for conversion line
            show_conversion_line: Whether to add conversion line below nomogram

        Returns:
            Matplotlib Figure
        """
        n_features = len(features)

        # Choose figure creation method based on conversion line
        if show_conversion_line and intercept is not None:
            fig, gs = _LayoutManager.create_single_column_figure_with_conversion(n_features)
        else:
            fig, gs = _LayoutManager.create_single_column_figure(n_features)

        base_params = self._get_column_axis_params("left", surround_axes, is_single_column=True)

        for idx, feature_dict in enumerate(features):
            ax = fig.add_subplot(gs[idx, 0])

            # Track first/last for proper axis formatting
            is_first = idx == 0
            is_last = idx == n_features - 1
            subplot_params = {**base_params, 'is_first': is_first, 'is_last': is_last}

            if feature_dict['type'] == 'univariate':
                self._render_univariate(ax, feature_dict['info'], subplot_params, x_range)
            elif feature_dict['type'] == 'mixed':
                self._render_mixed(
                    ax, feature_dict['info'], subplot_params, legend_on_right, x_range
                )
            elif feature_dict['type'] == 'binary_group':
                self._render_binary_group(ax, feature_dict['group'], subplot_params, x_range)

            # Apply x-axis formatting for nomograms
            if x_range is not None:
                self._format_nomogram_x_axis(ax, x_range)

        # Add conversion line if requested
        if show_conversion_line and intercept is not None and x_range is not None:
            ax_conversion = fig.add_subplot(gs[n_features, 0])  # Last row
            self._render_conversion_line(ax_conversion, x_range, intercept)

        fig.tight_layout()
        return fig

    def _render_two_column_page(
        self,
        features: List[Dict],
        x_range: Optional[Tuple[float, float]],
        legend_on_right: bool,
        surround_axes: bool,
        intercept: Optional[float] = None,
        show_conversion_line: bool = False,
    ) -> plt.Figure:
        """Render a single page with two-column layout.

        Args:
            features: List of feature dictionaries for this page
            x_range: Global x-range for continuous features
            legend_on_right: Place legends on right side
            surround_axes: Show axes on both sides
            intercept: Optional LASSO intercept (beta_0) for conversion line
            show_conversion_line: Whether to add conversion line below nomogram

        Returns:
            Matplotlib Figure
        """
        n_features = len(features)

        # Choose figure creation method based on conversion line
        if show_conversion_line and intercept is not None:
            fig, gs = _LayoutManager.create_two_column_figure_with_conversion(n_features)
        else:
            fig, gs = _LayoutManager.create_two_column_figure(n_features)

        # Split features between columns
        n_per_column = (n_features + 1) // 2  # Left column gets extra if odd
        left_features = features[:n_per_column]
        right_features = features[n_per_column:]

        # Get column-specific base parameters
        left_base = self._get_column_axis_params("left", surround_axes)
        right_base = self._get_column_axis_params("right", surround_axes)

        # Render left column
        for idx, feature_dict in enumerate(left_features):
            ax = fig.add_subplot(gs[idx, 0])

            # Track first/last for left column
            is_first = idx == 0
            is_last = idx == len(left_features) - 1
            left_params = {**left_base, 'is_first': is_first, 'is_last': is_last}

            if feature_dict['type'] == 'univariate':
                self._render_univariate(ax, feature_dict['info'], left_params, x_range)
            elif feature_dict['type'] == 'mixed':
                self._render_mixed(ax, feature_dict['info'], left_params, legend_on_right, x_range)
            elif feature_dict['type'] == 'binary_group':
                self._render_binary_group(ax, feature_dict['group'], left_params, x_range)

            # Apply x-axis formatting for nomograms
            if x_range is not None:
                self._format_nomogram_x_axis(ax, x_range)

        # Render right column
        for idx, feature_dict in enumerate(right_features):
            ax = fig.add_subplot(gs[idx, 1])

            # Track first/last for right column
            is_first = idx == 0
            is_last = idx == len(right_features) - 1
            right_params = {**right_base, 'is_first': is_first, 'is_last': is_last}

            if feature_dict['type'] == 'univariate':
                self._render_univariate(ax, feature_dict['info'], right_params, x_range)
            elif feature_dict['type'] == 'mixed':
                self._render_mixed(
                    ax, feature_dict['info'], right_params, legend_on_right, x_range
                )
            elif feature_dict['type'] == 'binary_group':
                self._render_binary_group(ax, feature_dict['group'], right_params, x_range)

            # Apply x-axis formatting for nomograms
            if x_range is not None:
                self._format_nomogram_x_axis(ax, x_range)

        # Add conversion line spanning both columns
        if show_conversion_line and intercept is not None and x_range is not None:
            n_rows = (n_features + 1) // 2
            # Use slice to span both columns: gs[last_row, :]
            ax_conversion = fig.add_subplot(gs[n_rows, :])
            self._render_conversion_line(ax_conversion, x_range, intercept)

        fig.tight_layout()
        return fig

    def render_nomogram(
        self,
        legend_on_right: bool = False,
        surround_axes: bool = False,
        features_per_plot: Optional[int] = None,
        two_column: bool = False,
        intercept: Optional[float] = None,
        show_conversion_line: bool = False,
    ) -> Union[plt.Figure, List[plt.Figure]]:
        """Render main nomogram with univariate and mixed bivariate features.

        Automatically groups binary categorical features (2-3 per subplot) for more
        efficient use of space while maintaining clear visual presentation.

        Args:
            legend_on_right: Place legends on right side (for mixed bivariate)
            surround_axes: Show axes on all sides
            features_per_plot: Maximum features per page (or per column if two_column)
            two_column: Use two-column layout
            intercept: Optional LASSO intercept (beta_0) for conversion line
            show_conversion_line: Whether to add conversion line below nomogram

        Returns:
            Single Figure or list of Figures if paginated
        """
        # Collect univariate features
        features_to_plot = []
        for info in self.bundle.univariate_features():
            features_to_plot.append(
                {
                    'type': 'univariate',
                    'info': info,
                }
            )

        # Add mixed bivariate features (one categorical, one continuous)
        for pair_info in self.bundle.bivariate_pairs():
            if pair_info.skipped:
                continue

            # Use the is_categorical tuple directly from FeaturePairInfo
            is_cat_i, is_cat_j = pair_info.is_categorical

            # Check if mixed (one categorical, one continuous)
            is_mixed = is_cat_i != is_cat_j

            if is_mixed:
                features_to_plot.append(
                    {
                        'type': 'mixed',
                        'info': pair_info,
                    }
                )

        # NEW: Group binary features
        features_to_plot = self._group_binary_features(features_to_plot)

        if not features_to_plot:
            logger.warning("No features to plot")
            return plt.figure()

        # Calculate global x-range
        x_range = self._calculate_x_range(features_to_plot)

        # Distribute to pages
        pages = _LayoutManager.distribute_features(features_to_plot, features_per_plot, two_column)

        # Render each page
        figures = []
        for page_features in pages:
            if two_column:
                fig = self._render_two_column_page(
                    page_features,
                    x_range,
                    legend_on_right,
                    surround_axes,
                    intercept=intercept,
                    show_conversion_line=show_conversion_line,
                )
            else:
                fig = self._render_single_column_page(
                    page_features,
                    x_range,
                    legend_on_right,
                    surround_axes,
                    intercept=intercept,
                    show_conversion_line=show_conversion_line,
                )
            figures.append(fig)

        return figures[0] if len(figures) == 1 else figures

    def render_bivariate_heatmaps(
        self, features_per_plot: Optional[int] = None
    ) -> Union[plt.Figure, List[plt.Figure], None]:
        """Render bivariate heatmaps (catxcat and contxcont).

        Args:
            features_per_plot: Maximum heatmaps per page

        Returns:
            Single Figure, list of Figures, or None if no heatmaps
        """
        # Collect catxcat and contxcont pairs (non-mixed)
        heatmap_pairs = []
        for pair_info in self.bundle.bivariate_pairs():
            if pair_info.skipped:
                continue

            # Use the is_categorical tuple directly from FeaturePairInfo
            is_cat_i, is_cat_j = pair_info.is_categorical

            # Skip mixed (already in main nomogram)
            is_mixed = is_cat_i != is_cat_j
            if is_mixed:
                continue

            # Determine type
            both_cat = is_cat_i and is_cat_j

            heatmap_pairs.append(
                {
                    'info': pair_info,
                    'both_cat': both_cat,
                }
            )

        if not heatmap_pairs:
            return None

        # Determine heatmaps per page
        if features_per_plot is None:
            # All on one page, arranged in grid
            heatmaps_per_page = len(heatmap_pairs)
        else:
            heatmaps_per_page = features_per_plot

        # Distribute to pages
        pages = []
        for i in range(0, len(heatmap_pairs), heatmaps_per_page):
            pages.append(heatmap_pairs[i : i + heatmaps_per_page])

        # Render each page
        figures = []
        for page_pairs in pages:
            fig = self._render_heatmap_page(page_pairs)
            figures.append(fig)

        return figures[0] if len(figures) == 1 else figures

    def _render_heatmap_page(self, heatmap_pairs: List[Dict]) -> plt.Figure:
        """Render a page of bivariate heatmaps.

        Args:
            heatmap_pairs: List of heatmap pair dictionaries

        Returns:
            Matplotlib Figure
        """
        n_pairs = len(heatmap_pairs)

        # Calculate grid dimensions (prefer square-ish grid)
        n_cols = min(3, n_pairs)  # Max 3 columns
        n_rows = int(np.ceil(n_pairs / n_cols))

        # Create figure
        fig = plt.figure(figsize=(6 * n_cols, 5 * n_rows))
        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.4, wspace=0.4)

        # Render each heatmap
        for idx, pair_dict in enumerate(heatmap_pairs):
            row = idx // n_cols
            col = idx % n_cols
            ax = fig.add_subplot(gs[row, col])

            info = pair_dict['info']
            both_cat = pair_dict['both_cat']

            if both_cat:
                self._render_heatmap(ax, info)
            else:
                self._render_contour(ax, info)

        fig.tight_layout()
        return fig

    def _render_continuous_response_with_histogram(self, ax: plt.Axes, info: FeatureInfo) -> None:
        """Render continuous univariate response with histogram.

        Args:
            ax: Matplotlib axis to render on
            info: FeatureInfo with response and x_values
        """
        # Apply default formatting
        hist_ax = ax.twinx()
        for axis in [ax, hist_ax]:
            self.formatter.apply_defaults(axis)
            axis.grid(False)

        # Ensure primary axis (plots/legends) is on top of histogram axis
        ax.set_zorder(hist_ax.get_zorder() + 1)
        ax.set_facecolor('none')  # Transparent so histogram shows through

        # x_values are already denormalized by PlottingPipeline
        x_denormalized = info.x_values

        # Get response (apply odds ratio conversion if needed)
        response = info.response.copy()
        if self.use_odds_ratio:
            response = np.exp(response)
            # Keep linear scale for response plots - visual trends clearer on linear scale
            # Use adaptive formatter based on response range
            response_min, response_max = response.min(), response.max()

            # Configure custom tick positioning for ranges around 1.0
            range_span = response_max - response_min
            if range_span < 0.4:  # Range within ±0.2 of 1.0 - use custom ticks for clarity
                n_ticks = 5
                tick_values = np.linspace(response_min, response_max, n_ticks)

                # Ensure 1.0 is included if it's in the range
                if response_min <= 1.0 <= response_max:
                    idx = np.argmin(np.abs(tick_values - 1.0))
                    tick_values[idx] = 1.0

                from matplotlib.ticker import FixedLocator

                ax.yaxis.set_major_locator(FixedLocator(tick_values))
            # For wide ranges, use default positioning like normal plots

            ax.yaxis.set_major_formatter(
                create_response_value_formatter(response_min, response_max)
            )

        # Add histogram of original data (already denormalized by PlottingPipeline)
        # Filter to grid range to handle trim_quantile properly
        original_data = self.bundle.x_data[:, info.index]
        grid_min, grid_max = x_denormalized.min(), x_denormalized.max()
        hist_data = original_data[(original_data >= grid_min) & (original_data <= grid_max)]

        n_bins = min(50, len(np.unique(hist_data)))
        hist_ax.hist(
            hist_data,
            bins=n_bins,
            alpha=0.3,
            color='gray',
            linewidth=1,
            edgecolor='white',
            zorder=1,
        )
        hist_ax.set_ylabel('Count', color='gray')
        hist_ax.tick_params(axis='y', labelcolor='gray')
        hist_ax.yaxis.set_major_formatter(create_continuous_formatter(""))

        # Plot partial response
        ax.plot(x_denormalized, response, label='Partial Response', zorder=3)

        # Set and format x-ticks using smart tick positioning
        x_ticks = get_nice_ticks(
            x_denormalized.min(), x_denormalized.max(), n_ticks=6, feature_name=info.label
        )
        ax.set_xticks(x_ticks)
        ax.xaxis.set_major_formatter(create_continuous_formatter(info.label))

        # Set x-axis limits based on grid values with margin for padding
        x_margin = (grid_max - grid_min) * 0.05
        ax.set_xlim(grid_min - x_margin, grid_max + x_margin)

        ax.set_xlabel(truncate_label(info.label))
        ax.set_ylabel('Odds ratio' if self.use_odds_ratio else 'Log odds ratio')

        # Adjust y-axis for density
        ylim = hist_ax.get_ylim()
        hist_ax.set_ylim(0, ylim[1] * 1.2)

        # Add reference line
        ref_y = 1 if self.use_odds_ratio else 0
        ax.axhline(
            y=ref_y,
            color='k',
            linewidth=self.formatter.line_widths['axis'],
            alpha=0.8,
            zorder=2,
        )

    def _render_categorical_response_with_histogram(self, ax: plt.Axes, info: FeatureInfo) -> None:
        """Render categorical univariate response with histogram.

        Args:
            ax: Matplotlib axis to render on
            info: FeatureInfo with response and x_values
        """
        # Apply default formatting
        hist_ax = ax.twinx()
        for axis in [ax, hist_ax]:
            self.formatter.apply_defaults(axis)
            axis.grid(False)

        # Ensure primary axis (plots/legends) is on top of histogram axis
        ax.set_zorder(hist_ax.get_zorder() + 1)
        ax.set_facecolor('none')  # Transparent so histogram shows through

        # x_values are already denormalized by PlottingPipeline
        x_denormalized = info.x_values

        # Get response (apply odds ratio conversion if needed)
        response = info.response.copy()
        if self.use_odds_ratio:
            response = np.exp(response)
            # Keep linear scale for response plots - visual trends clearer on linear scale
            # Use adaptive formatter based on response range
            response_min, response_max = response.min(), response.max()

            # Configure custom tick positioning for ranges around 1.0
            range_span = response_max - response_min
            if range_span < 0.4:  # Range within ±0.2 of 1.0 - use custom ticks for clarity
                n_ticks = 5
                tick_values = np.linspace(response_min, response_max, n_ticks)

                # Ensure 1.0 is included if it's in the range
                if response_min <= 1.0 <= response_max:
                    idx = np.argmin(np.abs(tick_values - 1.0))
                    tick_values[idx] = 1.0

                from matplotlib.ticker import FixedLocator

                ax.yaxis.set_major_locator(FixedLocator(tick_values))
            # For wide ranges, use default positioning like normal plots

            ax.yaxis.set_major_formatter(
                create_response_value_formatter(response_min, response_max)
            )

        # Get unique categories and their counts from the full dataset
        # (already denormalized by PlottingPipeline)
        original_data = self.bundle.x_data[:, info.index]
        categories, counts = np.unique(original_data, return_counts=True)

        # Check if binary feature
        is_binary = (
            len(categories) == 2
            and any(np.isclose(categories, 0))
            and any(np.isclose(categories, 1))
        )

        # Create stem plot for log odds ratio or odds ratio
        ref_y = 1 if self.use_odds_ratio else 0
        markerline, stemlines, baseline = ax.stem(
            x_denormalized, response, basefmt=' ', bottom=ref_y
        )
        markerline.set_marker('_')
        markerline.set_markersize(10)
        markerline.set_markeredgewidth(1.5)
        markerline.set_zorder(3)
        stemlines.set_linewidth(1.5)
        stemlines.set_zorder(2)
        ax.set_ylabel('Odds ratio' if self.use_odds_ratio else 'Log odds ratio')

        # Create histogram
        hist_ax.bar(categories, counts, alpha=0.3, color='gray', width=0.8, zorder=1)
        hist_ax.set_ylabel('Count', color='gray')
        hist_ax.tick_params(axis='y', labelcolor='gray')

        # Set x-ticks: use text labels for binary, integer labels for multi-category
        ax.set_xticks(categories)
        if is_binary:
            x_labels = [
                self._format_category_label(info.name, val, is_binary=True) for val in categories
            ]
            ax.set_xticklabels(x_labels)
        else:
            ax.set_xticklabels([str(int(round(c))) for c in categories])

        ax.set_xlabel(truncate_label(info.label))

        # Adjust y-axis for count
        ylim = hist_ax.get_ylim()
        hist_ax.set_ylim(0, ylim[1] * 1.2)

        # Add reference line
        ref_y = 1 if self.use_odds_ratio else 0
        ax.axhline(
            y=ref_y,
            color='k',
            linewidth=self.formatter.line_widths['axis'],
            alpha=0.8,
        )

    def _render_mixed_response_with_histogram(
        self, ax: plt.Axes, pair_info: FeaturePairInfo
    ) -> None:
        """Render mixed (cat x cont) bivariate response with histogram.

        Args:
            ax: Matplotlib axis to render on
            pair_info: FeaturePairInfo with response and x_values
        """
        # Apply default formatting
        hist_ax = ax.twinx()
        for axis in [ax, hist_ax]:
            self.formatter.apply_defaults(axis)
            axis.grid(False)

        # Ensure primary axis (plots/legends) is on top of histogram axis
        ax.set_zorder(hist_ax.get_zorder() + 1)
        ax.set_facecolor('none')  # Transparent so histogram shows through

        i, j = pair_info.indices
        is_cat_i, is_cat_j = pair_info.is_categorical

        # Determine which is categorical and which is continuous
        cat_idx = 0 if is_cat_i else 1
        cont_idx = 1 - cat_idx

        cont_feature_idx = i if cont_idx == 0 else j

        # x_values are already denormalized by PlottingPipeline
        x_denormalized = pair_info.x_values

        # Get response (apply odds ratio conversion if needed)
        response = pair_info.response.copy()
        if self.use_odds_ratio:
            response = np.exp(response)
            # Keep linear scale for response plots - visual trends clearer on linear scale
            # Use adaptive formatter based on response range
            response_min, response_max = response.min(), response.max()

            # Configure custom tick positioning for ranges around 1.0
            range_span = response_max - response_min
            if range_span < 0.4:  # Range within ±0.2 of 1.0 - use custom ticks for clarity
                n_ticks = 5
                tick_values = np.linspace(response_min, response_max, n_ticks)

                # Ensure 1.0 is included if it's in the range
                if response_min <= 1.0 <= response_max:
                    idx = np.argmin(np.abs(tick_values - 1.0))
                    tick_values[idx] = 1.0

                from matplotlib.ticker import FixedLocator

                ax.yaxis.set_major_locator(FixedLocator(tick_values))
            # For wide ranges, use default positioning like normal plots

            ax.yaxis.set_major_formatter(
                create_response_value_formatter(response_min, response_max)
            )

        cat_values = np.unique(x_denormalized[:, cat_idx])
        cont_values = x_denormalized[:, cont_idx]

        # Check if binary categorical feature
        is_binary = (
            len(cat_values) == 2
            and any(np.isclose(cat_values, 0))
            and any(np.isclose(cat_values, 1))
        )

        # Get categorical feature name and label for legend
        cat_name = pair_info.names[cat_idx]
        cat_label = pair_info.labels[cat_idx]

        # Plot lines for each categorical value
        for cat_val in cat_values:
            # Use text labels for binary, integer labels for multi-category
            if is_binary:
                label = self._format_category_label(cat_name, cat_val, is_binary=True)
            else:
                label = str(int(round(cat_val)))

            mask = x_denormalized[:, cat_idx] == cat_val
            ax.plot(
                cont_values[mask],
                response[mask],
                label=label,
                zorder=3,
            )
        # Add histogram of original continuous data (already denormalized by PlottingPipeline)
        # Filter to grid range to handle trim_quantile properly
        original_cont_data = self.bundle.x_data[:, cont_feature_idx]
        grid_min, grid_max = cont_values.min(), cont_values.max()
        hist_data = original_cont_data[
            (original_cont_data >= grid_min) & (original_cont_data <= grid_max)
        ]

        n_bins = min(50, len(np.unique(hist_data)))
        hist_ax.hist(
            hist_data,
            bins=n_bins,
            alpha=0.3,
            color='gray',
            linewidth=1,
            edgecolor='white',
            zorder=1,
        )
        hist_ax.set_ylabel('Count', color='gray')
        hist_ax.tick_params(axis='y', labelcolor='gray')

        # Set x-ticks using smart tick positioning
        cont_name = pair_info.names[cont_idx]
        cont_label = pair_info.labels[cont_idx]
        x_ticks = get_nice_ticks(
            cont_values.min(), cont_values.max(), n_ticks=6, feature_name=cont_name
        )
        ax.set_xticks(x_ticks)
        ax.xaxis.set_major_formatter(create_continuous_formatter(cont_name))

        # Set x-axis limits based on grid values with margin for padding
        x_margin = (grid_max - grid_min) * 0.05
        ax.set_xlim(grid_min - x_margin, grid_max + x_margin)

        ax.set_xlabel(truncate_label(cont_label))
        ax.set_ylabel('Odds ratio' if self.use_odds_ratio else 'Log odds ratio')

        # Add legend with categorical feature label as title (matching nomogram format)
        self.formatter._add_legend(
            ax, truncate_label(cat_label), loc='lower left', bbox_to_anchor=(0.05, 0.05)
        )

        # Adjust y-axis for density
        ylim = hist_ax.get_ylim()
        hist_ax.set_ylim(0, ylim[1] * 1.2)

        # Add reference line
        ref_y = 1 if self.use_odds_ratio else 0
        ax.axhline(
            y=ref_y,
            color='k',
            linewidth=self.formatter.line_widths['axis'],
            alpha=0.8,
            zorder=2,
        )

    def render_response_plots(
        self, subfig_size: float = 3.5, features_per_plot: Optional[int] = None
    ) -> Union[plt.Figure, List[plt.Figure]]:
        """Render response plots with histograms in 3-column grid.

        Generates a grid of response plots showing partial responses with histograms
        of the original data distribution. Handles univariate (categorical and continuous)
        and mixed bivariate features.

        Args:
            subfig_size: Size of individual subplots (default: 3.5)
            features_per_plot: Maximum features per page (enables pagination)

        Returns:
            Single Figure or list of Figures (if pagination enabled)
        """
        # Collect all features to plot
        features_to_plot = []

        # Add univariate features
        for info in self.bundle.univariate_features():
            features_to_plot.append(
                {
                    'type': 'univariate',
                    'info': info,
                }
            )

        # Add mixed bivariate pairs
        for pair_info in self.bundle.bivariate_pairs():
            if pair_info.skipped:
                continue

            # Use the is_categorical tuple directly from FeaturePairInfo
            is_cat_i, is_cat_j = pair_info.is_categorical

            # Only add mixed pairs (one categorical, one continuous)
            if is_cat_i != is_cat_j:
                features_to_plot.append(
                    {
                        'type': 'mixed',
                        'info': pair_info,
                    }
                )

        if not features_to_plot:
            logger.warning("No features to plot for response plots")
            return plt.figure()

        # Distribute features to pages
        if features_per_plot is None:
            pages = [features_to_plot]
        else:
            pages = [
                features_to_plot[i : i + features_per_plot]
                for i in range(0, len(features_to_plot), features_per_plot)
            ]

        # Render each page
        figures = []
        for page_features in pages:
            fig = self._render_response_page(page_features, subfig_size)
            figures.append(fig)

        return figures[0] if len(figures) == 1 else figures

    def _render_response_page(self, features: List[Dict], subfig_size: float) -> plt.Figure:
        """Render a page of response plots.

        Args:
            features: List of feature dictionaries to plot
            subfig_size: Size of individual subplots

        Returns:
            Matplotlib Figure
        """
        # Collect non-binary categorical features that need legend entries
        # (binary features use text labels directly in the plots)
        categorical_features = set()
        for feature_dict in features:
            feature_type = feature_dict['type']
            info = feature_dict['info']

            if feature_type == 'univariate' and info.is_categorical:
                # Check if binary - if so, skip (text labels used in plot)
                original_data = self.bundle.x_data[:, info.index]
                categories = np.unique(original_data)
                is_binary = (
                    len(categories) == 2
                    and any(np.isclose(categories, 0))
                    and any(np.isclose(categories, 1))
                )
                if not is_binary:
                    categorical_features.add((info.label, info.name, info.index))
            elif feature_type == 'mixed':
                # Add the categorical feature from the mixed pair (if not binary)
                is_cat_i, is_cat_j = info.is_categorical
                cat_idx = 0 if is_cat_i else 1
                i, j = info.indices
                feat_idx = i if cat_idx == 0 else j

                # Check if binary
                original_data = self.bundle.x_data[:, feat_idx]
                categories = np.unique(original_data)
                is_binary = (
                    len(categories) == 2
                    and any(np.isclose(categories, 0))
                    and any(np.isclose(categories, 1))
                )
                if not is_binary:
                    categorical_features.add((info.labels[cat_idx], info.names[cat_idx], feat_idx))

        # Calculate how many key subplots we need based on content
        n_plots = len(features)
        n_key_subplots = 0
        key_entries = []

        if categorical_features:
            # Pre-compute key entries to know total line count
            key_entries = self._build_category_key_entries(categorical_features)
            lines_per_subplot = 14  # Target lines per subplot
            # Calculate actual number of subplots needed by simulating the distribution
            # (simple division underestimates because entries are kept together)
            n_key_subplots = self._count_category_key_subplots(key_entries, lines_per_subplot)

        total_plots = n_plots + n_key_subplots
        n_cols = 3
        n_rows = max(1, (total_plots + n_cols - 1) // n_cols)

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(n_cols * subfig_size, n_rows * subfig_size * 0.8)
        )
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        # Render each feature
        for idx, feature_dict in enumerate(features):
            ax = axes[idx]
            feature_type = feature_dict['type']
            info = feature_dict['info']

            if feature_type == 'univariate':
                if info.is_categorical:
                    self._render_categorical_response_with_histogram(ax, info)
                else:
                    self._render_continuous_response_with_histogram(ax, info)
            elif feature_type == 'mixed':
                self._render_mixed_response_with_histogram(ax, info)

        # Add category key subplots if needed
        if n_key_subplots > 0:
            key_axes = [axes[n_plots + i] for i in range(n_key_subplots)]
            self._render_category_key(key_axes, key_entries)

        # Remove unused subplots
        start_remove = n_plots + n_key_subplots
        for idx in range(start_remove, len(axes)):
            fig.delaxes(axes[idx])

        fig.tight_layout()
        return fig

    def _build_category_key_entries(
        self, categorical_features: set
    ) -> List[Tuple[str, List[str]]]:
        """Build category key entries for all categorical features.

        Args:
            categorical_features: Set of (label, name, index) tuples

        Returns:
            List of (feature_label, [mapping_lines]) tuples
        """
        entries = []
        for label, name, feat_idx in sorted(categorical_features, key=lambda x: x[0]):
            # Get unique values for this feature from the data
            original_data = self.bundle.x_data[:, feat_idx]
            categories = np.unique(original_data)

            # Check if binary (should already be excluded, but double-check)
            is_binary = (
                len(categories) == 2
                and any(np.isclose(categories, 0))
                and any(np.isclose(categories, 1))
            )
            if is_binary:
                continue

            # Build mapping lines, but only if labels differ from integer values
            mapping_lines = []
            has_meaningful_labels = False
            for cat_val in categories:
                cat_label = self._format_category_label(name, cat_val, is_binary=False)
                # Replace newlines with spaces for compact display
                cat_label = cat_label.replace('\n', ' ')
                int_val = int(round(cat_val))  # Use round() to handle float imprecision
                # Check if label is meaningfully different from the integer
                # (not just the same number formatted as string)
                if cat_label != str(int_val) and cat_label != f"{cat_val:.4g}":
                    has_meaningful_labels = True
                mapping_lines.append(f"  {int_val} = {cat_label}")

            # Only include this feature if it has meaningful text labels
            if has_meaningful_labels:
                entries.append((label, mapping_lines))

        return entries

    def _count_category_key_subplots(
        self, key_entries: List[Tuple[str, List[str]]], lines_per_subplot: int = 14
    ) -> int:
        """Count actual number of subplots needed for category key.

        This simulates the distribution logic in _render_category_key to get
        an accurate count. Simple division (total_lines / lines_per_subplot)
        underestimates because entries are kept together and won't be split
        across subplots.

        Args:
            key_entries: List of (feature_label, [mapping_lines]) tuples
            lines_per_subplot: Maximum lines per subplot

        Returns:
            Number of subplots needed
        """
        if not key_entries:
            return 0

        n_subplots = 0
        current_line_count = 0

        for label, mapping_lines in key_entries:
            # Calculate lines needed: header + mappings + blank
            entry_line_count = 1 + len(mapping_lines) + 1

            # If adding this entry would exceed limit and we have content,
            # start a new subplot
            if (
                current_line_count > 0
                and current_line_count + entry_line_count > lines_per_subplot
            ):
                n_subplots += 1
                current_line_count = 0

            current_line_count += entry_line_count

        # Count the final subplot with remaining content
        if current_line_count > 0:
            n_subplots += 1

        return n_subplots

    def _render_category_key(
        self, axes: List[plt.Axes], key_entries: List[Tuple[str, List[str]]]
    ) -> None:
        """Render category key across one or more subplots.

        Args:
            axes: List of matplotlib axes to render on
            key_entries: List of (feature_label, [mapping_lines]) tuples
        """
        lines_per_subplot = 14  # Target lines per subplot

        # Group entries into subplots, keeping each category together
        subplot_contents = []
        current_subplot_lines = []
        current_line_count = 0

        for label, mapping_lines in key_entries:
            # Calculate lines needed for this entry (header + mappings + blank)
            entry_lines = [f"{label}:"] + mapping_lines + [""]
            entry_line_count = len(entry_lines)

            # If adding this entry would exceed limit and we have content,
            # start a new subplot (but always add at least one entry per subplot)
            if (
                current_line_count > 0
                and current_line_count + entry_line_count > lines_per_subplot
            ):
                # Remove trailing blank line before saving
                if current_subplot_lines and current_subplot_lines[-1] == "":
                    current_subplot_lines.pop()
                subplot_contents.append(current_subplot_lines)
                current_subplot_lines = []
                current_line_count = 0

            current_subplot_lines.extend(entry_lines)
            current_line_count += entry_line_count

        # Add remaining content
        if current_subplot_lines:
            if current_subplot_lines[-1] == "":
                current_subplot_lines.pop()
            subplot_contents.append(current_subplot_lines)

        # Render each subplot
        for ax_idx, ax in enumerate(axes):
            ax.axis('off')

            # Title only on first subplot
            if ax_idx == 0:
                ax.set_title('Category Key', fontsize=10, fontweight='bold', loc='left')

            # Get content for this subplot
            if ax_idx < len(subplot_contents):
                text = '\n'.join(subplot_contents[ax_idx])
                ax.text(
                    0.02,
                    0.98,
                    text,
                    transform=ax.transAxes,
                    fontsize=8,
                    fontfamily='sans-serif',
                    verticalalignment='top',
                    horizontalalignment='left',
                )
