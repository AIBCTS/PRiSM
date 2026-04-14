"""
Plot formatting utilities for NomogramRenderer.

This module contains:
- PlotStyle: Base class for plot styling configuration
- PlotFormatter: Handles plot formatting with integrated feature formatting
- Tick formatters: Smart numerical formatting for axes
- Response value formatters: Range-aware formatting for log odds/odds ratios
- Smart tick positioning: Uses matplotlib's MaxNLocator for "nice" tick values

Note: The formatters use feature-name-aware logic to handle special cases like
"age", "year", "duration" which are logged when special formatting is applied.
"""

import logging
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter, MaxNLocator

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# Feature Name Detection for Smart Formatting
# =============================================================================

# Keywords that indicate integer-friendly features (no decimals preferred)
INTEGER_FEATURE_KEYWORDS = frozenset(['age', 'year', 'count', 'number', 'num_', 'n_'])

# Keywords that indicate duration/time features (may need decimal)
DURATION_FEATURE_KEYWORDS = frozenset(['duration', 'months', 'days', 'weeks', 'time'])


def _is_integer_feature(feature_name: str) -> bool:
    """Check if feature name suggests integer values are preferred.

    Args:
        feature_name: Name of the feature

    Returns:
        True if feature should prefer integer formatting
    """
    name_lower = feature_name.lower()
    return any(keyword in name_lower for keyword in INTEGER_FEATURE_KEYWORDS)


def _is_duration_feature(feature_name: str) -> bool:
    """Check if feature name suggests duration/time values.

    Args:
        feature_name: Name of the feature

    Returns:
        True if feature is a duration/time feature
    """
    name_lower = feature_name.lower()
    return any(keyword in name_lower for keyword in DURATION_FEATURE_KEYWORDS)


# Default maximum label length for truncation
DEFAULT_MAX_LABEL_LENGTH = 35


def truncate_label(text: str, max_length: int = DEFAULT_MAX_LABEL_LENGTH) -> str:
    """Truncate a label to a maximum length, preserving the last character.

    If the text exceeds max_length, it is truncated to (max_length - 4) characters
    followed by '...' and the last character of the original string.

    Args:
        text: The label text to potentially truncate
        max_length: Maximum allowed length (default: 35)

    Returns:
        Original text if within limit, otherwise truncated with '...x' suffix
        where x is the last character of the original text.

    Examples:
        >>> truncate_label("Short label")  # Returns unchanged
        'Short label'
        >>> truncate_label("A" * 50)  # 50 chars -> truncated
        'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA...A'  # 35 chars total
    """
    if len(text) <= max_length:
        return text

    # Truncate to (max_length - 4) chars + "..." + last char = max_length total
    last_char = text[-1]
    truncated = text[: max_length - 4] + "..." + last_char
    return truncated


# =============================================================================
# Smart Tick Formatters
# =============================================================================


def format_value(x: float, feature_name: str = "") -> str:
    """Format a single numeric value intelligently based on its magnitude.

    This is the core formatting function used by all tick formatters.
    It handles special cases like:
    - Zero values
    - Integer-preferring features (age, year, count)
    - Duration features
    - Very small values (scientific notation)
    - General continuous values (adaptive decimal places)

    Args:
        x: Value to format
        feature_name: Optional feature name for context-aware formatting

    Returns:
        Formatted string representation
    """
    # Handle zero specially
    if x == 0:
        return "0"

    # Check for integer-preferring features
    if feature_name and _is_integer_feature(feature_name):
        # For integer features, round to nearest integer if close
        if abs(x - round(x)) < 0.01:
            return f"{int(round(x))}"
        # Otherwise allow one decimal for features like "age in years"
        return f"{x:.1f}".rstrip('0').rstrip('.')

    # Standard magnitude-based formatting
    abs_x = abs(x)
    if abs_x >= 10:
        # Large values: no decimals
        return f"{x:.0f}"
    elif abs_x >= 1:
        # Medium values: up to 1 decimal
        return f"{x:.1f}".rstrip('0').rstrip('.')
    elif abs_x >= 0.01:
        # Small values: up to 2 decimals
        return f"{x:.2f}".rstrip('0').rstrip('.')
    else:
        # Very small values: scientific notation
        return f"{x:.2e}"


def create_continuous_formatter(feature_name: str = "") -> FuncFormatter:
    """Create a formatter for continuous variable axes.

    The formatter adapts to value magnitude and uses feature-name-aware
    formatting when feature_name contains keywords like 'age', 'year', etc.

    When special formatting is applied based on feature name, a debug log
    message is emitted.

    Args:
        feature_name: Name of the feature being plotted (for smart formatting)

    Returns:
        A matplotlib FuncFormatter that formats ticks appropriately
    """
    # Log if special feature-aware formatting will be used
    if feature_name and _is_integer_feature(feature_name):
        logger.debug(f"Using integer-friendly formatting for feature: '{feature_name}'")

    def format_function(x: float, pos: int) -> str:
        return format_value(x, feature_name)

    return FuncFormatter(format_function)


def create_log_formatter() -> FuncFormatter:
    """Create a formatter for log scale axis ticks (odds ratio scale).

    Returns:
        A matplotlib FuncFormatter that formats log scale ticks appropriately,
        using scientific notation for values below 0.01 and above 99
    """

    def format_function(x: float, pos: int) -> str:
        if x < 0.01 or x > 99:
            return f"{x:.2e}"
        elif abs(x) >= 10:
            return f"{x:.0f}"
        elif abs(x) >= 1:
            return f"{x:.1f}".rstrip('0').rstrip('.')
        else:
            return f"{x:.2f}".rstrip('0').rstrip('.')

    return FuncFormatter(format_function)


# =============================================================================
# Response Value Formatters (Range-Aware)
# =============================================================================


def calculate_required_precision(vmin: float, vmax: float) -> int:
    """Determine decimal places needed to distinguish values in range [vmin, vmax].

    Calculates the optimal number of decimal places based on the data range span.
    Tighter ranges require more precision to distinguish nearby values.

    Args:
        vmin: Minimum value in the data range
        vmax: Maximum value in the data range

    Returns:
        Number of decimal places (0-4) needed for clear value distinction

    Examples:
        >>> calculate_required_precision(0, 15)
        0  # Wide range -> no decimals needed

        >>> calculate_required_precision(0.95, 1.05)
        2  # Tight range near 1 -> 2 decimals: "0.95", "1.00", "1.05"

        >>> calculate_required_precision(0.985, 1.015)
        3  # Very tight range -> 3 decimals: "0.985", "1.000", "1.015"
    """
    # Handle edge case: all values identical
    if vmin == vmax:
        return 2  # Default precision for consistency

    # Calculate the range span
    value_range = abs(vmax - vmin)

    # Determine precision based on range magnitude:
    # - Very tight ranges (< 0.01) need 3-4 decimals
    # - Tight ranges (< 0.1) need 2-3 decimals
    # - Moderate ranges (< 1) need 1-2 decimals
    # - Wide ranges (>= 1) need 0-1 decimals

    if value_range >= 10:
        return 0  # Range like [0, 15] -> "0", "5", "10", "15"
    elif value_range >= 1:
        return 1  # Range like [0.5, 2.5] -> "0.5", "1.0", "1.5", "2.0"
    elif value_range >= 0.1:
        return 2  # Range like [0.95, 1.05] -> "0.95", "1.00", "1.05"
    elif value_range >= 0.01:
        return 3  # Range like [0.985, 1.015] -> "0.985", "1.000", "1.015"
    else:
        return 4  # Very tight range -> "0.9985", "1.0000", "1.0015"


def format_value_adaptive(x: float, precision: int) -> str:
    """Format a value with specified precision, handling special cases.

    Uses range-aware precision calculated from the full data range to ensure
    all values in a heatmap/contour are distinguishable. Handles extreme values
    with scientific notation using 1 decimal place.

    Args:
        x: Value to format
        precision: Number of decimal places (from calculate_required_precision)

    Returns:
        Formatted string representation with appropriate precision

    Examples:
        >>> format_value_adaptive(0.98, precision=2)
        '0.98'

        >>> format_value_adaptive(1.0, precision=2)
        '1'  # Trailing zeros removed

        >>> format_value_adaptive(0.0005, precision=2)
        '5.0e-4'  # Extreme value -> scientific notation with 1 decimal

        >>> format_value_adaptive(1500, precision=2)
        '1.5e+3'  # Extreme value -> scientific notation
    """
    # Special case: exact zero
    if x == 0:
        return "0"

    # Scientific notation for extreme values (|x| < 0.001 or |x| > 1000)
    # User requested 1 decimal in scientific notation (e.g., 5.2e-3, not 5.21e-3)
    if abs(x) < 0.001 or abs(x) > 1000:
        return f"{x:.1e}"

    # Standard formatting with adaptive precision
    formatted = f"{x:.{precision}f}"

    # Remove trailing zeros and decimal point if not needed
    if precision > 0:
        formatted = formatted.rstrip('0').rstrip('.')

    return formatted


def format_value_for_annotation(x: float, precision: int) -> str:
    """Format value for heatmap cell annotations with threshold notation for extremes.

    Uses threshold notation (e.g., ">1000", "~0") instead of scientific notation
    for extreme values to improve readability in small annotation boxes.

    Args:
        x: Value to format
        precision: Number of decimal places (from calculate_required_precision)

    Returns:
        Formatted string with threshold notation for extremes

    Examples:
        >>> format_value_for_annotation(0.0005, precision=2)
        '~0'

        >>> format_value_for_annotation(1500, precision=2)
        '>1000'

        >>> format_value_for_annotation(-2000, precision=2)
        '<-1000'

        >>> format_value_for_annotation(1.02, precision=2)
        '1.02'
    """
    # Handle exact zero
    if x == 0:
        return "0"

    # Threshold notation for extreme values
    if x > 1000:
        return ">1000"
    elif x < -1000:
        return "<-1000"
    elif abs(x) < 0.001:  # Very small values near zero
        return "~0"

    # Standard adaptive formatting for normal range
    formatted = f"{x:.{precision}f}"
    if precision > 0:
        formatted = formatted.rstrip('0').rstrip('.')
    return formatted


def create_response_value_formatter(vmin: float, vmax: float) -> FuncFormatter:
    """Create adaptive formatter for response values (log odds/odds ratio).

    This formatter calculates the optimal precision based on the response value range
    and creates a formatter function with that precision captured in a closure.
    Designed for response axes, heatmap colorbars, and axis tick labels. Uses
    scientific notation for extreme values (standard for axis labels).

    This provides unified response value formatting across heatmaps, nomograms,
    and response plots with range-aware precision ensuring consistency.

    Args:
        vmin: Minimum response value in the data
        vmax: Maximum response value in the data

    Returns:
        FuncFormatter instance that formats response values with range-aware precision

    Examples:
        >>> # For tight odds ratio range (0.98 to 1.02)
        >>> formatter = create_response_value_formatter(0.98, 1.02)
        >>> formatter(0.98, 0)  # Returns "0.98"
        >>> formatter(1.00, 1)  # Returns "1"
        >>> formatter(1.02, 2)  # Returns "1.02"

        >>> # For wide range (0.5 to 2.5)
        >>> formatter = create_response_value_formatter(0.5, 2.5)
        >>> formatter(0.5, 0)   # Returns "0.5"
        >>> formatter(1.0, 1)   # Returns "1"
        >>> formatter(2.0, 2)   # Returns "2"
    """
    # Calculate precision once based on range
    precision = calculate_required_precision(vmin, vmax)

    # Create formatter function with precision captured in closure
    def format_function(x: float, pos: int) -> str:
        return format_value_adaptive(x, precision)

    return FuncFormatter(format_function)


# =============================================================================
# Smart Tick Positioning
# =============================================================================


def get_nice_ticks(
    data_min: float,
    data_max: float,
    n_ticks: int = 5,
    feature_name: str = "",
    force_integers: bool = False,
) -> np.ndarray:
    """Generate aesthetically pleasing tick values using matplotlib's algorithm.

    This replaces `np.linspace(min, max, n)` which produces awkward values
    like [0, 21.75, 43.5, 65.25, 87] with nice values like [0, 20, 40, 60, 80].

    For integer-preferring features (age, year, count), forces integer ticks.

    Args:
        data_min: Minimum data value
        data_max: Maximum data value
        n_ticks: Approximate number of ticks desired (may vary slightly)
        feature_name: Optional feature name for context-aware tick generation
        force_integers: Force integer tick values regardless of feature name

    Returns:
        Array of nice tick values within [data_min, data_max] range

    Example:
        >>> get_nice_ticks(0, 87, 5)
        array([ 0., 20., 40., 60., 80.])

        >>> get_nice_ticks(0, 87, 5, "age")  # Integer-preferring
        array([ 0., 20., 40., 60., 80.])
    """
    # Determine if we should use integer ticks
    use_integers = force_integers or (feature_name and _is_integer_feature(feature_name))

    if use_integers:
        logger.debug(f"Using integer tick generation for feature: '{feature_name}'")

    # Use MaxNLocator to find nice tick locations
    locator = MaxNLocator(
        nbins=n_ticks,
        integer=use_integers,
        steps=[1, 2, 2.5, 5, 10],  # Allowed step multiples for nice numbers
        prune=None,  # Don't prune edge ticks
    )

    # Get raw tick values
    raw_ticks = locator.tick_values(data_min, data_max)

    # Filter ticks with relaxed tolerance to allow ticks slightly outside data range
    # This ensures we get 4-5 ticks rather than 3, with ticks near the edges
    data_range = data_max - data_min
    tolerance = data_range * 0.10  # Allow ticks up to 10% outside data range
    nice_ticks = raw_ticks[
        (raw_ticks >= data_min - tolerance) & (raw_ticks <= data_max + tolerance)
    ]

    # Ensure we have at least 3 ticks for good visual appearance
    if len(nice_ticks) < 3:
        # Fall back to include more ticks with even more relaxed tolerance
        tolerance = data_range * 0.20
        nice_ticks = raw_ticks[
            (raw_ticks >= data_min - tolerance) & (raw_ticks <= data_max + tolerance)
        ]

    # Final fallback: ensure at least 2 ticks
    if len(nice_ticks) < 2:
        nice_ticks = np.array([data_min, data_max])

    return nice_ticks


class PlotStyle:
    """Base class for plot styling configuration."""

    def __init__(self):
        # Plot dimensions and sizes
        self.marker_size: int = 20
        self.padding_ratio: float = 0.15
        self.margin_ratios = {'linear': 0.15, 'log': 0.10}
        self.line_widths = {'grid': 0.5, 'axis': 0.5, 'plot': 1.2}
        self.tick_params = {
            'major': {'length': 3, 'width': 1.0},
            'minor': {'length': 2, 'width': 0.5},
        }

        # Colors and transparency
        self.alphas = {'default': 0.3, 'grid': 0.5, 'minor_grid': 0.3, 'reference_line': 0.8}
        self.colors = {'grid': "#b8b8b8", 'line': 'blue'}

        # Font settings
        self.fonts = {
            'family': 'sans-serif',
            'sans': ['DejaVu Sans', 'Arial', 'Helvetica'],
            'sizes': {'label': 8, 'tick': 8, 'title': 10},
        }

        # Label positioning configuration (for categorical plots)
        # For nomograms with ylim=[0, 1] and y_value=0.5:
        self.label_offsets = {
            'base': 0.08,  # Base offset from marker line
            'level_spacing': 0.10,  # Vertical spacing between stagger levels
            'max_levels': 4,  # Maximum stagger levels
            'overlap_margin': 10.0,  # Horizontal margin in pixels
            'min_marker_spacing': 25.0,  # Min marker distance before staggering (pixels)
        }

    def get_tick_params(self, tick_type: str) -> Dict[str, float]:
        """Get tick parameters for major/minor ticks."""
        return self.tick_params[tick_type]

    def get_font_size(self, element_type: str) -> int:
        """Get font size for different elements."""
        return self.fonts['sizes'][element_type]

    def apply_global_defaults(self, ax: Optional[Axes] = None) -> None:
        """Apply default styling to matplotlib or specific axis."""
        plot_params = {
            'font.family': self.fonts['family'],
            'font.sans-serif': self.fonts['sans'],
            'axes.labelsize': self.get_font_size('label'),
            'xtick.labelsize': self.get_font_size('tick'),
            'ytick.labelsize': self.get_font_size('tick'),
            'axes.titlesize': self.get_font_size('title'),
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'xtick.major.size': self.tick_params['major']['length'],
            'ytick.major.size': self.tick_params['major']['length'],
            'figure.dpi': 150,
            'axes.linewidth': self.line_widths['axis'],
            'grid.linewidth': self.line_widths['grid'],
        }

        if ax is None:
            plt.rcParams.update(plot_params)
        else:
            ax.grid(
                True,
                alpha=self.alphas['grid'],
                color=self.colors['grid'],
                linewidth=self.line_widths['grid'],
            )
            ax.set_axisbelow(True)
            self._apply_tick_params(ax)

    def apply_defaults(self, ax: Axes) -> None:
        """Apply all styling to a specific matplotlib axis without global changes."""
        if ax is None:
            raise ValueError("ax parameter is required")

        # Get figure for figure-level settings
        fig = ax.get_figure()
        fig.set_dpi(150)  # Set DPI for this figure only

        # Grid properties
        ax.grid(
            True,
            alpha=self.alphas['grid'],
            color=self.colors['grid'],
            linewidth=self.line_widths['grid'],
        )
        ax.set_axisbelow(True)

        # Axis line widths
        for spine in ax.spines.values():
            spine.set_linewidth(self.line_widths['axis'])

        # Set comprehensive tick parameters
        common_tick_params = {
            'direction': 'in',
            'labelsize': self.get_font_size('tick'),
            'labelcolor': 'black',
            'labelrotation': 0,
        }

        # Major ticks
        ax.tick_params(
            axis='both',
            which='major',
            length=self.tick_params['major']['length'],
            width=self.tick_params['major']['width'],
            **common_tick_params,
        )

        # Minor ticks
        ax.tick_params(
            axis='both',
            which='minor',
            length=self.tick_params['minor']['length'],
            width=self.tick_params['minor']['width'],
            **common_tick_params,
        )

        # Ensure consistent tick sizes on both axes
        ax.xaxis.set_tick_params(which='major', size=self.tick_params['major']['length'])
        ax.yaxis.set_tick_params(which='major', size=self.tick_params['major']['length'])

        # Set default label sizes and other text properties
        ax.tick_params(axis='both', labelsize=self.get_font_size('tick'))
        ax.xaxis.set_tick_params(labelsize=self.get_font_size('tick'))
        ax.yaxis.set_tick_params(labelsize=self.get_font_size('tick'))

        text_props = {
            'fontfamily': self.fonts['family'],
            'fontname': self.fonts['sans'][0],  # Use first sans-serif font
        }

        # Set label properties with defaults (even if no label exists yet)
        ax.set_xlabel(ax.get_xlabel() or '', fontsize=self.get_font_size('label'), **text_props)
        ax.set_ylabel(ax.get_ylabel() or '', fontsize=self.get_font_size('label'), **text_props)
        ax.set_title(ax.get_title() or '', fontsize=self.get_font_size('title'), **text_props)

        # Tick label fonts are now handled in tick_params

    def _apply_tick_params(self, ax: Axes) -> None:
        """Apply tick parameters to an axis."""
        major_params = self.get_tick_params('major')
        minor_params = self.get_tick_params('minor')

        ax.tick_params(
            axis='both',
            which='major',
            length=major_params['length'],
            width=major_params['width'],
        )
        ax.tick_params(
            axis='both',
            which='minor',
            length=minor_params['length'],
            width=minor_params['width'],
        )

    def calculate_axis_margins(
        self, values: np.ndarray, margin_ratio: Optional[float] = None, scale: str = 'linear'
    ) -> Tuple[float, float]:
        """Calculate axis limits with appropriate margins.

        Args:
            values: Array of values to calculate margins for
            margin_ratio: Optional custom margin ratio (default: uses instance defaults)
            scale: Scale type ('linear' or 'log')

        Returns:
            Tuple of (min_value, max_value) with margins applied
        """
        if margin_ratio is None:
            margin_ratio = self.margin_ratios['linear' if scale == 'linear' else 'log']

        if scale == 'linear':
            value_range = values.max() - values.min()
            margin = margin_ratio * value_range
            return values.min() - margin, values.max() + margin
        else:
            min_val = values.min()
            max_val = values.max()
            return (min_val / (1 + margin_ratio), max_val * (1 + margin_ratio))

    def apply_categorical_style(self, ax: Axes) -> None:
        """Apply styling specific to categorical plots."""
        ax.scatter([], [], s=self.marker_size, zorder=3, marker='_', linewidth=3)
        ax.grid(
            True,
            color=self.colors['grid'],
            linewidth=self.line_widths['grid'],
            alpha=self.alphas['grid'],
        )
        ax.set_axisbelow(True)

    def apply_continuous_style(self, ax: Axes) -> None:
        """Apply styling specific to continuous plots."""
        ax.plot([], [], color=self.colors['line'], linewidth=self.line_widths['plot'])
        ax.grid(
            True,
            color=self.colors['grid'],
            linewidth=self.line_widths['grid'],
            alpha=self.alphas['grid'],
        )
        ax.set_axisbelow(True)


class PlotFormatter(PlotStyle):
    """Handles plot formatting with integrated feature formatting."""

    def __init__(
        self,
        use_odds_ratio: bool = False,
        binary_labels: Optional[Union[Dict[int, str], List[Dict[int, str]]]] = None,
        categorical_labels: Optional[
            Union[Dict[str, Dict[float, str]], List[Dict[str, Dict[float, str]]]]
        ] = None,
    ):
        super().__init__()
        self.use_odds_ratio = use_odds_ratio

        self.is_comparison = isinstance(binary_labels, Sequence) or isinstance(
            categorical_labels, Sequence
        )

        # For single dictionary case in comparison mode, replicate it for all models
        if self.is_comparison:
            # Handle binary labels
            if isinstance(binary_labels, Sequence):
                self.binary_labels = [labels or {0: "No", 1: "Yes"} for labels in binary_labels]
            else:
                # Single dictionary - use same labels for all models
                default_binary = binary_labels or {0: "No", 1: "Yes"}
                self.binary_labels = [
                    default_binary
                    for _ in range(
                        len(categorical_labels) if isinstance(categorical_labels, Sequence) else 1
                    )
                ]

            # Handle categorical labels
            if isinstance(categorical_labels, Sequence):
                self.categorical_labels = [labels or {} for labels in categorical_labels]
            else:
                # Single dictionary - use same labels for all models
                default_categorical = categorical_labels or {}
                self.categorical_labels = [
                    default_categorical
                    for _ in range(
                        len(binary_labels) if isinstance(binary_labels, Sequence) else 1
                    )
                ]
        else:
            # Non-comparison mode - use single dictionary as is
            self.binary_labels = binary_labels or {0: "No", 1: "Yes"}
            self.categorical_labels = categorical_labels or {}

    def format_feature_label(
        self,
        feature_name: str,
        value: float,
        is_binary: bool = False,
        precision: int = 2,
        model_index: Optional[int] = None,
    ) -> str:
        """Format feature value label (integrated from FeatureFormatter)."""

        # Get appropriate labels based on whether this is a comparison plot
        if self.is_comparison and model_index is not None:
            categorical_labels = self.categorical_labels[model_index]
            binary_labels = self.binary_labels[model_index]
        else:
            categorical_labels = self.categorical_labels
            binary_labels = self.binary_labels

        if feature_name in categorical_labels:
            key = int(np.rint(value))
            label_dict = categorical_labels[feature_name]
            # Use .get() to gracefully handle missing labels (e.g., when reference
            # category was dropped during one-hot encoding and value 0 has no label)
            val = label_dict.get(key)
            if val is not None:
                return truncate_label(val)
            # Fall back to numeric representation if label not found
            logger.info(
                f"No label found for '{feature_name}' category {key}. "
                f"Using numeric value. To add labels, provide reference_columns "
                f"to OneHotGroupManager or pass categorical_labels to the plot function."
            )
            return f"{value:.{precision}g}"
        elif is_binary:
            if np.isclose(value, 0):
                return binary_labels[0]
            elif np.isclose(value, 1):
                return binary_labels[1]
        return f"{value:.{precision}g}"

    def _configure_axes(
        self,
        ax: Axes,
        is_first: bool,
        is_last: bool,
        y_axis_position: str,
        y_label_position: str,
        x_axis_position: str,
        x_ticks_position: str,
        x_tick_labels_position: str,
        x_label_position: str,
    ) -> None:
        """Configure axis visibility and positioning."""
        # Configure y-axis position
        ax.spines['left'].set_visible(y_axis_position in ['left', 'both'])
        ax.spines['right'].set_visible(y_axis_position in ['right', 'both'])
        ax.yaxis.set_ticks_position(y_axis_position)
        ax.yaxis.set_label_position(y_label_position)

        # For middle subplots, hide all x-axis elements
        if not (is_first or is_last):
            ax.tick_params(
                axis='x', which='both', bottom=False, top=False, labelbottom=False, labeltop=False
            )
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            return

        # Handle x-axis elements for first/last subplots
        if is_first:
            ax.spines['top'].set_visible(x_axis_position in ['top', 'both'])
            ax.spines['bottom'].set_visible(False)
            show_ticks = x_ticks_position in ['top', 'both']
            show_labels = x_tick_labels_position in ['top', 'both']
            ax.tick_params(
                axis='x',
                which='both',
                top=show_ticks,
                bottom=False,
                labeltop=show_labels,
                labelbottom=False,
            )
            if x_label_position in ['top', 'both']:
                ax.xaxis.set_label_position('top')

        elif is_last:
            ax.spines['bottom'].set_visible(x_axis_position in ['bottom', 'both'])
            ax.spines['top'].set_visible(False)
            show_ticks = x_ticks_position in ['bottom', 'both']
            show_labels = x_tick_labels_position in ['bottom', 'both']
            ax.tick_params(
                axis='x',
                which='both',
                bottom=show_ticks,
                top=False,
                labelbottom=show_labels,
                labeltop=False,
            )
            if x_label_position in ['bottom', 'both']:
                ax.xaxis.set_label_position('bottom')

    def _set_labels(
        self,
        ax: Axes,
        title: Optional[str],
        xlabel: Optional[str],
        ylabel: Optional[str],
        is_first: bool = False,
        is_last: bool = False,
        x_label_position: str = 'bottom',
    ) -> None:
        """Set axis labels and title."""
        if title:
            ax.set_title(title)
        if xlabel:
            # Handle x-label positioning based on subplot position
            if x_label_position == 'bottom' and is_last:
                ax.set_xlabel(xlabel)
            elif x_label_position == 'top' and is_first:
                ax.set_xlabel(xlabel)
            elif x_label_position == 'both':
                if is_first or is_last:
                    ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

    def format_subplot(
        self,
        ax: Axes,
        is_first: bool = False,
        is_last: bool = False,
        y_axis_position: str = 'left',
        y_label_position: str = 'left',
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        x_axis_position: str = 'bottom',
        x_ticks_position: str = 'bottom',
        x_tick_labels_position: str = 'bottom',
        x_label_position: str = 'bottom',
    ) -> None:
        """Format subplot with integrated styling."""
        self.apply_defaults(ax)

        # Set default x label if not provided
        if xlabel is None:
            xlabel = "Odds ratio" if self.use_odds_ratio else "Log odds ratio"

        self._configure_axes(
            ax,
            is_first,
            is_last,
            y_axis_position,
            y_label_position,
            x_axis_position,
            x_ticks_position,
            x_tick_labels_position,
            x_label_position,
        )

        self._set_labels(ax, title, xlabel, ylabel, is_first, is_last, x_label_position)

    def get_log_tick_locations(self, min_x: float, max_x: float) -> List[float]:
        """Calculate tick locations for log scale.

        Args:
            min_x: Minimum x value (must be positive)
            max_x: Maximum x value (must be positive)

        Returns:
            List of tick locations

        Raises:
            ValueError: If input values are invalid for log scale
        """
        # Validate inputs are positive and finite
        if not (np.isfinite(min_x) and np.isfinite(max_x)):
            raise ValueError(f"Log scale requires finite values, got min_x={min_x}, max_x={max_x}")
        if min_x <= 0 or max_x <= 0:
            raise ValueError(
                f"Log scale requires positive values, got min_x={min_x}, max_x={max_x}"
            )

        # Calculate log10 orders with validation
        min_order = np.floor(np.log10(min_x))
        max_order = np.ceil(np.log10(max_x))

        if not (np.isfinite(min_order) and np.isfinite(max_order)):
            raise ValueError(
                f"Invalid values for log scale, got min_order={min_order}, max_order={max_order}"
            )

        try:
            tick_locations = [10**i for i in range(int(min_order), int(max_order) + 1)]
            if max_order - min_order <= 2:
                tick_locations += [x * 5 for x in tick_locations[:-1]]

            return sorted([x for x in tick_locations if min_x <= x <= max_x])
        except Exception as e:
            raise ValueError(
                f"Error generating tick locations: {e}, min_x={min_x}, max_x={max_x}, min_order={min_order}, max_order={max_order}"
            )

    def set_x_scale_and_limits(
        self, ax: Axes, min_x: float, max_x: float, skip_margins: bool = False
    ) -> None:
        """Set x-axis scale and limits with appropriate padding.

        Args:
            ax: Matplotlib axes to configure
            min_x: Minimum x value
            max_x: Maximum x value
            skip_margins: If True, use exact limits without margin calculation.
                        Used for comparison plots where margins are pre-calculated.
        """
        if not skip_margins:
            values = np.array([min_x, max_x])
            scale = 'log' if self.use_odds_ratio else 'linear'
            min_x, max_x = self.calculate_axis_margins(values, scale=scale)

        if self.use_odds_ratio:
            ax.set_xscale('log')
            ax.set_xticks(self.get_log_tick_locations(min_x, max_x))
            # Use log formatter for odds ratio scale
            ax.xaxis.set_major_formatter(create_log_formatter())
        else:
            # Use continuous formatter for linear scale
            ax.xaxis.set_major_formatter(create_continuous_formatter(""))

        ax.set_xlim(min_x, max_x)

    def _add_legend(
        self,
        ax: Axes,
        title: str,
        loc: str = 'lower left',
        bbox_to_anchor: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Add a formatted legend to the plot."""
        legend = ax.legend(
            title=title,
            fontsize=self.get_font_size('tick'),
            loc=loc,
            bbox_to_anchor=bbox_to_anchor or (0.05, 0.05),
            borderaxespad=0.0,
        )

        legend.get_frame().set_alpha(0.7)
        legend.get_frame().set_edgecolor('none')

        plt.setp(
            legend.get_title(),
            fontsize=self.get_font_size('tick'),
            fontfamily=self.fonts['family'],
            fontweight=plt.rcParams['axes.labelweight'],
        )
        plt.setp(legend.get_texts(), fontfamily=self.fonts['family'])

    def style_legend(self, ax: Axes, loc: str = 'lower right', font_scale: float = 0.9) -> None:
        """Apply consistent legend styling.

        Args:
            ax: The matplotlib axes to style the legend for
            loc: Legend location (default: 'lower right')
            font_scale: Scaling factor for font size relative to tick size (default: 0.9)
        """
        legend = ax.legend(loc=loc, fontsize=self.get_font_size('tick') * font_scale)
        plt.setp(legend.get_title(), fontsize=self.get_font_size('tick') * font_scale)

    def format_categorical_plot(
        self,
        ax: Axes,
        response: np.ndarray,
        y_value: float,
        line_color: Optional[str],
        feature_name: str,
        feature_values: np.ndarray,
        is_binary: bool = False,
        label: Optional[str] = None,
        model_index: Optional[int] = None,
    ) -> None:
        """Apply standard categorical plot formatting."""
        self.apply_categorical_style(ax)
        # Draw line and markers for current category with consistent color
        (line,) = ax.plot(response, np.full_like(response, y_value), label=label)
        final_color = line_color if line_color is not None else line.get_color()
        line.set_color(final_color)

        # Add vertical line markers for categorical values
        ax.scatter(
            response,
            np.full_like(response, y_value),
            marker="|",
            color=final_color,
            s=self.marker_size * 4,
            linewidth=1,
            zorder=3,
        )

        # Sort by response value to alternate labels based on x-position
        sorted_indices = np.argsort(response)

        # Collect all labels
        all_labels = []
        for i in sorted_indices:
            value = feature_values[i]
            label = self.format_feature_label(
                feature_name, value, is_binary, model_index=model_index
            )
            all_labels.append(label)

        # Create initial annotations with horizontal, centered labels
        annotations_above, annotations_below = self._create_initial_annotations(
            ax, sorted_indices, response, y_value, all_labels, feature_name, model_index
        )

        # Force rendering and get bounding boxes
        self._force_rendering(ax)

        # Detect overlaps and assign stagger levels
        self._assign_stagger_levels(annotations_above, annotations_below)

        ax.set_yticks([])
        ax.set_ylabel(truncate_label(feature_name), rotation=90, loc="center", labelpad=5)

    def _bboxes_overlap_horizontal(self, bbox1, bbox2, margin: float = 5.0) -> bool:
        """Check if two bounding boxes overlap horizontally in display coordinates.

        Args:
            bbox1, bbox2: Bounding boxes in display coordinates (pixels)
            margin: Minimum gap in pixels (adds safety margin)

        Returns:
            True if boxes overlap (with margin)
        """
        # bbox coordinates: x0, y0, x1, y1 (left, bottom, right, top)
        # Overlap occurs if: bbox1.x1 + margin > bbox2.x0 AND bbox2.x1 + margin > bbox1.x0
        return (bbox1.x1 + margin > bbox2.x0) and (bbox2.x1 + margin > bbox1.x0)

    def _force_rendering(self, ax: Axes) -> None:
        """Force matplotlib to render the figure to compute bounding boxes.

        This is required before calling get_window_extent() on annotations.
        Uses the figure's canvas renderer.
        """
        fig = ax.get_figure()

        # Try to get existing renderer, or create one by drawing
        try:
            fig.canvas.get_renderer()
        except AttributeError:
            # Fallback: draw the canvas to create renderer
            fig.canvas.draw()
            fig.canvas.get_renderer()

    def _create_initial_annotations(
        self,
        ax: Axes,
        sorted_indices: np.ndarray,
        response: np.ndarray,
        y_value: float,
        all_labels: List[str],
        feature_name: str,
        model_index: Optional[int],
    ) -> Tuple[List[Dict], List[Dict]]:
        """Create annotations with horizontal, centered text at base offset levels.

        Args:
            ax: Matplotlib axes
            sorted_indices: Indices sorted by response value
            response: Response values for each category
            y_value: Y-coordinate for the horizontal line
            all_labels: Formatted label strings for each category
            feature_name: Name of the feature (for logging)
            model_index: Model index for comparison plots

        Returns:
            Tuple of (annotations_above, annotations_below) as lists of dicts.
            Each dict contains: {index, x_pos, label_text, annotation, bbox, assigned_level}
        """
        base_offset = self.label_offsets['base']
        annotations_above = []
        annotations_below = []

        for rank, i in enumerate(sorted_indices):
            label_text = all_labels[rank].replace('\n', ' ')
            x_pos = response[i]

            if rank % 2 == 0:  # Above
                y_offset = base_offset
                va = 'bottom'
                group = annotations_above
            else:  # Below
                y_offset = -base_offset
                va = 'top'
                group = annotations_below

            ann = ax.annotate(
                label_text,
                (x_pos, y_value),
                xytext=(x_pos, y_value + y_offset),
                ha='center',  # Changed from 'left'
                va=va,
                fontsize=self.get_font_size('tick'),
                fontfamily=self.fonts['family'],
                rotation=0,  # Always horizontal
            )

            label_info = {
                'index': i,
                'x_pos': x_pos,
                'label_text': label_text,
                'annotation': ann,
                'bbox': None,
                'assigned_level': 0,
            }
            group.append(label_info)

        return annotations_above, annotations_below

    def _assign_levels_to_group(
        self, label_infos: List[Dict], base_offset: float, direction: int
    ) -> None:
        """Assign stagger levels to a group of labels (all above or all below).

        Modifies label_infos in-place by updating 'assigned_level' and repositioning
        annotations to avoid overlaps.

        Args:
            label_infos: List of label info dicts
            base_offset: Base vertical offset (0.03 or -0.03)
            direction: +1 for above (increasing offset), -1 for below (decreasing)
        """
        if len(label_infos) <= 1:
            return  # No staggering needed

        max_levels = self.label_offsets['max_levels']
        level_spacing = self.label_offsets['level_spacing']
        overlap_margin = self.label_offsets['overlap_margin']

        # Get renderer and axes for coordinate transforms
        fig = label_infos[0]['annotation'].axes.get_figure()
        ax = label_infos[0]['annotation'].axes
        renderer = fig.canvas.get_renderer()

        # Update bboxes and get marker positions in display coordinates
        for info in label_infos:
            info['bbox'] = info['annotation'].get_window_extent(renderer)
            # Transform marker x position to display coordinates
            marker_pos_display = ax.transData.transform([(info['x_pos'], 0)])[0]
            info['marker_x_display'] = marker_pos_display[0]

        # Sort by x position
        label_infos.sort(key=lambda info: info['x_pos'])

        # Check for overlaps using both text bbox and marker proximity
        # Get configured minimum marker spacing
        min_marker_spacing = self.label_offsets.get('min_marker_spacing', 25.0)

        has_overlap = False
        for i in range(len(label_infos) - 1):
            # Check text bbox overlap
            text_overlap = self._bboxes_overlap_horizontal(
                label_infos[i]['bbox'], label_infos[i + 1]['bbox'], overlap_margin
            )
            # Check marker proximity (are markers too close?)
            marker_distance = abs(
                label_infos[i + 1]['marker_x_display'] - label_infos[i]['marker_x_display']
            )
            markers_too_close = marker_distance < min_marker_spacing

            if text_overlap or markers_too_close:
                has_overlap = True
                break

        if not has_overlap:
            return  # No staggering needed

        # Define levels
        levels = [base_offset + direction * i * level_spacing for i in range(max_levels)]

        # Track assignments
        level_assignments = {i: [] for i in range(max_levels)}

        # Greedy assignment
        for info in label_infos:
            assigned = False
            for level_idx in range(max_levels):
                # Check overlaps with labels at this level
                has_overlap_at_level = False
                for other_info in level_assignments[level_idx]:
                    if self._bboxes_overlap_horizontal(
                        info['bbox'], other_info['bbox'], overlap_margin
                    ):
                        has_overlap_at_level = True
                        break

                if not has_overlap_at_level:
                    info['assigned_level'] = level_idx
                    level_assignments[level_idx].append(info)
                    assigned = True
                    break

            # Fallback: use least crowded level
            if not assigned:
                least_crowded = min(
                    level_assignments.keys(), key=lambda k: len(level_assignments[k])
                )
                info['assigned_level'] = least_crowded
                level_assignments[least_crowded].append(info)

        # Reposition annotations and add leader lines
        ax = label_infos[0]['annotation'].axes

        # Check if any staggering occurred (any label not at base level)
        any_staggered = any(info['assigned_level'] > 0 for info in label_infos)

        for info in label_infos:
            new_y_offset = levels[info['assigned_level']]
            xy = info['annotation'].xy
            label_y = xy[1] + new_y_offset

            # Use set_position to update the xyann (text position) of the annotation
            info['annotation'].set_position((info['x_pos'], label_y))

            # Add leader line from label to marker (thin, subtle line)
            # If ANY staggering occurred, draw lines for ALL labels for consistency
            if any_staggered:
                ax.plot(
                    [info['x_pos'], info['x_pos']],  # Vertical line
                    [xy[1], label_y],  # From marker to label
                    color='gray',
                    linewidth=0.5,
                    alpha=0.75,
                    zorder=2,  # Below markers but above grid
                    clip_on=False,  # Don't clip at axes boundaries
                )

    def _assign_stagger_levels(
        self, annotations_above: List[Dict], annotations_below: List[Dict]
    ) -> None:
        """Detect overlaps and assign stagger levels to labels.

        Processes above and below groups independently. Modifies annotation
        positions in-place.

        Args:
            annotations_above: List of label info dicts for above labels
            annotations_below: List of label info dicts for below labels
        """
        # Process each group independently using configured base offset
        base = self.label_offsets['base']
        self._assign_levels_to_group(annotations_above, base_offset=base, direction=1)
        self._assign_levels_to_group(annotations_below, base_offset=-base, direction=-1)

    def format_continuous_plot(
        self,
        ax: Axes,
        response: np.ndarray,
        data_range: np.ndarray,
        feature_name: str,
        line_color: Optional[Union[str, np.ndarray]] = None,
        label: Optional[str] = None,
        num_ticks: int = 5,
        y_range: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Apply standard continuous plot formatting."""
        self.apply_continuous_style(ax)

        # matplotlib's plot() handles both string colors and RGBA arrays
        color = line_color

        (line,) = ax.plot(response, data_range, color=color, label=label)
        line_color = color if color is not None else line.get_color()

        # Set y-axis limits and ticks
        if y_range is not None:
            # y_range is raw range from comparison, calculate margins from it
            y_raw_min, y_raw_max = y_range
            y_ticks = get_nice_ticks(
                y_raw_min, y_raw_max, n_ticks=num_ticks, feature_name=feature_name
            )
            values_for_margins = np.array([y_raw_min, y_raw_max])
            logger.debug(f"Using raw y-range for ticks: [{y_raw_min}, {y_raw_max}]")
        else:
            # Use data range directly
            y_raw_min, y_raw_max = data_range.min(), data_range.max()
            y_ticks = get_nice_ticks(
                y_raw_min, y_raw_max, n_ticks=num_ticks, feature_name=feature_name
            )
            values_for_margins = data_range
            logger.debug(f"Using data range for ticks: [{y_raw_min}, {y_raw_max}]")

        # Calculate margins for limits in both cases
        y_min, y_max = self.calculate_axis_margins(values_for_margins)
        logger.debug(f"Setting y limits with margins: [{y_min}, {y_max}]")

        ax.set_ylim(y_min, y_max)
        ax.set_yticks(y_ticks)

        # Apply specialized continuous formatter for y-axis ticks
        ax.yaxis.set_major_formatter(create_continuous_formatter(feature_name))

        ax.set_ylabel(truncate_label(feature_name), rotation=90, loc="center", labelpad=5)

    def format_mixed_response_plot(
        self,
        ax: Axes,
        response_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
        cat_feature_name: str,
        cont_feature_name: str,
        legend_on_right: bool = False,
        set_xlabel: bool = True,
    ) -> None:
        """Apply standard mixed response plot formatting.

        Args:
            ax: Matplotlib axes to format
            response_data: Dict mapping category labels to (x_data, y_data) tuples
            cat_feature_name: Name of categorical feature (for legend title)
            cont_feature_name: Name of continuous feature (for y-axis label)
            legend_on_right: Place legend on right side
            set_xlabel: Ignored - x-axis label is always handled by format_subplot()
                to respect is_last positioning. Parameter kept for API compatibility.
        """
        first_values = next(iter(response_data.values()))[1]
        y_min, y_max = self.calculate_axis_margins(first_values)

        for label, (x_data, y_data) in response_data.items():
            (line,) = ax.plot(x_data, y_data, label=label, linewidth=self.line_widths['plot'])

        ax.set_ylim(y_min, y_max)
        # Note: x-axis label is handled by format_subplot() to respect is_last positioning
        ax.set_ylabel(truncate_label(cont_feature_name))

        # Apply specialized formatters for both axes
        ax.yaxis.set_major_formatter(create_continuous_formatter(cont_feature_name))
        if self.use_odds_ratio:
            ax.xaxis.set_major_formatter(create_log_formatter())
        else:
            ax.xaxis.set_major_formatter(create_continuous_formatter(""))

        legend_loc = 'lower right' if legend_on_right else 'lower left'
        anchor = (0.95, 0.05) if legend_on_right else (0.05, 0.05)
        self._add_legend(ax, cat_feature_name, loc=legend_loc, bbox_to_anchor=anchor)

    def apply_grid_settings(
        self,
        ax: Axes,
        use_x_grid: bool = True,
        use_y_grid: bool = True,
    ) -> None:
        """Apply consistent grid settings across all plots."""
        if use_x_grid:
            ax.grid(
                True,
                axis='x',
                color=self.colors['grid'],
                linewidth=self.line_widths['grid'],
                alpha=self.alphas['grid'],
            )
            if self.use_odds_ratio:
                ax.grid(
                    True,
                    axis='x',
                    which='minor',
                    color=self.colors['grid'],
                    linewidth=self.line_widths['grid'],
                    alpha=self.alphas['minor_grid'],
                )

        if use_y_grid:
            ax.grid(
                True,
                axis='y',
                color=self.colors['grid'],
                linewidth=self.line_widths['grid'],
                alpha=self.alphas['grid'],
            )

        ax.set_axisbelow(True)
