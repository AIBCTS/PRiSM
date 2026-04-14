"""
Tests for response value plotting formatters.

Tests cover:
- calculate_required_precision() - precision calculation based on data range
- format_value_adaptive() - adaptive value formatting with precision
- create_response_value_formatter() - matplotlib FuncFormatter creation
- Edge cases: extreme values, negative values, equal min/max, etc.
"""

import numpy as np
from matplotlib.ticker import FuncFormatter

from prism.plotting.formatter import (
    calculate_required_precision,
    create_response_value_formatter,
    format_value_adaptive,
    format_value_for_annotation,
)


class TestCalculateRequiredPrecision:
    """Test precision calculation based on data range."""

    def test_wide_range_10_or_more(self):
        """Wide range (>= 10) requires 0 decimals."""
        assert calculate_required_precision(0, 15) == 0
        assert calculate_required_precision(-5, 10) == 0
        assert calculate_required_precision(100, 150) == 0

    def test_moderate_range_1_to_10(self):
        """Moderate range ([1, 10)) requires 1 decimal."""
        assert calculate_required_precision(0.5, 2.5) == 1
        assert calculate_required_precision(-1, 1) == 1
        assert calculate_required_precision(0, 5) == 1

    def test_tight_range_0_1_to_1(self):
        """Tight range ([0.1, 1)) requires 2 decimals."""
        assert calculate_required_precision(0.95, 1.05) == 2
        assert calculate_required_precision(0, 0.5) == 2
        assert calculate_required_precision(-0.2, 0.3) == 2

    def test_very_tight_range_0_01_to_0_1(self):
        """Very tight range ([0.01, 0.1)) requires 3 decimals."""
        assert calculate_required_precision(0.985, 1.015) == 3
        assert calculate_required_precision(0, 0.05) == 3
        assert calculate_required_precision(-0.03, 0.02) == 3

    def test_extremely_tight_range_less_than_0_01(self):
        """Extremely tight range (< 0.01) requires 4 decimals."""
        assert calculate_required_precision(0.9985, 1.0015) == 4
        assert calculate_required_precision(0, 0.005) == 4
        assert calculate_required_precision(-0.002, 0.003) == 4

    def test_equal_values_returns_default(self):
        """Equal min/max returns default precision 2."""
        assert calculate_required_precision(1.0, 1.0) == 2
        assert calculate_required_precision(0, 0) == 2
        assert calculate_required_precision(-5.5, -5.5) == 2

    def test_negative_range(self):
        """Negative ranges work correctly (uses abs)."""
        # Range from -2 to -1 = range of 1
        assert calculate_required_precision(-2, -1) == 1
        # Range from -10 to 5 = range of 15
        assert calculate_required_precision(-10, 5) == 0

    def test_boundary_values(self):
        """Test boundary values for precision thresholds."""
        # Exactly 10 -> 0 decimals
        assert calculate_required_precision(0, 10) == 0
        # Just under 10 -> 1 decimal
        assert calculate_required_precision(0, 9.99) == 1

        # Exactly 1 -> 1 decimal
        assert calculate_required_precision(0, 1) == 1
        # Just under 1 -> 2 decimals
        assert calculate_required_precision(0, 0.99) == 2

        # Exactly 0.1 -> 2 decimals
        assert calculate_required_precision(0, 0.1) == 2
        # Just under 0.1 -> 3 decimals
        assert calculate_required_precision(0, 0.099) == 3

        # Exactly 0.01 -> 3 decimals
        assert calculate_required_precision(0, 0.01) == 3
        # Just under 0.01 -> 4 decimals
        assert calculate_required_precision(0, 0.009) == 4


class TestFormatValueAdaptive:
    """Test adaptive value formatting with specified precision."""

    def test_zero_always_returns_zero_string(self):
        """Zero is always formatted as '0' regardless of precision."""
        assert format_value_adaptive(0, precision=0) == "0"
        assert format_value_adaptive(0, precision=2) == "0"
        assert format_value_adaptive(0, precision=4) == "0"

    def test_extreme_small_values_scientific_notation(self):
        """Values |x| < 0.001 use scientific notation with 1 decimal."""
        assert format_value_adaptive(0.0005, precision=2) == "5.0e-04"
        assert format_value_adaptive(0.00052, precision=2) == "5.2e-04"
        assert format_value_adaptive(-0.0005, precision=2) == "-5.0e-04"
        assert format_value_adaptive(0.0001, precision=3) == "1.0e-04"

    def test_extreme_large_values_scientific_notation(self):
        """Values |x| > 1000 use scientific notation with 1 decimal."""
        assert format_value_adaptive(1500, precision=2) == "1.5e+03"
        assert format_value_adaptive(12000, precision=1) == "1.2e+04"
        assert format_value_adaptive(-2500, precision=0) == "-2.5e+03"

    def test_precision_0_formats_as_integers(self):
        """Precision 0 formats values as integers."""
        assert format_value_adaptive(5, precision=0) == "5"
        assert format_value_adaptive(10.7, precision=0) == "11"  # Rounded
        assert format_value_adaptive(-3.2, precision=0) == "-3"

    def test_precision_1_formats_with_one_decimal(self):
        """Precision 1 formats with up to 1 decimal, trailing zeros stripped."""
        assert format_value_adaptive(0.5, precision=1) == "0.5"
        assert format_value_adaptive(1.0, precision=1) in ["1", "1.0"]  # May strip
        assert format_value_adaptive(1.5, precision=1) == "1.5"
        assert format_value_adaptive(2.0, precision=1) in ["2", "2.0"]

    def test_precision_2_formats_with_two_decimals(self):
        """Precision 2 formats with up to 2 decimals, trailing zeros stripped."""
        assert format_value_adaptive(0.98, precision=2) == "0.98"
        assert format_value_adaptive(1.00, precision=2) in ["1", "1.0", "1.00"]
        assert format_value_adaptive(1.02, precision=2) == "1.02"
        assert format_value_adaptive(1.50, precision=2) in ["1.5", "1.50"]

    def test_precision_3_formats_with_three_decimals(self):
        """Precision 3 formats with up to 3 decimals, trailing zeros stripped."""
        assert format_value_adaptive(0.985, precision=3) == "0.985"
        assert format_value_adaptive(1.000, precision=3) in ["1", "1.0", "1.00", "1.000"]
        assert format_value_adaptive(1.015, precision=3) == "1.015"
        assert format_value_adaptive(1.100, precision=3) in ["1.1", "1.10", "1.100"]

    def test_precision_4_formats_with_four_decimals(self):
        """Precision 4 formats with up to 4 decimals."""
        assert format_value_adaptive(0.9985, precision=4) == "0.9985"
        assert format_value_adaptive(1.0000, precision=4) in [
            "1",
            "1.0",
            "1.00",
            "1.000",
            "1.0000",
        ]
        assert format_value_adaptive(1.0015, precision=4) == "1.0015"

    def test_negative_values_formatted_correctly(self):
        """Negative values are formatted correctly with sign."""
        assert format_value_adaptive(-0.98, precision=2) == "-0.98"
        assert format_value_adaptive(-1.0, precision=2) in ["-1", "-1.0"]
        assert format_value_adaptive(-5, precision=0) == "-5"

    def test_distinguishable_values_in_tight_range(self):
        """Values in tight range should be distinguishable."""
        # Tight odds ratio range
        values = [0.98, 1.00, 1.02]
        precision = 2
        formatted = [format_value_adaptive(v, precision) for v in values]
        # All three should be distinguishable (at least 2 different strings)
        assert len(set(formatted)) >= 2
        assert formatted[0] == "0.98"
        assert formatted[2] == "1.02"

    def test_rstrip_removes_trailing_zeros(self):
        """Trailing zeros and decimal point are removed when appropriate."""
        # 1.00 -> "1.00" -> rstrip('0') -> "1." -> rstrip('.') -> "1"
        result = format_value_adaptive(1.0, precision=2)
        assert result in ["1", "1.0", "1.00"]  # Implementation may vary

        # 1.50 -> "1.50" -> rstrip('0') -> "1.5"
        result = format_value_adaptive(1.5, precision=2)
        assert result in ["1.5", "1.50"]

    def test_values_near_scientific_notation_threshold(self):
        """Values near scientific notation thresholds handled correctly."""
        # Just above threshold (0.001) - use decimal
        assert format_value_adaptive(0.001, precision=3) == "0.001"
        assert format_value_adaptive(0.0011, precision=3) == "0.001"  # Rounded

        # Just below threshold - use scientific
        assert format_value_adaptive(0.0009, precision=3) == "9.0e-04"

        # Just below 1000 - use decimal
        assert format_value_adaptive(999, precision=0) == "999"

        # Just above 1000 - use scientific
        assert format_value_adaptive(1001, precision=0) == "1.0e+03"


class TestFormatValueForAnnotation:
    """Test annotation formatting with threshold notation for extremes."""

    def test_zero_returns_zero_string(self):
        """Zero is always formatted as '0'."""
        assert format_value_for_annotation(0, precision=2) == "0"

    def test_very_small_positive_value_near_zero(self):
        """Very small positive values near zero return '~0'."""
        assert format_value_for_annotation(0.0005, precision=2) == "~0"
        assert format_value_for_annotation(0.0009, precision=3) == "~0"
        assert format_value_for_annotation(0.00001, precision=4) == "~0"

    def test_very_small_negative_value_near_zero(self):
        """Very small negative values near zero return '~0'."""
        assert format_value_for_annotation(-0.0005, precision=2) == "~0"
        assert format_value_for_annotation(-0.0009, precision=3) == "~0"
        assert format_value_for_annotation(-0.00001, precision=4) == "~0"

    def test_threshold_at_0_001_positive(self):
        """Values at exactly 0.001 use normal formatting (not '~0')."""
        # 0.001 is NOT < 0.001, so should format normally
        result = format_value_for_annotation(0.001, precision=3)
        assert result == "0.001"

    def test_threshold_at_0_001_negative(self):
        """Values at exactly -0.001 use normal formatting (not '~0')."""
        result = format_value_for_annotation(-0.001, precision=3)
        assert result == "-0.001"

    def test_very_large_positive_value(self):
        """Very large positive values return '>1000'."""
        assert format_value_for_annotation(1500, precision=2) == ">1000"
        assert format_value_for_annotation(10000, precision=1) == ">1000"
        assert format_value_for_annotation(999999, precision=0) == ">1000"

    def test_very_large_negative_value(self):
        """Very large negative values return '<-1000'."""
        assert format_value_for_annotation(-1500, precision=2) == "<-1000"
        assert format_value_for_annotation(-10000, precision=1) == "<-1000"
        assert format_value_for_annotation(-999999, precision=0) == "<-1000"

    def test_threshold_at_1000_positive(self):
        """Value at exactly 1000 uses normal formatting (not '>1000')."""
        result = format_value_for_annotation(1000, precision=0)
        assert result == "1000"

    def test_threshold_at_1000_negative(self):
        """Value at exactly -1000 uses normal formatting (not '<-1000')."""
        result = format_value_for_annotation(-1000, precision=0)
        assert result == "-1000"

    def test_normal_range_values_formatted_adaptively(self):
        """Values in normal range [0.001, 1000] use adaptive formatting."""
        # Tight odds ratio range
        assert format_value_for_annotation(0.98, precision=2) == "0.98"
        assert format_value_for_annotation(1.00, precision=2) in ["1", "1.0"]
        assert format_value_for_annotation(1.02, precision=2) == "1.02"

        # Moderate range
        assert format_value_for_annotation(5.5, precision=1) == "5.5"
        assert format_value_for_annotation(10, precision=0) == "10"
        assert format_value_for_annotation(500, precision=0) == "500"

    def test_negative_normal_range_values(self):
        """Negative values in normal range formatted correctly."""
        assert format_value_for_annotation(-0.5, precision=1) == "-0.5"
        assert format_value_for_annotation(-5, precision=0) == "-5"
        assert format_value_for_annotation(-500, precision=0) == "-500"

    def test_precision_0_in_normal_range(self):
        """Precision 0 rounds to integers in normal range."""
        assert format_value_for_annotation(5.7, precision=0) == "6"
        assert format_value_for_annotation(10.2, precision=0) == "10"

    def test_precision_2_in_normal_range(self):
        """Precision 2 formats with up to 2 decimals, trailing zeros stripped."""
        assert format_value_for_annotation(1.50, precision=2) in ["1.5", "1.50"]
        assert format_value_for_annotation(2.00, precision=2) in ["2", "2.0"]
        assert format_value_for_annotation(3.14, precision=2) == "3.14"

    def test_comparison_with_format_value_adaptive(self):
        """Annotation formatter differs from adaptive formatter for extremes."""
        # Extreme small value
        assert format_value_for_annotation(0.0005, precision=2) == "~0"
        assert format_value_adaptive(0.0005, precision=2) == "5.0e-04"

        # Extreme large value
        assert format_value_for_annotation(1500, precision=2) == ">1000"
        assert format_value_adaptive(1500, precision=2) == "1.5e+03"

        # Normal range value (same)
        assert format_value_for_annotation(1.02, precision=2) == "1.02"
        assert format_value_adaptive(1.02, precision=2) == "1.02"

    def test_edge_cases_near_thresholds(self):
        """Test values very close to thresholds."""
        # Just above 1000
        assert format_value_for_annotation(1000.1, precision=0) == ">1000"
        assert format_value_for_annotation(1001, precision=0) == ">1000"

        # Just below -1000
        assert format_value_for_annotation(-1000.1, precision=0) == "<-1000"
        assert format_value_for_annotation(-1001, precision=0) == "<-1000"

        # Just below 0.001 (absolute value)
        assert format_value_for_annotation(0.0009, precision=4) == "~0"
        assert format_value_for_annotation(-0.0009, precision=4) == "~0"


class TestCreateResponseValueFormatter:
    """Test creation and behavior of response value formatters."""

    def test_returns_func_formatter(self):
        """create_response_value_formatter returns matplotlib FuncFormatter."""
        formatter = create_response_value_formatter(0, 10)
        assert isinstance(formatter, FuncFormatter)

    def test_formatter_callable_with_two_args(self):
        """Formatter is callable with (value, position) signature."""
        formatter = create_response_value_formatter(0.95, 1.05)
        result = formatter(1.0, 0)  # position argument required
        assert isinstance(result, str)

    def test_formatter_uses_correct_precision_tight_range(self):
        """Formatter calculates and uses correct precision for tight range."""
        formatter = create_response_value_formatter(0.98, 1.02)  # Range = 0.04 -> precision 2

        assert formatter(0.98, 0) == "0.98"
        assert formatter(1.00, 1) in ["1", "1.0"]
        assert formatter(1.02, 2) == "1.02"

    def test_formatter_uses_correct_precision_wide_range(self):
        """Formatter calculates and uses correct precision for wide range."""
        formatter = create_response_value_formatter(0, 15)  # Range = 15 -> precision 0

        assert formatter(0, 0) == "0"
        assert formatter(5, 1) == "5"
        assert formatter(10, 2) == "10"
        assert formatter(15, 3) == "15"

    def test_formatter_captures_precision_in_closure(self):
        """Formatter captures precision in closure (not recalculated each call)."""
        # Create formatter with tight range -> precision 2
        formatter = create_response_value_formatter(0.95, 1.05)

        # Even when calling with out-of-range values, uses captured precision
        # (This tests that precision is calculated once, not per value)
        result = formatter(10.0, 0)  # Large value, but precision=2 from range
        # Should format as "10" (precision 2, trailing zeros removed)
        assert result in ["10", "10.0", "10.00"]

    def test_formatter_handles_edge_cases(self):
        """Formatter handles edge cases correctly."""
        formatter = create_response_value_formatter(0.95, 1.05)

        # Zero
        assert formatter(0, 0) == "0"

        # Extreme small value (scientific notation)
        assert formatter(0.0005, 1) == "5.0e-04"

        # Extreme large value (scientific notation)
        assert formatter(1500, 2) == "1.5e+03"

    def test_different_formatters_different_precisions(self):
        """Different formatters with different ranges use different precisions."""
        formatter_tight = create_response_value_formatter(0.98, 1.02)  # precision 2
        formatter_wide = create_response_value_formatter(0, 10)  # range=10, precision 0

        # Same value formatted differently
        val = 1.5
        result_tight = formatter_tight(val, 0)
        result_wide = formatter_wide(val, 0)

        # Tight range should have more decimals, wide range rounds to integer
        assert result_tight == "1.5" or result_tight == "1.50"
        assert result_wide == "2"  # precision 0 rounds 1.5 to 2

    def test_formatter_with_equal_min_max(self):
        """Formatter handles equal min/max (defaults to precision 2)."""
        formatter = create_response_value_formatter(1.0, 1.0)

        assert formatter(1.0, 0) in ["1", "1.0", "1.00"]
        assert formatter(0.98, 1) == "0.98"
        assert formatter(1.02, 2) == "1.02"


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    def test_odds_ratio_tight_around_one(self):
        """Odds ratios tight around 1.0 are distinguishable."""
        vmin, vmax = 0.98, 1.02
        precision = calculate_required_precision(vmin, vmax)
        values = np.linspace(vmin, vmax, 5)
        formatted = [format_value_adaptive(v, precision) for v in values]

        # All 5 values should be distinguishable
        unique_formatted = set(formatted)
        assert len(unique_formatted) >= 4  # At least 4 distinct values

    def test_log_odds_around_zero(self):
        """Log odds around 0 formatted correctly."""
        vmin, vmax = -0.5, 0.5
        precision = calculate_required_precision(vmin, vmax)
        assert precision == 1  # Range = 1.0

        values = [-0.5, 0.0, 0.5]
        formatted = [format_value_adaptive(v, precision) for v in values]

        assert formatted[0] == "-0.5"
        assert formatted[1] == "0"
        assert formatted[2] == "0.5"

    def test_wide_odds_ratio_range(self):
        """Wide odds ratio range uses fewer decimals."""
        vmin, vmax = 0.1, 10
        precision = calculate_required_precision(vmin, vmax)
        assert precision == 1  # Range = 9.9 < 10, so precision = 1

        formatter = create_response_value_formatter(vmin, vmax)
        assert formatter(0.5, 0) == "0.5"
        assert formatter(5, 1) == "5"  # 5.0 -> "5.0" -> rstrip -> "5"
        assert formatter(10, 2) == "10"

    def test_very_tight_range_requires_high_precision(self):
        """Very tight range automatically increases precision."""
        vmin, vmax = 0.9985, 1.0015
        precision = calculate_required_precision(vmin, vmax)
        assert precision == 4  # Range = 0.003

        formatter = create_response_value_formatter(vmin, vmax)
        result_min = formatter(vmin, 0)
        result_max = formatter(vmax, 1)

        # Both should be formatted with 4 decimals and distinguishable
        assert "9985" in result_min or "998" in result_min
        assert "0015" in result_max or "001" in result_max
        assert result_min != result_max


class TestTruncateLabel:
    """Test label truncation for long feature/category names."""

    def test_short_label_unchanged(self):
        """Labels within limit are returned unchanged."""
        from prism.plotting.formatter import DEFAULT_MAX_LABEL_LENGTH, truncate_label

        assert truncate_label("Short label") == "Short label"
        assert truncate_label("A" * DEFAULT_MAX_LABEL_LENGTH) == "A" * DEFAULT_MAX_LABEL_LENGTH
        assert truncate_label("") == ""

    def test_long_label_truncated_with_ellipsis_and_last_char(self):
        """Labels exceeding limit are truncated with ...x suffix."""
        from prism.plotting.formatter import DEFAULT_MAX_LABEL_LENGTH, truncate_label

        result = truncate_label("A" * (DEFAULT_MAX_LABEL_LENGTH + 10))
        assert len(result) == DEFAULT_MAX_LABEL_LENGTH
        assert result.endswith("...A")
        # (max - 4) A's + "..." + last A = max chars
        assert result == "A" * (DEFAULT_MAX_LABEL_LENGTH - 4) + "...A"

    def test_preserves_last_character(self):
        """Last character of original string is preserved."""
        from prism.plotting.formatter import DEFAULT_MAX_LABEL_LENGTH, truncate_label

        # Create a string longer than max that ends with Z
        long_label = "A" * (DEFAULT_MAX_LABEL_LENGTH + 10) + "Z"
        result = truncate_label(long_label)
        assert result.endswith("...Z")
        assert len(result) == DEFAULT_MAX_LABEL_LENGTH

    def test_custom_max_length(self):
        """Custom max_length parameter works correctly."""
        from prism.plotting.formatter import truncate_label

        result = truncate_label("ABCDEFGHIJ", max_length=8)
        # 4 chars + "..." + last char = 8
        assert result == "ABCD...J"
        assert len(result) == 8

    def test_boundary_at_max_plus_one(self):
        """Label one character over limit is truncated."""
        from prism.plotting.formatter import DEFAULT_MAX_LABEL_LENGTH, truncate_label

        label = "A" * (DEFAULT_MAX_LABEL_LENGTH + 1)
        result = truncate_label(label)
        assert len(result) == DEFAULT_MAX_LABEL_LENGTH
        assert result == "A" * (DEFAULT_MAX_LABEL_LENGTH - 4) + "...A"

    def test_realistic_feature_name(self):
        """Realistic long feature name is handled correctly."""
        from prism.plotting.formatter import DEFAULT_MAX_LABEL_LENGTH, truncate_label

        long_name = "Recipient Previous Cardiac Surgery Type (CABG/Valve)"
        result = truncate_label(long_name)
        assert len(result) <= DEFAULT_MAX_LABEL_LENGTH
        assert result.endswith("...)")  # Last char is ')'
