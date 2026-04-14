"""
Unit tests for binary feature grouping in nomograms.

Tests the functionality that groups up to 3 binary categorical features
in a single subplot with proper positioning and formatting.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')  # Use non-interactive backend for testing
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt

from prism.plotting.formatter import PlotFormatter
from prism.plotting.renderer import NomogramRenderer
from prism.plotting_data import BinaryFeatureGroup, FeatureInfo, PlottingDataBundle


class TestBinaryFeatureGroup:
    """Test the BinaryFeatureGroup dataclass."""

    def test_init_two_features(self):
        """Test BinaryFeatureGroup with 2 features calculates correct y-positions."""
        feat1 = FeatureInfo(index=0, name="bin1", label="Binary 1", is_categorical=True)
        feat2 = FeatureInfo(index=1, name="bin2", label="Binary 2", is_categorical=True)

        group = BinaryFeatureGroup(features=[feat1, feat2])

        assert group.n_features == 2
        assert group.y_positions == [0.33, 0.66]

    def test_init_three_features(self):
        """Test BinaryFeatureGroup with 3 features calculates correct y-positions."""
        feat1 = FeatureInfo(index=0, name="bin1", label="Binary 1", is_categorical=True)
        feat2 = FeatureInfo(index=1, name="bin2", label="Binary 2", is_categorical=True)
        feat3 = FeatureInfo(index=2, name="bin3", label="Binary 3", is_categorical=True)

        group = BinaryFeatureGroup(features=[feat1, feat2, feat3])

        assert group.n_features == 3
        assert group.y_positions == [0.17, 0.5, 0.83]

    def test_init_one_feature(self):
        """Test BinaryFeatureGroup with 1 feature calculates correct y-position."""
        feat1 = FeatureInfo(index=0, name="bin1", label="Binary 1", is_categorical=True)

        group = BinaryFeatureGroup(features=[feat1])

        assert group.n_features == 1
        assert group.y_positions == [0.5]

    def test_init_explicit_positions(self):
        """Test BinaryFeatureGroup respects explicitly provided y-positions."""
        feat1 = FeatureInfo(index=0, name="bin1", label="Binary 1", is_categorical=True)
        feat2 = FeatureInfo(index=1, name="bin2", label="Binary 2", is_categorical=True)

        custom_positions = [0.25, 0.75]
        group = BinaryFeatureGroup(features=[feat1, feat2], y_positions=custom_positions)

        assert group.y_positions == custom_positions

    def test_init_fallback_positions(self):
        """Test BinaryFeatureGroup fallback for unusual number of features."""
        features = []
        for i in range(4):  # Test with 4 features (edge case)
            features.append(
                FeatureInfo(index=i, name=f"bin{i}", label=f"Binary {i}", is_categorical=True)
            )

        group = BinaryFeatureGroup(features=features)

        # Should use fallback calculation: step = 1/(n+1) = 1/5 = 0.2
        assert group.n_features == 4
        # Use np.testing for floating point comparison
        np.testing.assert_array_almost_equal(group.y_positions, [0.2, 0.4, 0.6, 0.8])


class TestBinaryFeatureDetection:
    """Test binary feature detection logic in NomogramRenderer."""

    @pytest.fixture
    def mock_renderer(self):
        """Create a mock NomogramRenderer for testing."""

        class MockNomogramRenderer:
            def _is_binary_feature(self, info: FeatureInfo) -> bool:
                """Check if a feature is binary (0/1 values)."""
                if not info.is_categorical or info.x_values is None:
                    return False

                unique_values = np.unique(info.x_values)
                return (
                    len(unique_values) == 2
                    and any(np.isclose(unique_values, 0))
                    and any(np.isclose(unique_values, 1))
                )

        return MockNomogramRenderer()

    def test_detect_binary_feature_0_1(self, mock_renderer):
        """Test detection of binary feature with 0/1 values."""
        info = FeatureInfo(
            index=0,
            name="binary_1",
            label="Binary 1",
            is_categorical=True,
            x_values=np.array([0.0, 1.0]),
        )
        assert mock_renderer._is_binary_feature(info) is True

    def test_detect_binary_feature_1_2(self, mock_renderer):
        """Test non-detection of feature with 1/2 values."""
        info = FeatureInfo(
            index=0,
            name="not_binary_1",
            label="Not Binary 1",
            is_categorical=True,
            x_values=np.array([1.0, 2.0]),
        )
        assert mock_renderer._is_binary_feature(info) is False

    def test_detect_continuous_feature(self, mock_renderer):
        """Test non-detection of continuous feature."""
        info = FeatureInfo(
            index=0,
            name="continuous",
            label="Continuous",
            is_categorical=False,
            x_values=np.linspace(0, 10, 100),
        )
        assert mock_renderer._is_binary_feature(info) is False

    def test_detect_categorical_with_three_levels(self, mock_renderer):
        """Test non-detection of categorical with >2 levels."""
        info = FeatureInfo(
            index=0,
            name="three_level",
            label="Three Level",
            is_categorical=True,
            x_values=np.array([0.0, 1.0, 2.0]),
        )
        assert mock_renderer._is_binary_feature(info) is False


class TestBinaryFeatureGrouping:
    """Test the binary feature grouping logic."""

    @pytest.fixture
    def mock_renderer_with_grouping(self):
        """Create a mock renderer with _group_binary_features method."""

        class MockNomogramRenderer:
            def _is_binary_feature(self, info: FeatureInfo) -> bool:
                if not info.is_categorical or info.x_values is None:
                    return False
                unique_values = np.unique(info.x_values)
                return (
                    len(unique_values) == 2
                    and any(np.isclose(unique_values, 0))
                    and any(np.isclose(unique_values, 1))
                )

            def _group_binary_features(self, features):
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
                        # Keep single binary features as-is
                        grouped_features.append(group_features[0])
                    else:
                        # Create a grouped feature entry
                        group = BinaryFeatureGroup(features=[f['info'] for f in group_features])
                        grouped_features.append({'type': 'binary_group', 'group': group})

                # Add non-binary features
                grouped_features.extend(non_binary_features)

                return grouped_features

        return MockNomogramRenderer()

    def test_group_three_binary_features(self, mock_renderer_with_grouping):
        """Test grouping 3 binary features into one group."""
        features = []
        for i in range(3):
            info = FeatureInfo(
                index=i,
                name=f"binary_{i}",
                label=f"Binary {i}",
                is_categorical=True,
                x_values=np.array([0.0, 1.0]),
            )
            features.append({'type': 'univariate', 'info': info})

        grouped = mock_renderer_with_grouping._group_binary_features(features)

        assert len(grouped) == 1
        assert grouped[0]['type'] == 'binary_group'
        assert grouped[0]['group'].n_features == 3
        assert grouped[0]['group'].y_positions == [0.17, 0.5, 0.83]

    def test_group_two_binary_features(self, mock_renderer_with_grouping):
        """Test grouping 2 binary features into one group."""
        features = []
        for i in range(2):
            info = FeatureInfo(
                index=i,
                name=f"binary_{i}",
                label=f"Binary {i}",
                is_categorical=True,
                x_values=np.array([0.0, 1.0]),
            )
            features.append({'type': 'univariate', 'info': info})

        grouped = mock_renderer_with_grouping._group_binary_features(features)

        assert len(grouped) == 1
        assert grouped[0]['type'] == 'binary_group'
        assert grouped[0]['group'].n_features == 2
        assert grouped[0]['group'].y_positions == [0.33, 0.66]

    def test_group_four_binary_features(self, mock_renderer_with_grouping):
        """Test grouping 4 binary features into 1 group of 3 and 1 single."""
        features = []
        for i in range(4):
            info = FeatureInfo(
                index=i,
                name=f"binary_{i}",
                label=f"Binary {i}",
                is_categorical=True,
                x_values=np.array([0.0, 1.0]),
            )
            features.append({'type': 'univariate', 'info': info})

        grouped = mock_renderer_with_grouping._group_binary_features(features)

        assert len(grouped) == 2
        assert grouped[0]['type'] == 'binary_group'
        assert grouped[0]['group'].n_features == 3
        assert grouped[1]['type'] == 'univariate'
        assert grouped[1]['info'].index == 3

    def test_mixed_features(self, mock_renderer_with_grouping):
        """Test grouping mixed binary and non-binary features."""
        features = []

        # Add 2 binary features
        for i in range(2):
            info = FeatureInfo(
                index=i,
                name=f"binary_{i}",
                label=f"Binary {i}",
                is_categorical=True,
                x_values=np.array([0.0, 1.0]),
            )
            features.append({'type': 'univariate', 'info': info})

        # Add a continuous feature
        info = FeatureInfo(
            index=2,
            name="continuous",
            label="Continuous",
            is_categorical=False,
            x_values=np.linspace(0, 10, 100),
        )
        features.append({'type': 'univariate', 'info': info})

        grouped = mock_renderer_with_grouping._group_binary_features(features)

        assert len(grouped) == 2
        assert grouped[0]['type'] == 'binary_group'
        assert grouped[0]['group'].n_features == 2
        assert grouped[1]['type'] == 'univariate'
        assert grouped[1]['info'].name == "continuous"

    def test_mixed_bivariate_features(self, mock_renderer_with_grouping):
        """Test that bivariate features are not grouped."""
        features = []

        # Add 2 binary features
        for i in range(2):
            info = FeatureInfo(
                index=i,
                name=f"binary_{i}",
                label=f"Binary {i}",
                is_categorical=True,
                x_values=np.array([0.0, 1.0]),
            )
            features.append({'type': 'univariate', 'info': info})

        # Add a bivariate feature
        info = FeatureInfo(index=2, name="feature1", label="Feature 1", is_categorical=True)
        features.append({'type': 'mixed', 'info': info})

        grouped = mock_renderer_with_grouping._group_binary_features(features)

        assert len(grouped) == 2
        assert grouped[0]['type'] == 'binary_group'
        assert grouped[1]['type'] == 'mixed'


class TestBinaryFeatureRendering:
    """Test the rendering of binary feature groups."""

    @pytest.fixture
    def mock_bundle(self):
        """Create a mock PlottingDataBundle for testing rendering."""
        features = []
        for i in range(3):
            feat = FeatureInfo(
                index=i,
                name=f"binary_{i}",
                label=f"Binary Feature {i+1}",
                is_categorical=True,
                response=np.array([-0.5 + i * 0.1, 0.3 + i * 0.05]),
                x_values=np.array([0.0, 1.0]),
            )
            features.append(feat)

        class MockMetadata:
            def __init__(self):
                self.is_categorical = True

        class MockMetadataRegistry:
            def get_by_collapsed(self, index):
                return MockMetadata()

        class MockIndexMapper:
            pass

        bundle = PlottingDataBundle(
            all_feature_names=[f"binary_{i}" for i in range(3)],
            _univariate_info=features,
            _bivariate_info=[],
            scaler=None,
            x_data=np.random.randn(100, 3),
            n_steps=50,
            categorical_threshold=15,
            index_mapper=MockIndexMapper(),
            metadata_registry=MockMetadataRegistry(),
        )

        return bundle

    @pytest.fixture
    def mock_renderer(self, mock_bundle):
        """Create a mock NomogramRenderer."""
        formatter = PlotFormatter(use_odds_ratio=False, binary_labels={0: "No", 1: "Yes"})

        renderer = NomogramRenderer(mock_bundle, formatter, use_odds_ratio=False)
        return renderer

    def test_render_binary_group_suppresses_yticks(self, mock_renderer, mock_bundle):
        """Test that binary group rendering suppresses y-axis ticks."""
        # Create a binary feature group
        features = mock_bundle.univariate_features()[:2]
        group = BinaryFeatureGroup(features=features)

        # Create mock axes
        fig, ax = plt.subplots(figsize=(10, 2))

        # Render the group
        mock_renderer._render_binary_group(ax=ax, group=group, subplot_params={}, x_range=(-1, 1))

        # Check that yticks are suppressed
        assert len(ax.get_yticks()) == 0

        plt.close(fig)

    def test_render_binary_group_plot_properties(self, mock_renderer, mock_bundle):
        """Test that binary group rendering sets correct plot properties."""
        # Create a binary feature group
        features = mock_bundle.univariate_features()[:2]
        group = BinaryFeatureGroup(features=features)

        # Create mock axes
        fig, ax = plt.subplots(figsize=(10, 2))

        # Render the group
        mock_renderer._render_binary_group(ax=ax, group=group, subplot_params={}, x_range=(-1, 1))

        # Check properties
        assert ax.get_ylim() == (0, 1)
        assert ax.get_xlim() == (-1, 1)
        assert len(ax.get_yticks()) == 0  # Ticks suppressed

        # Check for line plots (should have 2 lines for 2 features)
        lines = ax.get_lines()
        assert len(lines) == 2

        # Check all lines use the same color
        colors = [line.get_color() for line in lines]
        assert len(set(colors)) == 1  # All colors should be the same

        plt.close(fig)

    def test_calculate_x_range_with_groups(self, mock_renderer):
        """Test x-range calculation includes binary groups."""
        # Create feature list with a binary group
        feat1 = FeatureInfo(
            index=0,
            name="bin1",
            label="Bin 1",
            is_categorical=True,
            response=np.array([-0.5, 0.3]),
            x_values=np.array([0, 1]),
        )
        feat2 = FeatureInfo(
            index=1,
            name="bin2",
            label="Bin 2",
            is_categorical=True,
            response=np.array([-0.2, 0.4]),
            x_values=np.array([0, 1]),
        )

        group = BinaryFeatureGroup(features=[feat1, feat2])

        features_to_plot = [{'type': 'binary_group', 'group': group}]

        x_range = mock_renderer._calculate_x_range(features_to_plot)

        # Should include both binary features' response ranges
        assert x_range is not None
        assert x_range[0] <= -0.5  # Should include minimum
        assert x_range[1] >= 0.4  # Should include maximum

    @patch('matplotlib.figure.Figure.savefig')
    def test_render_nomogram_with_binary_groups(self, mock_savefig, mock_renderer):
        """Test full nomogram rendering with binary groups."""
        fig = mock_renderer.render_nomogram(
            legend_on_right=False,
            surround_axes=True,
            features_per_plot=None,
            two_column=False,
        )

        assert fig is not None
        # Should create a figure, not a list (single page)
        assert not isinstance(fig, list)

        # Count subplots - should be 1 for the grouped 3 binary features
        axes = fig.get_axes()
        assert len(axes) >= 1

        # Verify binary group subplot exists and has correct properties
        binary_subplot = None
        for ax in axes:
            if len(ax.get_lines()) >= 3:  # Binary group should have 3+ lines
                binary_subplot = ax
                break

        assert binary_subplot is not None, "Binary group subplot not found"
        # Should have at least 3 lines (one for each binary feature)
        assert len(binary_subplot.get_lines()) >= 3  # 3 binary features
        assert len(binary_subplot.get_yticks()) == 0  # Ticks suppressed

        plt.close(fig)

    def test_newline_stripping_in_rendering(self, mock_renderer):
        """Test that newlines are stripped from labels during rendering."""
        # Create feature with newlines in label
        feat = FeatureInfo(
            index=0,
            name="binary_with_newlines",
            label="Feature With\n\nNewlines",
            is_categorical=True,
            response=np.array([-0.5, 0.3]),
            x_values=np.array([0.0, 1.0]),
        )

        group = BinaryFeatureGroup(features=[feat])

        # Create mock axes
        fig, ax = plt.subplots(figsize=(10, 2))

        # Render the group
        mock_renderer._render_binary_group(ax=ax, group=group, subplot_params={}, x_range=(-1, 1))

        # Check text objects - should have stripped newlines
        # In matplotlib, text objects are accessible through ax.texts
        label_text = None
        for text in ax.texts:
            if 'Feature With' in text.get_text():
                label_text = text.get_text()
                break

        assert label_text is not None, f"No label text found in {len(ax.texts)} text objects"
        assert '\n' not in label_text, f"Newlines found in: {repr(label_text)}"
        assert '\r' not in label_text, f"Carriage returns found in: {repr(label_text)}"
        # Check that the newlines were stripped (might have extra spaces)
        assert 'Feature With' in label_text and 'Newlines' in label_text
        assert (
            label_text.replace(' ', '') == 'FeatureWithNewlines'
        ), f"Unexpected text: {repr(label_text)}"

        plt.close(fig)


class TestBinaryGroupLayout:
    """Test layout management for binary feature groups."""

    def test_layout_manager_counts_groups_as_one(self):
        """Test that LayoutManager treats binary groups as single units."""
        from prism.plotting.renderer import _LayoutManager

        # Create features: 2 features + 1 binary group (with 3 features) + 1 feature
        features = [
            {'type': 'univariate', 'info': MagicMock()},
            {'type': 'univariate', 'info': MagicMock()},
            {'type': 'binary_group', 'group': MagicMock()},
            {'type': 'univariate', 'info': MagicMock()},
        ]

        # Single column with 3 features per plot
        pages = _LayoutManager.distribute_features(
            features=features, features_per_plot=3, two_column=False
        )

        # Should create 2 pages:
        # Page 1: univariate, univariate, binary_group (3 units)
        # Page 2: univariate (1 unit)
        assert len(pages) == 2
        assert len(pages[0]) == 3
        assert len(pages[1]) == 1
        assert pages[0][2]['type'] == 'binary_group'

    def test_layout_manager_two_column_with_groups(self):
        """Test LayoutManager with two columns and binary groups."""
        from prism.plotting.renderer import _LayoutManager

        features = [
            {'type': 'univariate', 'info': MagicMock()},
            {'type': 'binary_group', 'group': MagicMock()},
            {'type': 'univariate', 'info': MagicMock()},
            {'type': 'binary_group', 'group': MagicMock()},
        ]

        # Two column with 2 features per column
        pages = _LayoutManager.distribute_features(
            features=features, features_per_plot=2, two_column=True
        )

        # Should create 1 page with auto-balancing
        assert len(pages) == 1
        assert len(pages[0]) == 4
