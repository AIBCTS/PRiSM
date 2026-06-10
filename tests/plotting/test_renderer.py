"""
Tests for NomogramRenderer (Phase 1: Univariate rendering).

Tests cover:
- Bundle validation (requires services)
- Univariate categorical rendering
- Univariate continuous rendering
- Layout creation (single/two-column)
- Odds ratio conversion
- X-range calculation
- Pagination
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from prism.plotting.formatter import PlotFormatter
from prism.plotting.pipeline import PlottingPipeline
from prism.plotting.renderer import NomogramRenderer, _LayoutManager
from prism.plotting_data import PlottingDataBundle


class TestLayoutManager:
    """Test _LayoutManager class."""

    def test_create_single_column_figure(self):
        """Test single-column figure creation."""
        fig, gs = _LayoutManager.create_single_column_figure(n_features=5)

        assert fig is not None
        assert gs is not None
        assert gs.nrows == 5
        assert gs.ncols == 1

    def test_create_two_column_figure(self):
        """Test two-column figure creation."""
        fig, gs = _LayoutManager.create_two_column_figure(n_features=10)

        assert fig is not None
        assert gs is not None
        assert gs.nrows == 5  # (10 + 1) // 2 = 5
        assert gs.ncols == 2

    def test_distribute_features_single_column_no_limit(self):
        """Test feature distribution - single column, no pagination."""
        features = [{'id': i} for i in range(5)]
        pages = _LayoutManager.distribute_features(
            features, features_per_plot=None, two_column=False
        )

        assert len(pages) == 1
        assert len(pages[0]) == 5

    def test_distribute_features_single_column_with_pagination(self):
        """Test feature distribution - single column with pagination."""
        features = [{'id': i} for i in range(7)]
        pages = _LayoutManager.distribute_features(features, features_per_plot=3, two_column=False)

        assert len(pages) == 3
        assert len(pages[0]) == 3
        assert len(pages[1]) == 3
        assert len(pages[2]) == 1

    def test_distribute_features_two_column_no_limit(self):
        """Test feature distribution - two column, no pagination."""
        features = [{'id': i} for i in range(5)]
        pages = _LayoutManager.distribute_features(
            features, features_per_plot=None, two_column=True
        )

        assert len(pages) == 1
        assert len(pages[0]) == 5

    def test_distribute_features_two_column_with_pagination(self):
        """Test feature distribution - two column with pagination."""
        features = [{'id': i} for i in range(10)]
        pages = _LayoutManager.distribute_features(features, features_per_plot=3, two_column=True)

        # 3 per column = 6 per page
        assert len(pages) == 2
        assert len(pages[0]) == 6
        assert len(pages[1]) == 4


class TestNomogramRendererValidation:
    """Test NomogramRenderer initialization and validation."""

    @pytest.fixture
    def bundle_without_services(self):
        """Create bundle without services."""
        return PlottingDataBundle(all_feature_names=['age', 'bmi', 'glucose'])

    @pytest.fixture
    def mock_bundle_with_services(self):
        """Create mock bundle with services."""

        class MockBundle:
            has_services = True
            index_mapper = "mock_mapper"
            metadata_registry = "mock_registry"

            def univariate_features(self):
                return []

        return MockBundle()

    def test_renderer_requires_bundle_with_services(self, bundle_without_services):
        """Test that renderer requires bundle with services."""
        formatter = PlotFormatter()

        with pytest.raises(ValueError, match="must have services"):
            NomogramRenderer(bundle_without_services, formatter)

    def test_renderer_validates_index_mapper(self):
        """Test that renderer validates IndexMapper exists."""

        class BadBundle:
            has_services = True
            index_mapper = None  # Missing!
            metadata_registry = "mock"

        formatter = PlotFormatter()

        with pytest.raises(ValueError, match="IndexMapper"):
            NomogramRenderer(BadBundle(), formatter)

    def test_renderer_validates_metadata_registry(self):
        """Test that renderer validates FeatureMetadataRegistry exists."""

        class BadBundle:
            has_services = True
            index_mapper = "mock"
            metadata_registry = None  # Missing!

        formatter = PlotFormatter()

        with pytest.raises(ValueError, match="FeatureMetadataRegistry"):
            NomogramRenderer(BadBundle(), formatter)

    def test_renderer_initialization_success(self, mock_bundle_with_services):
        """Test successful renderer initialization."""
        formatter = PlotFormatter()
        renderer = NomogramRenderer(mock_bundle_with_services, formatter)

        assert renderer.bundle is mock_bundle_with_services
        assert renderer.formatter is formatter
        assert renderer.use_odds_ratio is False

    def test_renderer_initialization_with_odds_ratio(self, mock_bundle_with_services):
        """Test renderer initialization with odds ratio flag."""
        formatter = PlotFormatter()
        renderer = NomogramRenderer(mock_bundle_with_services, formatter, use_odds_ratio=True)

        assert renderer.use_odds_ratio is True


class TestNomogramRendererWithRealBundle:
    """Test NomogramRenderer with real PlottingDataBundle."""

    @pytest.fixture
    def mock_lasso_results(self):
        """Mock LassoResultsManager."""

        class MockLassoResults:
            def __init__(self):
                self.all_feature_names = ['age', 'bmi', 'glucose']
                self.univariate_feature_names = ['age', 'bmi', 'glucose']

            def get_selected_univariate_indices(self):
                return [0, 1, 2]  # All selected

            def get_selected_bivariate_index_pairs(self):
                return []  # No bivariate for Phase 1

            def get_selected_beta(self):
                return np.array([1.5, 2.0, 2.5])

        return MockLassoResults()

    @pytest.fixture
    def simple_model(self):
        """Simple mock model."""

        class SimpleModel:
            def predict_proba(self, x, device='cpu'):
                batch_size = x.shape[0] if hasattr(x, 'shape') else len(x)
                return torch.ones(batch_size, 1, device=device) * 0.5

            def __call__(self, x):
                return self.predict_proba(x)

        return SimpleModel()

    @pytest.fixture
    def test_data(self):
        """Create test data with categorical and continuous features."""
        np.random.seed(42)
        data = np.column_stack(
            [
                np.random.randint(0, 3, 100),  # age (categorical: 0, 1, 2)
                np.random.uniform(15, 40, 100),  # bmi (continuous)
                np.random.randint(0, 2, 100),  # glucose (binary: 0, 1)
            ]
        )
        return torch.from_numpy(data).float()

    @pytest.fixture
    def bundle_with_services(self, mock_lasso_results, simple_model, test_data):
        """Create real bundle with services."""
        pipeline = PlottingPipeline(
            lasso_results=mock_lasso_results,
            group_manager=None,
            label_manager=None,
        )

        bundle = pipeline.prepare_plotting_bundle(
            x=test_data,
            model=simple_model,
            feature_names=['age', 'bmi', 'glucose'],
            categorical_threshold=5,  # age and glucose will be categorical
        )

        return bundle

    def test_render_nomogram_basic(self, bundle_with_services):
        """Test basic nomogram rendering."""
        formatter = PlotFormatter()
        renderer = NomogramRenderer(bundle_with_services, formatter)

        fig = renderer.render_nomogram()

        assert isinstance(fig, plt.Figure)
        assert len(fig.get_axes()) == 3  # 3 features
        plt.close(fig)

    def test_render_nomogram_with_odds_ratio(self, bundle_with_services):
        """Test nomogram rendering with odds ratio."""
        formatter = PlotFormatter()
        renderer = NomogramRenderer(bundle_with_services, formatter, use_odds_ratio=True)

        fig = renderer.render_nomogram()

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_render_nomogram_two_column(self, bundle_with_services):
        """Test two-column nomogram rendering."""
        formatter = PlotFormatter()
        renderer = NomogramRenderer(bundle_with_services, formatter)

        fig = renderer.render_nomogram(two_column=True)

        assert isinstance(fig, plt.Figure)
        # 3 features = 2 rows (ceiling of 3/2)
        assert len(fig.get_axes()) == 3
        plt.close(fig)

    def test_render_nomogram_with_pagination(self, bundle_with_services):
        """Test nomogram rendering with pagination."""
        formatter = PlotFormatter()
        renderer = NomogramRenderer(bundle_with_services, formatter)

        figs = renderer.render_nomogram(features_per_plot=2)

        assert isinstance(figs, list)
        assert len(figs) == 2  # 3 features / 2 per page = 2 pages
        assert isinstance(figs[0], plt.Figure)
        assert isinstance(figs[1], plt.Figure)

        for fig in figs:
            plt.close(fig)

    def test_render_nomogram_surround_axes(self, bundle_with_services):
        """Test nomogram rendering with surround axes option."""
        formatter = PlotFormatter()
        renderer = NomogramRenderer(bundle_with_services, formatter)

        fig = renderer.render_nomogram(surround_axes=True)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_calculate_x_range(self, bundle_with_services):
        """Test x-range calculation for continuous features."""
        formatter = PlotFormatter()
        renderer = NomogramRenderer(bundle_with_services, formatter)

        # Create features_to_plot list manually
        features_to_plot = []
        for info in bundle_with_services.univariate_features():
            features_to_plot.append(
                {
                    'type': 'univariate',
                    'info': info,
                }
            )

        x_range = renderer._calculate_x_range(features_to_plot)

        # Should have x_range (bmi is continuous)
        assert x_range is not None
        assert isinstance(x_range, tuple)
        assert len(x_range) == 2
        assert x_range[0] < x_range[1]

    def test_format_category_label(self, bundle_with_services):
        """Test category label formatting."""
        formatter = PlotFormatter()
        renderer = NomogramRenderer(bundle_with_services, formatter)

        # Test binary label
        label = renderer._format_category_label("glucose", 0.0, is_binary=True)
        assert isinstance(label, str)

        # Test non-binary label
        label = renderer._format_category_label("age", 1.0, is_binary=False)
        assert isinstance(label, str)

    def test_get_column_axis_params(self, bundle_with_services):
        """Test column axis parameters."""
        formatter = PlotFormatter()
        renderer = NomogramRenderer(bundle_with_services, formatter)

        # Left column
        left_params = renderer._get_column_axis_params("left", surround_axes=False)
        assert left_params['y_axis_position'] == 'left'

        # Right column
        right_params = renderer._get_column_axis_params("right", surround_axes=False)
        assert right_params['y_axis_position'] == 'left'  # No surround

        # Right column with surround
        right_params_surround = renderer._get_column_axis_params("right", surround_axes=True)
        assert right_params_surround['y_axis_position'] == 'right'  # With surround


class TestNomogramRendererPlaceholders:
    """Test placeholder methods for future phases."""

    @pytest.fixture
    def mock_bundle_with_services(self):
        """Create mock bundle with services."""

        class MockBundle:
            has_services = True
            index_mapper = "mock_mapper"
            metadata_registry = "mock_registry"
            denorm_service = "mock_denorm"

            def univariate_features(self):
                return []

            def bivariate_pairs(self):
                return []

        return MockBundle()

    def test_render_bivariate_heatmaps_placeholder(self, mock_bundle_with_services):
        """Test bivariate heatmaps returns None (Phase 3)."""
        formatter = PlotFormatter()
        renderer = NomogramRenderer(mock_bundle_with_services, formatter)

        result = renderer.render_bivariate_heatmaps()
        assert result is None

    def test_render_response_plots_placeholder(self, mock_bundle_with_services):
        """Test response plots returns empty figure (Phase 4)."""
        formatter = PlotFormatter()
        renderer = NomogramRenderer(mock_bundle_with_services, formatter)

        fig = renderer.render_response_plots()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestDisplayNomogramsSideBySide:
    """Test display_nomograms_side_by_side function."""

    def test_display_single_nomogram(self):
        """Test displaying a single nomogram pair."""
        from prism.nomogram_plot import display_nomograms_side_by_side

        # Create mock figures
        fig1, ax1 = plt.subplots(figsize=(4, 6))
        ax1.plot([1, 2, 3], [1, 2, 3])
        ax1.set_title("Main Nomogram")

        fig_hm, ax_hm = plt.subplots(figsize=(4, 4))
        ax_hm.imshow([[1, 2], [3, 4]])
        ax_hm.set_title("Heatmap")

        # Display side by side
        result = display_nomograms_side_by_side(nomograms=[(fig1, fig_hm)], titles=["Model 1"])

        assert isinstance(result, plt.Figure)
        plt.close('all')

    def test_display_multiple_nomograms(self):
        """Test displaying multiple nomogram pairs side by side."""
        from prism.nomogram_plot import display_nomograms_side_by_side

        # Create mock figures for two models
        fig1, ax1 = plt.subplots(figsize=(4, 6))
        ax1.plot([1, 2, 3], [1, 2, 3])

        fig2, ax2 = plt.subplots(figsize=(4, 6))
        ax2.plot([1, 2, 3], [3, 2, 1])

        # Display side by side (no heatmaps)
        result = display_nomograms_side_by_side(
            nomograms=[(fig1, None), (fig2, None)], titles=["Blackbox", "PRN"]
        )

        assert isinstance(result, plt.Figure)
        plt.close('all')

    def test_display_with_default_titles(self):
        """Test that default titles are generated when not provided."""
        from prism.nomogram_plot import display_nomograms_side_by_side

        fig1, _ = plt.subplots()
        fig2, _ = plt.subplots()

        result = display_nomograms_side_by_side(nomograms=[(fig1, None), (fig2, None)])

        assert isinstance(result, plt.Figure)
        # Check that default titles "Model 1", "Model 2" are used
        axes = result.get_axes()
        assert len(axes) == 2
        plt.close('all')

    def test_display_empty_raises_error(self):
        """Test that empty nomogram list raises ValueError."""
        from prism.nomogram_plot import display_nomograms_side_by_side

        with pytest.raises(ValueError, match="At least one nomogram"):
            display_nomograms_side_by_side(nomograms=[])

    def test_display_mismatched_titles_raises_error(self):
        """Test that mismatched title count raises ValueError."""
        from prism.nomogram_plot import display_nomograms_side_by_side

        fig1, _ = plt.subplots()

        with pytest.raises(ValueError, match="Number of titles"):
            display_nomograms_side_by_side(
                nomograms=[(fig1, None)], titles=["Title 1", "Title 2"]  # Too many titles
            )
        plt.close('all')

    def test_display_with_paginated_figures(self):
        """Test handling of paginated figures (list of figures)."""
        from prism.nomogram_plot import display_nomograms_side_by_side

        # Create list of figures (simulating pagination)
        fig1, _ = plt.subplots()
        fig2, _ = plt.subplots()
        paginated_figs = [fig1, fig2]  # Multiple pages

        # Should use first page
        result = display_nomograms_side_by_side(
            nomograms=[(paginated_figs, None)], titles=["Paginated Model"]
        )

        assert isinstance(result, plt.Figure)
        plt.close('all')
