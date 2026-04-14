"""
Tests for NomogramRenderer heatmap and contour rendering (Phase 3).

Tests cover:
- Catxcat heatmap rendering
- Contxcont contour rendering
- Colorbar formatting
- Text annotations on small grids
- Pagination for multiple heatmaps
- Odds ratio conversion
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from matplotlib.ticker import FuncFormatter

from prism.plotting.formatter import (
    PlotFormatter,
    calculate_required_precision,
    create_response_value_formatter,
    format_value_adaptive,
    format_value_for_annotation,
)
from prism.plotting.pipeline import PlottingPipeline
from prism.plotting.renderer import NomogramRenderer


class TestHeatmapRendering:
    """Test categoricalxcategorical heatmap rendering."""

    @pytest.fixture
    def mock_lasso_results_cat_cat(self):
        """Mock LassoResultsManager with catxcat bivariate pairs."""

        class MockLassoResults:
            def __init__(self):
                self.univariate_feature_names = ['gender', 'smoking', 'diagnosis', 'age']
                # all_feature_names includes univariate + ALL bivariate pair names
                # 4 univariate + 6 bivariate pairs (4 choose 2) = 10 total
                self.all_feature_names = self.univariate_feature_names + [
                    'gender : smoking',
                    'gender : diagnosis',
                    'gender : age',
                    'smoking : diagnosis',
                    'smoking : age',
                    'diagnosis : age',
                ]

            def get_selected_univariate_indices(self):
                return [0, 1, 2, 3]

            def get_selected_bivariate_index_pairs(self):
                # gender (binary) x smoking (binary), diagnosis (3-cat) x gender (binary)
                return [(0, 1), (0, 2)]

            def get_selected_beta(self):
                # Full beta: 4 univariate + 6 bivariate = 10 total
                # Only selected pairs have non-zero bivariate betas
                beta = np.zeros(10)
                beta[:4] = [1.0, 1.5, 2.0, 1.2]  # Univariate betas
                beta[4] = 0.5  # gender : smoking (pair index 0)
                beta[5] = 0.3  # gender : diagnosis (pair index 1)
                return beta

        return MockLassoResults()

    @pytest.fixture
    def simple_model(self):
        """Simple mock model."""

        class SimpleModel:
            def predict(self, x, device='cpu'):
                batch_size = x.shape[0] if hasattr(x, 'shape') else len(x)
                return torch.ones(batch_size, 1, device=device) * 0.5

            def __call__(self, x):
                return self.predict(x)

        return SimpleModel()

    @pytest.fixture
    def test_data_cat_cat(self):
        """Create test data with categorical features."""
        np.random.seed(42)
        data = np.column_stack(
            [
                np.random.randint(0, 2, 100),  # gender (binary)
                np.random.randint(0, 2, 100),  # smoking (binary)
                np.random.randint(0, 3, 100),  # diagnosis (3 categories)
                np.random.uniform(18, 90, 100),  # age (continuous, for variety)
            ]
        )
        return torch.from_numpy(data).float()

    @pytest.fixture
    def bundle_with_cat_cat(self, mock_lasso_results_cat_cat, simple_model, test_data_cat_cat):
        """Create bundle with catxcat bivariate pairs."""
        pipeline = PlottingPipeline(
            lasso_results=mock_lasso_results_cat_cat,
            group_manager=None,
            label_manager=None,
        )

        bundle = pipeline.prepare_plotting_bundle(
            x=test_data_cat_cat,
            model=simple_model,
            feature_names=['gender', 'smoking', 'diagnosis', 'age'],
            categorical_threshold=5,
        )

        bundle = pipeline.apply_beta_scaling(bundle)
        return bundle

    def test_render_bivariate_heatmaps_cat_cat(self, bundle_with_cat_cat):
        """Test rendering catxcat heatmaps."""
        formatter = PlotFormatter()
        renderer = NomogramRenderer(bundle_with_cat_cat, formatter)

        fig = renderer.render_bivariate_heatmaps()

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        # Should have 2 catxcat pairs
        assert len(fig.get_axes()) >= 2  # At least 2 (main axes + colorbars)
        plt.close(fig)

    def test_render_heatmap_directly(self, bundle_with_cat_cat):
        """Test _render_heatmap method directly."""
        formatter = PlotFormatter()
        renderer = NomogramRenderer(bundle_with_cat_cat, formatter)

        # Get first catxcat pair
        cat_cat_pair = None
        for pair_info in bundle_with_cat_cat.bivariate_pairs():
            if pair_info.skipped:
                continue

            i, j = pair_info.indices
            metadata_i = bundle_with_cat_cat.metadata_registry.get_by_collapsed(i)
            metadata_j = bundle_with_cat_cat.metadata_registry.get_by_collapsed(j)

            if metadata_i.is_categorical and metadata_j.is_categorical:
                cat_cat_pair = pair_info
                break

        assert cat_cat_pair is not None

        # Create figure and axis
        fig, ax = plt.subplots()

        # Render heatmap
        renderer._render_heatmap(ax, cat_cat_pair)

        # Check that heatmap was created
        assert len(ax.images) > 0  # Has image (heatmap)
        plt.close(fig)

    def test_heatmap_with_odds_ratio(self, bundle_with_cat_cat):
        """Test heatmap rendering with odds ratio conversion."""
        formatter = PlotFormatter()
        renderer = NomogramRenderer(bundle_with_cat_cat, formatter, use_odds_ratio=True)

        fig = renderer.render_bivariate_heatmaps()

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_heatmap_with_pagination(self, bundle_with_cat_cat):
        """Test heatmap rendering with pagination."""
        formatter = PlotFormatter()
        renderer = NomogramRenderer(bundle_with_cat_cat, formatter)

        # 1 heatmap per page: 2 total = 2 pages
        figs = renderer.render_bivariate_heatmaps(features_per_plot=1)

        assert isinstance(figs, list)
        assert len(figs) == 2
        assert all(isinstance(fig, plt.Figure) for fig in figs)

        for fig in figs:
            plt.close(fig)


class TestContourRendering:
    """Test continuousxcontinuous contour rendering."""

    @pytest.fixture
    def mock_lasso_results_cont_cont(self):
        """Mock LassoResultsManager with contxcont bivariate pairs."""

        class MockLassoResults:
            def __init__(self):
                self.univariate_feature_names = ['age', 'bmi', 'glucose', 'cholesterol']
                # all_feature_names includes univariate + ALL bivariate pair names
                # 4 univariate + 6 bivariate pairs (4 choose 2) = 10 total
                self.all_feature_names = self.univariate_feature_names + [
                    'age : bmi',
                    'age : glucose',
                    'age : cholesterol',
                    'bmi : glucose',
                    'bmi : cholesterol',
                    'glucose : cholesterol',
                ]

            def get_selected_univariate_indices(self):
                return [0, 1, 2, 3]

            def get_selected_bivariate_index_pairs(self):
                # age x bmi, glucose x cholesterol (all continuous)
                return [(0, 1), (2, 3)]

            def get_selected_beta(self):
                # Full beta: 4 univariate + 6 bivariate = 10 total
                # Only selected pairs have non-zero bivariate betas
                beta = np.zeros(10)
                beta[:4] = [1.0, 1.5, 2.0, 1.2]  # Univariate betas
                beta[4] = 0.5  # age : bmi (pair index 0)
                beta[9] = 0.3  # glucose : cholesterol (pair index 5)
                return beta

        return MockLassoResults()

    @pytest.fixture
    def simple_model(self):
        """Simple mock model."""

        class SimpleModel:
            def predict(self, x, device='cpu'):
                batch_size = x.shape[0] if hasattr(x, 'shape') else len(x)
                return torch.ones(batch_size, 1, device=device) * 0.5

            def __call__(self, x):
                return self.predict(x)

        return SimpleModel()

    @pytest.fixture
    def test_data_cont_cont(self):
        """Create test data with continuous features."""
        np.random.seed(42)
        data = np.column_stack(
            [
                np.random.uniform(18, 90, 100),  # age
                np.random.uniform(15, 40, 100),  # bmi
                np.random.uniform(70, 200, 100),  # glucose
                np.random.uniform(100, 300, 100),  # cholesterol
            ]
        )
        return torch.from_numpy(data).float()

    @pytest.fixture
    def bundle_with_cont_cont(
        self, mock_lasso_results_cont_cont, simple_model, test_data_cont_cont
    ):
        """Create bundle with contxcont bivariate pairs."""
        pipeline = PlottingPipeline(
            lasso_results=mock_lasso_results_cont_cont,
            group_manager=None,
            label_manager=None,
        )

        bundle = pipeline.prepare_plotting_bundle(
            x=test_data_cont_cont,
            model=simple_model,
            feature_names=['age', 'bmi', 'glucose', 'cholesterol'],
            categorical_threshold=5,  # All will be continuous
        )

        bundle = pipeline.apply_beta_scaling(bundle)
        return bundle

    def test_render_bivariate_heatmaps_cont_cont(self, bundle_with_cont_cont):
        """Test rendering contxcont contour plots."""
        formatter = PlotFormatter()
        renderer = NomogramRenderer(bundle_with_cont_cont, formatter)

        fig = renderer.render_bivariate_heatmaps()

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        # Should have 2 contxcont pairs
        assert len(fig.get_axes()) >= 2
        plt.close(fig)

    def test_render_contour_directly(self, bundle_with_cont_cont):
        """Test _render_contour method directly."""
        formatter = PlotFormatter()
        renderer = NomogramRenderer(bundle_with_cont_cont, formatter)

        # Get first contxcont pair
        cont_cont_pair = None
        for pair_info in bundle_with_cont_cont.bivariate_pairs():
            if pair_info.skipped:
                continue

            i, j = pair_info.indices
            metadata_i = bundle_with_cont_cont.metadata_registry.get_by_collapsed(i)
            metadata_j = bundle_with_cont_cont.metadata_registry.get_by_collapsed(j)

            if not metadata_i.is_categorical and not metadata_j.is_categorical:
                cont_cont_pair = pair_info
                break

        assert cont_cont_pair is not None

        # Create figure and axis
        fig, ax = plt.subplots()

        # Render contour
        renderer._render_contour(ax, cont_cont_pair)

        # Check that contour was created
        assert len(ax.collections) > 0  # Has contour collections
        plt.close(fig)

    def test_contour_with_odds_ratio(self, bundle_with_cont_cont):
        """Test contour rendering with odds ratio conversion."""
        formatter = PlotFormatter()
        renderer = NomogramRenderer(bundle_with_cont_cont, formatter, use_odds_ratio=True)

        fig = renderer.render_bivariate_heatmaps()

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestMixedBivariateExclusion:
    """Test that mixed bivariate pairs are excluded from heatmaps."""

    @pytest.fixture
    def mock_lasso_results_mixed(self):
        """Mock LassoResultsManager with only mixed bivariate pairs."""

        class MockLassoResults:
            def __init__(self):
                self.all_feature_names = ['age', 'bmi', 'gender']
                self.univariate_feature_names = ['age', 'bmi', 'gender']

            def get_selected_univariate_indices(self):
                return [0, 1, 2]

            def get_selected_bivariate_index_pairs(self):
                # Only mixed pairs: age(cat)xbmi(cont), gender(cat)xbmi(cont)
                return [(0, 1), (2, 1)]

            def get_selected_beta(self):
                return np.array([1.0, 1.5, 2.0, 0.5, 0.3])

        return MockLassoResults()

    @pytest.fixture
    def simple_model(self):
        """Simple mock model."""

        class SimpleModel:
            def predict(self, x, device='cpu'):
                batch_size = x.shape[0] if hasattr(x, 'shape') else len(x)
                return torch.ones(batch_size, 1, device=device) * 0.5

            def __call__(self, x):
                return self.predict(x)

        return SimpleModel()

    @pytest.fixture
    def test_data_mixed(self):
        """Create test data with mixed features."""
        np.random.seed(42)
        data = np.column_stack(
            [
                np.random.randint(0, 3, 100),  # age (categorical)
                np.random.uniform(15, 40, 100),  # bmi (continuous)
                np.random.randint(0, 2, 100),  # gender (binary)
            ]
        )
        return torch.from_numpy(data).float()

    def test_render_bivariate_heatmaps_no_non_mixed(
        self, mock_lasso_results_mixed, simple_model, test_data_mixed
    ):
        """Test that only mixed pairs returns None for heatmaps."""
        pipeline = PlottingPipeline(
            lasso_results=mock_lasso_results_mixed,
            group_manager=None,
            label_manager=None,
        )

        bundle = pipeline.prepare_plotting_bundle(
            x=test_data_mixed,
            model=simple_model,
            feature_names=['age', 'bmi', 'gender'],
            categorical_threshold=5,
        )

        formatter = PlotFormatter()
        renderer = NomogramRenderer(bundle, formatter)

        fig = renderer.render_bivariate_heatmaps()

        # Should return None since all pairs are mixed
        assert fig is None


class TestHeatmapPageLayout:
    """Test heatmap page layout and grid arrangement."""

    @pytest.fixture
    def mock_lasso_results_many_pairs(self):
        """Mock LassoResultsManager with many bivariate pairs."""

        class MockLassoResults:
            def __init__(self):
                # 5 categorical features = many catxcat pairs possible
                self.all_feature_names = ['f1', 'f2', 'f3', 'f4', 'f5']
                self.univariate_feature_names = ['f1', 'f2', 'f3', 'f4', 'f5']

            def get_selected_univariate_indices(self):
                return [0, 1, 2, 3, 4]

            def get_selected_bivariate_index_pairs(self):
                # 5 catxcat pairs
                return [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)]

            def get_selected_beta(self):
                # 5 univariate + 5 bivariate
                return np.array([1.0] * 10)

        return MockLassoResults()

    @pytest.fixture
    def simple_model(self):
        """Simple mock model."""

        class SimpleModel:
            def predict(self, x, device='cpu'):
                batch_size = x.shape[0] if hasattr(x, 'shape') else len(x)
                return torch.ones(batch_size, 1, device=device) * 0.5

            def __call__(self, x):
                return self.predict(x)

        return SimpleModel()

    @pytest.fixture
    def test_data_many_cat(self):
        """Create test data with many categorical features."""
        np.random.seed(42)
        data = np.column_stack(
            [
                np.random.randint(0, 2, 100),  # f1
                np.random.randint(0, 2, 100),  # f2
                np.random.randint(0, 2, 100),  # f3
                np.random.randint(0, 2, 100),  # f4
                np.random.randint(0, 2, 100),  # f5
            ]
        )
        return torch.from_numpy(data).float()

    def test_heatmap_grid_layout(
        self, mock_lasso_results_many_pairs, simple_model, test_data_many_cat
    ):
        """Test that multiple heatmaps are arranged in grid."""
        pipeline = PlottingPipeline(
            lasso_results=mock_lasso_results_many_pairs,
            group_manager=None,
            label_manager=None,
        )

        bundle = pipeline.prepare_plotting_bundle(
            x=test_data_many_cat,
            model=simple_model,
            feature_names=['f1', 'f2', 'f3', 'f4', 'f5'],
            categorical_threshold=5,
        )

        formatter = PlotFormatter()
        renderer = NomogramRenderer(bundle, formatter)

        fig = renderer.render_bivariate_heatmaps()

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        # 5 pairs = at least 5 main axes (plus colorbars)
        assert len(fig.get_axes()) >= 5
        plt.close(fig)


class TestAdaptiveFormatting:
    """Test range-adaptive number formatting for heatmaps."""

    def test_calculate_required_precision_wide_range(self):
        """Wide range (>= 10) requires 0 decimals."""
        precision = calculate_required_precision(0, 15)
        assert precision == 0

    def test_calculate_required_precision_moderate_range(self):
        """Moderate range (>= 1) requires 1 decimal."""
        precision = calculate_required_precision(0.5, 2.5)
        assert precision == 1

    def test_calculate_required_precision_tight_range(self):
        """Tight range (>= 0.1) requires 2 decimals."""
        precision = calculate_required_precision(0.95, 1.05)
        assert precision == 2

    def test_calculate_required_precision_very_tight_range(self):
        """Very tight range (>= 0.01) requires 3 decimals."""
        precision = calculate_required_precision(0.985, 1.015)
        assert precision == 3

    def test_calculate_required_precision_extreme_tight(self):
        """Extremely tight range (< 0.01) requires 4 decimals."""
        precision = calculate_required_precision(0.9985, 1.0015)
        assert precision == 4

    def test_calculate_required_precision_equal_values(self):
        """Equal min/max returns default precision 2."""
        precision = calculate_required_precision(1.0, 1.0)
        assert precision == 2

    def test_format_value_adaptive_tight_odds_ratio(self):
        """Tight OR range formats with sufficient precision."""
        precision = calculate_required_precision(0.98, 1.02)  # precision=2
        formatted = [format_value_adaptive(x, precision) for x in [0.98, 1.00, 1.02]]
        # All values should be distinguishable
        assert len(set(formatted)) == 3
        assert formatted[0] == "0.98"
        assert formatted[1] in ["1", "1.0"]  # rstrip may remove trailing .0
        assert formatted[2] == "1.02"

    def test_format_value_adaptive_extreme_values(self):
        """Extreme values use scientific notation with 1 decimal."""
        precision = 2
        assert format_value_adaptive(0.0005, precision) == "5.0e-04"
        assert format_value_adaptive(1500, precision) == "1.5e+03"
        assert format_value_adaptive(-0.0005, precision) == "-5.0e-04"

    def test_format_value_adaptive_zero(self):
        """Zero is always formatted as '0'."""
        assert format_value_adaptive(0, precision=2) == "0"

    def test_format_value_adaptive_with_precision_0(self):
        """Precision 0 formats as integers."""
        formatted = [format_value_adaptive(x, precision=0) for x in [0, 5, 10, 15]]
        assert formatted == ["0", "5", "10", "15"]

    def test_format_value_adaptive_with_precision_1(self):
        """Precision 1 formats with up to 1 decimal."""
        formatted = [format_value_adaptive(x, precision=1) for x in [0.5, 1.0, 1.5, 2.0]]
        # After rstrip, 1.0 and 2.0 become "1" and "2"
        assert formatted[0] == "0.5"
        assert formatted[1] in ["1", "1.0"]
        assert formatted[2] == "1.5"
        assert formatted[3] in ["2", "2.0"]

    def test_format_value_adaptive_with_precision_3(self):
        """Precision 3 formats with up to 3 decimals."""
        formatted = [format_value_adaptive(x, precision=3) for x in [0.985, 1.000, 1.015]]
        assert formatted[0] == "0.985"
        assert formatted[1] in ["1", "1.0", "1.00", "1.000"]  # rstrip removes trailing zeros
        assert formatted[2] == "1.015"

    def test_create_response_value_formatter_returns_func_formatter(self):
        """Response value formatter returns matplotlib FuncFormatter."""
        formatter = create_response_value_formatter(0.95, 1.05)
        assert isinstance(formatter, FuncFormatter)

    def test_create_response_value_formatter_formats_correctly(self):
        """Response value formatter formats values correctly."""
        formatter = create_response_value_formatter(0.98, 1.02)  # precision=2
        # FuncFormatter returns a callable that takes (value, position)
        format_func = formatter

        # Test formatting
        result_1 = format_func(0.98, 0)
        result_2 = format_func(1.00, 1)
        result_3 = format_func(1.02, 2)

        assert result_1 == "0.98"
        assert result_2 in ["1", "1.0"]
        assert result_3 == "1.02"

    def test_format_value_for_annotation_threshold_notation(self):
        """Annotation formatter uses threshold notation for extremes."""
        precision = 2

        # Very small values near zero
        assert format_value_for_annotation(0.0005, precision) == "~0"
        assert format_value_for_annotation(-0.0005, precision) == "~0"

        # Very large values
        assert format_value_for_annotation(1500, precision) == ">1000"
        assert format_value_for_annotation(-1500, precision) == "<-1000"

        # Normal range (same as adaptive)
        assert format_value_for_annotation(0.98, precision) == "0.98"
        assert format_value_for_annotation(1.02, precision) == "1.02"

    def test_annotation_formatter_differs_from_colorbar_formatter(self):
        """Annotation formatter differs from colorbar formatter for extremes."""
        # Colorbar formatter uses scientific notation
        assert format_value_adaptive(0.0005, precision=2) == "5.0e-04"
        assert format_value_adaptive(1500, precision=2) == "1.5e+03"

        # Annotation formatter uses threshold notation
        assert format_value_for_annotation(0.0005, precision=2) == "~0"
        assert format_value_for_annotation(1500, precision=2) == ">1000"

        # Both formatters agree in normal range
        assert format_value_adaptive(1.02, precision=2) == format_value_for_annotation(
            1.02, precision=2
        )
