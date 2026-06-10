"""
Tests for NomogramRenderer mixed bivariate rendering (Phase 2).

Tests cover:
- Mixed bivariate (cat x cont) in both directions
- Legend positioning (right vs top)
- Response grouping by category
- Category label formatting
- Integration with nomogram rendering
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from prism.plotting.formatter import PlotFormatter
from prism.plotting.pipeline import PlottingPipeline
from prism.plotting.renderer import NomogramRenderer


class TestMixedBivariateRendering:
    """Test mixed bivariate feature rendering."""

    @pytest.fixture
    def mock_lasso_results_with_mixed(self):
        """Mock LassoResultsManager with mixed bivariate pairs."""

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
                return [0, 1, 2, 3]  # All selected

            def get_selected_bivariate_index_pairs(self):
                # age (cat) x bmi (cont), glucose (binary) x cholesterol (cont)
                return [(0, 1), (2, 3)]

            def get_selected_beta(self):
                # Full beta: 4 univariate + 6 bivariate = 10 total
                # Only selected pairs have non-zero bivariate betas
                beta = np.zeros(10)
                beta[:4] = [1.5, 2.0, 1.0, 2.5]  # Univariate betas
                beta[4] = 0.5  # age : bmi (pair index 0)
                beta[9] = 0.3  # glucose : cholesterol (pair index 5)
                return beta

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
    def test_data_with_mixed(self):
        """Create test data with categorical and continuous features."""
        np.random.seed(42)
        data = np.column_stack(
            [
                np.random.randint(0, 3, 100),  # age (categorical: 0, 1, 2)
                np.random.uniform(15, 40, 100),  # bmi (continuous)
                np.random.randint(0, 2, 100),  # glucose (binary: 0, 1)
                np.random.uniform(100, 300, 100),  # cholesterol (continuous)
            ]
        )
        return torch.from_numpy(data).float()

    @pytest.fixture
    def bundle_with_mixed(self, mock_lasso_results_with_mixed, simple_model, test_data_with_mixed):
        """Create bundle with mixed bivariate pairs."""
        pipeline = PlottingPipeline(
            lasso_results=mock_lasso_results_with_mixed,
            group_manager=None,
            label_manager=None,
        )

        bundle = pipeline.prepare_plotting_bundle(
            x=test_data_with_mixed,
            model=simple_model,
            feature_names=['age', 'bmi', 'glucose', 'cholesterol'],
            categorical_threshold=5,  # age and glucose will be categorical
        )

        # Apply beta scaling for realistic responses
        bundle = pipeline.apply_beta_scaling(bundle)

        return bundle

    def test_render_nomogram_with_mixed_bivariate(self, bundle_with_mixed):
        """Test nomogram rendering includes mixed bivariate features."""
        formatter = PlotFormatter()
        renderer = NomogramRenderer(bundle_with_mixed, formatter)

        fig = renderer.render_nomogram()

        assert isinstance(fig, plt.Figure)
        # Should have 4 univariate + 2 mixed bivariate = 6 features
        axes = fig.get_axes()
        assert len(axes) == 6
        plt.close(fig)

    def test_render_nomogram_mixed_with_legend_on_right(self, bundle_with_mixed):
        """Test mixed bivariate with legend on right."""
        formatter = PlotFormatter()
        renderer = NomogramRenderer(bundle_with_mixed, formatter)

        fig = renderer.render_nomogram(legend_on_right=True)

        assert isinstance(fig, plt.Figure)
        assert len(fig.get_axes()) == 6
        plt.close(fig)

    def test_render_nomogram_mixed_two_column(self, bundle_with_mixed):
        """Test mixed bivariate with two-column layout."""
        formatter = PlotFormatter()
        renderer = NomogramRenderer(bundle_with_mixed, formatter)

        fig = renderer.render_nomogram(two_column=True)

        assert isinstance(fig, plt.Figure)
        # 6 features in two columns = still 6 axes
        assert len(fig.get_axes()) == 6
        plt.close(fig)

    def test_render_nomogram_mixed_with_odds_ratio(self, bundle_with_mixed):
        """Test mixed bivariate with odds ratio conversion."""
        formatter = PlotFormatter()
        renderer = NomogramRenderer(bundle_with_mixed, formatter, use_odds_ratio=True)

        fig = renderer.render_nomogram()

        assert isinstance(fig, plt.Figure)
        assert len(fig.get_axes()) == 6
        plt.close(fig)

    def test_render_nomogram_mixed_with_pagination(self, bundle_with_mixed):
        """Test mixed bivariate with pagination."""
        formatter = PlotFormatter()
        renderer = NomogramRenderer(bundle_with_mixed, formatter)

        # 3 features per page: 6 total / 3 per page = 2 pages
        figs = renderer.render_nomogram(features_per_plot=3)

        assert isinstance(figs, list)
        assert len(figs) == 2
        assert isinstance(figs[0], plt.Figure)
        assert isinstance(figs[1], plt.Figure)

        for fig in figs:
            plt.close(fig)

    def test_mixed_bivariate_count(self, bundle_with_mixed):
        """Test that bundle contains expected mixed bivariate pairs."""
        # Count mixed pairs
        mixed_count = 0
        for pair_info in bundle_with_mixed.bivariate_pairs():
            if pair_info.skipped:
                continue

            i, j = pair_info.indices
            metadata_i = bundle_with_mixed.metadata_registry.get_by_collapsed(i)
            metadata_j = bundle_with_mixed.metadata_registry.get_by_collapsed(j)

            if metadata_i.is_categorical != metadata_j.is_categorical:
                mixed_count += 1

        assert mixed_count == 2  # agexbmi and glucosexcholesterol

    def test_render_mixed_directly(self, bundle_with_mixed):
        """Test _render_mixed method directly."""
        formatter = PlotFormatter()
        renderer = NomogramRenderer(bundle_with_mixed, formatter)

        # Get first mixed bivariate pair
        mixed_pair = None
        for pair_info in bundle_with_mixed.bivariate_pairs():
            if pair_info.skipped:
                continue

            i, j = pair_info.indices
            metadata_i = bundle_with_mixed.metadata_registry.get_by_collapsed(i)
            metadata_j = bundle_with_mixed.metadata_registry.get_by_collapsed(j)

            if metadata_i.is_categorical != metadata_j.is_categorical:
                mixed_pair = pair_info
                break

        assert mixed_pair is not None

        # Create figure and axis
        fig, ax = plt.subplots()
        subplot_params = renderer._get_column_axis_params("left", surround_axes=False)

        # Render mixed bivariate
        renderer._render_mixed(ax, mixed_pair, subplot_params, legend_on_right=False)

        # Check that plot was created
        assert len(ax.lines) > 0 or len(ax.collections) > 0  # Has lines or scatter points
        plt.close(fig)

    def test_format_category_label_for_mixed(self, bundle_with_mixed):
        """Test category label formatting used in mixed rendering."""
        formatter = PlotFormatter()
        renderer = NomogramRenderer(bundle_with_mixed, formatter)

        # Test with different values
        label_binary = renderer._format_category_label("glucose", 0.0, is_binary=True)
        assert isinstance(label_binary, str)

        label_multicat = renderer._format_category_label("age", 1.0, is_binary=False)
        assert isinstance(label_multicat, str)


class TestMixedBivariateEdgeCases:
    """Test edge cases for mixed bivariate rendering."""

    @pytest.fixture
    def mock_lasso_results_no_bivariate(self):
        """Mock LassoResultsManager with no bivariate pairs."""

        class MockLassoResults:
            def __init__(self):
                self.all_feature_names = ['age', 'bmi']
                self.univariate_feature_names = ['age', 'bmi']

            def get_selected_univariate_indices(self):
                return [0, 1]

            def get_selected_bivariate_index_pairs(self):
                return []  # No bivariate

            def get_selected_beta(self):
                return np.array([1.5, 2.0])

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
    def test_data_simple(self):
        """Create simple test data."""
        np.random.seed(42)
        data = np.column_stack(
            [
                np.random.randint(0, 3, 100),  # age (categorical)
                np.random.uniform(15, 40, 100),  # bmi (continuous)
            ]
        )
        return torch.from_numpy(data).float()

    def test_render_nomogram_no_bivariate(
        self, mock_lasso_results_no_bivariate, simple_model, test_data_simple
    ):
        """Test nomogram rendering with no bivariate features."""
        pipeline = PlottingPipeline(
            lasso_results=mock_lasso_results_no_bivariate,
            group_manager=None,
            label_manager=None,
        )

        bundle = pipeline.prepare_plotting_bundle(
            x=test_data_simple,
            model=simple_model,
            feature_names=['age', 'bmi'],
            categorical_threshold=5,
        )

        formatter = PlotFormatter()
        renderer = NomogramRenderer(bundle, formatter)

        fig = renderer.render_nomogram()

        assert isinstance(fig, plt.Figure)
        # Only 2 univariate features
        assert len(fig.get_axes()) == 2
        plt.close(fig)
