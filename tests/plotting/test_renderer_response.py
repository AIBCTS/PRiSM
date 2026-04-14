"""
Tests for NomogramRenderer response plot rendering (Phase 4).

Tests cover:
- Continuous univariate response with histogram
- Categorical univariate response with histogram
- Mixed bivariate response with histogram
- render_response_plots() method
- Pagination
- Odds ratio conversion
"""

import matplotlib
import numpy as np
import pytest
import torch

matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from prism.plotting.formatter import PlotFormatter
from prism.plotting.pipeline import PlottingPipeline
from prism.plotting.renderer import NomogramRenderer


class TestContinuousResponseWithHistogram:
    """Test continuous univariate response rendering with histogram."""

    @pytest.fixture
    def mock_lasso_results_continuous(self):
        """Mock LassoResultsManager with continuous features."""

        class MockLassoResults:
            def __init__(self):
                self.all_feature_names = ['age', 'bmi', 'glucose']
                self.univariate_feature_names = self.all_feature_names

            def get_selected_univariate_indices(self):
                return [0, 1, 2]

            def get_selected_bivariate_index_pairs(self):
                return []

            def get_selected_beta(self):
                return np.array([1.0, 1.5, 2.0])

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
    def test_data_continuous(self):
        """Create test data with continuous features."""
        np.random.seed(42)
        data = np.column_stack(
            [
                np.random.uniform(20, 80, 100),  # age
                np.random.uniform(18, 35, 100),  # bmi
                np.random.uniform(70, 200, 100),  # glucose
            ]
        )
        return torch.from_numpy(data).float()

    @pytest.fixture
    def bundle_continuous(self, mock_lasso_results_continuous, simple_model, test_data_continuous):
        """Create bundle with continuous features."""
        pipeline = PlottingPipeline(
            lasso_results=mock_lasso_results_continuous,
            group_manager=None,
            label_manager=None,
        )

        bundle = pipeline.prepare_plotting_bundle(
            x=test_data_continuous,
            model=simple_model,
            feature_names=['age', 'bmi', 'glucose'],
            categorical_threshold=5,
        )

        bundle = pipeline.apply_beta_scaling(bundle)
        return bundle

    def test_render_continuous_response_directly(self, bundle_continuous):
        """Test _render_continuous_response_with_histogram method directly."""
        formatter = PlotFormatter()
        renderer = NomogramRenderer(bundle_continuous, formatter)

        # Get first univariate feature
        info = bundle_continuous.univariate_features()[0]

        # Create figure and axis
        fig, ax = plt.subplots()

        # Render continuous response with histogram
        renderer._render_continuous_response_with_histogram(ax, info)

        # Check that plot and histogram were created
        assert len(ax.lines) > 0  # Has response line
        assert len(fig.get_axes()) == 2  # Main ax + histogram ax
        plt.close(fig)

    def test_continuous_response_with_odds_ratio(self, bundle_continuous):
        """Test continuous response with odds ratio conversion."""
        formatter = PlotFormatter()
        renderer = NomogramRenderer(bundle_continuous, formatter, use_odds_ratio=True)

        info = bundle_continuous.univariate_features()[0]
        fig, ax = plt.subplots()

        renderer._render_continuous_response_with_histogram(ax, info)

        # Check that y-axis is log scale
        assert ax.get_yscale() == 'linear'  # Changed from log to linear for clearer response plots
        plt.close(fig)


class TestCategoricalResponseWithHistogram:
    """Test categorical univariate response rendering with histogram."""

    @pytest.fixture
    def mock_lasso_results_categorical(self):
        """Mock LassoResultsManager with categorical features."""

        class MockLassoResults:
            def __init__(self):
                self.all_feature_names = ['gender', 'smoking', 'diagnosis']
                self.univariate_feature_names = self.all_feature_names

            def get_selected_univariate_indices(self):
                return [0, 1, 2]

            def get_selected_bivariate_index_pairs(self):
                return []

            def get_selected_beta(self):
                return np.array([1.0, 1.5, 2.0])

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
    def test_data_categorical(self):
        """Create test data with categorical features."""
        np.random.seed(42)
        data = np.column_stack(
            [
                np.random.randint(0, 2, 100),  # gender (binary)
                np.random.randint(0, 2, 100),  # smoking (binary)
                np.random.randint(0, 3, 100),  # diagnosis (3 categories)
            ]
        )
        return torch.from_numpy(data).float()

    @pytest.fixture
    def bundle_categorical(
        self, mock_lasso_results_categorical, simple_model, test_data_categorical
    ):
        """Create bundle with categorical features."""
        pipeline = PlottingPipeline(
            lasso_results=mock_lasso_results_categorical,
            group_manager=None,
            label_manager=None,
        )

        bundle = pipeline.prepare_plotting_bundle(
            x=test_data_categorical,
            model=simple_model,
            feature_names=['gender', 'smoking', 'diagnosis'],
            categorical_threshold=5,
        )

        bundle = pipeline.apply_beta_scaling(bundle)
        return bundle

    def test_render_categorical_response_directly(self, bundle_categorical):
        """Test _render_categorical_response_with_histogram method directly."""
        formatter = PlotFormatter()
        renderer = NomogramRenderer(bundle_categorical, formatter)

        # Get first univariate feature
        info = bundle_categorical.univariate_features()[0]

        # Create figure and axis
        fig, ax = plt.subplots()

        # Render categorical response with histogram
        renderer._render_categorical_response_with_histogram(ax, info)

        # Check that scatter plot and histogram were created
        assert len(ax.collections) > 0  # Has scatter points
        assert len(fig.get_axes()) == 2  # Main ax + histogram ax
        plt.close(fig)

    def test_categorical_response_with_odds_ratio(self, bundle_categorical):
        """Test categorical response with odds ratio conversion."""
        formatter = PlotFormatter()
        renderer = NomogramRenderer(bundle_categorical, formatter, use_odds_ratio=True)

        info = bundle_categorical.univariate_features()[0]
        fig, ax = plt.subplots()

        renderer._render_categorical_response_with_histogram(ax, info)

        # Check that y-axis is log scale
        assert ax.get_yscale() == 'linear'  # Changed from log to linear for clearer response plots
        plt.close(fig)


class TestMixedResponseWithHistogram:
    """Test mixed bivariate response rendering with histogram."""

    @pytest.fixture
    def mock_lasso_results_mixed(self):
        """Mock LassoResultsManager with mixed bivariate pairs."""

        class MockLassoResults:
            def __init__(self):
                self.univariate_feature_names = ['age', 'bmi', 'gender', 'glucose']
                # all_feature_names includes univariate + ALL bivariate pair names
                # 4 univariate + 6 bivariate pairs (4 choose 2) = 10 total
                self.all_feature_names = self.univariate_feature_names + [
                    'age : bmi',
                    'age : gender',
                    'age : glucose',
                    'bmi : gender',
                    'bmi : glucose',
                    'gender : glucose',
                ]

            def get_selected_univariate_indices(self):
                return [0, 1, 2, 3]

            def get_selected_bivariate_index_pairs(self):
                # age (cat) x bmi (cont), gender (binary) x glucose (cont)
                return [(0, 1), (2, 3)]

            def get_selected_beta(self):
                # Full beta: 4 univariate + 6 bivariate = 10 total
                # Only selected pairs have non-zero bivariate betas
                beta = np.zeros(10)
                beta[:4] = [1.0, 1.5, 2.0, 1.2]  # Univariate betas
                beta[4] = 0.5  # age : bmi (pair index 0)
                beta[9] = 0.3  # gender : glucose (pair index 5)
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
    def test_data_mixed(self):
        """Create test data with mixed features."""
        np.random.seed(42)
        data = np.column_stack(
            [
                np.random.randint(0, 3, 100),  # age (categorical: 0, 1, 2)
                np.random.uniform(18, 35, 100),  # bmi (continuous)
                np.random.randint(0, 2, 100),  # gender (binary)
                np.random.uniform(70, 200, 100),  # glucose (continuous)
            ]
        )
        return torch.from_numpy(data).float()

    @pytest.fixture
    def bundle_mixed(self, mock_lasso_results_mixed, simple_model, test_data_mixed):
        """Create bundle with mixed bivariate pairs."""
        pipeline = PlottingPipeline(
            lasso_results=mock_lasso_results_mixed,
            group_manager=None,
            label_manager=None,
        )

        bundle = pipeline.prepare_plotting_bundle(
            x=test_data_mixed,
            model=simple_model,
            feature_names=['age', 'bmi', 'gender', 'glucose'],
            categorical_threshold=5,
        )

        bundle = pipeline.apply_beta_scaling(bundle)
        return bundle

    def test_render_mixed_response_directly(self, bundle_mixed):
        """Test _render_mixed_response_with_histogram method directly."""
        formatter = PlotFormatter()
        renderer = NomogramRenderer(bundle_mixed, formatter)

        # Get first mixed bivariate pair
        mixed_pair = None
        for pair_info in bundle_mixed.bivariate_pairs():
            if pair_info.skipped:
                continue

            i, j = pair_info.indices
            metadata_i = bundle_mixed.metadata_registry.get_by_collapsed(i)
            metadata_j = bundle_mixed.metadata_registry.get_by_collapsed(j)

            if metadata_i.is_categorical != metadata_j.is_categorical:
                mixed_pair = pair_info
                break

        assert mixed_pair is not None

        # Create figure and axis
        fig, ax = plt.subplots()

        # Render mixed response with histogram
        renderer._render_mixed_response_with_histogram(ax, mixed_pair)

        # Check that plot and histogram were created
        assert len(ax.lines) > 0  # Has response lines
        assert len(fig.get_axes()) == 2  # Main ax + histogram ax
        plt.close(fig)

    def test_mixed_response_with_odds_ratio(self, bundle_mixed):
        """Test mixed response with odds ratio conversion."""
        formatter = PlotFormatter()
        renderer = NomogramRenderer(bundle_mixed, formatter, use_odds_ratio=True)

        # Get first mixed pair
        mixed_pair = None
        for pair_info in bundle_mixed.bivariate_pairs():
            if pair_info.skipped:
                continue

            i, j = pair_info.indices
            metadata_i = bundle_mixed.metadata_registry.get_by_collapsed(i)
            metadata_j = bundle_mixed.metadata_registry.get_by_collapsed(j)

            if metadata_i.is_categorical != metadata_j.is_categorical:
                mixed_pair = pair_info
                break

        fig, ax = plt.subplots()
        renderer._render_mixed_response_with_histogram(ax, mixed_pair)

        # Check that y-axis is log scale
        assert ax.get_yscale() == 'linear'  # Changed from log to linear for clearer response plots
        plt.close(fig)


class TestRenderResponsePlots:
    """Test render_response_plots() method."""

    @pytest.fixture
    def mock_lasso_results_comprehensive(self):
        """Mock LassoResultsManager with all feature types."""

        class MockLassoResults:
            def __init__(self):
                self.univariate_feature_names = ['age', 'bmi', 'gender', 'glucose', 'cholesterol']
                # all_feature_names includes univariate + ALL bivariate pair names
                # 5 univariate + 10 bivariate pairs (5 choose 2) = 15 total
                self.all_feature_names = self.univariate_feature_names + [
                    'age : bmi',
                    'age : gender',
                    'age : glucose',
                    'age : cholesterol',
                    'bmi : gender',
                    'bmi : glucose',
                    'bmi : cholesterol',
                    'gender : glucose',
                    'gender : cholesterol',
                    'glucose : cholesterol',
                ]

            def get_selected_univariate_indices(self):
                return [0, 1, 2, 3, 4]

            def get_selected_bivariate_index_pairs(self):
                # Mixed: age (cat) x bmi (cont), gender (binary) x glucose (cont)
                # Non-mixed: bmi x glucose (for heatmaps)
                return [(0, 1), (2, 3), (1, 3)]

            def get_selected_beta(self):
                # Full beta: 5 univariate + 10 bivariate = 15 total
                # Only selected pairs have non-zero bivariate betas
                beta = np.zeros(15)
                beta[:5] = [1.0, 1.5, 2.0, 1.2, 1.8]  # Univariate betas
                beta[5] = 0.5  # age : bmi (pair index 0)
                beta[12] = 0.3  # gender : glucose (pair index 7)
                beta[10] = 0.4  # bmi : glucose (pair index 5)
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
    def test_data_comprehensive(self):
        """Create comprehensive test data."""
        np.random.seed(42)
        data = np.column_stack(
            [
                np.random.randint(0, 3, 100),  # age (categorical: 0, 1, 2)
                np.random.uniform(18, 35, 100),  # bmi (continuous)
                np.random.randint(0, 2, 100),  # gender (binary)
                np.random.uniform(70, 200, 100),  # glucose (continuous)
                np.random.uniform(100, 300, 100),  # cholesterol (continuous)
            ]
        )
        return torch.from_numpy(data).float()

    @pytest.fixture
    def bundle_comprehensive(
        self, mock_lasso_results_comprehensive, simple_model, test_data_comprehensive
    ):
        """Create comprehensive bundle."""
        pipeline = PlottingPipeline(
            lasso_results=mock_lasso_results_comprehensive,
            group_manager=None,
            label_manager=None,
        )

        bundle = pipeline.prepare_plotting_bundle(
            x=test_data_comprehensive,
            model=simple_model,
            feature_names=['age', 'bmi', 'gender', 'glucose', 'cholesterol'],
            categorical_threshold=5,
        )

        bundle = pipeline.apply_beta_scaling(bundle)
        return bundle

    def test_render_response_plots_basic(self, bundle_comprehensive):
        """Test basic response plot rendering."""
        formatter = PlotFormatter()
        renderer = NomogramRenderer(bundle_comprehensive, formatter)

        fig = renderer.render_response_plots()

        assert isinstance(fig, plt.Figure)
        # Should have 5 univariate + 2 mixed = 7 total
        axes = fig.get_axes()
        # Each response plot has 2 axes (main + histogram), so 7 * 2 = 14
        # Note: No key generated because 'age' has default integer labels (0, 1, 2)
        assert len(axes) == 14
        plt.close(fig)

    def test_render_response_plots_with_labels(self, bundle_comprehensive):
        """Test response plot rendering with meaningful labels (triggers key generation)."""
        # Create a label manager with meaningful labels for the categorical feature 'age'
        # 'age' has values 0, 1, 2. We map them to "Young", "Middle", "Old"
        # Note: FeatureLabelManager maps column names to labels, but here we need
        # to map CATEGORY VALUES to labels.
        # The renderer uses formatter.format_feature_label which uses label_manager.
        # However, FeatureLabelManager currently only maps feature names, not values.
        # The renderer's _format_category_label method uses formatter.format_feature_label.
        # Let's check PlotFormatter.format_feature_label.

        # Actually, looking at renderer.py:
        # cat_label = self._format_category_label(name, cat_val, is_binary=False)
        # return self.formatter.format_feature_label(feature_name, value, is_binary, precision=4)

        # And PlotFormatter.format_feature_label:
        # if self.categorical_labels and feature_name in self.categorical_labels:
        #     return self.categorical_labels[feature_name].get(value, str(value))

        # So we need to inject categorical_labels into the formatter, NOT the label_manager.

        categorical_labels = {'age': {0.0: 'Young', 1.0: 'Middle', 2.0: 'Old'}}

        formatter = PlotFormatter(categorical_labels=categorical_labels)
        renderer = NomogramRenderer(bundle_comprehensive, formatter)

        fig = renderer.render_response_plots()

        assert isinstance(fig, plt.Figure)
        axes = fig.get_axes()

        # Should have:
        # 7 feature plots * 2 axes (main + histogram) = 14 axes
        # + 1 key subplot for 'age' because it now has meaningful labels
        # Total = 15 axes
        assert len(axes) == 15
        plt.close(fig)

    def test_render_response_plots_with_pagination(self, bundle_comprehensive):
        """Test response plot rendering with pagination."""
        formatter = PlotFormatter()
        renderer = NomogramRenderer(bundle_comprehensive, formatter)

        # 3 features per page
        figs = renderer.render_response_plots(features_per_plot=3)

        assert isinstance(figs, list)
        # 7 total features / 3 per page = 3 pages
        assert len(figs) == 3
        assert all(isinstance(fig, plt.Figure) for fig in figs)

        for fig in figs:
            plt.close(fig)

    def test_render_response_plots_with_odds_ratio(self, bundle_comprehensive):
        """Test response plots with odds ratio conversion."""
        formatter = PlotFormatter()
        renderer = NomogramRenderer(bundle_comprehensive, formatter, use_odds_ratio=True)

        fig = renderer.render_response_plots()

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_render_response_plots_custom_subfig_size(self, bundle_comprehensive):
        """Test response plots with custom subplot size."""
        formatter = PlotFormatter()
        renderer = NomogramRenderer(bundle_comprehensive, formatter)

        fig = renderer.render_response_plots(subfig_size=5.0)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestResponsePlotsEdgeCases:
    """Test edge cases for response plot rendering."""

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

    def test_render_response_plots_no_features(self, simple_model):
        """Test response plots with no features."""

        class EmptyLassoResults:
            def __init__(self):
                self.all_feature_names = []
                self.univariate_feature_names = []

            def get_selected_univariate_indices(self):
                return []

            def get_selected_bivariate_index_pairs(self):
                return []

            def get_selected_beta(self):
                return np.array([])

        pipeline = PlottingPipeline(
            lasso_results=EmptyLassoResults(),
            group_manager=None,
            label_manager=None,
        )

        np.random.seed(42)
        data = np.random.uniform(0, 1, (50, 3))

        bundle = pipeline.prepare_plotting_bundle(
            x=torch.from_numpy(data).float(),
            model=simple_model,
            feature_names=['f1', 'f2', 'f3'],
            categorical_threshold=5,
        )

        formatter = PlotFormatter()
        renderer = NomogramRenderer(bundle, formatter)

        fig = renderer.render_response_plots()

        # Should return empty figure with warning
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_render_response_plots_only_univariate(self, simple_model):
        """Test response plots with only univariate features."""

        class UnivariateOnlyResults:
            def __init__(self):
                self.all_feature_names = ['f1', 'f2']
                self.univariate_feature_names = self.all_feature_names

            def get_selected_univariate_indices(self):
                return [0, 1]

            def get_selected_bivariate_index_pairs(self):
                return []

            def get_selected_beta(self):
                return np.array([1.0, 1.5])

        pipeline = PlottingPipeline(
            lasso_results=UnivariateOnlyResults(),
            group_manager=None,
            label_manager=None,
        )

        np.random.seed(42)
        data = np.random.uniform(0, 1, (50, 2))

        bundle = pipeline.prepare_plotting_bundle(
            x=torch.from_numpy(data).float(),
            model=simple_model,
            feature_names=['f1', 'f2'],
            categorical_threshold=5,
        )

        bundle = pipeline.apply_beta_scaling(bundle)

        formatter = PlotFormatter()
        renderer = NomogramRenderer(bundle, formatter)

        fig = renderer.render_response_plots()

        assert isinstance(fig, plt.Figure)
        # 2 univariate features, each with 2 axes (main + histogram)
        assert len(fig.get_axes()) == 4
        plt.close(fig)

    def test_render_response_plots_with_non_mixed_bivariate(self, simple_model):
        """Test response plots exclude non-mixed bivariate features."""

        class MixedAndNonMixedResults:
            def __init__(self):
                self.univariate_feature_names = ['cat', 'cont1', 'cont2']
                # all_feature_names includes univariate + ALL bivariate pair names
                # 3 univariate + 3 bivariate pairs (3 choose 2) = 6 total
                self.all_feature_names = self.univariate_feature_names + [
                    'cat : cont1',
                    'cat : cont2',
                    'cont1 : cont2',
                ]

            def get_selected_univariate_indices(self):
                return [0, 1, 2]  # All univariate

            def get_selected_bivariate_index_pairs(self):
                # Mixed: cat x cont1
                # Non-mixed: cont1 x cont2 (should be excluded from response plots)
                return [(0, 1), (1, 2)]

            def get_selected_beta(self):
                # Full beta: 3 univariate + 3 bivariate = 6 total
                # Only selected pairs have non-zero bivariate betas
                beta = np.zeros(6)
                beta[:3] = [1.0, 1.5, 2.0]  # Univariate betas
                beta[3] = 0.5  # cat : cont1 (pair index 0)
                beta[5] = 0.3  # cont1 : cont2 (pair index 2)
                return beta

        pipeline = PlottingPipeline(
            lasso_results=MixedAndNonMixedResults(),
            group_manager=None,
            label_manager=None,
        )

        np.random.seed(42)
        data = np.column_stack(
            [
                np.random.randint(0, 2, 50),  # categorical
                np.random.uniform(0, 1, 50),  # continuous
                np.random.uniform(0, 1, 50),  # continuous
            ]
        )

        bundle = pipeline.prepare_plotting_bundle(
            x=torch.from_numpy(data).float(),
            model=simple_model,
            feature_names=['cat', 'cont1', 'cont2'],
            categorical_threshold=5,
        )

        bundle = pipeline.apply_beta_scaling(bundle)

        formatter = PlotFormatter()
        renderer = NomogramRenderer(bundle, formatter)

        fig = renderer.render_response_plots()

        assert isinstance(fig, plt.Figure)
        # 3 univariate + 1 mixed = 4 plots, each with 2 axes (main + histogram) = 8 axes
        assert len(fig.get_axes()) == 8
        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
