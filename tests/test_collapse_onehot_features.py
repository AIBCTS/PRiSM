"""
Tests for one-hot group histogram data in partial response plots.

These tests verify that collapse_onehot_features correctly handles both
scaled and unscaled data, ensuring histograms display the correct category
distribution.

Key fix (Issue #9): Changed from == 1 check to argmax-based detection,
which works for both scaled and unscaled one-hot data.
"""

import numpy as np
import pytest
import torch

from prism.partial_responses import PartialResponseCalculator
from prism.preprocessing import NoScaler, OneHotGroupManager, PRiSMScaler, collapse_onehot_features


class TestCollapseOnehotFeatures:
    """Tests for collapse_onehot_features function."""

    @pytest.fixture
    def sample_data_with_onehot(self):
        """Create sample data with one-hot encoded groups."""
        np.random.seed(42)
        n_samples = 100

        continuous_1 = np.random.randn(n_samples)
        continuous_2 = np.random.randn(n_samples)

        # Categorical with 3 categories (0=reference, 1=cat_A, 2=cat_B)
        categorical = np.random.randint(0, 3, n_samples)
        cat_A = (categorical == 1).astype(float)
        cat_B = (categorical == 2).astype(float)

        X_onehot = np.column_stack([continuous_1, continuous_2, cat_A, cat_B])
        feature_names = ['cont1', 'cont2', 'cat_A', 'cat_B']

        group_manager = OneHotGroupManager(
            groups_dict={'cat': ['cat_A', 'cat_B']}, reference_columns={'cat': 'cat_ref'}
        )

        return {
            'X_onehot': X_onehot,
            'feature_names': feature_names,
            'group_manager': group_manager,
            'categorical_values': categorical,
        }

    def test_collapse_produces_correct_categorical_values(self, sample_data_with_onehot):
        """Verify that collapse_onehot_features produces correct categorical values."""
        data = sample_data_with_onehot

        X_collapsed, collapsed_names = collapse_onehot_features(
            data['X_onehot'], data['group_manager'], data['feature_names']
        )

        cat_idx = collapsed_names.index('cat')
        collapsed_cat_values = X_collapsed[:, cat_idx]

        # Verify mapping: reference=0, cat_A=1, cat_B=2
        np.testing.assert_array_equal(
            collapsed_cat_values,
            data['categorical_values'],
            err_msg="Collapsed values should match ground truth categories",
        )

    def test_collapse_column_order_matches_calculator(self, sample_data_with_onehot):
        """Verify collapse_onehot_features order matches PartialResponseCalculator order."""
        data = sample_data_with_onehot

        _, collapsed_names_from_collapse = collapse_onehot_features(
            data['X_onehot'], data['group_manager'], data['feature_names']
        )

        class MockModel:
            def predict(self, x, device=None):
                if isinstance(x, torch.Tensor):
                    return torch.sigmoid(x.sum(dim=1))
                return torch.sigmoid(torch.tensor(x).sum(dim=1))

            def __call__(self, x):
                return self.predict(x)

        calculator = PartialResponseCalculator(
            model=MockModel(),
            method='dirac',
            device='cpu',
            input_dim=4,
            group_manager=data['group_manager'],
            feature_names=data['feature_names'],
            scaler=NoScaler(),
        )

        assert collapsed_names_from_collapse == calculator.collapsed_feature_names

    def test_histogram_data_matches_ground_truth(self, sample_data_with_onehot):
        """Test that histogram data matches ground truth distribution."""
        data = sample_data_with_onehot

        X_collapsed, collapsed_names = collapse_onehot_features(
            data['X_onehot'], data['group_manager'], data['feature_names']
        )

        cat_idx = collapsed_names.index('cat')
        histogram_data = X_collapsed[:, cat_idx]
        categories, counts = np.unique(histogram_data, return_counts=True)

        gt_categories, gt_counts = np.unique(data['categorical_values'], return_counts=True)

        np.testing.assert_array_equal(categories, gt_categories)
        np.testing.assert_array_equal(counts, gt_counts)

    def test_scaled_data_collapse_works_correctly(self):
        """
        Test that collapse_onehot_features works correctly on SCALED data.

        The argmax-based detection finds the active one-hot column regardless
        of whether data has been scaled (Issue #9 fix).
        """
        np.random.seed(42)
        n_samples = 100

        # Create one-hot encoded data
        categories = np.random.choice([0, 1, 2], size=n_samples, p=[0.4, 0.3, 0.3])
        cat_A = (categories == 1).astype(float)
        cat_B = (categories == 2).astype(float)
        continuous = np.random.randn(n_samples)

        X_original = np.column_stack([cat_A, cat_B, continuous])
        feature_names = ['cat_A', 'cat_B', 'continuous']

        group_manager = OneHotGroupManager(
            groups_dict={'cat': ['cat_A', 'cat_B']}, reference_columns={'cat': 'cat_REF'}
        )

        # Collapse UNSCALED data
        X_collapsed_unscaled, _ = collapse_onehot_features(
            X_original, group_manager, feature_names
        )

        # Scale and collapse SCALED data
        scaler = PRiSMScaler(scaler='median_std')
        scaler.fit(X_original)
        X_scaled = scaler.transform(X_original)

        X_collapsed_scaled, _ = collapse_onehot_features(X_scaled, group_manager, feature_names)

        # Both should produce identical categorical values
        np.testing.assert_array_equal(
            X_collapsed_unscaled[:, 0],
            X_collapsed_scaled[:, 0],
            err_msg="Scaled and unscaled collapse should give identical results",
        )

        # Verify all 3 categories present
        assert len(np.unique(X_collapsed_scaled[:, 0])) == 3


class TestColumnOrder:
    """Tests for collapsed column ordering."""

    @pytest.fixture
    def multi_group_data(self):
        """Create data with multiple one-hot groups."""
        np.random.seed(42)
        n_samples = 100

        cont1 = np.random.randn(n_samples)
        cont2 = np.random.randn(n_samples)
        cont3 = np.random.randn(n_samples)

        catA = np.random.randint(0, 3, n_samples)
        groupA_1 = (catA == 1).astype(float)
        groupA_2 = (catA == 2).astype(float)

        catB = np.random.randint(0, 4, n_samples)
        groupB_1 = (catB == 1).astype(float)
        groupB_2 = (catB == 2).astype(float)
        groupB_3 = (catB == 3).astype(float)

        X_onehot = np.column_stack(
            [cont1, cont2, groupA_1, groupA_2, cont3, groupB_1, groupB_2, groupB_3]
        )
        feature_names = [
            'cont1',
            'cont2',
            'groupA_1',
            'groupA_2',
            'cont3',
            'groupB_1',
            'groupB_2',
            'groupB_3',
        ]

        groups_dict = {
            'groupA': ['groupA_1', 'groupA_2'],
            'groupB': ['groupB_1', 'groupB_2', 'groupB_3'],
        }
        group_manager = OneHotGroupManager(groups_dict=groups_dict)

        return {
            'X_onehot': X_onehot,
            'feature_names': feature_names,
            'group_manager': group_manager,
            'catA': catA,
            'catB': catB,
            'cont1': cont1,
            'cont2': cont2,
            'cont3': cont3,
        }

    def test_column_order_with_groups_first(self, multi_group_data):
        """Verify that groups come first in collapsed order."""
        data = multi_group_data

        X_collapsed, collapsed_names = collapse_onehot_features(
            data['X_onehot'], data['group_manager'], data['feature_names']
        )

        # Groups should come first
        group_names = list(data['group_manager'].groups_dict.keys())
        for i, gname in enumerate(group_names):
            assert collapsed_names[i] == gname

        # Verify each column has expected data
        for i, name in enumerate(collapsed_names):
            col_data = X_collapsed[:, i]
            if name == 'groupA':
                np.testing.assert_array_equal(col_data, data['catA'])
            elif name == 'groupB':
                np.testing.assert_array_equal(col_data, data['catB'])

    def test_index_mapper_matches_collapse(self, multi_group_data):
        """Verify IndexMapper uses same order as collapse_onehot_features."""
        from prism.plotting.index_mapper import IndexMapper

        data = multi_group_data

        _, collapsed_names = collapse_onehot_features(
            data['X_onehot'], data['group_manager'], data['feature_names']
        )

        selected_indices = list(range(len(collapsed_names)))

        index_mapper = IndexMapper(
            original_names=data['feature_names'],
            collapsed_names=collapsed_names,
            selected_indices=selected_indices,
            group_manager=data['group_manager'],
        )

        for dense_idx in range(len(collapsed_names)):
            collapsed_idx = index_mapper.dense_to_collapsed(dense_idx)
            assert collapsed_idx == selected_indices[dense_idx]


class TestPipelineIntegration:
    """Integration tests for histogram data in plotting pipeline."""

    def test_bundle_x_data_has_correct_categorical_values(self):
        """Test that PlottingDataBundle.x_data has correct categorical values."""
        from prism.plotting.pipeline import PlottingPipeline

        np.random.seed(42)
        n_samples = 100

        cont1 = np.random.randn(n_samples)
        cont2 = np.random.randn(n_samples)
        categorical = np.random.randint(0, 3, n_samples)
        cat_A = (categorical == 1).astype(float)
        cat_B = (categorical == 2).astype(float)

        X_onehot = np.column_stack([cont1, cont2, cat_A, cat_B])
        X_tensor = torch.tensor(X_onehot, dtype=torch.float32)
        feature_names = ['cont1', 'cont2', 'cat_A', 'cat_B']

        group_manager = OneHotGroupManager(groups_dict={'cat': ['cat_A', 'cat_B']})

        _, collapsed_names = collapse_onehot_features(X_onehot, group_manager, feature_names)

        class MockLasso:
            def __init__(self, collapsed_names):
                self.univariate_feature_names = collapsed_names
                self.n_univ = len(collapsed_names)
                self.all_feature_names = collapsed_names.copy()
                for i in range(len(collapsed_names)):
                    for j in range(i + 1, len(collapsed_names)):
                        self.all_feature_names.append(
                            f"{collapsed_names[i]} : {collapsed_names[j]}"
                        )

            def get_selected_univariate_indices(self):
                return [0]  # Select 'cat'

            def get_selected_bivariate_index_pairs(self):
                return []

            def get_selected_beta(self):
                return np.array([1.0])

        class MockModel:
            def predict(self, x, device=None):
                if isinstance(x, torch.Tensor):
                    return torch.sigmoid(x.sum(dim=1))
                return torch.sigmoid(torch.tensor(x).sum(dim=1))

            def __call__(self, x):
                return self.predict(x)

        pipeline = PlottingPipeline(
            lasso_results=MockLasso(collapsed_names),
            group_manager=group_manager,
            label_manager=None,
        )

        bundle = pipeline.prepare_plotting_bundle(
            x=X_tensor,
            model=MockModel(),
            scaler=NoScaler(),
            n_steps=50,
            method='dirac',
            x_train=None,
            device='cpu',
            categorical_threshold=15,
            subtract_univariate=True,
            feature_names=feature_names,
        )

        # Verify categorical column has correct values
        cat_col = bundle.x_data[:, 0]
        np.testing.assert_array_equal(
            np.unique(cat_col).astype(int),
            np.unique(categorical),
            err_msg="bundle.x_data should contain categorical values [0, 1, 2]",
        )

    def test_scaled_data_in_pipeline(self):
        """Test pipeline correctly handles scaled data for histograms."""
        from prism.plotting.pipeline import PlottingPipeline

        np.random.seed(42)
        n_samples = 100

        categories = np.random.choice([0, 1, 2], size=n_samples, p=[0.4, 0.3, 0.3])
        cat_A = (categories == 1).astype(float)
        cat_B = (categories == 2).astype(float)
        continuous = np.random.randn(n_samples)

        X_original = np.column_stack([cat_A, cat_B, continuous])
        feature_names = ['cat_A', 'cat_B', 'continuous']

        group_manager = OneHotGroupManager(
            groups_dict={'cat': ['cat_A', 'cat_B']}, reference_columns={'cat': 'cat_REF'}
        )

        # Scale the data
        scaler = PRiSMScaler(scaler='median_std')
        scaler.fit(X_original)
        X_scaled = scaler.transform(X_original)
        X_scaled_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        _, collapsed_names = collapse_onehot_features(X_original, group_manager, feature_names)

        class MockLasso:
            def __init__(self, collapsed_names):
                self.univariate_feature_names = collapsed_names
                self.n_univ = len(collapsed_names)
                self.all_feature_names = collapsed_names.copy()
                for i in range(len(collapsed_names)):
                    for j in range(i + 1, len(collapsed_names)):
                        self.all_feature_names.append(
                            f"{collapsed_names[i]} : {collapsed_names[j]}"
                        )

            def get_selected_univariate_indices(self):
                return [0]

            def get_selected_bivariate_index_pairs(self):
                return []

            def get_selected_beta(self):
                return np.array([1.0])

        class MockModel:
            def predict(self, x, device=None):
                if isinstance(x, torch.Tensor):
                    return torch.sigmoid(x.sum(dim=1))
                return torch.sigmoid(torch.tensor(x).sum(dim=1))

            def __call__(self, x):
                return self.predict(x)

        pipeline = PlottingPipeline(
            lasso_results=MockLasso(collapsed_names),
            group_manager=group_manager,
            label_manager=None,
        )

        bundle = pipeline.prepare_plotting_bundle(
            x=X_scaled_tensor,
            model=MockModel(),
            scaler=scaler,
            n_steps=50,
            method='dirac',
            x_train=None,
            device='cpu',
            categorical_threshold=15,
            subtract_univariate=True,
            feature_names=feature_names,
        )

        # Verify all 3 categories are present in histogram data
        cat_column = bundle.x_data[:, 0]
        unique_categories = np.unique(cat_column)

        assert len(unique_categories) == 3, f"Should have 3 categories but got {unique_categories}"
        assert 0 in unique_categories, "Should have reference category (0)"
        assert 1 in unique_categories, "Should have cat_A category (1)"
        assert 2 in unique_categories, "Should have cat_B category (2)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
