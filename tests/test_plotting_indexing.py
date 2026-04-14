"""
Unit tests for diagnosing and fixing index space misalignment in plotting.

These tests are designed to expose the specific bugs where:
1. Feature labels are mismatched with feature data
2. Scaler indices don't match collapsed feature space
3. Dense list indices are confused with sparse feature indices
"""

import logging
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from prism.plotting_data import PlottingDataBundle
from prism.preprocessing import OneHotGroupManager, PRiSMScaler

logger = logging.getLogger(__name__)


# Module-level fixtures for shared use
@pytest.fixture
def realistic_htx_data():
    """
    Create realistic HTX-like data with one-hot groups.

    Features:
    - recageyear (continuous): age in years
    - reccrcl (continuous): creatinine clearance
    - dialpretx_0, dialpretx_1 (one-hot): dialysis pre-transplant (No=0, Yes=1)

    Collapsed features (order determined by collapse_onehot_features):
    - dialpretx (idx 0) - collapsed categorical (groups first!)
    - recageyear (idx 1)
    - reccrcl (idx 2)

    This mimics the structure causing the bug where:
    - 3rd plot shows binary but labelled "Age (year)"
    - 10th plot labelled "Dialysis" shows creatinine clearance data
    """
    np.random.seed(42)
    n_samples = 200

    # Continuous features (raw values)
    raw_recageyear = np.random.uniform(20, 70, n_samples)  # Age 20-70
    raw_reccrcl = np.random.uniform(30, 120, n_samples)  # Creatinine clearance

    # One-hot encoded dialysis (2 columns in expanded space)
    raw_dialysis = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    dialpretx_0 = (raw_dialysis == 0).astype(float)  # Reference: No dialysis
    dialpretx_1 = (raw_dialysis == 1).astype(float)  # Yes dialysis

    # RAW expanded (one-hot) representation: 4 columns
    X_expanded_raw = np.column_stack([raw_recageyear, raw_reccrcl, dialpretx_0, dialpretx_1])
    expanded_names = ['recageyear', 'reccrcl', 'dialpretx_0', 'dialpretx_1']

    # OneHotGroupManager
    group_manager = OneHotGroupManager(
        groups_dict={'dialpretx': ['dialpretx_0', 'dialpretx_1']},
        reference_columns={'dialpretx': 'dialpretx_0'},
    )

    # Get actual collapsed order from collapse_onehot_features
    from prism.preprocessing import collapse_onehot_features

    X_collapsed_raw, collapsed_names = collapse_onehot_features(
        X_expanded_raw, group_manager, expanded_names
    )
    # collapsed_names is ['dialpretx', 'recageyear', 'reccrcl']
    # X_collapsed_raw[:, 0] = dialysis (0 or 1)
    # X_collapsed_raw[:, 1] = recageyear
    # X_collapsed_raw[:, 2] = reccrcl

    # Scaler fitted on RAW EXPANDED data (original space)
    expanded_scaler = PRiSMScaler()
    expanded_scaler.fit(X_expanded_raw)

    # Create collapsed scaler from expanded scaler
    collapsed_scaler = group_manager.create_collapsed_scaler(expanded_scaler, expanded_names)

    # SCALED collapsed data
    X_collapsed_scaled = collapsed_scaler.transform(X_collapsed_raw)

    # SCALED expanded data (what model sees)
    X_expanded_scaled = expanded_scaler.transform(X_expanded_raw)

    return {
        'X_expanded': torch.tensor(X_expanded_scaled, dtype=torch.float32),
        'X_collapsed': torch.tensor(X_collapsed_scaled, dtype=torch.float32),
        'X_expanded_raw': X_expanded_raw,
        'X_collapsed_raw': X_collapsed_raw,
        'expanded_names': expanded_names,
        'collapsed_names': collapsed_names,  # ['dialpretx', 'recageyear', 'reccrcl']
        'group_manager': group_manager,
        'expanded_scaler': expanded_scaler,
        'collapsed_scaler': collapsed_scaler,
        'raw_recageyear': raw_recageyear,
        'raw_reccrcl': raw_reccrcl,
        'raw_dialysis': raw_dialysis,
    }


@pytest.fixture
def mock_lasso_results(realistic_htx_data):
    """Create LassoResultsManager mock with collapsed feature names."""
    collapsed_names = realistic_htx_data['collapsed_names']
    n_collapsed = len(collapsed_names)

    # Select all 3 features
    selected_univ = [0, 1, 2]  # All three features selected
    selected_biv = [(0, 1), (0, 2)]  # Some bivariate interactions

    mock = MagicMock()
    mock.univariate_feature_names = collapsed_names
    mock.n_univ = n_collapsed
    mock.get_selected_univariate_indices.return_value = selected_univ
    mock.get_selected_bivariate_index_pairs.return_value = selected_biv
    mock.all_feature_names = collapsed_names + [
        f'{collapsed_names[i]} : {collapsed_names[j]}' for i, j in selected_biv
    ]

    # Mock beta values
    beta = np.array([0.5, 0.3, 0.2, 0.1, 0.05])  # 3 univ + 2 biv
    mock.get_selected_beta.return_value = beta

    return mock


class TestIndexSpaceAlignment:
    """Tests that index spaces are properly aligned throughout the plotting pipeline."""

    def test_scaler_dimension_matches_collapsed_space(self, realistic_htx_data):
        """
        Test that collapsed scaler has correct dimensions.

        This catches the bug where scaler has 4 features (expanded)
        but is used with 3-feature collapsed indices.
        """
        collapsed_scaler = realistic_htx_data['collapsed_scaler']
        collapsed_names = realistic_htx_data['collapsed_names']

        # Get scaler dimension
        if hasattr(collapsed_scaler.scaler, 'median_'):
            scaler_dim = len(collapsed_scaler.scaler.median_)
        else:
            pytest.skip("Scaler doesn't have median_ attribute")

        assert scaler_dim == len(collapsed_names), (
            f"Scaler dimension ({scaler_dim}) doesn't match "
            f"collapsed feature count ({len(collapsed_names)})"
        )

    def test_denormalize_with_correct_indices(self, realistic_htx_data):
        """
        Test that denormalization uses correct indices in collapsed space.

        This catches the bug where feature_index from collapsed space
        is used with a scaler in expanded space.
        """
        collapsed_scaler = realistic_htx_data['collapsed_scaler']
        X_collapsed = realistic_htx_data['X_collapsed'].numpy()
        raw_recageyear = realistic_htx_data['raw_recageyear']
        collapsed_names = realistic_htx_data['collapsed_names']

        # Find index of recageyear in collapsed space
        age_idx = collapsed_names.index('recageyear')  # Should be 1

        # Denormalize recageyear from collapsed space
        # X_collapsed already contains SCALED values
        normalized_age = X_collapsed[:, age_idx]

        # Create dummy array for inverse transform
        dummy = np.zeros((len(normalized_age), 3))
        dummy[:, age_idx] = normalized_age

        denormalized = collapsed_scaler.inverse_transform(dummy)
        denormalized_age = denormalized[:, age_idx]

        # Should be close to original age values
        np.testing.assert_allclose(
            denormalized_age,
            raw_recageyear,
            rtol=0.01,
            err_msg="Denormalized age doesn't match raw age - index mismatch?",
        )

    def test_feature_label_matches_feature_data(self, realistic_htx_data, mock_lasso_results):
        """
        Test that feature labels in PlottingDataBundle match feature data.

        This catches the bug where:
        - Label says "Age (year)"
        - Data shows binary values (dialysis)
        """
        X_collapsed = realistic_htx_data['X_collapsed'].numpy()
        collapsed_names = realistic_htx_data['collapsed_names']

        # collapsed_names is ['dialpretx', 'recageyear', 'reccrcl']
        # idx 0 = dialpretx (categorical)
        # idx 1 = recageyear (continuous)
        # idx 2 = reccrcl (continuous)

        # Create mock responses and x_values
        # These would come from partial_responses_subset in dense order
        selected_indices = [0, 1, 2]

        # Create responses aligned with selected_indices (dense order)
        univariate_responses = [
            np.array([0.0, 0.3]),  # Response for feature 0 (dialpretx - categorical)
            np.linspace(-1, 1, 50),  # Response for feature 1 (recageyear)
            np.linspace(-0.5, 0.5, 50),  # Response for feature 2 (reccrcl)
        ]

        # X values: continuous have many values, categorical have few
        x_univariate = [
            np.array([0.0, 1.0]),  # idx 0 dialpretx: Categorical, only 2 values
            np.linspace(X_collapsed[:, 1].min(), X_collapsed[:, 1].max(), 50),  # idx 1 recageyear
            np.linspace(X_collapsed[:, 2].min(), X_collapsed[:, 2].max(), 50),  # idx 2 reccrcl
        ]

        # Create bundle
        bundle = PlottingDataBundle.from_partial_responses(
            univariate_responses=univariate_responses,
            bivariate_responses=[],
            x_univariate=x_univariate,
            x_bivariate=[],
            selected_univariate_indices=selected_indices,
            selected_bivariate_pairs=[],
            all_feature_names=collapsed_names,
            is_categorical=[True, False, False],  # dialpretx, recageyear, reccrcl
        )

        # Verify alignments
        for info in bundle.univariate_features():
            expected_name = collapsed_names[info.index]
            actual_name = info.name

            assert actual_name == expected_name, (
                f"Feature at index {info.index}: "
                f"expected name '{expected_name}', got '{actual_name}'"
            )

            # Verify categorical flag matches data characteristics
            n_unique = len(np.unique(info.x_values))
            if info.is_categorical:
                assert (
                    n_unique <= 15
                ), f"Feature '{info.name}' marked categorical but has {n_unique} unique values"
            else:
                assert (
                    n_unique > 2
                ), f"Feature '{info.name}' marked continuous but has only {n_unique} unique values"

    def test_dense_to_sparse_mapping_consistency(self, realistic_htx_data):
        """
        Test that dense list indices correctly map to sparse feature indices.

        The key insight: partial_responses_subset returns DENSE lists where
        responses[0] corresponds to selected_features[0], not feature index 0.
        """
        # Use actual collapsed names from the fixture
        collapsed_names = realistic_htx_data['collapsed_names']
        # ['dialpretx', 'recageyear', 'reccrcl']

        # Select in non-sequential order: idx 2 (reccrcl), idx 0 (dialpretx)
        selected_features = [2, 0]

        # Mock responses in DENSE order (aligned with selected_features)
        # selected_features[0] = 2 = reccrcl (continuous)
        # selected_features[1] = 0 = dialpretx (categorical)
        univariate_responses = [
            np.linspace(-0.5, 0.5, 50),  # Dense idx 0 → feature 2 (reccrcl)
            np.array([0.0, 0.5]),  # Dense idx 1 → feature 0 (dialpretx)
        ]

        x_univariate = [
            np.linspace(30, 120, 50),  # reccrcl continuous values
            np.array([0.0, 1.0]),  # dialpretx categorical values
        ]

        bundle = PlottingDataBundle.from_partial_responses(
            univariate_responses=univariate_responses,
            bivariate_responses=[],
            x_univariate=x_univariate,
            x_bivariate=[],
            selected_univariate_indices=selected_features,
            selected_bivariate_pairs=[],
            all_feature_names=collapsed_names,
            is_categorical=[False, True],  # Aligned with selected_features order
        )

        # Verify bundle correctly maps dense → sparse
        features = list(bundle.univariate_features())

        # Dense position 0 should map to feature index 2 (reccrcl)
        assert features[0].index == 2, f"Expected index 2, got {features[0].index}"
        assert features[0].name == 'reccrcl', f"Expected 'reccrcl', got {features[0].name}"
        assert features[0].is_categorical is False

        # Dense position 1 should map to feature index 0 (dialpretx)
        assert features[1].index == 0, f"Expected index 0, got {features[1].index}"
        assert features[1].name == 'dialpretx', f"Expected 'dialpretx', got {features[1].name}"
        assert features[1].is_categorical is True

    def test_scaler_denormalization_with_collapsed_indices(
        self, realistic_htx_data, mock_lasso_results
    ):
        """
        Test that PRiSMScaler denormalization works correctly with collapsed data.

        This tests the core denormalization logic that the new architecture uses,
        verifying that scaler.inverse_transform properly recovers original values.
        """
        import numpy as np

        X_collapsed = realistic_htx_data['X_collapsed'].numpy()
        collapsed_scaler = realistic_htx_data['collapsed_scaler']
        collapsed_names = realistic_htx_data['collapsed_names']
        raw_recageyear = realistic_htx_data['raw_recageyear']

        # collapsed_names is ['dialpretx', 'recageyear', 'reccrcl']
        age_idx = collapsed_names.index('recageyear')  # Should be 1
        dial_idx = collapsed_names.index('dialpretx')  # Should be 0
        n_features = X_collapsed.shape[1]

        # Test denormalization for recageyear (continuous, idx 1)
        # Use actual normalized values from the data
        normalized_age_sample = X_collapsed[:3, age_idx]

        # Create dummy array for denormalization (like DenormalizationService does)
        dummy_array = np.zeros((len(normalized_age_sample), n_features))
        dummy_array[:, age_idx] = normalized_age_sample
        denorm_result = collapsed_scaler.inverse_transform(dummy_array)
        denorm_age = denorm_result[:, age_idx]

        # Denormalized age should match the raw age values
        np.testing.assert_allclose(
            denorm_age,
            raw_recageyear[:3],
            rtol=0.01,
            err_msg="Scaler denormalization doesn't recover original age",
        )

        # Test denormalization for dialpretx (categorical, idx 0)
        dial_values = np.array([0.0, 1.0])
        dummy_array_dial = np.zeros((len(dial_values), n_features))
        dummy_array_dial[:, dial_idx] = dial_values
        denorm_dial_result = collapsed_scaler.inverse_transform(dummy_array_dial)
        denorm_dial = denorm_dial_result[:, dial_idx]

        # Dialysis is binary categorical - should stay as 0, 1
        np.testing.assert_array_almost_equal(denorm_dial, [0.0, 1.0])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
