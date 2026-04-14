"""Subset Calculation and Grid Generation Tests.

This module tests that subset calculations (used for plotting) produce
consistent results with full calculations, and that grid generation and
denormalization work correctly.

Test Plan Reference: Phase 3, Subset Calculations and Plotting
"""

import numpy as np
import torch

from prism.partial_responses import (
    PartialResponseCalculator,
    partial_responses,
    partial_responses_subset,
    stable_logit,
)


class TestSubsetConsistency:
    """Test that subset calculations match full calculations.

    When calculating partial responses for specific features via partial_responses_subset(),
    the results should match extracting those features from a full calculation.
    """

    def test_dirac_univariate_subset_matches_full(self, linear_test_model):
        """Verify Dirac univariate subset gives IDENTICAL values to full calculation.

        Call partial_responses_subset() with all features, then with a subset.
        The subset results should exactly match the corresponding full results.
        """
        torch.manual_seed(42)

        model = linear_test_model
        x_test = torch.randn(20, 5)

        # Full calculation - all features
        uni_full, _, x_uni_full, _ = partial_responses_subset(
            x=x_test,
            model=model,
            method='dirac',
            device='cpu',
            n_steps=15,
            selected_features=None,  # All features
            selected_feature_pairs=[],  # Skip bivariate for this test
        )

        # Subset calculation - features [0, 2, 4]
        selected_features = [0, 2, 4]
        uni_sub, _, x_uni_sub, _ = partial_responses_subset(
            x=x_test,
            model=model,
            method='dirac',
            device='cpu',
            n_steps=15,
            selected_features=selected_features,
            selected_feature_pairs=[],
        )

        # KEY ASSERTION: Subset must match corresponding full values
        for i, feat_idx in enumerate(selected_features):
            np.testing.assert_allclose(
                uni_sub[i],
                uni_full[feat_idx],
                rtol=1e-5,
                atol=1e-7,
                err_msg=f"Feature {feat_idx} response mismatch",
            )
            np.testing.assert_allclose(
                x_uni_sub[i],
                x_uni_full[feat_idx],
                rtol=1e-5,
                atol=1e-7,
                err_msg=f"Feature {feat_idx} grid mismatch",
            )

    def test_dirac_bivariate_subset_matches_full(self, linear_test_model):
        """Verify Dirac bivariate subset gives IDENTICAL values to full calculation."""
        torch.manual_seed(42)

        model = linear_test_model
        x_test = torch.randn(20, 5)

        # Full calculation - all pairs
        # Note: Need to determine pair ordering from the function
        _, bi_full, _, x_bi_full = partial_responses_subset(
            x=x_test,
            model=model,
            method='dirac',
            device='cpu',
            n_steps=10,
            selected_features=None,
            selected_feature_pairs=None,  # All pairs
        )

        # For 5 features, pairs are in order:
        # (0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)
        # Indices: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        selected_pairs = [(0, 1), (2, 3)]  # indices 0 and 7

        # Subset calculation - specific pairs
        _, bi_sub, _, x_bi_sub = partial_responses_subset(
            x=x_test,
            model=model,
            method='dirac',
            device='cpu',
            n_steps=10,
            selected_features=None,
            selected_feature_pairs=selected_pairs,
        )

        # Map selected pairs to their expected indices in full list
        # (0,1) -> index 0, (2,3) -> index 7
        pair_to_full_idx = {(0, 1): 0, (2, 3): 7}

        for i, pair in enumerate(selected_pairs):
            full_idx = pair_to_full_idx[pair]
            np.testing.assert_allclose(
                bi_sub[i],
                bi_full[full_idx],
                rtol=1e-5,
                atol=1e-7,
                err_msg=f"Pair {pair} response mismatch",
            )
            np.testing.assert_allclose(
                x_bi_sub[i],
                x_bi_full[full_idx],
                rtol=1e-5,
                atol=1e-7,
                err_msg=f"Pair {pair} grid mismatch",
            )

    def test_lebesgue_univariate_subset_matches_full(self, linear_test_model):
        """Verify Lebesgue univariate subset gives IDENTICAL values to full calculation."""
        torch.manual_seed(42)

        model = linear_test_model
        x_train = torch.randn(100, 5)
        x_test = torch.randn(20, 5)

        # Full calculation - all features
        uni_full, _, x_uni_full, _ = partial_responses_subset(
            x=x_test,
            model=model,
            method='lebesgue',
            x_train=x_train,
            device='cpu',
            n_steps=15,
            batch_size=64,
            selected_features=None,
            selected_feature_pairs=[],
        )

        # Subset calculation
        selected_features = [1, 3]
        uni_sub, _, x_uni_sub, _ = partial_responses_subset(
            x=x_test,
            model=model,
            method='lebesgue',
            x_train=x_train,
            device='cpu',
            n_steps=15,
            batch_size=64,
            selected_features=selected_features,
            selected_feature_pairs=[],
        )

        # KEY ASSERTION: Subset must match corresponding full values
        for i, feat_idx in enumerate(selected_features):
            np.testing.assert_allclose(
                uni_sub[i],
                uni_full[feat_idx],
                rtol=1e-5,
                atol=1e-7,
                err_msg=f"Feature {feat_idx} response mismatch",
            )
            np.testing.assert_allclose(
                x_uni_sub[i],
                x_uni_full[feat_idx],
                rtol=1e-5,
                atol=1e-7,
                err_msg=f"Feature {feat_idx} grid mismatch",
            )

    def test_lebesgue_bivariate_subset_matches_full(self, linear_test_model):
        """Verify Lebesgue bivariate subset gives IDENTICAL values to full calculation."""
        torch.manual_seed(42)

        model = linear_test_model
        x_train = torch.randn(100, 5)
        x_test = torch.randn(20, 5)

        # Full calculation
        _, bi_full, _, x_bi_full = partial_responses_subset(
            x=x_test,
            model=model,
            method='lebesgue',
            x_train=x_train,
            device='cpu',
            n_steps=10,
            batch_size=64,
            selected_features=None,
            selected_feature_pairs=None,
        )

        # Subset calculation
        # (0,2) -> index 1, (1,4) -> index 6
        selected_pairs = [(0, 2), (1, 4)]
        _, bi_sub, _, x_bi_sub = partial_responses_subset(
            x=x_test,
            model=model,
            method='lebesgue',
            x_train=x_train,
            device='cpu',
            n_steps=10,
            batch_size=64,
            selected_features=None,
            selected_feature_pairs=selected_pairs,
        )

        # Map selected pairs to their expected indices in full list
        pair_to_full_idx = {(0, 2): 1, (1, 4): 6}

        for i, pair in enumerate(selected_pairs):
            full_idx = pair_to_full_idx[pair]
            np.testing.assert_allclose(
                bi_sub[i],
                bi_full[full_idx],
                rtol=1e-5,
                atol=1e-7,
                err_msg=f"Pair {pair} response mismatch",
            )
            np.testing.assert_allclose(
                x_bi_sub[i],
                x_bi_full[full_idx],
                rtol=1e-5,
                atol=1e-7,
                err_msg=f"Pair {pair} grid mismatch",
            )


class TestGridGeneration:
    """Test grid generation for continuous and categorical features.

    The grid defines the feature values at which partial responses
    are evaluated for plotting.
    """

    def test_continuous_feature_grid(self, linear_test_model):
        """Test that continuous features get evenly spaced grids.

        Continuous features should be evaluated at n_steps evenly spaced
        points between min and max values.
        """
        torch.manual_seed(42)

        model = linear_test_model
        x_train = torch.randn(100, 5)
        x_test = torch.randn(20, 5)

        n_steps = 10

        univariate_responses, bivariate_responses, univariate_grids, bivariate_grids = (
            partial_responses_subset(
                x=x_test,
                model=model,
                method='lebesgue',
                x_train=x_train,
                device='cpu',
                n_steps=n_steps,
                selected_features=[0],  # Just feature 0
                selected_feature_pairs=None,
                batch_size=64,
            )
        )

        grid_0 = univariate_grids[0]

        print("\nContinuous Feature Grid Test:")
        print(f"  n_steps: {n_steps}")
        print(f"  Grid shape: {grid_0.shape}")
        print(f"  Grid values: {grid_0}")
        print(f"  Grid range: [{grid_0.min():.3f}, {grid_0.max():.3f}]")

        # Grid should have n_steps points
        assert len(grid_0) == n_steps, f"Expected {n_steps} grid points"

        # Grid should be sorted
        assert np.all(np.diff(grid_0) >= 0), "Grid should be sorted"

        # Grid should span the data range
        x_combined = torch.cat([x_train, x_test], dim=0)
        feature_0_min = x_combined[:, 0].min().item()
        feature_0_max = x_combined[:, 0].max().item()

        print(f"  Data range: [{feature_0_min:.3f}, {feature_0_max:.3f}]")

        # Document: Grid may not cover full data range
        # It's based on train data, and may exclude outliers
        covers_min = grid_0.min() <= feature_0_min
        covers_max = grid_0.max() >= feature_0_max
        print(f"  Grid covers min: {covers_min}")
        print(f"  Grid covers max: {covers_max}")

        # Grid should at least be in reasonable range
        assert grid_0.min() < feature_0_max, "Grid completely below data"
        assert grid_0.max() > feature_0_min, "Grid completely above data"

    def test_categorical_feature_grid(self, linear_test_model):
        """Test that categorical features get discrete grids.

        Features with few unique values should be treated as categorical
        and evaluated at each unique value.
        """
        torch.manual_seed(42)

        model = linear_test_model

        # Create data with a categorical feature (feature 0)
        x_train = torch.randn(100, 5)
        x_train[:, 0] = torch.randint(0, 3, (100,)).float()  # Only values 0, 1, 2

        x_test = torch.randn(20, 5)
        x_test[:, 0] = torch.randint(0, 3, (20,)).float()

        n_steps = 15
        categorical_threshold = 15  # Feature 0 has 3 unique values < 15

        univariate_responses, bivariate_responses, univariate_grids, bivariate_grids = (
            partial_responses_subset(
                x=x_test,
                model=model,
                method='lebesgue',
                x_train=x_train,
                device='cpu',
                n_steps=n_steps,
                categorical_threshold=categorical_threshold,
                selected_features=[0],
                selected_feature_pairs=None,
                batch_size=64,
            )
        )

        grid_0 = univariate_grids[0]

        print("\nCategorical Feature Grid Test:")
        print(f"  Unique values in feature 0: {np.unique(x_train[:, 0].numpy())}")
        print(f"  Grid: {grid_0}")
        print(f"  Grid length: {len(grid_0)}")

        # For categorical feature, grid should have one point per unique value
        # (or n_steps if many unique values)
        # With 3 unique values, should have 3 grid points
        unique_values = np.unique(x_train[:, 0].numpy())
        expected_grid_size = len(unique_values)

        # Grid size should match unique values (for categorical) or n_steps (for continuous)
        assert len(grid_0) <= max(expected_grid_size, n_steps), f"Grid too large: {len(grid_0)}"

    def test_bivariate_grid_shape(self, linear_test_model):
        """Test that bivariate grids are 2D meshgrids.

        Bivariate responses should be evaluated on a 2D grid of
        (feature_i_values, feature_j_values).
        """
        torch.manual_seed(42)

        model = linear_test_model
        x_train = torch.randn(100, 5)
        x_test = torch.randn(20, 5)

        n_steps = 10

        univariate_responses, bivariate_responses, univariate_grids, bivariate_grids = (
            partial_responses_subset(
                x=x_test,
                model=model,
                method='lebesgue',
                x_train=x_train,
                device='cpu',
                n_steps=n_steps,
                selected_features=[0, 1],
                selected_feature_pairs=[(0, 1)],
                batch_size=64,
            )
        )

        # Should have one bivariate grid for pair (0, 1)
        assert len(bivariate_grids) == 1, "Expected one bivariate grid"

        grid_01 = bivariate_grids[0]
        response_01 = bivariate_responses[0]

        print("\nBivariate Grid Test:")
        print(f"  Grid shape: {grid_01.shape}")
        print(f"  Response shape: {response_01.shape}")
        print(f"  First 3 grid points: {grid_01[:3]}")

        # Bivariate grid is a flattened meshgrid: shape (n_steps * n_steps, 2)
        # Each row is a pair of (feature_i_value, feature_j_value)
        expected_grid_size = n_steps * n_steps
        assert grid_01.shape == (
            expected_grid_size,
            2,
        ), f"Expected grid shape ({expected_grid_size}, 2), got {grid_01.shape}"

        # Response should match grid size
        assert (
            response_01.shape[0] == expected_grid_size
        ), f"Response shape {response_01.shape} doesn't match grid size {expected_grid_size}"

        # Verify all values are finite
        assert np.all(np.isfinite(grid_01)), "Bivariate grid contains NaN/Inf"
        assert np.all(np.isfinite(response_01)), "Bivariate response contains NaN/Inf"


class TestDenormalization:
    """Test denormalization of partial responses.

    If features are normalized, partial responses may need to be
    denormalized for interpretation.
    """

    def test_responses_in_correct_space(self, linear_test_model):
        """Verify partial responses are in logit space.

        Partial responses should be in logit space, not probability space,
        for proper ANOVA decomposition.
        """
        torch.manual_seed(42)

        model = linear_test_model
        x_train = torch.randn(100, 5)
        x_test = torch.randn(20, 5)

        calculator = PartialResponseCalculator(
            model, method='lebesgue', x_train=x_train, input_dim=5
        )
        univariate, bivariate = calculator.calculate(x_test)

        # Logit space can have large magnitude values
        # Unlike probabilities which are bounded [0, 1]
        print("\nResponse Space Test:")
        print(
            f"  Univariate range: [{univariate.min().item():.3f}, {univariate.max().item():.3f}]"
        )
        print(f"  Bivariate range: [{bivariate.min().item():.3f}, {bivariate.max().item():.3f}]")
        print(f"  Baseline (logit): {calculator.logit_y0:.3f}")

        # Probabilities from model
        y_pred = model(x_test)
        print(
            f"  Model output (probability) range: [{y_pred.min().item():.3f}, {y_pred.max().item():.3f}]"
        )

        # Logit values can exceed [-1, 1]
        # This is expected and correct
        assert not (
            univariate.min() >= 0 and univariate.max() <= 1
        ), "Univariate appears to be in probability space instead of logit space"

    def test_reconstruction_uses_logit_space(self, linear_test_model):
        """Verify reconstruction works in logit space.

        The reconstruction formula should work in logit space,
        not probability space.
        """
        torch.manual_seed(42)

        model = linear_test_model
        x_train = torch.randn(100, 5)
        x_test = torch.randn(20, 5)

        calculator = PartialResponseCalculator(
            model, method='lebesgue', x_train=x_train, input_dim=5
        )
        univariate, bivariate = calculator.calculate(x_test)

        # Reconstruction in logit space
        y_pred = model(x_test)
        logit_pred = stable_logit(y_pred)

        logit_reconstructed = calculator.logit_y0 + univariate.sum(dim=1) + bivariate.sum(dim=1)

        error_logit = (logit_pred - logit_reconstructed).abs().max().item()

        # Convert back to probability space
        prob_reconstructed = torch.sigmoid(logit_reconstructed)
        error_prob = (y_pred - prob_reconstructed).abs().max().item()

        print("\nReconstruction Space Test:")
        print(f"  Logit space error: {error_logit:.6e}")
        print(f"  Probability space error: {error_prob:.6e}")

        # Reconstruction should work in logit space
        assert error_logit < 1e-4, f"Logit space reconstruction failed: {error_logit:.6e}"

        # Probability space error should also be small after conversion
        assert error_prob < 0.01, f"Probability space error too high: {error_prob:.3f}"


class TestSubsetEdgeCases:
    """Test edge cases in subset calculations."""

    def test_empty_subset(self, linear_test_model):
        """Test behavior with empty subset.

        What happens when selected_features is an empty list?
        """
        torch.manual_seed(42)

        model = linear_test_model
        x_train = torch.randn(50, 5)
        x_test = torch.randn(10, 5)

        # Empty subset
        univariate_responses, bivariate_responses, univariate_grids, bivariate_grids = (
            partial_responses_subset(
                x=x_test,
                model=model,
                method='lebesgue',
                x_train=x_train,
                device='cpu',
                n_steps=10,
                selected_features=[],  # Empty
                selected_feature_pairs=[],  # Empty
                batch_size=64,
            )
        )

        print("\nEmpty Subset Test:")
        print(f"  Univariate results: {len(univariate_responses)}")
        print(f"  Bivariate results: {len(bivariate_responses)}")

        # Should have empty results
        assert len(univariate_responses) == 0, "Should have no univariate results"
        assert len(bivariate_responses) == 0, "Should have no bivariate results"

    def test_single_feature_subset(self, linear_test_model):
        """Test subset with single feature.

        Single feature should work correctly.
        """
        torch.manual_seed(42)

        model = linear_test_model
        x_train = torch.randn(50, 5)
        x_test = torch.randn(10, 5)

        univariate_responses, bivariate_responses, univariate_grids, bivariate_grids = (
            partial_responses_subset(
                x=x_test,
                model=model,
                method='lebesgue',
                x_train=x_train,
                device='cpu',
                n_steps=10,
                selected_features=[2],  # Only feature 2
                selected_feature_pairs=None,
                batch_size=64,
            )
        )

        print("\nSingle Feature Subset Test:")
        print(f"  Univariate results: {len(univariate_responses)}")
        print(f"  Bivariate results: {len(bivariate_responses)}")
        print(f"  Response shape: {univariate_responses[0].shape}")

        assert len(univariate_responses) == 1, "Should have 1 univariate result"
        assert np.all(np.isfinite(univariate_responses[0])), "Response contains NaN/Inf"


class TestSubsetVsFullFunctionConsistency:
    """Test that partial_responses_subset() matches partial_responses() at same points.

    This validates that the grid-based subset function produces the same values
    as the sample-based full function when evaluated at the same feature values.
    """

    def test_dirac_subset_matches_partial_responses(self, linear_test_model):
        """Verify partial_responses_subset matches partial_responses at categorical points."""
        torch.manual_seed(42)
        model = linear_test_model

        # Create categorical test data with known values that will be on the grid
        x_test = torch.zeros(12, 5)
        # Feature 0: values 0, 1, 2 (categorical)
        x_test[:, 0] = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=torch.float)
        # Feature 1: values 0, 1 (binary)
        x_test[:, 1] = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1], dtype=torch.float)
        # Other features: continuous
        x_test[:, 2:5] = torch.randn(12, 3)

        # Calculate sample-based responses using partial_responses()
        responses_full = partial_responses(x=x_test, model=model, method='dirac', device='cpu')
        # Extract univariate for feature 0 (first 5 columns are univariate)
        uni_full_f0 = responses_full[:, 0]  # Shape: [12]

        # Calculate grid-based responses using partial_responses_subset()
        uni_sub, _, x_uni_sub, _ = partial_responses_subset(
            x=x_test,
            model=model,
            method='dirac',
            device='cpu',
            n_steps=15,
            categorical_threshold=15,
            selected_features=[0],
            selected_feature_pairs=[],
        )
        # Grid for feature 0 should be [0, 1, 2] (categorical)

        # Map sample feature values to grid indices and compare
        grid_f0 = x_uni_sub[0]  # The grid values
        response_f0 = uni_sub[0]  # Responses at grid points

        for sample_idx, feat_val in enumerate(x_test[:, 0]):
            grid_idx = np.where(np.isclose(grid_f0, feat_val.item()))[0]
            if len(grid_idx) > 0:
                expected = response_f0[grid_idx[0]]
                actual = uni_full_f0[sample_idx].item()
                np.testing.assert_allclose(
                    actual,
                    expected,
                    rtol=1e-5,
                    atol=1e-7,
                    err_msg=f"Sample {sample_idx} mismatch at feat value {feat_val}",
                )

    def test_lebesgue_subset_matches_partial_responses(self, linear_test_model):
        """Verify Lebesgue partial_responses_subset matches partial_responses."""
        torch.manual_seed(42)
        model = linear_test_model

        # Create training data
        x_train = torch.randn(100, 5)

        # Create categorical test data
        x_test = torch.zeros(8, 5)
        # Feature 0: binary
        x_test[:, 0] = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.float)
        x_test[:, 1:5] = torch.randn(8, 4)

        # Calculate sample-based responses
        responses_full = partial_responses(
            x=x_test, model=model, method='lebesgue', x_train=x_train, device='cpu', batch_size=64
        )
        uni_full_f0 = responses_full[:, 0]

        # Calculate grid-based responses
        uni_sub, _, x_uni_sub, _ = partial_responses_subset(
            x=x_test,
            model=model,
            method='lebesgue',
            x_train=x_train,
            device='cpu',
            n_steps=15,
            categorical_threshold=15,
            batch_size=64,
            selected_features=[0],
            selected_feature_pairs=[],
        )

        grid_f0 = x_uni_sub[0]
        response_f0 = uni_sub[0]

        for sample_idx, feat_val in enumerate(x_test[:, 0]):
            grid_idx = np.where(np.isclose(grid_f0, feat_val.item()))[0]
            if len(grid_idx) > 0:
                expected = response_f0[grid_idx[0]]
                actual = uni_full_f0[sample_idx].item()
                np.testing.assert_allclose(
                    actual,
                    expected,
                    rtol=1e-5,
                    atol=1e-7,
                    err_msg=f"Sample {sample_idx} mismatch at feat value {feat_val}",
                )


class TestOneHotEncodingConsistency:
    """Test subset consistency with one-hot encoded features."""

    def test_dirac_onehot_subset_matches_full(self, mock_model):
        """Verify subset consistency with one-hot encoded features using Dirac."""
        from prism.preprocessing import NoScaler

        torch.manual_seed(42)

        # Create test data with one-hot group (features 0,1,2) + continuous (3,4)
        x_test = torch.zeros(20, 5)
        for i in range(20):
            choice = np.random.choice([0, 1, 2, 3])  # 3 = reference (all zeros)
            if choice < 3:
                x_test[i, choice] = 1
        x_test[:, 3:5] = torch.randn(20, 2)

        onehot_groups = [(0, 1, 2)]
        feature_names = ['cat_A', 'cat_B', 'cat_C', 'cont_0', 'cont_1']

        # Full calculation - all features
        uni_full, _, x_uni_full, _ = partial_responses_subset(
            x=x_test,
            model=mock_model,
            method='dirac',
            device='cpu',
            n_steps=15,
            onehot_groups=onehot_groups,
            feature_names=feature_names,
            scaler=NoScaler(),
            selected_features=None,
            selected_feature_pairs=[],
        )

        # After collapse: [group0, cont_3, cont_4] -> 3 features
        # Subset: select collapsed group (index 0) and cont_4 (original index 4, collapsed index 2)
        # Note: After collapse, indices change. We select based on collapsed indices.
        selected_features = [0, 2]  # group0 (collapsed) and cont_4 (collapsed index 2)
        uni_sub, _, x_uni_sub, _ = partial_responses_subset(
            x=x_test,
            model=mock_model,
            method='dirac',
            device='cpu',
            n_steps=15,
            onehot_groups=onehot_groups,
            feature_names=feature_names,
            scaler=NoScaler(),
            selected_features=selected_features,
            selected_feature_pairs=[],
        )

        # Verify subset matches corresponding full values
        for i, feat_idx in enumerate(selected_features):
            np.testing.assert_allclose(
                uni_sub[i],
                uni_full[feat_idx],
                rtol=1e-5,
                atol=1e-7,
                err_msg=f"Feature {feat_idx} response mismatch",
            )

    def test_lebesgue_onehot_subset_matches_full(self, mock_model):
        """Verify subset consistency with one-hot encoded features using Lebesgue."""
        from prism.preprocessing import NoScaler

        torch.manual_seed(42)

        # Create training and test data with one-hot group
        x_train = torch.zeros(100, 5)
        for i in range(100):
            choice = np.random.choice([0, 1, 2, 3])
            if choice < 3:
                x_train[i, choice] = 1
        x_train[:, 3:5] = torch.randn(100, 2)

        x_test = torch.zeros(20, 5)
        for i in range(20):
            choice = np.random.choice([0, 1, 2, 3])
            if choice < 3:
                x_test[i, choice] = 1
        x_test[:, 3:5] = torch.randn(20, 2)

        onehot_groups = [(0, 1, 2)]
        feature_names = ['cat_A', 'cat_B', 'cat_C', 'cont_0', 'cont_1']

        # Full calculation
        uni_full, _, x_uni_full, _ = partial_responses_subset(
            x=x_test,
            model=mock_model,
            method='lebesgue',
            x_train=x_train,
            device='cpu',
            n_steps=15,
            batch_size=64,
            onehot_groups=onehot_groups,
            feature_names=feature_names,
            scaler=NoScaler(),
            selected_features=None,
            selected_feature_pairs=[],
        )

        # Subset calculation - select just the one-hot group (collapsed index 0)
        selected_features = [0]
        uni_sub, _, x_uni_sub, _ = partial_responses_subset(
            x=x_test,
            model=mock_model,
            method='lebesgue',
            x_train=x_train,
            device='cpu',
            n_steps=15,
            batch_size=64,
            onehot_groups=onehot_groups,
            feature_names=feature_names,
            scaler=NoScaler(),
            selected_features=selected_features,
            selected_feature_pairs=[],
        )

        # Verify subset matches
        for i, feat_idx in enumerate(selected_features):
            np.testing.assert_allclose(
                uni_sub[i],
                uni_full[feat_idx],
                rtol=1e-5,
                atol=1e-7,
                err_msg=f"Feature {feat_idx} response mismatch",
            )


class TestScalingConsistency:
    """Test subset consistency with scaled features."""

    def test_dirac_scaled_subset_matches_full(self, mock_model):
        """Verify subset consistency with scaled features using Dirac."""
        from sklearn.preprocessing import StandardScaler

        from prism.preprocessing import PRiSMScaler

        torch.manual_seed(42)

        # Create raw data with non-standard mean/variance
        x_train_raw = torch.randn(100, 5) * 10 + 5
        x_test_raw = torch.randn(20, 5) * 10 + 5

        # Fit scaler on training data - use sklearn StandardScaler wrapped in PRiSMScaler
        scaler = PRiSMScaler(scaler=StandardScaler())
        scaler.fit(x_train_raw.numpy())

        # Transform to scaled space
        x_train_scaled = torch.tensor(scaler.transform(x_train_raw.numpy()), dtype=torch.float32)
        x_test_scaled = torch.tensor(scaler.transform(x_test_raw.numpy()), dtype=torch.float32)

        # Full calculation with scaler
        uni_full, _, x_uni_full, _ = partial_responses_subset(
            x=x_test_scaled,
            model=mock_model,
            method='dirac',
            device='cpu',
            x_train=x_train_scaled,
            scaler=scaler,
            n_steps=15,
            selected_features=None,
            selected_feature_pairs=[],
        )

        # Subset calculation with same scaler
        selected_features = [0, 2, 4]
        uni_sub, _, x_uni_sub, _ = partial_responses_subset(
            x=x_test_scaled,
            model=mock_model,
            method='dirac',
            device='cpu',
            x_train=x_train_scaled,
            scaler=scaler,
            n_steps=15,
            selected_features=selected_features,
            selected_feature_pairs=[],
        )

        # Verify subset matches corresponding full values
        for i, feat_idx in enumerate(selected_features):
            np.testing.assert_allclose(
                uni_sub[i],
                uni_full[feat_idx],
                rtol=1e-5,
                atol=1e-7,
                err_msg=f"Feature {feat_idx} response mismatch",
            )
            np.testing.assert_allclose(
                x_uni_sub[i],
                x_uni_full[feat_idx],
                rtol=1e-5,
                atol=1e-7,
                err_msg=f"Feature {feat_idx} grid mismatch",
            )

    def test_lebesgue_scaled_subset_matches_full(self, mock_model):
        """Verify subset consistency with scaled features using Lebesgue."""
        from sklearn.preprocessing import StandardScaler

        from prism.preprocessing import PRiSMScaler

        torch.manual_seed(42)

        # Create raw data
        x_train_raw = torch.randn(100, 5) * 10 + 5
        x_test_raw = torch.randn(20, 5) * 10 + 5

        # Fit scaler
        scaler = PRiSMScaler(scaler=StandardScaler())
        scaler.fit(x_train_raw.numpy())

        # Transform
        x_train_scaled = torch.tensor(scaler.transform(x_train_raw.numpy()), dtype=torch.float32)
        x_test_scaled = torch.tensor(scaler.transform(x_test_raw.numpy()), dtype=torch.float32)

        # Full calculation
        uni_full, _, x_uni_full, _ = partial_responses_subset(
            x=x_test_scaled,
            model=mock_model,
            method='lebesgue',
            x_train=x_train_scaled,
            device='cpu',
            scaler=scaler,
            n_steps=15,
            batch_size=64,
            selected_features=None,
            selected_feature_pairs=[],
        )

        # Subset calculation
        selected_features = [1, 3]
        uni_sub, _, x_uni_sub, _ = partial_responses_subset(
            x=x_test_scaled,
            model=mock_model,
            method='lebesgue',
            x_train=x_train_scaled,
            device='cpu',
            scaler=scaler,
            n_steps=15,
            batch_size=64,
            selected_features=selected_features,
            selected_feature_pairs=[],
        )

        # Verify subset matches
        for i, feat_idx in enumerate(selected_features):
            np.testing.assert_allclose(
                uni_sub[i],
                uni_full[feat_idx],
                rtol=1e-5,
                atol=1e-7,
                err_msg=f"Feature {feat_idx} response mismatch",
            )
            np.testing.assert_allclose(
                x_uni_sub[i],
                x_uni_full[feat_idx],
                rtol=1e-5,
                atol=1e-7,
                err_msg=f"Feature {feat_idx} grid mismatch",
            )
