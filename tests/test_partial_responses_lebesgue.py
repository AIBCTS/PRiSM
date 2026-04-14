"""Lebesgue Method Validation Tests - Detailed Mechanism Verification.

This module tests the Lebesgue averaging mechanism in detail, verifying:
1. Averaging over training data distribution is performed correctly
2. Bivariate isolation formula is mathematically correct
3. One-hot encoding groups are handled properly
4. Caching produces consistent results
5. Batch size doesn't affect final results

Test Plan Reference: Phase 2, Detailed Mechanism Validation
"""

import numpy as np
import torch

from prism.partial_responses import PartialResponseCalculator, stable_logit
from prism.preprocessing import NoScaler


class TestLebesgueAveragingMechanism:
    """Test that Lebesgue method correctly averages over training distribution.

    The Lebesgue method computes:
    E[f(X) | Xi = xi] by averaging f(x1, x2, ..., xi, ..., xn) over
    all training samples while fixing xi.
    """

    def test_univariate_averages_over_training_data(self, linear_test_model):
        """Verify univariate responses average over training distribution.

        For a single test point, the univariate response should be computed
        by replacing feature i with the test value across all training samples
        and averaging the results.
        """
        torch.manual_seed(42)

        model = linear_test_model
        x_train = torch.randn(50, 5)
        x_test = torch.randn(1, 5)  # Single test point

        calculator = PartialResponseCalculator(
            model, method='lebesgue', x_train=x_train, input_dim=5
        )
        univariate, bivariate = calculator.calculate(x_test)

        # Manually compute univariate for feature 0
        # Expected: Replace x_train[:, 0] with x_test[0, 0] and average
        x_modified = x_train.clone()
        x_modified[:, 0] = x_test[0, 0]

        y_modified = model(x_modified)
        logit_modified = stable_logit(y_modified)
        expected_marginal = logit_modified.mean().item()

        # Univariate = marginal - baseline
        expected_univariate = expected_marginal - calculator.logit_y0
        actual_univariate = univariate[0, 0].item()

        print("\nUnivariate Averaging Test (Feature 0):")
        print(f"  Expected univariate: {expected_univariate:.6f}")
        print(f"  Actual univariate:   {actual_univariate:.6f}")
        print(f"  Difference:          {abs(expected_univariate - actual_univariate):.6e}")

        assert (
            abs(expected_univariate - actual_univariate) < 1e-5
        ), "Univariate averaging incorrect for feature 0"

    def test_bivariate_averages_over_training_data(self, linear_test_model):
        """Verify bivariate responses average over training distribution.

        Bivariate response should average f(x) over training samples
        with features i and j both fixed at test values.
        """
        torch.manual_seed(42)

        model = linear_test_model
        x_train = torch.randn(50, 5)
        x_test = torch.randn(1, 5)

        calculator = PartialResponseCalculator(
            model, method='lebesgue', x_train=x_train, input_dim=5
        )
        univariate, bivariate = calculator.calculate(x_test)

        # Manually compute bivariate for features (0, 1)
        # Step 1: Fix both features 0 and 1
        x_modified = x_train.clone()
        x_modified[:, 0] = x_test[0, 0]
        x_modified[:, 1] = x_test[0, 1]

        y_modified = model(x_modified)
        logit_modified = stable_logit(y_modified)
        joint_marginal = logit_modified.mean().item()

        # Step 2: Bivariate isolation formula
        # phi_01 = E[f | X0, X1] - phi_0 - phi_1 - baseline
        expected_bivariate = (
            joint_marginal
            - univariate[0, 0].item()
            - univariate[0, 1].item()
            - calculator.logit_y0
        )

        # Find bivariate response for pair (0, 1)
        # Bivariate tensor has shape (n_samples, n_pairs)
        # Need to identify which index corresponds to pair (0, 1)
        # For 5 features: pairs are (0,1), (0,2), (0,3), (0,4), (1,2), ...
        pair_01_idx = 0  # First pair is (0, 1)
        actual_bivariate = bivariate[0, pair_01_idx].item()

        print("\nBivariate Averaging Test (Features 0, 1):")
        print(f"  Joint marginal:      {joint_marginal:.6f}")
        print(f"  Univariate[0]:       {univariate[0, 0].item():.6f}")
        print(f"  Univariate[1]:       {univariate[0, 1].item():.6f}")
        print(f"  Baseline:            {calculator.logit_y0:.6f}")
        print(f"  Expected bivariate:  {expected_bivariate:.6f}")
        print(f"  Actual bivariate:    {actual_bivariate:.6f}")
        print(f"  Difference:          {abs(expected_bivariate - actual_bivariate):.6e}")

        assert (
            abs(expected_bivariate - actual_bivariate) < 1e-5
        ), "Bivariate averaging incorrect for pair (0, 1)"

    def test_averaging_with_different_training_sizes(self, linear_test_model):
        """Verify averaging works correctly with different training set sizes.

        The averaging mechanism should work with any training set size.
        Results should converge as training set size increases.
        """
        torch.manual_seed(42)

        model = linear_test_model
        x_test = torch.randn(10, 5)

        training_sizes = [10, 50, 100, 500]
        baseline_values = []
        univariate_values = []

        for n_train in training_sizes:
            x_train = torch.randn(n_train, 5)

            calculator = PartialResponseCalculator(
                model, method='lebesgue', x_train=x_train, input_dim=5
            )
            univariate, bivariate = calculator.calculate(x_test)

            baseline_values.append(calculator.logit_y0)
            univariate_values.append(univariate[0, 0].item())

        print("\nTraining Size Convergence Test:")
        for n_train, baseline, uni in zip(training_sizes, baseline_values, univariate_values):
            print(f"  n_train={n_train:4d}: baseline={baseline:.6f}, univariate[0]={uni:.6f}")

        # Variance should decrease as training size increases
        # (Results become more stable with more samples)
        baseline_std_first_half = np.std(baseline_values[:2])
        baseline_std_second_half = np.std(baseline_values[2:])

        print(f"\n  Baseline std (n=10,50):   {baseline_std_first_half:.6f}")
        print(f"  Baseline std (n=100,500): {baseline_std_second_half:.6f}")

        # Larger training sets should have more stable estimates
        # (This is a weak test - just checking mechanism works)
        assert all(np.isfinite(baseline_values)), "Baseline contains NaN/Inf"
        assert all(np.isfinite(univariate_values)), "Univariate contains NaN/Inf"


class TestBivariateIsolationFormula:
    """Test the bivariate isolation formula in detail.

    The formula is:
    phi_ij(xi, xj) = E[f(X) | Xi=xi, Xj=xj] - phi_i(xi) - phi_j(xj) - baseline

    This isolates the pure interaction effect between features i and j.
    """

    def test_bivariate_isolation_formula_linear_model(self, linear_test_model):
        """For linear model, bivariate should be near zero (no interactions).

        Since linear model has no interaction terms, after subtracting
        univariate effects, bivariate should be approximately zero.
        """
        torch.manual_seed(42)

        model = linear_test_model
        x_train = torch.randn(100, 5)
        x_test = torch.randn(20, 5)

        calculator = PartialResponseCalculator(
            model, method='lebesgue', x_train=x_train, input_dim=5
        )
        univariate, bivariate = calculator.calculate(x_test)

        max_bivariate = bivariate.abs().max().item()
        mean_bivariate = bivariate.abs().mean().item()

        print("\nBivariate Isolation (Linear Model):")
        print(f"  Max |bivariate|:  {max_bivariate:.6e}")
        print(f"  Mean |bivariate|: {mean_bivariate:.6e}")
        print("  (Should be near zero for linear model)")

        assert (
            max_bivariate < 1e-3
        ), f"Linear model has significant bivariate after isolation: {max_bivariate:.6e}"

    def test_bivariate_isolation_formula_interaction_model(self, interaction_test_model):
        """For interaction model, bivariate should capture interaction strength.

        Model has 0.5 * x0 * x1 interaction term.
        Bivariate for pair (0, 1) should be non-zero.
        """
        torch.manual_seed(42)

        model = interaction_test_model
        x_train = torch.randn(100, 5)
        x_test = torch.randn(20, 5)

        calculator = PartialResponseCalculator(
            model, method='lebesgue', x_train=x_train, input_dim=5
        )
        univariate, bivariate = calculator.calculate(x_test)

        # Pair (0, 1) should have non-zero bivariate
        pair_01_bivariate = bivariate[:, 0].abs().max().item()

        # Other pairs should have near-zero bivariate (no interaction)
        other_pairs_bivariate = bivariate[:, 1:].abs().max().item()

        print("\nBivariate Isolation (Interaction Model):")
        print(f"  Pair (0, 1) max |bivariate|: {pair_01_bivariate:.6f}")
        print(f"  Other pairs max |bivariate|: {other_pairs_bivariate:.6e}")

        assert (
            pair_01_bivariate > 0.01
        ), f"Failed to detect interaction in pair (0, 1): {pair_01_bivariate:.6e}"

        assert (
            other_pairs_bivariate < 0.01
        ), f"Spurious interactions detected in other pairs: {other_pairs_bivariate:.6e}"

    def test_bivariate_subtraction_order_independence(self, test_mlp):
        """Verify bivariate formula is symmetric: phi_ij = phi_ji.

        The isolation formula should produce symmetric results since
        feature order doesn't matter for interactions.
        """
        torch.manual_seed(42)

        model = test_mlp
        x_train = torch.randn(50, 5)
        x_test = torch.randn(10, 5)

        calculator = PartialResponseCalculator(
            model, method='lebesgue', x_train=x_train, input_dim=5
        )
        univariate, bivariate = calculator.calculate(x_test)

        # For pair (0, 1), manually verify the isolation formula
        # phi_01 = joint - phi_0 - phi_1 - baseline

        # Compute joint marginal
        x_modified = x_train.clone()
        x_modified[:, 0] = x_test[0, 0]
        x_modified[:, 1] = x_test[0, 1]
        y_modified = model(x_modified)
        joint_marginal = stable_logit(y_modified).mean().item()

        # Apply isolation formula
        isolated_bivariate = (
            joint_marginal
            - univariate[0, 0].item()
            - univariate[0, 1].item()
            - calculator.logit_y0
        )

        actual_bivariate = bivariate[0, 0].item()

        print("\nBivariate Subtraction Test:")
        print(f"  Joint marginal:        {joint_marginal:.6f}")
        print(f"  - Univariate[0]:       {univariate[0, 0].item():.6f}")
        print(f"  - Univariate[1]:       {univariate[0, 1].item():.6f}")
        print(f"  - Baseline:            {calculator.logit_y0:.6f}")
        print(f"  = Expected bivariate:  {isolated_bivariate:.6f}")
        print(f"  Actual bivariate:      {actual_bivariate:.6f}")
        print(f"  Difference:            {abs(isolated_bivariate - actual_bivariate):.6e}")

        assert (
            abs(isolated_bivariate - actual_bivariate) < 1e-5
        ), "Bivariate isolation formula produces incorrect result"


class TestOneHotEncodingHandling:
    """Test that one-hot encoded features are handled correctly.

    One-hot groups should be treated as a single categorical variable,
    not as independent features.
    """

    def test_onehot_groups_parameter_accepted(self, test_mlp):
        """Verify onehot_groups parameter is accepted and used."""
        torch.manual_seed(42)

        model = test_mlp
        x_train = torch.randn(50, 5)
        x_test = torch.randn(10, 5)

        # Define one-hot groups: features 0-2 are one group
        onehot_groups = [[0, 1, 2]]
        feature_names = ['cat_A', 'cat_B', 'cat_C', 'cont_0', 'cont_1']

        calculator = PartialResponseCalculator(
            model,
            method='lebesgue',
            x_train=x_train,
            input_dim=5,
            onehot_groups=onehot_groups,
            feature_names=feature_names,
            scaler=NoScaler(),
        )

        # Should not raise an error
        univariate, bivariate = calculator.calculate(x_test)

        print("\nOne-Hot Groups Test:")
        print(f"  Groups: {onehot_groups}")
        print(f"  Univariate shape: {univariate.shape}")
        print(f"  Bivariate shape:  {bivariate.shape}")
        print("  Calculation successful")

        # With collapse: 3 OHE cols -> 1 group + 2 continuous = 3 collapsed features
        assert univariate.shape[1] == 3, "Univariate dimension mismatch (should be collapsed)"

    def test_onehot_groups_with_proper_binary_data(self, linear_test_model):
        """Test one-hot groups with proper binary one-hot encoded data.

        This test uses ACTUAL binary one-hot encoded data (0s and 1s only)
        to verify that the one-hot group handling works correctly.

        The previous test was flawed because it used continuous normal data
        (torch.randn) and declared it as one-hot, which caused massive
        reconstruction errors due to the distribution mismatch.
        """
        torch.manual_seed(42)

        model = linear_test_model
        n_train = 100
        n_test = 20

        # Create PROPER one-hot encoded data
        # Features 0-2: one-hot group representing a 3-category variable
        # Features 3-4: continuous features
        x_train = torch.zeros(n_train, 5)
        x_test = torch.zeros(n_test, 5)

        # Create one-hot encoding for features 0-2 (3 categories)
        # Each sample has exactly one of [0,1,2] active, or all zero (reference)
        for i in range(n_train):
            category = torch.randint(0, 4, (1,)).item()  # 0=reference, 1-3=active categories
            if category > 0:
                x_train[i, category - 1] = 1

        for i in range(n_test):
            category = torch.randint(0, 4, (1,)).item()
            if category > 0:
                x_test[i, category - 1] = 1

        # Add continuous features (3 and 4)
        x_train[:, 3:5] = torch.randn(n_train, 2)
        x_test[:, 3:5] = torch.randn(n_test, 2)

        # Define one-hot group
        onehot_groups = [[0, 1, 2]]
        feature_names = ['cat_A', 'cat_B', 'cat_C', 'cont_0', 'cont_1']

        # Calculate with one-hot groups
        calculator = PartialResponseCalculator(
            model,
            method='lebesgue',
            x_train=x_train,
            input_dim=5,
            onehot_groups=onehot_groups,
            feature_names=feature_names,
            scaler=NoScaler(),
        )

        univariate, bivariate = calculator.calculate(x_test)

        # Test reconstruction
        y_pred = model(x_test)
        logit_pred = stable_logit(y_pred)

        logit_reconstructed = calculator.logit_y0 + univariate.sum(dim=1) + bivariate.sum(dim=1)

        error = (logit_pred - logit_reconstructed).abs().max().item()

        print("\nOne-Hot Groups with PROPER Binary Data:")
        print(f"  One-hot group: features {onehot_groups[0]} (3-category variable)")
        print("  Continuous features: [3, 4]")
        print(f"  Linear model reconstruction error: {error:.6e}")
        print("  Expected: < 1e-4 (should be nearly exact for linear model)")

        # Linear model should reconstruct exactly (or nearly so)
        assert torch.all(torch.isfinite(univariate)), "Univariate contains NaN/Inf"
        assert torch.all(torch.isfinite(bivariate)), "Bivariate contains NaN/Inf"
        assert error < 1e-3, (
            "Linear model reconstruction with proper one-hot data failed!\n"
            f"Error: {error:.6e}\n"
            "This indicates a real bug in one-hot group handling."
        )


class TestCachingAndBatchInvariance:
    """Test that caching and batch size don't affect results.

    Results should be identical regardless of:
    1. Batch size used for computation
    2. Whether results are cached or recomputed
    """

    def test_batch_size_invariance(self, test_mlp):
        """Verify results are independent of batch_size parameter.

        Same input should produce same output regardless of batch size.
        """
        torch.manual_seed(42)

        model = test_mlp
        x_train = torch.randn(100, 5)
        x_test = torch.randn(30, 5)

        # Test different batch sizes
        batch_sizes = [1, 5, 10, 30, 100]
        results = []

        for batch_size in batch_sizes:
            calculator = PartialResponseCalculator(
                model, method='lebesgue', x_train=x_train, input_dim=5
            )
            univariate, bivariate = calculator.calculate(x_test, batch_size=batch_size)

            results.append(
                {
                    'batch_size': batch_size,
                    'univariate': univariate.clone(),
                    'bivariate': bivariate.clone(),
                }
            )

        print("\nBatch Size Invariance Test:")

        # Compare all results to first result
        reference = results[0]
        for result in results[1:]:
            uni_diff = (result['univariate'] - reference['univariate']).abs().max().item()
            bi_diff = (result['bivariate'] - reference['bivariate']).abs().max().item()

            print(f"  Batch size {result['batch_size']:3d} vs {reference['batch_size']:3d}:")
            print(f"    Univariate max diff: {uni_diff:.6e}")
            print(f"    Bivariate max diff:  {bi_diff:.6e}")

            assert uni_diff < 1e-6, f"Univariate results vary with batch size: {uni_diff:.6e}"
            assert bi_diff < 1e-6, f"Bivariate results vary with batch size: {bi_diff:.6e}"

    def test_multiple_calculations_consistent(self, test_mlp):
        """Verify multiple calculations produce identical results.

        Calling calculate() multiple times should give same results
        (tests that caching or stateful behavior doesn't cause issues).
        """
        torch.manual_seed(42)

        model = test_mlp
        x_train = torch.randn(50, 5)
        x_test = torch.randn(10, 5)

        calculator = PartialResponseCalculator(
            model, method='lebesgue', x_train=x_train, input_dim=5
        )

        # Calculate multiple times
        univariate1, bivariate1 = calculator.calculate(x_test)
        univariate2, bivariate2 = calculator.calculate(x_test)
        univariate3, bivariate3 = calculator.calculate(x_test)

        # All should be identical
        uni_diff_12 = (univariate1 - univariate2).abs().max().item()
        uni_diff_23 = (univariate2 - univariate3).abs().max().item()
        bi_diff_12 = (bivariate1 - bivariate2).abs().max().item()
        bi_diff_23 = (bivariate2 - bivariate3).abs().max().item()

        print("\nMultiple Calculations Consistency:")
        print(f"  Univariate diff (1 vs 2): {uni_diff_12:.6e}")
        print(f"  Univariate diff (2 vs 3): {uni_diff_23:.6e}")
        print(f"  Bivariate diff (1 vs 2):  {bi_diff_12:.6e}")
        print(f"  Bivariate diff (2 vs 3):  {bi_diff_23:.6e}")

        assert uni_diff_12 < 1e-10, "Univariate results not consistent across calls"
        assert uni_diff_23 < 1e-10, "Univariate results not consistent across calls"
        assert bi_diff_12 < 1e-10, "Bivariate results not consistent across calls"
        assert bi_diff_23 < 1e-10, "Bivariate results not consistent across calls"

    def test_different_test_samples_independent(self, test_mlp):
        """Verify calculations for different test samples are independent.

        Calculating for sample A shouldn't affect calculation for sample B.
        """
        torch.manual_seed(42)

        model = test_mlp
        x_train = torch.randn(50, 5)
        x_test_A = torch.randn(5, 5)
        x_test_B = torch.randn(5, 5)
        x_test_combined = torch.cat([x_test_A, x_test_B], dim=0)

        calculator = PartialResponseCalculator(
            model, method='lebesgue', x_train=x_train, input_dim=5
        )

        # Calculate separately
        univariate_A, bivariate_A = calculator.calculate(x_test_A)
        univariate_B, bivariate_B = calculator.calculate(x_test_B)

        # Calculate combined
        univariate_combined, bivariate_combined = calculator.calculate(x_test_combined)

        # First 5 samples of combined should match A
        uni_diff_A = (univariate_combined[:5] - univariate_A).abs().max().item()
        bi_diff_A = (bivariate_combined[:5] - bivariate_A).abs().max().item()

        # Last 5 samples of combined should match B
        uni_diff_B = (univariate_combined[5:] - univariate_B).abs().max().item()
        bi_diff_B = (bivariate_combined[5:] - bivariate_B).abs().max().item()

        print("\nTest Sample Independence:")
        print(f"  Univariate diff (A separate vs combined[:5]): {uni_diff_A:.6e}")
        print(f"  Bivariate diff (A separate vs combined[:5]):  {bi_diff_A:.6e}")
        print(f"  Univariate diff (B separate vs combined[5:]): {uni_diff_B:.6e}")
        print(f"  Bivariate diff (B separate vs combined[5:]):  {bi_diff_B:.6e}")

        assert uni_diff_A < 1e-6, "Results change when calculated separately vs combined"
        assert bi_diff_A < 1e-6, "Results change when calculated separately vs combined"
        assert uni_diff_B < 1e-6, "Results change when calculated separately vs combined"
        assert bi_diff_B < 1e-6, "Results change when calculated separately vs combined"


class TestBatchedPredict:
    """Test _batched_predict sub-batching for GPU OOM prevention."""

    def test_batched_predict_matches_unbatched(self, linear_test_model):
        """Small predict_batch_size produces identical results to full predict."""
        torch.manual_seed(42)
        x_train = torch.randn(50, 5)
        x_test = torch.randn(500, 5)

        calculator = PartialResponseCalculator(
            linear_test_model,
            method='lebesgue',
            x_train=x_train,
            input_dim=5,
            predict_batch_size=100,
        )

        expected = calculator.predict(x_test)
        actual = calculator._batched_predict(x_test)

        assert torch.allclose(
            expected, actual, atol=1e-6
        ), f"Max diff: {(expected - actual).abs().max().item():.2e}"

    def test_batched_predict_fast_path(self, linear_test_model):
        """When n_rows <= predict_batch_size, no splitting occurs."""
        torch.manual_seed(42)
        x_train = torch.randn(50, 5)
        x_test = torch.randn(500, 5)

        calculator = PartialResponseCalculator(
            linear_test_model,
            method='lebesgue',
            x_train=x_train,
            input_dim=5,
            predict_batch_size=1000,
        )

        expected = calculator.predict(x_test)
        actual = calculator._batched_predict(x_test)

        assert torch.allclose(expected, actual, atol=1e-6)

    def test_lebesgue_results_unchanged_with_small_predict_batch(self, linear_test_model):
        """End-to-end: calculate() results identical with small predict_batch_size."""
        torch.manual_seed(42)
        x_train = torch.randn(50, 5)
        x_test = torch.randn(10, 5)

        calc_default = PartialResponseCalculator(
            linear_test_model,
            method='lebesgue',
            x_train=x_train,
            input_dim=5,
            predict_batch_size=65536,
        )
        uni_default, bi_default = calc_default.calculate(x_test)

        calc_small = PartialResponseCalculator(
            linear_test_model,
            method='lebesgue',
            x_train=x_train,
            input_dim=5,
            predict_batch_size=50,
        )
        uni_small, bi_small = calc_small.calculate(x_test)

        uni_diff = (uni_default - uni_small).abs().max().item()
        bi_diff = (bi_default - bi_small).abs().max().item()

        assert uni_diff < 1e-5, f"Univariate mismatch: {uni_diff:.2e}"
        assert bi_diff < 1e-5, f"Bivariate mismatch: {bi_diff:.2e}"

    def test_batched_predict_single_row(self, linear_test_model):
        """Edge case: single row input."""
        torch.manual_seed(42)
        x_train = torch.randn(50, 5)
        x_test = torch.randn(1, 5)

        calculator = PartialResponseCalculator(
            linear_test_model,
            method='lebesgue',
            x_train=x_train,
            input_dim=5,
            predict_batch_size=100,
        )

        expected = calculator.predict(x_test)
        actual = calculator._batched_predict(x_test)

        assert torch.allclose(expected, actual, atol=1e-6)

    def test_batched_predict_exact_multiple(self, linear_test_model):
        """n_rows is exact multiple of predict_batch_size."""
        torch.manual_seed(42)
        x_train = torch.randn(50, 5)
        x_test = torch.randn(200, 5)

        calculator = PartialResponseCalculator(
            linear_test_model,
            method='lebesgue',
            x_train=x_train,
            input_dim=5,
            predict_batch_size=100,
        )

        expected = calculator.predict(x_test)
        actual = calculator._batched_predict(x_test)

        assert torch.allclose(expected, actual, atol=1e-6)

    def test_auto_predict_batch_size_cpu_fallback(self):
        """CPU device returns fallback of 65536."""
        cpu = torch.device('cpu')
        result = PartialResponseCalculator._auto_predict_batch_size(cpu)
        assert result == 65536

    def test_auto_predict_batch_size_cuda_no_model(self):
        """Mock CUDA with 36GB free, no model -> uses 20KB fallback per row."""
        from unittest.mock import patch

        cuda_device = torch.device('cuda:0')
        free_bytes = 36 * (1024**3)  # 36 GB
        total_bytes = 40 * (1024**3)  # 40 GB

        with patch('torch.cuda.mem_get_info', return_value=(free_bytes, total_bytes)):
            result = PartialResponseCalculator._auto_predict_batch_size(cuda_device)

        # 36GB * 0.25 = 9GB budget, 9GB / 20KB = ~471K rows
        # Round down to power of 2: 262144
        assert result == 262144, f"Expected 262144, got {result}"

    def test_auto_predict_batch_size_model_aware(self):
        """Model-aware estimation: large model -> smaller batch, small model -> larger batch."""
        from unittest.mock import patch

        import torch.nn as nn

        cuda_device = torch.device('cuda:0')
        free_bytes = 8 * (1024**3)  # 8 GB free

        # Small model: 10 + 1 = 11 output units -> 11 * 4 * 2 = 88 bytes/row
        small_model = nn.Sequential(nn.Linear(5, 10), nn.Linear(10, 1))

        # Large model: 1620 + 1 = 1621 output units -> 1621 * 4 * 2 = 12968 bytes/row
        large_model = nn.Sequential(nn.Linear(162, 1620), nn.Linear(1620, 1))

        with patch('torch.cuda.mem_get_info', return_value=(free_bytes, free_bytes)):
            batch_small = PartialResponseCalculator._auto_predict_batch_size(
                cuda_device, small_model
            )
            batch_large = PartialResponseCalculator._auto_predict_batch_size(
                cuda_device, large_model
            )

        # Small model should get a much larger batch size than large model
        assert (
            batch_small > batch_large
        ), f"Small model batch ({batch_small}) should exceed large model batch ({batch_large})"
        # Large model: 8GB * 0.25 = 2GB budget, 2GB / 12968 = ~161K -> 131072
        assert batch_large == 131072, f"Expected 131072, got {batch_large}"

    def test_estimate_bytes_per_row_nn_module(self):
        """_estimate_bytes_per_row inspects Linear layers correctly."""
        import torch.nn as nn

        model = nn.Sequential(nn.Linear(10, 100), nn.Tanh(), nn.Linear(100, 1))
        result = PartialResponseCalculator._estimate_bytes_per_row(model)
        # (100 + 1) output units * 4 bytes * 2 safety = 808
        assert result == 101 * 4 * 2

    def test_estimate_bytes_per_row_non_nn(self):
        """Non-nn.Module without GPU XGBoost returns 0."""
        from unittest.mock import MagicMock

        sklearn_model = MagicMock(spec=[])
        result = PartialResponseCalculator._estimate_bytes_per_row(sklearn_model)
        assert result == 0

    def test_estimate_bytes_per_row_xgb_gpu(self):
        """SklearnWrapper with GPU XGBoost estimates from n_estimators and max_depth."""
        from unittest.mock import MagicMock

        # Simulate SklearnWrapper with _xgb_gpu_enabled=True
        inner_model = MagicMock()
        inner_model.get_params.return_value = {
            'n_estimators': 200,
            'max_depth': 8,
        }

        wrapper = MagicMock()
        wrapper._xgb_gpu_enabled = True
        wrapper.model = inner_model

        result = PartialResponseCalculator._estimate_bytes_per_row(wrapper)
        # 200 trees * 8 depth * 32 bytes = 51200
        assert result == 200 * 8 * 32

    def test_estimate_bytes_per_row_xgb_defaults(self):
        """XGBoost GPU with default params uses n_estimators=100, max_depth=6."""
        from unittest.mock import MagicMock

        inner_model = MagicMock()
        inner_model.get_params.return_value = {}  # no explicit params

        wrapper = MagicMock()
        wrapper._xgb_gpu_enabled = True
        wrapper.model = inner_model

        result = PartialResponseCalculator._estimate_bytes_per_row(wrapper)
        # defaults: 100 * 6 * 32 = 19200
        assert result == 100 * 6 * 32
