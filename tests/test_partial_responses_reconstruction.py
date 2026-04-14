"""Reconstruction Tests for Partial Responses - Mathematical Correctness Validation.

This module contains tests for validating the mathematical correctness of the
ANOVA decomposition implementation. The primary test verifies that partial
responses can reconstruct the original model predictions.

KEY INSIGHT: The ANOVA decomposition is truncated at 2nd order (univariate +
bivariate terms). For models with higher-order interactions (like MLPs), this
truncation means reconstruction will NOT be exact.

Expected behavior:
- Linear models: Exact reconstruction (< 1e-4 error) because they have no
  higher-order terms
- MLPs: Approximate reconstruction (typically 0.01-0.15 error) due to
  3rd-order and higher terms being omitted

The linear model tests in test_partial_responses_mathematical.py verify the
implementation is correct. MLP reconstruction errors document the inherent
limitation of 2nd-order truncation, not implementation bugs.

Test Plan Reference: Phase 1, Test 1.2
"""

import torch

from prism.partial_responses import PartialResponseCalculator, stable_logit


class TestExactReconstruction:
    """Test suite for verifying exact reconstruction from partial responses.

    The ANOVA decomposition should allow exact reconstruction of model
    predictions from the baseline and partial responses.
    """

    def test_reconstruction_lebesgue_mlp(self, test_mlp):
        """Verify partial responses reconstruct MLP predictions (Lebesgue).

        MLPs have higher-order interactions beyond 2nd order, so the truncated
        ANOVA decomposition will NOT achieve exact reconstruction. This test
        verifies the error is within expected bounds for 2nd-order truncation.

        Expected: Reconstruction error < 0.2 (typical range: 0.01-0.15)
        For exact reconstruction tests, see test_partial_responses_mathematical.py
        which uses linear models with no higher-order terms.
        """
        torch.manual_seed(42)

        # Setup
        n_features = 5
        model = test_mlp
        x_train = torch.randn(100, n_features)
        x_test = torch.randn(20, n_features)

        # Calculate partial responses
        calculator = PartialResponseCalculator(
            model, method='lebesgue', x_train=x_train, input_dim=n_features
        )
        univariate, bivariate = calculator.calculate(x_test)

        # Get true predictions in logit space
        y_pred = model(x_test)
        logit_pred = stable_logit(y_pred)

        # Reconstruct from partial responses
        # Formula: f(x) = baseline + sum(φᵢ) + sum(φᵢⱼ)
        logit_reconstructed = calculator.logit_y0 + univariate.sum(dim=1) + bivariate.sum(dim=1)

        # Calculate reconstruction error
        reconstruction_error = (logit_pred - logit_reconstructed).abs()
        max_error = reconstruction_error.max().item()
        mean_error = reconstruction_error.mean().item()
        std_error = reconstruction_error.std().item()

        # Detailed error reporting
        print("\nReconstruction Test Results (Lebesgue, MLP):")
        print(f"  Max error:  {max_error:.6e}")
        print(f"  Mean error: {mean_error:.6e}")
        print(f"  Std error:  {std_error:.6e}")
        print(f"  Baseline:   {calculator.logit_y0:.6f}")
        print("  Note: MLP has higher-order terms, so 2nd-order truncation is approximate")

        # Relaxed tolerance for MLP (higher-order terms cause irreducible error)
        assert max_error < 0.2, (
            f"Reconstruction error unexpectedly high: {max_error:.6e}. "
            "Expected < 0.2 for 2nd-order truncated ANOVA on MLP."
        )

    def test_reconstruction_dirac_mlp(self, test_mlp):
        """Verify partial responses reconstruct MLP predictions (Dirac).

        Tests Dirac method. Like Lebesgue, Dirac is also a 2nd-order truncation
        so MLPs will have reconstruction error from omitted higher-order terms.
        Dirac typically has slightly higher error than Lebesgue.
        """
        torch.manual_seed(42)

        # Setup
        n_features = 5
        model = test_mlp
        x_test = torch.randn(20, n_features)

        # Calculate partial responses (Dirac doesn't need x_train)
        calculator = PartialResponseCalculator(
            model, method='dirac', x_train=None, input_dim=n_features
        )
        univariate, bivariate = calculator.calculate(x_test)

        # Get true predictions in logit space
        y_pred = model(x_test)
        logit_pred = stable_logit(y_pred)

        # Reconstruct from partial responses
        logit_reconstructed = calculator.logit_y0 + univariate.sum(dim=1) + bivariate.sum(dim=1)

        # Calculate reconstruction error
        reconstruction_error = (logit_pred - logit_reconstructed).abs()
        max_error = reconstruction_error.max().item()
        mean_error = reconstruction_error.mean().item()

        print("\nReconstruction Test Results (Dirac, MLP):")
        print(f"  Max error:  {max_error:.6e}")
        print(f"  Mean error: {mean_error:.6e}")
        print(f"  Baseline:   {calculator.logit_y0:.6f}")
        print("  Note: MLP has higher-order terms, so 2nd-order truncation is approximate")

        # Relaxed tolerance for MLP (Dirac often slightly worse than Lebesgue)
        assert max_error < 0.25, (
            f"Reconstruction error unexpectedly high: {max_error:.6e}. "
            "Expected < 0.25 for Dirac method on MLP."
        )

    def test_reconstruction_lebesgue_larger_mlp(self, test_mlp_10d):
        """Test reconstruction with larger 10-feature MLP.

        Verifies reconstruction error remains bounded for higher-dimensional
        models. With more features, there are more potential higher-order
        interactions, but error should remain reasonable.
        """
        torch.manual_seed(42)

        n_features = 10
        n_pairs = n_features * (n_features - 1) // 2  # C(10,2) = 45
        model = test_mlp_10d
        x_train = torch.randn(200, n_features)
        x_test = torch.randn(50, n_features)

        calculator = PartialResponseCalculator(
            model, method='lebesgue', x_train=x_train, input_dim=n_features
        )
        univariate, bivariate = calculator.calculate(x_test)

        y_pred = model(x_test)
        logit_pred = stable_logit(y_pred)

        logit_reconstructed = calculator.logit_y0 + univariate.sum(dim=1) + bivariate.sum(dim=1)

        reconstruction_error = (logit_pred - logit_reconstructed).abs()
        max_error = reconstruction_error.max().item()

        print("\nReconstruction Test Results (Lebesgue, 10D MLP):")
        print(f"  Max error:  {max_error:.6e}")
        print(f"  Features:   {n_features}")
        print(f"  Pairs:      {n_pairs}")
        print("  Note: Higher dimensions may have more higher-order interactions")

        # Relaxed tolerance for MLP
        assert (
            max_error < 0.25
        ), f"Reconstruction error unexpectedly high for 10D model: {max_error:.6e}"

    def test_reconstruction_with_different_batch_sizes(self, test_mlp):
        """Verify reconstruction is independent of batch size.

        Reconstruction should produce identical results regardless of the
        batch_size parameter used during calculation.
        """
        torch.manual_seed(42)

        model = test_mlp
        x_train = torch.randn(100, 5)
        x_test = torch.randn(20, 5)

        y_pred = model(x_test)
        logit_pred = stable_logit(y_pred)

        batch_sizes = [1, 5, 10, 20, 100]
        errors = []

        for batch_size in batch_sizes:
            calculator = PartialResponseCalculator(
                model, method='lebesgue', x_train=x_train, input_dim=5
            )
            univariate, bivariate = calculator.calculate(x_test, batch_size=batch_size)

            logit_reconstructed = (
                calculator.logit_y0 + univariate.sum(dim=1) + bivariate.sum(dim=1)
            )

            error = (logit_pred - logit_reconstructed).abs().max().item()
            errors.append(error)

        print("\nReconstruction errors across batch sizes:")
        for bs, err in zip(batch_sizes, errors):
            print(f"  Batch size {bs:3d}: {err:.6e}")

        # All should be below threshold (relaxed for MLP)
        for bs, err in zip(batch_sizes, errors):
            assert err < 0.2, f"Reconstruction failed for batch_size={bs}: error={err:.6e}"

        # All should be nearly identical (batch size shouldn't affect result)
        error_range = max(errors) - min(errors)
        assert (
            error_range < 1e-5
        ), f"Reconstruction varies with batch size: range={error_range:.6e}"


class TestBaselineCalculation:
    """Test suite for validating baseline calculation.

    The baseline is E[f(X)] for Lebesgue, f(0) for Dirac.
    This is critical for correct ANOVA decomposition.

    Investigation: Is current implementation mean(logit(y)) correct,
    or should it be logit(mean(y))?
    """

    def test_baseline_lebesgue_is_mean_logit(self, test_mlp):
        """Verify Lebesgue baseline is computed as mean(logit(predictions)).

        Current implementation: baseline = stable_logit(y0).mean()
        where y0 = model(x_train)

        This test documents the current behavior.
        """
        torch.manual_seed(42)

        model = test_mlp
        x_train = torch.randn(100, 5)

        calculator = PartialResponseCalculator(
            model, method='lebesgue', x_train=x_train, input_dim=5
        )

        # Manually compute what baseline should be
        y_train = model(x_train)
        expected_baseline = stable_logit(y_train).mean().item()

        print("\nBaseline Calculation (Lebesgue):")
        print(f"  Calculator baseline: {calculator.logit_y0:.6f}")
        print(f"  Expected (mean(logit)): {expected_baseline:.6f}")

        assert (
            abs(calculator.logit_y0 - expected_baseline) < 1e-6
        ), "Baseline calculation doesn't match expected mean(logit(y))"

    def test_baseline_dirac_is_zero_input(self, test_mlp):
        """Verify Dirac baseline is f(0, 0, ..., 0).

        Note: For Dirac method, logit_y0 is computed lazily during calculate().
        """
        model = test_mlp

        calculator = PartialResponseCalculator(model, method='dirac', x_train=None, input_dim=5)

        # Trigger baseline calculation by calling calculate
        x_test = torch.randn(5, 5)
        calculator.calculate(x_test)

        # Manually compute expected baseline
        x_zero = torch.zeros((1, 5))
        y_zero = model(x_zero)
        expected_baseline = stable_logit(y_zero).item()

        print("\nBaseline Calculation (Dirac):")
        print(f"  Calculator baseline: {calculator.logit_y0:.6f}")
        print(f"  Expected (f(0)): {expected_baseline:.6f}")

        assert abs(calculator.logit_y0 - expected_baseline) < 1e-6


class TestReconstructionEdgeCases:
    """Test reconstruction under edge cases."""

    def test_reconstruction_single_sample(self, test_mlp):
        """Test reconstruction with single test sample.

        Verifies the calculation works with batch size of 1.
        """
        torch.manual_seed(42)

        model = test_mlp
        x_train = torch.randn(50, 5)
        x_test = torch.randn(1, 5)  # Single sample

        calculator = PartialResponseCalculator(
            model, method='lebesgue', x_train=x_train, input_dim=5
        )
        univariate, bivariate = calculator.calculate(x_test)

        y_pred = model(x_test)
        logit_pred = stable_logit(y_pred)

        logit_reconstructed = calculator.logit_y0 + univariate.sum(dim=1) + bivariate.sum(dim=1)

        error = (logit_pred - logit_reconstructed).abs().item()

        print(f"\nSingle Sample Reconstruction Error: {error:.6e}")

        # Tightened from 0.2 to 0.05 after recent corrections (actual ~2.7e-3)
        assert error < 0.05

    def test_reconstruction_small_training_set(self, test_mlp):
        """Test reconstruction with small training set.

        With very few training samples, Lebesgue should still produce
        reasonable results, though error may be slightly higher due to
        less representative averaging.
        """
        torch.manual_seed(42)

        model = test_mlp
        x_train = torch.randn(5, 5)  # Only 5 training samples
        x_test = torch.randn(10, 5)

        calculator = PartialResponseCalculator(
            model, method='lebesgue', x_train=x_train, input_dim=5
        )
        univariate, bivariate = calculator.calculate(x_test)

        y_pred = model(x_test)
        logit_pred = stable_logit(y_pred)

        logit_reconstructed = calculator.logit_y0 + univariate.sum(dim=1) + bivariate.sum(dim=1)

        error = (logit_pred - logit_reconstructed).abs().max().item()

        print(f"\nSmall Training Set (n=5) Reconstruction Error: {error:.6e}")

        # Relaxed tolerance for MLP with small training set
        assert error < 0.2

    def test_reconstruction_extreme_predictions(self, linear_test_model):
        """Test reconstruction when model makes very confident predictions.

        This tests numerical stability of logit transformation with extreme
        probability values (near 0 or 1). The main check is that no NaN/Inf
        values are produced.

        Note: With extreme inputs, linear models can push probabilities to
        extreme values where logit numerical precision may degrade.
        """
        torch.manual_seed(42)

        model = linear_test_model

        # Create data that leads to extreme predictions
        x_train = torch.randn(50, 5) * 10  # Large scale
        x_test = torch.randn(10, 5) * 10

        calculator = PartialResponseCalculator(
            model, method='lebesgue', x_train=x_train, input_dim=5
        )
        univariate, bivariate = calculator.calculate(x_test)

        y_pred = model(x_test)
        logit_pred = stable_logit(y_pred)

        # Primary check: no NaN or Inf values
        assert torch.all(torch.isfinite(logit_pred)), "Predicted logits contain NaN/Inf"
        assert torch.all(torch.isfinite(univariate)), "Univariate responses contain NaN/Inf"
        assert torch.all(torch.isfinite(bivariate)), "Bivariate responses contain NaN/Inf"

        logit_reconstructed = calculator.logit_y0 + univariate.sum(dim=1) + bivariate.sum(dim=1)

        assert torch.all(
            torch.isfinite(logit_reconstructed)
        ), "Reconstructed logits contain NaN/Inf"

        error = (logit_pred - logit_reconstructed).abs().max().item()

        print(f"\nExtreme Predictions Reconstruction Error: {error:.6e}")
        print(f"  Min probability: {y_pred.min().item():.6e}")
        print(f"  Max probability: {y_pred.max().item():.6e}")

        # Relaxed tolerance - extreme values may have higher numerical error
        # Main goal is numerical stability (no NaN/Inf), not tight reconstruction
        assert error < 1.0, "Reconstruction completely fails with extreme predictions"
