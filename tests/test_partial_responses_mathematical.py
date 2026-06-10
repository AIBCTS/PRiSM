"""Mathematical Validation Tests for Partial Responses - Ground Truth Testing.

This module tests partial response calculations against models with known
mathematical properties (ground truth). This allows us to validate correctness
by comparing computed responses to analytically known values.

CRITICAL DIAGNOSTIC: test_linear_model_exact_reconstruction
This test determines whether reconstruction failure is due to:
1. Implementation bug (if linear model fails)
2. 2nd-order truncation limitation (if linear model passes)

Test Plan Reference: Phase 1, Tests 1.3
"""

import torch

from prism.partial_responses import PartialResponseCalculator, stable_logit


class TestLinearModelGroundTruth:
    """Test partial responses with purely linear models (no interactions).

    Linear models should:
    1. Have bivariate responses ≈ 0 (no interactions)
    2. Allow exact reconstruction (no higher-order terms)
    3. Have univariate responses proportional to coefficients
    """

    def test_linear_model_exact_reconstruction(self, linear_test_model):
        """CRITICAL TEST: Linear model should reconstruct exactly.

        This is THE KEY DIAGNOSTIC TEST.

        If this PASSES: Reconstruction failure for MLP is due to 2nd-order truncation
        If this FAILS: There is a bug in the implementation

        Linear model has NO higher-order terms, so 2nd-order decomposition
        should be exact.
        """
        torch.manual_seed(42)

        model = linear_test_model
        x_train = torch.randn(100, 5)
        x_test = torch.randn(20, 5)

        # Calculate partial responses
        calculator = PartialResponseCalculator(
            model, method='lebesgue', x_train=x_train, input_dim=5
        )
        univariate, bivariate = calculator.calculate(x_test)

        # Get true predictions
        y_pred = model(x_test)
        logit_pred = stable_logit(y_pred)

        # Reconstruct
        logit_reconstructed = calculator.logit_y0 + univariate.sum(dim=1) + bivariate.sum(dim=1)

        # Calculate error
        error = (logit_pred - logit_reconstructed).abs()
        max_error = error.max().item()
        mean_error = error.mean().item()

        print("\nLinear Model Reconstruction Test:")
        print(f"  Max error:  {max_error:.6e}")
        print(f"  Mean error: {mean_error:.6e}")
        print(f"  Baseline:   {calculator.logit_y0:.6f}")
        print("  Model coefficients: [0.3, 0.5, -0.2, 0.1, 0.4]")

        # For linear model, reconstruction should be exact
        # Tightened from 1e-4 to 1e-5 after recent corrections (actual ~4e-7)
        assert max_error < 1e-5, (
            f"Linear model reconstruction failed! Error: {max_error:.6e}\n"
            "This indicates a BUG in the implementation.\n"
            "Expected: < 1e-5 for linear model with no interactions."
        )

        print("  PASS: Linear model reconstructs exactly")
        print("  Conclusion: MLP failure is due to 2nd-order truncation limitation")

    def test_linear_model_no_interactions(self, linear_test_model):
        """Linear model should have bivariate responses ≈ 0 (no interactions)."""
        torch.manual_seed(42)

        model = linear_test_model
        x_train = torch.randn(100, 5)
        x_test = torch.randn(20, 5)

        calculator = PartialResponseCalculator(
            model, method='lebesgue', x_train=x_train, input_dim=5
        )
        univariate, bivariate = calculator.calculate(x_test)

        # Bivariate should be near zero for purely additive model
        max_bivariate = bivariate.abs().max().item()
        mean_bivariate = bivariate.abs().mean().item()

        print("\nLinear Model Bivariate Test:")
        print(f"  Max bivariate:  {max_bivariate:.6e}")
        print(f"  Mean bivariate: {mean_bivariate:.6e}")

        # Tightened from 1e-3 to 1e-5 after recent corrections (actual ~1.5e-7)
        assert max_bivariate < 1e-5, (
            f"Linear model has significant bivariate responses: {max_bivariate:.6e}\n"
            "Expected near zero (< 1e-5) since model is purely additive."
        )

        print("  PASS: No spurious interactions detected")

    def test_linear_model_dirac_reconstruction(self, linear_test_model):
        """Test Dirac method reconstruction with linear model."""
        torch.manual_seed(42)

        model = linear_test_model
        x_test = torch.randn(20, 5)

        calculator = PartialResponseCalculator(model, method='dirac', x_train=None, input_dim=5)
        univariate, bivariate = calculator.calculate(x_test)

        y_pred = model(x_test)
        logit_pred = stable_logit(y_pred)

        logit_reconstructed = calculator.logit_y0 + univariate.sum(dim=1) + bivariate.sum(dim=1)

        error = (logit_pred - logit_reconstructed).abs()
        max_error = error.max().item()

        print("\nLinear Model Dirac Reconstruction:")
        print(f"  Max error: {max_error:.6e}")

        assert max_error < 1e-4, f"Dirac reconstruction failed for linear model: {max_error:.6e}"

    def test_linear_model_coefficients_match(self, linear_test_model):
        """Univariate responses should relate to model coefficients.

        For a linear model f(x) = w·x + b in logit space,
        the partial response should capture the linear contribution.

        This is a qualitative test - we expect to see the pattern of
        coefficients reflected in the average univariate responses.
        """
        torch.manual_seed(42)

        model = linear_test_model
        coeffs = torch.tensor([0.3, 0.5, -0.2, 0.1, 0.4])

        # Use values centered at 0 for cleaner interpretation
        x_train = torch.randn(200, 5)
        x_test = torch.zeros(1, 5)  # Test at origin

        calculator = PartialResponseCalculator(
            model, method='lebesgue', x_train=x_train, input_dim=5
        )
        univariate, bivariate = calculator.calculate(x_test)

        print("\nLinear Model Coefficient Analysis:")
        print(f"  True coefficients: {coeffs.tolist()}")
        print(f"  Univariate at origin: {univariate[0].tolist()}")
        print("  (Should be close to zero at origin for zero-centered data)")

        # At origin with zero-centered data, univariate responses should be small
        assert univariate[0].abs().max() < 0.5, "Responses too large at origin"


class TestInteractionModelGroundTruth:
    """Test partial responses with models containing known interactions."""

    def test_interaction_model_detects_interaction(self, interaction_test_model):
        """Model with x₀*x₁ interaction should have non-zero bivariate response.

        Model: logit = x₀ + x₁ + 0.5*x₀*x₁

        Expected:
        - φ₀(x₀) and φ₁(x₁) capture linear terms
        - φ₀₁(x₀, x₁) captures 0.5*x₀*x₁ interaction
        """
        torch.manual_seed(42)

        model = interaction_test_model
        x_train = torch.randn(100, 5)
        x_test = torch.randn(20, 5)

        calculator = PartialResponseCalculator(
            model, method='lebesgue', x_train=x_train, input_dim=5
        )
        univariate, bivariate = calculator.calculate(x_test)

        # Model has interaction between features 0 and 1
        # Bivariate responses should be non-zero
        max_bivariate = bivariate.abs().max().item()
        mean_bivariate = bivariate.abs().mean().item()

        print("\nInteraction Model Detection Test:")
        print(f"  Max bivariate:  {max_bivariate:.6e}")
        print(f"  Mean bivariate: {mean_bivariate:.6e}")
        print("  Model has 0.5*x0*x1 interaction")

        assert max_bivariate > 1e-2, (
            f"Failed to detect known interaction! Max bivariate: {max_bivariate:.6e}\n"
            "Expected > 0.01 since model has explicit x\u2080*x\u2081 term"
        )

        print("  PASS: Interaction successfully detected")

    def test_interaction_model_reconstruction(self, interaction_test_model):
        """Interaction model should reconstruct exactly (no higher-order terms)."""
        torch.manual_seed(42)

        model = interaction_test_model
        x_train = torch.randn(100, 5)
        x_test = torch.randn(20, 5)

        calculator = PartialResponseCalculator(
            model, method='lebesgue', x_train=x_train, input_dim=5
        )
        univariate, bivariate = calculator.calculate(x_test)

        y_pred = model(x_test)
        logit_pred = stable_logit(y_pred)

        logit_reconstructed = calculator.logit_y0 + univariate.sum(dim=1) + bivariate.sum(dim=1)

        error = (logit_pred - logit_reconstructed).abs()
        max_error = error.max().item()

        print("\nInteraction Model Reconstruction:")
        print(f"  Max error: {max_error:.6e}")

        # Tightened from 1e-3 to 1e-5 after recent corrections (actual ~2.4e-7)
        assert max_error < 1e-5, (
            f"Interaction model reconstruction error too high: {max_error:.6e}\n"
            "Expected < 1e-5 for 2nd-order polynomial model"
        )


class TestAdditiveModelProperty:
    """Test that purely additive models have zero bivariate responses."""

    def test_additive_model_zero_interactions(self):
        """Purely additive model should have bivariate approx 0 everywhere.

        Tests the fundamental property that bivariate terms should only
        be non-zero when there are actual interactions in the model.
        """
        torch.manual_seed(42)

        # Create additive model: each feature contributes independently
        class AdditiveModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weights = torch.nn.Parameter(
                    torch.tensor([0.5, 0.3, 0.2, 0.1, 0.4]), requires_grad=False
                )

            def forward(self, x):
                # f(x) = 0.5*x0 + 0.3*x1 + 0.2*x2 + 0.1*x3 + 0.4*x4
                logits = torch.matmul(x, self.weights)
                return torch.sigmoid(logits)

            def predict_proba(self, x, device=None):
                if device is not None:
                    x = x.to(device)
                    self.to(device)
                return self.forward(x)

        model = AdditiveModel()
        x_train = torch.randn(100, 5)
        x_test = torch.randn(20, 5)

        calculator = PartialResponseCalculator(
            model, method='lebesgue', x_train=x_train, input_dim=5
        )
        univariate, bivariate = calculator.calculate(x_test)

        max_bivariate = bivariate.abs().max().item()

        print("\nAdditive Model Test:")
        print(f"  Max bivariate: {max_bivariate:.6e}")

        # Tightened from 1e-3 to 1e-5 after recent corrections (actual ~2.4e-7)
        assert max_bivariate < 1e-5, f"Additive model has non-zero bivariate: {max_bivariate:.6e}"


class TestMLPHigherOrderTerms:
    """Document MLP reconstruction error as expected behavior."""

    def test_mlp_reconstruction_error_documented(self, test_mlp):
        """Document that MLP reconstruction error is expected behavior.

        This test documents the reconstruction error for MLPs and
        confirms it's due to higher-order terms, not a bug.

        This test is expected to have reconstruction error > 1e-4.
        """
        torch.manual_seed(42)

        model = test_mlp
        x_train = torch.randn(100, 5)
        x_test = torch.randn(20, 5)

        calculator = PartialResponseCalculator(
            model, method='lebesgue', x_train=x_train, input_dim=5
        )
        univariate, bivariate = calculator.calculate(x_test)

        y_pred = model(x_test)
        logit_pred = stable_logit(y_pred)

        logit_reconstructed = calculator.logit_y0 + univariate.sum(dim=1) + bivariate.sum(dim=1)

        error = (logit_pred - logit_reconstructed).abs()
        residual = logit_pred - logit_reconstructed

        print("\nMLP Higher-Order Residual Analysis:")
        print(f"  Max error: {error.max().item():.6e}")
        print(f"  Mean error: {error.mean().item():.6e}")
        print(f"  Std error: {error.std().item():.6e}")
        print(f"  Mean residual: {residual.mean().item():.6f}")
        print(f"  Std residual: {residual.std().item():.6f}")
        print("\n  This error represents higher-order interactions")
        print("  not captured by 2nd-order truncation.")
        print("  This is EXPECTED BEHAVIOR, not a bug.")

        # This should have error > 1e-4 (confirming higher-order terms exist)
        assert error.max().item() > 1e-4, (
            "MLP unexpectedly has perfect reconstruction - " "may indicate test issue"
        )

        # But error should be reasonable (not extreme)
        assert (
            error.max().item() < 0.5
        ), f"Reconstruction error unreasonably high: {error.max().item():.3f}"

    def test_mlp_residual_vs_linear(self, test_mlp, linear_test_model):
        """Compare MLP residual to linear model - should be much larger."""
        torch.manual_seed(42)

        x_train = torch.randn(100, 5)
        x_test = torch.randn(20, 5)

        # MLP residual
        calc_mlp = PartialResponseCalculator(
            test_mlp, method='lebesgue', x_train=x_train, input_dim=5
        )
        uni_mlp, bi_mlp = calc_mlp.calculate(x_test)
        y_mlp = test_mlp(x_test)
        logit_mlp = stable_logit(y_mlp)
        recon_mlp = calc_mlp.logit_y0 + uni_mlp.sum(dim=1) + bi_mlp.sum(dim=1)
        error_mlp = (logit_mlp - recon_mlp).abs().max().item()

        # Linear residual
        calc_lin = PartialResponseCalculator(
            linear_test_model, method='lebesgue', x_train=x_train, input_dim=5
        )
        uni_lin, bi_lin = calc_lin.calculate(x_test)
        y_lin = linear_test_model(x_test)
        logit_lin = stable_logit(y_lin)
        recon_lin = calc_lin.logit_y0 + uni_lin.sum(dim=1) + bi_lin.sum(dim=1)
        error_lin = (logit_lin - recon_lin).abs().max().item()

        print("\nResidual Comparison:")
        print(f"  Linear model: {error_lin:.6e}")
        print(f"  MLP model:    {error_mlp:.6e}")
        print(f"  Ratio:        {error_mlp / error_lin:.1f}x")

        # MLP should have significantly more residual
        assert (
            error_mlp > 10 * error_lin
        ), "MLP should have much more reconstruction error than linear model"
