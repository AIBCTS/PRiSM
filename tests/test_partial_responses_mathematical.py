"""Ground-truth validation of the functional ANOVA decomposition.

Models with known mathematical structure (linear, explicit pairwise
interaction, explicit third-order term) make the exact decomposition known in
closed form, separating implementation bugs from legitimate second-order
truncation error: a linear model must reconstruct exactly, so any
reconstruction failure on it is a bug, while residual error on a model with
genuine higher-order terms is expected truncation, not a defect.
"""

import inspect

import torch

from prism.partial_responses import PartialResponseCalculator, stable_logit


class TestLinearModelGroundTruth:
    """Test partial responses with purely linear models (no interactions).

    Linear models should:
    1. Have bivariate responses close to 0 (no interactions)
    2. Allow exact reconstruction (no higher-order terms)
    3. Have univariate responses proportional to coefficients
    """

    def test_linear_model_exact_reconstruction(self, linear_test_model):
        """A linear model must reconstruct exactly: the key diagnostic.

        A linear model has no higher-order terms, so the 2nd-order
        decomposition is exact. If this test fails the implementation is
        broken; if it passes, reconstruction error on nonlinear models is
        attributable to truncation, not bugs.
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

        # For a linear model reconstruction is exact up to float error
        # (observed max ~4e-7).
        assert max_error < 1e-5, (
            f"Linear model reconstruction failed! Error: {max_error:.6e}\n"
            f"This indicates a BUG in the implementation.\n"
            f"Expected: < 1e-5 for linear model with no interactions."
        )

        print("  PASS: Linear model reconstructs exactly")
        print("  Conclusion: MLP failure is due to 2nd-order truncation limitation")

    def test_linear_model_no_interactions(self, linear_test_model):
        """Linear model should have bivariate responses close to 0 (no interactions)."""
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

        # Exact up to float error (observed max ~1.5e-7).
        assert max_bivariate < 1e-5, (
            f"Linear model has significant bivariate responses: {max_bivariate:.6e}\n"
            f"Expected near zero (< 1e-5) since model is purely additive."
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

        For a linear model f(x) = w . x + b in logit space,
        the partial response should capture the linear contribution.

        This is a qualitative test: we expect to see the pattern of
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
        """Model with x_0*x_1 interaction should have non-zero bivariate response.

        Model: logit = x_0 + x_1 + 0.5*x_0*x_1

        Expected:
        - φ_0(x_0) and φ_1(x_1) capture linear terms
        - φ_01(x_0, x_1) captures 0.5*x_0*x_1 interaction
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
            f"Expected > 0.01 since model has explicit x₀*x₁ term"
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

        # Exact up to float error (observed max ~2.4e-7).
        assert max_error < 1e-5, (
            f"Interaction model reconstruction error too high: {max_error:.6e}\n"
            f"Expected < 1e-5 for 2nd-order polynomial model"
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

        # Exact up to float error (observed max ~2.4e-7).
        assert max_bivariate < 1e-5, f"Additive model has non-zero bivariate: {max_bivariate:.6e}"


class ThirdOrderModel(torch.nn.Module):
    """Model with a controllable, known third-order term: a.x + c*(x0*x1*x2).

    The linear part is exactly reconstructable by 2nd-order ANOVA, so the entire
    reconstruction residual is attributable to the third-order term scaled by `c`.
    This gives a mathematically grounded, threshold-free way to validate the
    truncation: residual is ~0 at c=0 and grows monotonically with |c|.
    """

    def __init__(self, linear_coeffs, third_order_coeff, triple=(0, 1, 2)):
        super().__init__()
        self.linear_coeffs = torch.nn.Parameter(
            torch.tensor(linear_coeffs, dtype=torch.float32), requires_grad=False
        )
        self.c = float(third_order_coeff)
        self.triple = triple

    def forward(self, x):
        i, j, k = self.triple
        logits = torch.matmul(x, self.linear_coeffs) + self.c * x[:, i] * x[:, j] * x[:, k]
        return torch.sigmoid(logits)

    def predict_proba(self, x, device=None):
        if device is not None:
            x = x.to(device)
            self.to(device)
        return self.forward(x)


class TestKnownHigherOrderTruncation:
    """Second-order truncation validated against a known third-order term.

    Instead of asserting a magic error bound on a random MLP, use a model
    whose only non-2nd-order content is a single known third-order term and
    assert structural properties the math guarantees: zero residual at c=0
    and monotonically growing residual as |c| increases.
    """

    @staticmethod
    def _max_reconstruction_error(model, x_train, x_test):
        calculator = PartialResponseCalculator(
            model, method='lebesgue', x_train=x_train, input_dim=x_train.shape[1]
        )
        univariate, bivariate = calculator.calculate(x_test)
        logit_pred = stable_logit(model(x_test))
        logit_recon = calculator.logit_y0 + univariate.sum(dim=1) + bivariate.sum(dim=1)
        return (logit_pred - logit_recon).abs().max().item()

    def test_residual_zero_without_third_order_term(self):
        """c=0 -> purely additive -> exact 2nd-order reconstruction."""
        torch.manual_seed(42)
        x_train = torch.randn(200, 5)
        x_test = torch.randn(40, 5)
        model = ThirdOrderModel([0.3, 0.5, -0.2, 0.1, 0.4], third_order_coeff=0.0)

        err = self._max_reconstruction_error(model, x_train, x_test)
        assert err < 1e-5, (
            f"Additive model (c=0) failed exact reconstruction: {err:.3e} -- this is a bug, "
            f"not truncation error."
        )

    def test_residual_grows_monotonically_with_third_order_coeff(self):
        """Turning up the known third-order coefficient must increase the residual.

        This validates that 2nd-order truncation captures exactly the <=2nd-order
        content and nothing higher: threshold-free and robust to torch internals.
        """
        torch.manual_seed(42)
        x_train = torch.randn(200, 5)
        x_test = torch.randn(40, 5)
        linear = [0.3, 0.5, -0.2, 0.1, 0.4]

        coeffs = [0.0, 0.5, 1.0, 2.0]
        errors = [
            self._max_reconstruction_error(
                ThirdOrderModel(linear, third_order_coeff=c), x_train, x_test
            )
            for c in coeffs
        ]

        print("\nThird-order knob sweep (coeff -> max reconstruction residual):")
        for c, e in zip(coeffs, errors):
            print(f"  c={c:>4}: {e:.6e}")

        # c=0 is exact; every increase in |c| strictly increases the residual.
        assert errors[0] < 1e-5
        for a, b in zip(errors, errors[1:]):
            assert b > a, f"residual not monotonically increasing with third-order coeff: {errors}"

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


class TestDecompositionOutcomeIndependence:
    """Functional ANOVA depends only on the model, never on the outcome.

    The decomposition is a "translation of this model": it uses only the model's
    input->output behavior. Outcome labels enter the pipeline strictly at LASSO
    selection/weighting and PRN training, never in the partial responses themselves.
    """

    def test_partial_responses_independent_of_outcome(self, linear_test_model):
        """Recomputing the decomposition is bit-for-bit identical regardless of any
        outcome vector the surrounding pipeline happens to hold.

        The calculator never receives y, so a regression that leaked the outcome into
        the decomposition (or keyed a cache on it) would break this equality.
        """
        torch.manual_seed(42)
        model = linear_test_model
        x_train = torch.randn(100, 5)
        x_test = torch.randn(20, 5)

        calc = PartialResponseCalculator(model, method='lebesgue', x_train=x_train, input_dim=5)
        uni, bi = calc.calculate(x_test)

        for _ in range(3):
            # A different "outcome" each iteration: it is never passed in, so it
            # must not change the decomposition.
            _ = torch.randint(0, 2, (100,))
            calc2 = PartialResponseCalculator(
                model, method='lebesgue', x_train=x_train, input_dim=5
            )
            uni2, bi2 = calc2.calculate(x_test)
            torch.testing.assert_close(uni, uni2, rtol=0, atol=0)
            torch.testing.assert_close(bi, bi2, rtol=0, atol=0)

    def test_calculator_api_has_no_outcome_argument(self):
        """Structural guard: the decomposition API must not accept an outcome.

        A future refactor that threads y into the calculator would violate the
        model-only decomposition contract; this fails loudly if such a parameter
        is introduced.
        """
        forbidden = {"y", "y_train", "y_test", "target", "labels", "outcome"}

        ctor = set(inspect.signature(PartialResponseCalculator.__init__).parameters)
        calc = set(inspect.signature(PartialResponseCalculator.calculate).parameters)

        assert not (
            ctor & forbidden
        ), f"decomposition constructor leaks the outcome: {ctor & forbidden}"
        assert not (calc & forbidden), f"calculate() leaks the outcome: {calc & forbidden}"
