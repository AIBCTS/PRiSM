"""
Test Dirac Baseline with One-Hot Encoded Features.

This test validates:
1. PRiSMScaler correctly excludes binary/one-hot columns from scaling
2. Dirac baseline calculation is consistent with partial response calculations
3. Reconstruction accuracy with one-hot features

See: notes/DIRAC_BASELINE_SCALING_BUG.md
"""

import warnings

import numpy as np
import pytest
import torch

from prism.partial_responses import PartialResponseCalculator, _warn_if_scaled_onehot, stable_logit
from prism.preprocessing import PRiSMScaler


class MockLinearModelWithOnehot:
    """
    Linear model with known coefficients for testing.

    Model: logit = intercept + sum(coef_i * x_i)

    Supports both continuous and one-hot features.
    """

    def __init__(self, coefficients, intercept=0.0):
        """
        Parameters
        ----------
        coefficients : list or tensor
            Coefficients for each feature
        intercept : float
            Model intercept
        """
        self.coefficients = torch.tensor(coefficients, dtype=torch.float32)
        self.intercept = intercept

    def predict_proba(self, x, device='cpu'):
        """Return probability predictions."""
        x = x.to(device) if isinstance(x, torch.Tensor) else torch.tensor(x, device=device)
        logits = x @ self.coefficients + self.intercept
        return torch.sigmoid(logits)

    def parameters(self):
        """For device detection."""
        return iter([self.coefficients])


@pytest.mark.unit
class TestPRiSMScalerBinaryExclusion:
    """Test that PRiSMScaler correctly excludes binary columns from scaling."""

    def test_auto_detect_binary_columns(self):
        """Binary columns should be auto-detected and excluded from scaling."""
        # Create data with 2 continuous and 3 binary (one-hot) columns
        np.random.seed(42)
        X = np.column_stack(
            [
                np.random.randn(100),  # continuous 0
                np.random.randn(100) * 2,  # continuous 1
                np.array([1, 0, 0] * 33 + [1]),  # one-hot col 0
                np.array([0, 1, 0] * 33 + [0]),  # one-hot col 1
                np.array([0, 0, 1] * 33 + [0]),  # one-hot col 2
            ]
        )

        scaler = PRiSMScaler(scaler='median_std')
        X_scaled = scaler.fit_transform(X)

        # Binary columns (indices 2, 3, 4) should remain unchanged
        np.testing.assert_array_equal(X_scaled[:, 2], X[:, 2])
        np.testing.assert_array_equal(X_scaled[:, 3], X[:, 3])
        np.testing.assert_array_equal(X_scaled[:, 4], X[:, 4])

        # Continuous columns should be scaled (not equal to original)
        assert not np.allclose(X_scaled[:, 0], X[:, 0])
        assert not np.allclose(X_scaled[:, 1], X[:, 1])

        # Check excluded columns were recorded
        assert set(scaler.get_excluded_columns()) == {2, 3, 4}

    def test_explicit_onehot_columns(self):
        """User-specified one-hot columns should be excluded."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        X[:, 3:5] = np.random.randint(0, 2, (50, 2))  # Make last 2 binary

        scaler = PRiSMScaler(scaler='median_std', onehot_columns=[3, 4])
        X_scaled = scaler.fit_transform(X)

        # Specified columns should be unchanged
        np.testing.assert_array_equal(X_scaled[:, 3], X[:, 3])
        np.testing.assert_array_equal(X_scaled[:, 4], X[:, 4])

    def test_inverse_transform_preserves_binary(self):
        """Inverse transform should preserve binary values."""
        np.random.seed(42)
        X = np.column_stack(
            [
                np.random.randn(50),
                np.random.randint(0, 2, 50),
            ]
        )

        scaler = PRiSMScaler(scaler='median_std')
        X_scaled = scaler.fit_transform(X)
        X_recovered = scaler.inverse_transform(X_scaled)

        # Binary column should round-trip exactly
        np.testing.assert_array_almost_equal(X[:, 1], X_recovered[:, 1], decimal=10)


@pytest.mark.unit
class TestScaledOnehotDetection:
    """Test detection of scaled one-hot columns."""

    def test_warn_on_scaled_onehot(self):
        """Should warn when one-hot columns appear scaled."""
        # Create "scaled" one-hot data (values other than 0/1)
        X_scaled = torch.tensor(
            [
                [0.0, 0.0, 1.5],  # col 2 has non-0/1 value
                [0.0, 1.0, -0.5],  # col 2 has negative value
                [1.0, 0.0, 0.0],
            ]
        )
        onehot_groups = [(0, 1, 2)]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_if_scaled_onehot(X_scaled, onehot_groups)

            # Should have issued warnings (scaled values + two-active constraint)
            assert len(w) >= 1
            warning_messages = [str(warning.message).lower() for warning in w]
            assert any("scaled" in msg for msg in warning_messages)

    def test_warn_on_two_active_categories(self):
        """Should warn when multiple categories are active in a one-hot group."""
        # Create data with two-active constraint violation
        X_two_active = torch.tensor(
            [
                [1.0, 1.0, 0.0],  # row 0: two categories active (violation)
                [0.0, 1.0, 0.0],  # row 1: one category active (ok)
                [0.0, 0.0, 1.0],  # row 2: one category active (ok)
            ]
        )
        onehot_groups = [(0, 1, 2)]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_if_scaled_onehot(X_two_active, onehot_groups)

            # Should have issued a warning about two-active constraint
            assert len(w) == 1
            assert "constraint violated" in str(w[0].message).lower()

    def test_no_warn_on_proper_onehot(self):
        """Should not warn when one-hot columns are proper 0/1."""
        X_proper = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )
        onehot_groups = [(0, 1, 2)]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_if_scaled_onehot(X_proper, onehot_groups)

            # Should NOT have issued a warning
            assert len(w) == 0


@pytest.mark.unit
class TestDiracBaselineConsistency:
    """Test that Dirac baseline is consistent with partial response calculations."""

    def test_baseline_uses_scaled_zeros(self):
        """Baseline should use _create_baseline_input, not raw zeros."""
        # Create model and data
        coefficients = [0.5, 0.3, 0.2, 0.4, -0.1]  # 5 features
        model = MockLinearModelWithOnehot(coefficients)
        feature_names = ['cont_0', 'cont_1', 'oh_a', 'oh_b', 'oh_c']

        # Create data with continuous (0,1) and one-hot (2,3,4)
        np.random.seed(42)
        X = np.column_stack(
            [
                np.random.randn(30),
                np.random.randn(30),
                np.array([1, 0, 0] * 10),
                np.array([0, 1, 0] * 10),
                np.array([0, 0, 1] * 10),
            ]
        )

        # Scale with binary exclusion
        scaler = PRiSMScaler(scaler='median_std')
        X_scaled = scaler.fit_transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        # Create calculator
        calculator = PartialResponseCalculator(
            model=model,
            method='dirac',
            input_dim=5,
            onehot_groups=[(2, 3, 4)],
            feature_names=feature_names,
            scaler=scaler,
        )

        # Calculate baseline
        calculator.calculate_baseline(X_tensor)

        # The baseline should be computed at scaled zeros for continuous features
        # and raw zeros (= reference category) for one-hot features
        x0 = calculator._create_baseline_input(1, 5)

        # One-hot columns should be 0 (reference category)
        assert x0[0, 2].item() == 0.0
        assert x0[0, 3].item() == 0.0
        assert x0[0, 4].item() == 0.0

        # Continuous columns: check they have expected scaled_0 value
        # (will be 0 for columns with median=0, may be non-zero otherwise)
        expected_y0 = model.predict_proba(x0)
        expected_logit_y0 = stable_logit(expected_y0).item()

        # Baseline should match expected
        assert (
            abs(calculator.logit_y0 - expected_logit_y0) < 1e-6
        ), f"Baseline mismatch: {calculator.logit_y0} vs {expected_logit_y0}"


@pytest.mark.integration
class TestDiracReconstructionWithOnehot:
    """Test Dirac reconstruction accuracy with one-hot encoded features."""

    def test_linear_model_reconstruction(self):
        """Linear model with one-hot features should reconstruct exactly."""
        # Linear model: logit = 0.3*x0 + 0.5*x1 + 0.2*oh0 + 0.4*oh1 + 0.1*oh2
        coefficients = [0.3, 0.5, 0.2, 0.4, 0.1]
        model = MockLinearModelWithOnehot(coefficients, intercept=0.5)
        feature_names = ['cont_0', 'cont_1', 'oh_a', 'oh_b', 'oh_c']

        # Create data: 2 continuous + 3 one-hot
        np.random.seed(42)
        n_samples = 50
        X = np.column_stack(
            [
                np.random.randn(n_samples),
                np.random.randn(n_samples),
                np.zeros(n_samples),  # one-hot placeholder
                np.zeros(n_samples),
                np.zeros(n_samples),
            ]
        )

        # Create valid one-hot encoding (exactly one active per row)
        for i in range(n_samples):
            active_col = np.random.randint(0, 3)
            X[i, 2 + active_col] = 1.0

        # Scale (binary columns should be excluded)
        scaler = PRiSMScaler(scaler='median_std')
        X_scaled = scaler.fit_transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        # Create calculator
        calculator = PartialResponseCalculator(
            model=model,
            method='dirac',
            input_dim=5,
            onehot_groups=[(2, 3, 4)],
            feature_names=feature_names,
            scaler=scaler,
        )

        # Calculate partial responses
        univariate, bivariate = calculator.calculate(X_tensor)

        # Get true predictions
        y_pred = model.predict_proba(X_tensor)
        logit_pred = stable_logit(y_pred)

        # Reconstruct
        logit_reconstructed = calculator.logit_y0 + univariate.sum(dim=1) + bivariate.sum(dim=1)

        # Calculate error
        error = (logit_pred - logit_reconstructed).abs()
        max_error = error.max().item()
        mean_error = error.mean().item()

        print("\nLinear Model with One-Hot Reconstruction:")
        print(f"  Max error:  {max_error:.6e}")
        print(f"  Mean error: {mean_error:.6e}")
        print(f"  Baseline:   {calculator.logit_y0:.6f}")

        # For linear model, reconstruction should be exact (within numerical precision)
        assert max_error < 1e-4, (
            f"Linear model with one-hot reconstruction failed! Max error: {max_error:.6e}\n"
            "This indicates a bug in the Dirac baseline or partial response calculation."
        )

    def test_dirac_lebesgue_alignment(self):
        """Dirac and Lebesgue should produce similar results for simple models."""
        coefficients = [0.3, 0.5, 0.2, 0.4, 0.1]
        model = MockLinearModelWithOnehot(coefficients, intercept=0.0)
        feature_names = ['cont_0', 'cont_1', 'oh_a', 'oh_b', 'oh_c']

        # Create data
        np.random.seed(42)
        n_samples = 30
        X = np.column_stack(
            [
                np.random.randn(n_samples) * 0.5,  # smaller variance for stability
                np.random.randn(n_samples) * 0.5,
                np.zeros(n_samples),
                np.zeros(n_samples),
                np.zeros(n_samples),
            ]
        )

        for i in range(n_samples):
            active_col = np.random.randint(0, 3)
            X[i, 2 + active_col] = 1.0

        scaler = PRiSMScaler(scaler='median_std')
        X_scaled = scaler.fit_transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        # Dirac calculator
        calc_dirac = PartialResponseCalculator(
            model=model,
            method='dirac',
            input_dim=5,
            onehot_groups=[(2, 3, 4)],
            feature_names=feature_names,
            scaler=scaler,
        )
        univ_dirac, biv_dirac = calc_dirac.calculate(X_tensor)

        # Lebesgue calculator
        calc_lebesgue = PartialResponseCalculator(
            model=model,
            method='lebesgue',
            input_dim=5,
            x_train=X_tensor,
            onehot_groups=[(2, 3, 4)],
            feature_names=feature_names,
            scaler=scaler,
        )
        univ_lebesgue, biv_lebesgue = calc_lebesgue.calculate(X_tensor)

        # Both should reconstruct the same predictions
        y_pred = model.predict_proba(X_tensor)
        logit_pred = stable_logit(y_pred)

        recon_dirac = calc_dirac.logit_y0 + univ_dirac.sum(dim=1) + biv_dirac.sum(dim=1)
        recon_lebesgue = (
            calc_lebesgue.logit_y0 + univ_lebesgue.sum(dim=1) + biv_lebesgue.sum(dim=1)
        )

        error_dirac = (logit_pred - recon_dirac).abs().max().item()
        error_lebesgue = (logit_pred - recon_lebesgue).abs().max().item()

        print("\nDirac vs Lebesgue Comparison:")
        print(f"  Dirac max error:    {error_dirac:.6e}")
        print(f"  Lebesgue max error: {error_lebesgue:.6e}")
        print(f"  Dirac baseline:     {calc_dirac.logit_y0:.6f}")
        print(f"  Lebesgue baseline:  {calc_lebesgue.logit_y0:.6f}")

        # Both should have small reconstruction error
        assert error_dirac < 1e-4, f"Dirac reconstruction error too large: {error_dirac}"
        assert error_lebesgue < 1e-3, f"Lebesgue reconstruction error too large: {error_lebesgue}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
