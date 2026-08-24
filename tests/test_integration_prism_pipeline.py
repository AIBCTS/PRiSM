"""Integration tests for the complete PRiSM pipeline.

These tests verify that the major components work together correctly:
- Partial responses -> LASSO -> PRN training
"""

import numpy as np
import pytest
import torch
from sklearn.datasets import make_classification

from prism.lasso import LassoComputation, LassoConfig
from prism.maskedmlp import train_mlp
from prism.partial_responses import partial_responses
from tests.conftest import MockBinaryClassifier


@pytest.fixture
def synthetic_classification_data():
    """Generate synthetic binary classification dataset for testing."""
    X, y = make_classification(
        n_samples=200,
        n_features=8,
        n_informative=6,
        n_redundant=2,
        n_classes=2,
        random_state=42,
        flip_y=0.1,
    )

    # Split into train/test
    n_train = 150
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    # Convert to tensors
    X_train_tensor = torch.from_numpy(X_train).float()
    X_test_tensor = torch.from_numpy(X_test).float()
    y_train_tensor = torch.from_numpy(y_train).float()
    y_test_tensor = torch.from_numpy(y_test).float()

    return {
        'X_train': X_train_tensor,
        'X_test': X_test_tensor,
        'y_train': y_train_tensor,
        'y_test': y_test_tensor,
        'n_features': X.shape[1],
    }


@pytest.mark.integration
class TestLASSOwithPartialResponses:
    """Integration tests for LASSO with partial responses."""

    def test_lasso_fit_with_partial_responses(self, synthetic_classification_data):
        """Test LASSO fitting with actual partial response data."""
        data = synthetic_classification_data

        # Use mock model
        model = MockBinaryClassifier(n_features=data['n_features'])

        # Calculate partial responses
        responses_train = partial_responses(
            x=data['X_train'],
            model=model,
            method='dirac',
            device='cpu',
        )

        responses_test = partial_responses(
            x=data['X_test'],
            model=model,
            method='dirac',
            device='cpu',
        )

        # Fit LASSO
        config = LassoConfig(
            max_lambda=10.0,
            min_lambda=0.001,
            nlambda=20,
            batch_size=10,
            max_workers=1,
            real_time_plot=False,
        )

        lasso = LassoComputation(config)
        results, _ = lasso.fit(
            partial_responses_train=responses_train,
            partial_responses_test=responses_test,
            y_train=data['y_train'].numpy(),
            y_test=data['y_test'].numpy(),
        )

        # Verify results structure
        assert results is not None
        assert hasattr(results, 'lambdas')
        assert hasattr(results, 'betas')
        assert len(results.lambdas) == 20
        assert results.betas.shape[1] == 20  # nlambda

        # Verify we can extract features
        results.select_lambda(lambda_index=10)
        selected_features = results.get_selected_feature_indicies()
        assert isinstance(selected_features, np.ndarray)
        assert all(isinstance(int(f), int) for f in selected_features)

    def test_lasso_feature_selection_consistency(self, synthetic_classification_data):
        """Test that LASSO feature selection reduces features progressively."""
        data = synthetic_classification_data

        # Use mock model
        model = MockBinaryClassifier(n_features=data['n_features'])

        # Calculate partial responses
        responses_train = partial_responses(
            x=data['X_train'],
            model=model,
            method='dirac',
            device='cpu',
        )

        responses_test = partial_responses(
            x=data['X_test'],
            model=model,
            method='dirac',
            device='cpu',
        )

        n_total_features = responses_train.shape[1]

        # Fit LASSO with different regularization strengths
        config = LassoConfig(
            max_lambda=10.0,
            min_lambda=0.01,
            nlambda=15,
            batch_size=15,
            max_workers=1,
            real_time_plot=False,
        )

        lasso = LassoComputation(config)
        results, _ = lasso.fit(
            partial_responses_train=responses_train,
            partial_responses_test=responses_test,
            y_train=data['y_train'].numpy(),
            y_test=data['y_test'].numpy(),
        )

        # Check that stronger regularization (higher lambda) selects fewer features
        results.select_lambda(lambda_index=0)  # max lambda
        features_at_high_lambda = results.get_selected_feature_indicies()

        results.select_lambda(lambda_index=len(results.lambdas) - 1)  # min lambda
        features_at_low_lambda = results.get_selected_feature_indicies()

        assert len(features_at_high_lambda) <= len(features_at_low_lambda)
        assert len(features_at_low_lambda) < n_total_features  # Some features should be selected


@pytest.mark.integration
class TestMaskedMLPIntegration:
    """Integration tests for MaskedMLP training with LASSO-selected features."""

    def test_train_mlp_with_lasso_features(self, synthetic_classification_data):
        """Test training MaskedMLP (PRN) with LASSO-selected features."""
        data = synthetic_classification_data

        # Get partial responses
        model = MockBinaryClassifier(n_features=data['n_features'])

        responses_train = partial_responses(
            x=data['X_train'],
            model=model,
            method='dirac',
            device='cpu',
        )

        responses_test = partial_responses(
            x=data['X_test'],
            model=model,
            method='dirac',
            device='cpu',
        )

        # Run LASSO
        config = LassoConfig(
            max_lambda=5.0,
            min_lambda=0.01,
            nlambda=15,
            batch_size=15,
            max_workers=1,
            real_time_plot=False,
        )

        lasso = LassoComputation(config)
        results, _ = lasso.fit(
            partial_responses_train=responses_train,
            partial_responses_test=responses_test,
            y_train=data['y_train'].numpy(),
            y_test=data['y_test'].numpy(),
        )

        # Select features
        results.select_lambda(lambda_index=10)
        selected_features = results.get_selected_feature_indicies()

        # Create masked dataset
        X_train_masked = responses_train[:, selected_features]
        X_test_masked = responses_test[:, selected_features]

        # Train MaskedMLP (PRN) - train_mlp creates the model internally
        trained_model = train_mlp(
            x_tr=X_train_masked,
            y_tr=data['y_train'].numpy(),
            x_ts=X_test_masked,
            y_ts=data['y_test'].numpy(),
            n_hidden=16,
            mask=None,
            lr=0.01,
            max_iter=10,
            patience=10,
            batch_size=32,
            plot_loss=False,
        )

        # Verify model can make predictions
        with torch.no_grad():
            X_test_tensor = torch.as_tensor(X_test_masked, dtype=torch.float32)
            predictions = trained_model(X_test_tensor).squeeze()

        assert predictions.shape == (len(X_test_masked),)
        assert torch.all((predictions >= 0) & (predictions <= 1))


@pytest.mark.integration
class TestEndToEndPRiSMPipeline:
    """End-to-end integration tests for complete PRiSM workflow."""

    def test_complete_prism_pipeline(self, synthetic_classification_data):
        """Test complete PRiSM pipeline from partial responses to PRN."""
        data = synthetic_classification_data

        # Step 1: Get base model
        base_model = MockBinaryClassifier(n_features=data['n_features'])

        # Verify base model works
        base_pred = base_model.predict_proba(data['X_test'][:5], device='cpu')
        assert base_pred.shape == (5,)

        # Step 2: Calculate partial responses
        responses_train = partial_responses(
            x=data['X_train'],
            model=base_model,
            method='dirac',
            device='cpu',
        )

        responses_test = partial_responses(
            x=data['X_test'],
            model=base_model,
            method='dirac',
            device='cpu',
        )

        n_total_features = responses_train.shape[1]
        expected_features = (
            data['n_features'] + (data['n_features'] * (data['n_features'] - 1)) // 2
        )
        assert n_total_features == expected_features

        # Step 3: LASSO feature selection
        config = LassoConfig(
            max_lambda=5.0,
            min_lambda=0.01,
            nlambda=15,
            batch_size=15,
            max_workers=1,
            real_time_plot=False,  # Disable plotting for tests
        )

        lasso = LassoComputation(config)
        results, _ = lasso.fit(
            partial_responses_train=responses_train,
            partial_responses_test=responses_test,
            y_train=data['y_train'].numpy(),
            y_test=data['y_test'].numpy(),
        )

        # Select lambda with best test AUC
        results.select_lambda_max_test_auc()
        selected_features = results.get_selected_feature_indicies()

        assert len(selected_features) > 0
        assert len(selected_features) < n_total_features

        # Step 4: Train PRN with selected features
        X_train_masked = responses_train[:, selected_features]
        X_test_masked = responses_test[:, selected_features]

        trained_prn = train_mlp(
            x_tr=X_train_masked,
            y_tr=data['y_train'].numpy(),
            x_ts=X_test_masked,
            y_ts=data['y_test'].numpy(),
            n_hidden=16,
            mask=None,
            lr=0.01,
            max_iter=15,
            patience=10,
            batch_size=32,
            plot_loss=False,
        )

        # Verify PRN can make predictions
        X_test_sample = torch.as_tensor(X_test_masked[:5], dtype=torch.float32)
        prn_pred = trained_prn(X_test_sample).squeeze()
        assert prn_pred.shape == (5,)
        assert torch.all((prn_pred >= 0) & (prn_pred <= 1))

    def test_dirac_vs_lebesgue_pipeline(self, synthetic_classification_data):
        """Test that both Dirac and Lebesgue methods work in pipeline."""
        data = synthetic_classification_data

        # Use smaller subset for speed
        X_train_small = data['X_train'][:80]
        X_test_small = data['X_test'][:20]

        # Base model
        base_model = MockBinaryClassifier(n_features=data['n_features'])

        # Calculate partial responses with both methods
        responses_dirac = partial_responses(
            x=X_test_small,
            model=base_model,
            method='dirac',
            device='cpu',
        )

        responses_lebesgue = partial_responses(
            x=X_test_small,
            model=base_model,
            x_train=X_train_small,
            method='lebesgue',
            device='cpu',
            batch_size=32,
        )

        # Both should produce same shape
        assert responses_dirac.shape == responses_lebesgue.shape

        # Results should differ (Lebesgue considers training distribution)
        assert not torch.allclose(responses_dirac, responses_lebesgue, rtol=0.1)


class _LinearLogitModel:
    """Source model with logit = w . x (no interactions), used to control exactly
    which features the model exposes a (non-flat) partial response for."""

    def __init__(self, weights):
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def predict_proba(self, x, device=None):
        x = x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float32)
        return torch.sigmoid(x @ self.weights)

    def __call__(self, x):
        return self.predict_proba(x)


@pytest.mark.integration
class TestLassoOutcomeSupervision:
    """LASSO is supervised by the observed outcome, on partial-response features
    whose shapes are fixed by the source model.

    Consequence: selection can only choose among the signal the model *exposed*, and
    among that, it follows the *outcome*, not the source model's own importance.
    """

    def _decompose(self, model, x):
        return partial_responses(x=x, model=model, method='dirac', device='cpu')

    def test_selection_follows_outcome_not_model_importance(self):
        """Option 1: the model exposes both x0 (dominant) and x1, but the outcome is
        aligned with x1 alone. Selection must keep x1 and drop x0, proving beta
        tracks the outcome, not the model's internal weighting."""
        torch.manual_seed(0)
        n = 400
        x = torch.randn(n, 4)
        # Model uses x0 twice as strongly as x1; x2, x3 unused.
        model = _LinearLogitModel([2.0, 1.0, 0.0, 0.0])
        # Outcome depends ONLY on x1 (independent of x0).
        y = (x[:, 1] > 0).numpy().astype(float)

        pr = self._decompose(model, x)
        pr_tr, pr_te = pr[:300], pr[300:]
        y_tr, y_te = y[:300], y[300:]

        cfg = LassoConfig(
            max_lambda=5.0,
            min_lambda=1e-3,
            nlambda=25,
            batch_size=64,
            max_workers=1,
            real_time_plot=False,
        )
        results, _ = LassoComputation(cfg).fit(
            partial_responses_train=pr_tr,
            partial_responses_test=pr_te,
            y_train=y_tr,
            y_test=y_te,
        )
        results.select_lambda_max_test_auc()
        selected_univ = set(results.get_selected_univariate_indices())

        assert 1 in selected_univ, (
            "outcome-aligned feature x1 was not selected -- selection is not "
            "supervised by the outcome"
        )
        assert 0 not in selected_univ, (
            "model-dominant but outcome-irrelevant feature x0 was selected -- "
            "selection followed model importance instead of the outcome"
        )

    def test_selection_cannot_recover_signal_the_model_discarded(self):
        """Option 2 (limitation/boundary): the model ignores x1 entirely, so x1's
        partial response is flat. Even though x1 perfectly predicts the outcome, it
        cannot be selected; PRiSM translates *this* model and cannot conjure signal
        the model never learned."""
        torch.manual_seed(0)
        n = 400
        x = torch.randn(n, 4)
        # Model uses ONLY x0.
        model = _LinearLogitModel([2.0, 0.0, 0.0, 0.0])
        # Outcome depends ONLY on x1, which the model ignores.
        y = (x[:, 1] > 0).numpy().astype(float)

        pr = self._decompose(model, x)

        # x1's partial response (column 1) is flat: the model output is independent of x1.
        assert pr[:, 1].std().item() < 1e-6, (
            "model that ignores x1 produced a non-flat partial response for it -- "
            "the decomposition may be leaking the raw feature/outcome"
        )

        # Across the entire lambda path, x1's coefficient never clears the 0.1 filter,
        # so x1 can never be selected despite predicting the outcome.
        cfg = LassoConfig(
            max_lambda=5.0,
            min_lambda=1e-3,
            nlambda=25,
            batch_size=64,
            max_workers=1,
            real_time_plot=False,
        )
        results, _ = LassoComputation(cfg).fit(
            partial_responses_train=pr[:300],
            partial_responses_test=pr[300:],
            y_train=y[:300],
            y_test=y[300:],
        )
        assert np.all(np.abs(results.betas[1, :]) <= 0.1), (
            "a feature with a flat partial response was assigned a selectable "
            "coefficient -- selection should be unable to recover discarded signal"
        )


@pytest.mark.integration
class TestPhaseTwoRefinement:
    """Phase II preserves the Phase-I scaffold and relearns shapes.

    The full loop: Phase-I decomposition + selection -> build the PRN mask from that
    selection -> train the masked PRN on the RAW features against the outcome ->
    re-decompose the trained PRN -> a second LASSO produces the refined nomogram.
    Features dropped in Phase I cannot re-enter, because their input connections are
    masked to zero by construction.
    """

    def test_phase2_loop_scaffold_and_redecomposition(self):
        torch.manual_seed(0)
        n, d = 300, 5
        x = torch.randn(n, d)
        # Source model uses only x0 and x1; x2, x3, x4 are irrelevant.
        model = _LinearLogitModel([1.0, 0.7, 0.0, 0.0, 0.0])
        y = ((x[:, 0] + 0.7 * x[:, 1]) > 0).numpy().astype(float)

        x_tr, x_te = x[:220], x[220:]
        y_tr, y_te = y[:220], y[220:]

        cfg = LassoConfig(
            max_lambda=5.0,
            min_lambda=1e-3,
            nlambda=20,
            batch_size=64,
            max_workers=1,
            real_time_plot=False,
        )

        # --- Phase I: decompose source model + select ---
        pr1_tr = partial_responses(x=x_tr, model=model, method='dirac', device='cpu')
        pr1_te = partial_responses(x=x_te, model=model, method='dirac', device='cpu')
        res1, _ = LassoComputation(cfg).fit(
            partial_responses_train=pr1_tr,
            partial_responses_test=pr1_te,
            y_train=y_tr,
            y_test=y_te,
        )
        res1.select_lambda_max_test_auc()

        # --- Phase II.1: build the scaffold and train the masked PRN on RAW features ---
        subnet_nodes = 3
        mask, n_effects = res1.get_mask(subnet_nodes=subnet_nodes, verbose=False)
        assert mask.shape[0] == d  # one input row per raw feature

        # Irrelevant features should not be in the scaffold at all.
        dropped = [i for i in range(d) if not mask[i].any()]
        assert dropped, "expected Phase I to drop the irrelevant features (x2..x4)"

        prn = train_mlp(
            x_tr=x_tr.numpy(),
            y_tr=y_tr,
            x_ts=x_te.numpy(),
            y_ts=y_te,
            n_hidden=n_effects,
            subnet_nodes=subnet_nodes,
            mask=mask.T,
            lr=0.05,
            max_iter=60,
            patience=60,
            batch_size=32,
            plot_loss=False,
            verbose=False,
        )

        # Scaffold is enforced: dropped inputs have exactly-zero fc1 columns, so no
        # higher-order interaction through them can re-enter.
        w = prn.fc1.weight.detach().cpu().numpy()  # (hidden_units, input_dim)
        for i in dropped:
            np.testing.assert_array_equal(w[:, i], np.zeros(w.shape[0]))

        # --- Phase II.2: re-decompose the trained PRN (the previously missing step) ---
        pr2_tr = partial_responses(x=x_tr, model=prn, method='dirac', device='cpu')
        pr2_te = partial_responses(x=x_te, model=prn, method='dirac', device='cpu')
        assert pr2_tr.shape == pr1_tr.shape  # same decomposition layout

        # The PRN ignores the dropped inputs, so their re-decomposed univariate
        # responses are flat.
        for i in dropped:
            assert pr2_tr[:, i].std().item() < 1e-5

        # --- Phase II.3: second LASSO on the PRN's own partial responses ---
        res2, _ = LassoComputation(cfg).fit(
            partial_responses_train=pr2_tr,
            partial_responses_test=pr2_te,
            y_train=y_tr,
            y_test=y_te,
        )
        res2.select_lambda_max_test_auc()
        assert len(res2.get_selected_feature_indicies()) > 0
