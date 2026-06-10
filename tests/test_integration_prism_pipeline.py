"""
Integration tests for the complete PRiSM pipeline.

These tests verify that the major components work together correctly:
- Partial responses → LASSO → PRN training
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
