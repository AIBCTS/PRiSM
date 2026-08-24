"""Tests for LassoResultsManager.verify_prediction.

Regression coverage for numpy 2 compatibility: passing 2-D (n_samples, 1)
original_model_pred used to reach float() on a shape-(1,) array, which is a
TypeError under numpy 2 (deprecated since numpy 1.25).
"""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from prism.lasso.lasso_results import LassoResultsManager


@pytest.fixture
def fitted_manager():
    """LassoResultsManager with 2 univariate features and fitted models."""
    rng = np.random.default_rng(257)
    n_samples = 40
    n_univ = 2
    n_features = n_univ + 1  # one bivariate pair: (0, 1)

    phi = rng.normal(size=(n_samples, n_features))
    y = (phi @ np.array([1.0, -1.0, 0.5]) + rng.normal(scale=0.1, size=n_samples) > 0).astype(int)

    n_lambda = 2
    models = []
    betas = np.zeros((n_features, n_lambda))
    for i in range(n_lambda):
        model = LogisticRegression().fit(phi, y)
        models.append(model)
        betas[:, i] = model.coef_[0]

    manager = LassoResultsManager(
        lambdas=np.array([0.1, 1.0]),
        betas=betas,
        models=models,
        feature_names=[f'feature_{i}' for i in range(n_univ)],
        train_losses=np.array([0.1, 0.2]),
        test_losses=np.array([0.15, 0.25]),
        train_aucs=np.array([0.9, 0.85]),
        test_aucs=np.array([0.85, 0.8]),
    )
    manager.selected_lambda_index = 0
    return manager, phi, y


@pytest.mark.unit
class TestVerifyPrediction:
    def test_original_pred_2d_column(self, fitted_manager):
        """Shape (n_samples, 1) predictions must not raise on scalar conversion."""
        manager, phi, y = fitted_manager
        preds = manager.models[0].predict_proba(phi)[:, 1].reshape(-1, 1)

        results = manager.verify_prediction(phi, y, sample_idx=3, original_model_pred=preds)

        original = results['calculations']['original']
        assert isinstance(float(original['probability']), float)
        assert original['probability'] == pytest.approx(preds[3, 0])

    def test_original_pred_1d(self, fitted_manager):
        """Shape (n_samples,) predictions give the same result as the 2-D column."""
        manager, phi, y = fitted_manager
        preds = manager.models[0].predict_proba(phi)[:, 1]

        results = manager.verify_prediction(phi, y, sample_idx=3, original_model_pred=preds)

        original = results['calculations']['original']
        assert original['probability'] == pytest.approx(preds[3])

    def test_without_original_pred(self, fitted_manager):
        """LASSO and manual paths work without original model predictions."""
        manager, phi, y = fitted_manager

        results = manager.verify_prediction(phi, y, sample_idx=3)

        assert 'original' not in results['calculations']
        assert results['verification']['lasso_match']
