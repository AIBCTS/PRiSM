"""
Metric computation utilities for LASSO regression.

This module provides functions for computing various metrics used in LASSO regression.
"""

from typing import Tuple

import numpy as np
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score


def compute_metrics(
    model, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray
) -> Tuple[dict, dict]:
    """
    Compute comprehensive metrics for train and test sets.

    Parameters
    ----------
    model : sklearn.linear_model.LogisticRegression
        Fitted LASSO model
    X_train : np.ndarray
        Training features
    X_test : np.ndarray
        Test features
    y_train : np.ndarray
        Training labels
    y_test : np.ndarray
        Test labels

    Returns
    -------
    Tuple[dict, dict]
        Dictionaries containing train and test metrics respectively
    """
    # Get predictions
    train_pred = model.predict_proba(X_train)[:, 1]
    test_pred = model.predict_proba(X_test)[:, 1]

    # Clip predictions for numerical stability
    eps = 1e-15
    train_pred = np.clip(train_pred, eps, 1 - eps)
    test_pred = np.clip(test_pred, eps, 1 - eps)

    # Compute metrics
    train_metrics = {
        "loss": log_loss(y_train, train_pred),
        "auc": roc_auc_score(y_train, train_pred),
        "accuracy": accuracy_score(y_train, model.predict(X_train)),
    }

    test_metrics = {
        "loss": log_loss(y_test, test_pred),
        "auc": roc_auc_score(y_test, test_pred),
        "accuracy": accuracy_score(y_test, model.predict(X_test)),
    }

    return train_metrics, test_metrics
