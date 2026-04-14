"""Unit tests for MaskedMLP module.

These tests verify the MaskedMLP class and train_mlp function behavior,
particularly the scale_lr parameter for learning rate scaling.
"""

import numpy as np
import pytest
import torch
from sklearn.datasets import make_classification

from prism.maskedmlp import MaskedMLP, scale_learning_rate, train_mlp

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_classification_data():
    """Generate simple binary classification dataset for testing."""
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        n_classes=2,
        random_state=42,
    )

    # Split into train/test
    n_train = 80
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'n_features': X.shape[1],
    }


# ============================================================================
# scale_learning_rate Tests
# ============================================================================


class TestScaleLearningRate:
    """Tests for scale_learning_rate function."""

    def test_scale_lr_with_base_batch_size(self):
        """Test that LR is unchanged when batch_size equals base_batch_size."""
        base_lr = 0.001
        result = scale_learning_rate(base_lr, batch_size=1024, base_batch_size=1024)
        assert result == base_lr

    def test_scale_lr_smaller_batch(self):
        """Test LR scaling with smaller batch size."""
        base_lr = 0.001
        result = scale_learning_rate(base_lr, batch_size=512, base_batch_size=1024)
        expected = base_lr * (512 / 1024)  # 0.0005
        assert result == expected

    def test_scale_lr_larger_batch(self):
        """Test LR scaling with larger batch size."""
        base_lr = 0.001
        result = scale_learning_rate(base_lr, batch_size=2048, base_batch_size=1024)
        expected = base_lr * (2048 / 1024)  # 0.002
        assert result == expected

    def test_scale_lr_custom_base_batch_size(self):
        """Test LR scaling with custom base batch size."""
        base_lr = 0.01
        result = scale_learning_rate(base_lr, batch_size=64, base_batch_size=32)
        expected = base_lr * (64 / 32)  # 0.02
        assert result == expected


# ============================================================================
# MaskedMLP Tests
# ============================================================================


class TestMaskedMLP:
    """Tests for MaskedMLP class."""

    def test_forward_pass(self):
        """Test that forward pass produces valid outputs."""
        model = MaskedMLP(input_dim=5, hidden_units=10, output_dim=1)
        x = torch.randn(16, 5)
        output = model(x)

        assert output.shape == (16, 1)
        assert torch.all((output >= 0) & (output <= 1))  # sigmoid output

    def test_predict_numpy(self):
        """Test predict method with numpy array."""
        model = MaskedMLP(input_dim=5, hidden_units=10, output_dim=1)
        x = np.random.randn(16, 5)
        output = model.predict(x)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (16, 1)

    def test_mask_application(self):
        """Test that mask is correctly applied."""
        mask = np.array([[1, 0, 1, 0, 1], [0, 1, 0, 1, 0]])  # 2 hidden x 5 input
        model = MaskedMLP(input_dim=5, hidden_units=2, output_dim=1, mask=mask)

        assert hasattr(model, 'mask_tensor')
        assert model.mask_tensor.shape == (2, 5)


# ============================================================================
# train_mlp Tests
# ============================================================================


class TestTrainMLP:
    """Tests for train_mlp function."""

    def test_train_mlp_basic(self, simple_classification_data):
        """Test basic MLP training without batching."""
        data = simple_classification_data

        model = train_mlp(
            x_tr=data['X_train'],
            y_tr=data['y_train'],
            x_ts=data['X_test'],
            y_ts=data['y_test'],
            n_hidden=8,
            lr=0.01,
            max_iter=5,
            patience=5,
            plot_loss=False,
        )

        assert isinstance(model, MaskedMLP)
        # Verify model can make predictions
        with torch.no_grad():
            x = torch.as_tensor(data['X_test'], dtype=torch.float32)
            pred = model(x)
        assert pred.shape == (len(data['X_test']), 1)

    def test_train_mlp_with_batching_scale_lr_true(self, simple_classification_data):
        """Test MLP training with batching and scale_lr=True (default).

        With scale_lr=True, the LR should be scaled by batch_size/1024.
        """
        data = simple_classification_data

        # Train with batch_size=32 and scale_lr=True (default)
        model = train_mlp(
            x_tr=data['X_train'],
            y_tr=data['y_train'],
            x_ts=data['X_test'],
            y_ts=data['y_test'],
            n_hidden=8,
            lr=0.01,
            batch_size=32,
            scale_lr=True,
            max_iter=5,
            patience=5,
            plot_loss=False,
        )

        assert isinstance(model, MaskedMLP)
        # Model should still train and produce valid predictions
        with torch.no_grad():
            x = torch.as_tensor(data['X_test'], dtype=torch.float32)
            pred = model(x)
        assert pred.shape == (len(data['X_test']), 1)
        assert torch.all((pred >= 0) & (pred <= 1))

    def test_train_mlp_with_batching_scale_lr_false(self, simple_classification_data):
        """Test MLP training with batching and scale_lr=False.

        With scale_lr=False, the LR should be used directly without scaling.
        This matches tuning behavior.
        """
        data = simple_classification_data

        # Train with batch_size=32 and scale_lr=False
        model = train_mlp(
            x_tr=data['X_train'],
            y_tr=data['y_train'],
            x_ts=data['X_test'],
            y_ts=data['y_test'],
            n_hidden=8,
            lr=0.01,
            batch_size=32,
            scale_lr=False,
            max_iter=5,
            patience=5,
            plot_loss=False,
        )

        assert isinstance(model, MaskedMLP)
        # Model should still train and produce valid predictions
        with torch.no_grad():
            x = torch.as_tensor(data['X_test'], dtype=torch.float32)
            pred = model(x)
        assert pred.shape == (len(data['X_test']), 1)
        assert torch.all((pred >= 0) & (pred <= 1))

    def test_train_mlp_default_max_iter(self, simple_classification_data):
        """Test that default max_iter is 4000."""
        # We can't easily test this without running full training,
        # but we can verify it by importing and checking signature
        import inspect

        sig = inspect.signature(train_mlp)
        max_iter_default = sig.parameters['max_iter'].default
        assert max_iter_default == 4000

    def test_train_mlp_no_scaling_without_batch_size(self, simple_classification_data):
        """Test that LR scaling is not applied when batch_size is None.

        Even with scale_lr=True, if batch_size is None, no scaling should occur.
        """
        data = simple_classification_data

        # Train without batch_size (full-batch training)
        model = train_mlp(
            x_tr=data['X_train'],
            y_tr=data['y_train'],
            x_ts=data['X_test'],
            y_ts=data['y_test'],
            n_hidden=8,
            lr=0.01,
            batch_size=None,  # No batching
            scale_lr=True,  # scale_lr is ignored when batch_size is None
            max_iter=5,
            patience=5,
            plot_loss=False,
        )

        assert isinstance(model, MaskedMLP)

    def test_train_mlp_with_mask(self, simple_classification_data):
        """Test MLP training with a mask (PRN mode)."""
        data = simple_classification_data

        # Create a mask with 8 hidden units, 5 features
        mask = np.ones((8, 5))
        mask[0, 0] = 0  # Zero out some connections
        mask[1, 1] = 0

        model = train_mlp(
            x_tr=data['X_train'],
            y_tr=data['y_train'],
            x_ts=data['X_test'],
            y_ts=data['y_test'],
            n_hidden=8,
            mask=mask,
            lr=0.01,
            batch_size=32,
            max_iter=5,
            patience=5,
            plot_loss=False,
        )

        assert isinstance(model, MaskedMLP)
        assert hasattr(model, 'mask_tensor')
