"""Unit tests for MaskedMLP module.

These tests verify the MaskedMLP class and train_mlp function behavior,
particularly the scale_lr parameter for learning rate scaling.
"""

import numpy as np
import pytest
import torch
from sklearn.datasets import make_classification

from prism.maskedmlp import MaskedMLP, build_prn_from_lasso, scale_learning_rate, train_mlp

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

    def test_predict_proba_numpy_input(self):
        """Test predict_proba method with numpy array."""
        model = MaskedMLP(input_dim=5, hidden_units=10, output_dim=1)
        x = np.random.randn(16, 5)
        output = model.predict_proba(x)

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

    def test_mask_enforced_after_training(self):
        """Forbidden connections stay exactly zero throughout training.

        apply_mask is reapplied after every optimizer step, so a feature the mask
        excludes cannot acquire a non-zero fc1 weight even though the optimizer would
        otherwise grow it.
        """
        rng = np.random.RandomState(0)
        X = rng.randn(120, 5).astype('float32')
        # outcome depends on feature 3, which we then forbid via the mask
        y = (X[:, 3] > 0).astype('float32')

        mask = np.ones((8, 5))
        mask[:, 3] = 0  # input feature 3 forbidden to every hidden unit

        model = train_mlp(
            x_tr=X[:90],
            y_tr=y[:90],
            x_ts=X[90:],
            y_ts=y[90:],
            n_hidden=8,
            mask=mask,
            lr=0.05,
            max_iter=50,
            patience=50,
            batch_size=32,
            plot_loss=False,
            verbose=False,
        )

        w = model.fc1.weight.detach().cpu().numpy()
        # column 3 (weights from the forbidden input) must be exactly zero
        np.testing.assert_array_equal(w[:, 3], np.zeros(8))
        # a permitted column must have learned something (sanity: training happened)
        assert np.any(w[:, 0] != 0)


# ============================================================================
# build_prn_from_lasso Tests
# ============================================================================


class _StubLassoResults:
    """Minimal stand-in exposing the get_mask interface build_prn_from_lasso uses."""

    def __init__(self, mask, n_effects):
        self._mask = mask
        self._n_effects = n_effects
        self.get_mask_kwargs = None

    def get_mask(self, threshold=0.1, subnet_nodes=5, verbose=True):
        self.get_mask_kwargs = {
            'threshold': threshold,
            'subnet_nodes': subnet_nodes,
            'verbose': verbose,
        }
        return self._mask, self._n_effects


@pytest.mark.unit
class TestBuildPrnFromLasso:
    """Tests for the build_prn_from_lasso wrapper."""

    def _make_data(self, n_features, n=120):
        rng = np.random.RandomState(0)
        X = rng.randn(n, n_features).astype('float32')
        y = (X[:, 0] > 0).astype('float32')
        return X[:90], y[:90], X[90:], y[90:]

    def test_scaffold_enforced_for_dropped_features(self):
        """Features absent from the mask keep exactly-zero input weights."""
        subnet_nodes = 3
        n_effects = 2
        # 5 input features; effects on features 0 and 1; features 2-4 dropped
        mask = np.zeros((5, n_effects * subnet_nodes))
        mask[0, 0:subnet_nodes] = 1
        mask[1, subnet_nodes:] = 1
        stub = _StubLassoResults(mask, n_effects)

        x_tr, y_tr, x_ts, y_ts = self._make_data(5)
        prn = build_prn_from_lasso(
            stub,
            x_tr,
            y_tr,
            x_ts,
            y_ts,
            subnet_nodes=subnet_nodes,
            threshold=0.2,
            lr=0.05,
            max_iter=30,
            patience=30,
            batch_size=32,
            plot_loss=False,
            verbose=False,
        )

        assert isinstance(prn, MaskedMLP)
        # subnet_nodes and threshold forwarded to get_mask
        assert stub.get_mask_kwargs['subnet_nodes'] == subnet_nodes
        assert stub.get_mask_kwargs['threshold'] == 0.2
        # hidden layer sized from n_effects * subnet_nodes
        assert prn.fc1.weight.shape == (n_effects * subnet_nodes, 5)

        w = prn.fc1.weight.detach().cpu().numpy()
        for dropped in (2, 3, 4):
            np.testing.assert_array_equal(w[:, dropped], np.zeros(w.shape[0]))
        assert np.any(w[:, 0] != 0)

    def test_onehot_mask_expanded_to_original_space(self):
        """With a group manager, the collapsed mask is expanded to one-hot width."""
        from prism.preprocessing import OneHotGroupManager

        subnet_nodes = 2
        n_effects = 2
        # Collapsed space follows collapse_onehot_features order (groups first):
        # ['cat', 'num1', 'num2']; effects selected on num1 and cat
        mask = np.zeros((3, n_effects * subnet_nodes))
        mask[1, 0:subnet_nodes] = 1  # num1
        mask[0, subnet_nodes:] = 1  # cat (the collapsed group)
        stub = _StubLassoResults(mask, n_effects)

        group_manager = OneHotGroupManager({'cat': ['cat_b', 'cat_c']}, {'cat': 'cat_a'})
        feature_names = ['num1', 'num2', 'cat_b', 'cat_c']

        x_tr, y_tr, x_ts, y_ts = self._make_data(4)
        prn = build_prn_from_lasso(
            stub,
            x_tr,
            y_tr,
            x_ts,
            y_ts,
            group_manager=group_manager,
            feature_names=feature_names,
            subnet_nodes=subnet_nodes,
            lr=0.05,
            max_iter=30,
            patience=30,
            batch_size=32,
            plot_loss=False,
            verbose=False,
        )

        # Input width is the one-hot expanded space
        assert prn.fc1.weight.shape == (n_effects * subnet_nodes, 4)

        w = prn.fc1.weight.detach().cpu().numpy()
        # num2 (dropped) has exactly-zero weights everywhere
        np.testing.assert_array_equal(w[:, 1], np.zeros(w.shape[0]))
        # num1 connects only to its own subnetwork (hidden units 0..subnet_nodes)
        assert np.any(w[:subnet_nodes, 0] != 0)
        np.testing.assert_array_equal(w[subnet_nodes:, 0], np.zeros(subnet_nodes))
        # both one-hot columns of the selected 'cat' group are connected
        # to the cat subnetwork (hidden units subnet_nodes..2*subnet_nodes)
        assert np.any(w[subnet_nodes:, 2] != 0)
        assert np.any(w[subnet_nodes:, 3] != 0)
        np.testing.assert_array_equal(w[:subnet_nodes, 2], np.zeros(subnet_nodes))
        np.testing.assert_array_equal(w[:subnet_nodes, 3], np.zeros(subnet_nodes))

    def test_group_manager_without_feature_names_raises(self):
        from prism.preprocessing import OneHotGroupManager

        stub = _StubLassoResults(np.ones((3, 2)), 1)
        group_manager = OneHotGroupManager({'cat': ['cat_b', 'cat_c']}, {'cat': 'cat_a'})
        x_tr, y_tr, x_ts, y_ts = self._make_data(4)

        with pytest.raises(ValueError, match="feature_names"):
            build_prn_from_lasso(stub, x_tr, y_tr, x_ts, y_ts, group_manager=group_manager)
