"""Tests for the predict_proba API and the deprecated predict aliases.

All PRiSM model classes (MaskedMLP, LogisticRegression, IMPACTModel,
SklearnWrapper) expose predict_proba returning P(y=1) as a torch tensor.
The old predict name remains as a deprecated alias that emits
PrismDeprecationWarning and returns identical values.
"""

import io
import pickle

import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression as SkLogisticRegression

from prism._deprecation import PrismDeprecationWarning
from prism.impact import IMPACTModel
from prism.logreg import LogisticRegression
from prism.maskedmlp import MaskedMLP
from prism.wrapper import SklearnWrapper


@pytest.fixture
def mlp_model():
    torch.manual_seed(0)
    return MaskedMLP(input_dim=5, hidden_units=8, output_dim=1)


@pytest.fixture
def logreg_model():
    return LogisticRegression(input_features=5, random_seed=0)


@pytest.fixture
def wrapped_sklearn():
    X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=np.float64)
    y = np.array([0, 0, 1, 1])
    inner = SkLogisticRegression()
    inner.fit(X, y)
    return SklearnWrapper(inner)


class TestPredictProba:
    """predict_proba matches the forward pass / inner model exactly."""

    def test_maskedmlp_matches_forward(self, mlp_model):
        x = torch.randn(16, 5)
        with torch.no_grad():
            expected = mlp_model(x)
        result = mlp_model.predict_proba(x)
        assert torch.equal(result, expected)
        assert result.shape == (16, 1)

    def test_logreg_matches_forward(self, logreg_model):
        x = torch.randn(16, 5)
        logreg_model.eval()
        with torch.no_grad():
            expected = torch.sigmoid(logreg_model.forward(x))
        result = logreg_model.predict_proba(x)
        assert torch.equal(result, expected)

    def test_impact_matches_forward(self):
        model = IMPACTModel()
        x = torch.rand(8, 18)
        with torch.no_grad():
            expected = model.forward(x)['mortality_prob_logit']
        result = model.predict_proba(x)
        assert torch.equal(result, expected)

    def test_wrapper_matches_inner_predict_proba(self, wrapped_sklearn):
        X = np.array([[0.5, 0.5], [2.5, 2.5]])
        expected = wrapped_sklearn.model.predict_proba(X)[:, 1]
        result = wrapped_sklearn.predict_proba(X)
        assert isinstance(result, torch.Tensor)
        np.testing.assert_array_equal(result.numpy(), expected.astype(np.float32))
        assert result.shape == (2,)

    def test_maskedmlp_predict_proba_numpy(self, mlp_model):
        x = np.random.randn(16, 5)
        result = mlp_model.predict_proba_numpy(x)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, mlp_model.predict_proba(x).cpu().numpy())


class TestDeprecatedPredictAliases:
    """predict warns with PrismDeprecationWarning and returns identical values."""

    def test_maskedmlp_predict(self, mlp_model):
        x = torch.randn(16, 5)
        expected = mlp_model.predict_proba(x)
        with pytest.warns(PrismDeprecationWarning, match="MaskedMLP.predict is deprecated"):
            result = mlp_model.predict(x)
        assert torch.equal(result, expected)

    def test_maskedmlp_predict_numpy(self, mlp_model):
        x = np.random.randn(16, 5)
        expected = mlp_model.predict_proba_numpy(x)
        with pytest.warns(PrismDeprecationWarning, match="MaskedMLP.predict_numpy"):
            result = mlp_model.predict_numpy(x)
        np.testing.assert_array_equal(result, expected)

    def test_logreg_predict(self, logreg_model):
        x = torch.randn(16, 5)
        expected = logreg_model.predict_proba(x)
        with pytest.warns(PrismDeprecationWarning, match="LogisticRegression.predict"):
            result = logreg_model.predict(x)
        assert torch.equal(result, expected)

    def test_impact_predict(self):
        model = IMPACTModel()
        x = torch.rand(8, 18)
        expected = model.predict_proba(x)
        with pytest.warns(PrismDeprecationWarning, match="IMPACTModel.predict"):
            result = model.predict(x)
        assert torch.equal(result, expected)

    def test_wrapper_predict(self, wrapped_sklearn):
        X = np.array([[0.5, 0.5], [2.5, 2.5]])
        expected = wrapped_sklearn.predict_proba(X)
        with pytest.warns(PrismDeprecationWarning, match="SklearnWrapper.predict"):
            result = wrapped_sklearn.predict(X)
        assert torch.equal(result, expected)

    def test_calculator_predict(self, mlp_model):
        from prism.partial_responses import PartialResponseCalculator

        x_train = torch.rand(20, 5)
        calculator = PartialResponseCalculator(
            mlp_model, method='dirac', device='cpu', input_dim=5, x_train=x_train
        )
        expected = calculator.predict_proba(x_train)
        with pytest.warns(PrismDeprecationWarning, match="PartialResponseCalculator.predict"):
            result = calculator.predict(x_train)
        assert torch.equal(result, expected)


class TestDeviceHandling:
    """predict_proba honors the device argument on available devices."""

    def test_cpu_explicit(self, mlp_model):
        x = torch.randn(4, 5)
        result = mlp_model.predict_proba(x, device='cpu')
        assert result.device.type == 'cpu'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda(self, mlp_model):
        x = torch.randn(4, 5)
        result = mlp_model.predict_proba(x, device='cuda')
        assert result.is_cuda

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_mps(self, mlp_model):
        x = torch.randn(4, 5)
        result = mlp_model.predict_proba(x, device='mps')
        assert result.device.type == 'mps'


class TestLoadedArtifacts:
    """Old saved artifacts gain the new API automatically (methods live on the class)."""

    def test_maskedmlp_state_dict_roundtrip(self, mlp_model):
        buffer = io.BytesIO()
        torch.save(mlp_model.state_dict(), buffer)
        buffer.seek(0)
        loaded = MaskedMLP(input_dim=5, hidden_units=8, output_dim=1)
        loaded.load_state_dict(torch.load(buffer))
        x = torch.randn(4, 5)
        assert torch.equal(loaded.predict_proba(x), mlp_model.predict_proba(x))

    def test_wrapper_pickle_roundtrip(self, wrapped_sklearn):
        loaded = pickle.loads(pickle.dumps(wrapped_sklearn))
        X = np.array([[1.0, 1.0]])
        assert torch.equal(loaded.predict_proba(X), wrapped_sklearn.predict_proba(X))
