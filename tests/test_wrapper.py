"""Tests for model wrapper module."""

import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression

from prism.wrapper import SklearnWrapper


class TestSklearnWrapper:
    """Tests for SklearnWrapper class."""

    @pytest.fixture
    def sklearn_model(self):
        """Create a simple sklearn logistic regression model."""
        # Create and fit a simple model
        X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        y = np.array([0, 0, 1, 1])

        model = LogisticRegression()
        model.fit(X, y)

        return model

    @pytest.fixture
    def wrapped_model(self, sklearn_model):
        """Create a wrapped sklearn model."""
        return SklearnWrapper(sklearn_model)

    def test_initialization(self, sklearn_model):
        """Test wrapper initialization."""
        wrapper = SklearnWrapper(sklearn_model)

        assert wrapper.model is sklearn_model

    def test_predict_with_numpy_array(self, wrapped_model):
        """Test predict method with numpy array input."""
        x = np.array([[0, 0], [3, 3]])

        predictions = wrapped_model.predict(x)

        # Should return torch tensor
        assert isinstance(predictions, torch.Tensor)
        # Should be probabilities (between 0 and 1)
        assert torch.all(predictions >= 0) and torch.all(predictions <= 1)
        # Should have same number of samples as input
        assert len(predictions) == len(x)

    def test_predict_with_torch_tensor(self, wrapped_model):
        """Test predict method with torch tensor input."""
        x = torch.tensor([[0.0, 0.0], [3.0, 3.0]])

        predictions = wrapped_model.predict(x)

        # Should return torch tensor
        assert isinstance(predictions, torch.Tensor)
        # Should be probabilities
        assert torch.all(predictions >= 0) and torch.all(predictions <= 1)
        # Should have correct shape
        assert len(predictions) == len(x)

    def test_callable_interface(self, wrapped_model):
        """Test that wrapper is callable."""
        x = np.array([[0, 0], [3, 3]])

        # Should be callable
        predictions = wrapped_model(x)

        assert isinstance(predictions, torch.Tensor)
        assert len(predictions) == len(x)

    def test_predict_vs_callable_consistency(self, wrapped_model):
        """Test that predict() and __call__() return same results."""
        x = np.array([[1, 1], [2, 2]])

        predictions1 = wrapped_model.predict(x)
        predictions2 = wrapped_model(x)

        torch.testing.assert_close(predictions1, predictions2)

    def test_predict_single_sample(self, wrapped_model):
        """Test prediction with single sample."""
        x = np.array([[1, 1]])

        predictions = wrapped_model.predict(x)

        assert len(predictions) == 1
        assert 0 <= predictions[0] <= 1

    def test_predict_batch(self, wrapped_model):
        """Test prediction with batch of samples."""
        x = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])

        predictions = wrapped_model.predict(x)

        assert len(predictions) == 5
        assert all(0 <= p <= 1 for p in predictions)

    def test_tensor_conversion_preserves_values(self, wrapped_model):
        """Test that tensor conversion preserves input values."""
        x_numpy = np.array([[1.234, 2.345], [3.456, 4.567]])
        x_torch = torch.tensor(x_numpy, dtype=torch.float32)

        predictions_numpy = wrapped_model.predict(x_numpy)
        predictions_torch = wrapped_model.predict(x_torch)

        # Should produce same results regardless of input type
        torch.testing.assert_close(predictions_numpy, predictions_torch)

    def test_prediction_determinism(self, wrapped_model):
        """Test that predictions are deterministic."""
        x = np.array([[1, 1], [2, 2]])

        predictions1 = wrapped_model.predict(x)
        predictions2 = wrapped_model.predict(x)

        torch.testing.assert_close(predictions1, predictions2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_predict_with_cuda_tensor(self, wrapped_model):
        """Test predict with CUDA tensor (if available)."""
        x = torch.tensor([[1.0, 1.0], [2.0, 2.0]]).cuda()

        predictions = wrapped_model.predict(x)

        # Should work and return torch tensor
        assert isinstance(predictions, torch.Tensor)
        assert len(predictions) == 2


@pytest.mark.unit
class TestSklearnWrapperGPUDetection:
    """Tests for GPU detection and device sync."""

    @pytest.fixture
    def xgb_cpu_model(self):
        """XGBClassifier trained on CPU."""
        xgb = pytest.importorskip("xgboost")
        X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        y = np.array([0, 0, 1, 1])
        model = xgb.XGBClassifier(device='cpu', n_estimators=10, use_label_encoder=False)
        model.fit(X, y)
        return model

    def test_cpu_model_not_gpu_enabled(self, xgb_cpu_model):
        wrapper = SklearnWrapper(xgb_cpu_model)
        assert wrapper._xgb_gpu_enabled is False

    def test_refresh_gpu_detection(self, xgb_cpu_model):
        wrapper = SklearnWrapper(xgb_cpu_model)
        assert wrapper._xgb_gpu_enabled is False
        # Simulate device change without CUDA (just test the method exists and works)
        result = wrapper.refresh_gpu_detection()
        assert result is False  # Still CPU

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_refresh_after_device_change_to_cuda(self, xgb_cpu_model):
        wrapper = SklearnWrapper(xgb_cpu_model)
        assert wrapper._xgb_gpu_enabled is False
        wrapper.model.set_params(device='cuda')
        result = wrapper.refresh_gpu_detection()
        assert result is True

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_sync_xgb_device_refreshes_wrapper(self, xgb_cpu_model):
        from prism.notebook_utils import _sync_xgb_device

        wrapper = SklearnWrapper(xgb_cpu_model)
        assert wrapper._xgb_gpu_enabled is False
        _sync_xgb_device(wrapper, torch.device('cuda'))
        assert wrapper._xgb_gpu_enabled is True

    def test_predict_cpu_path_with_cpu_model(self, xgb_cpu_model):
        wrapper = SklearnWrapper(xgb_cpu_model)
        X = torch.tensor([[0.0, 0.0], [3.0, 3.0]])
        result = wrapper.predict(X)
        assert isinstance(result, torch.Tensor)
        assert len(result) == 2
        assert all(0 <= p <= 1 for p in result)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_predict_gpu_path_with_cuda_model(self, xgb_cpu_model):
        """Test GPU predict path after device sync."""
        wrapper = SklearnWrapper(xgb_cpu_model)
        wrapper.model.set_params(device='cuda')
        wrapper.refresh_gpu_detection()

        X = torch.tensor([[0.0, 0.0], [3.0, 3.0]], device='cuda')
        result = wrapper.predict(X, device='cuda')
        assert isinstance(result, torch.Tensor)
        assert len(result) == 2

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_cpu_predict_consistency(self, xgb_cpu_model):
        """GPU and CPU paths should produce same predictions."""
        wrapper_cpu = SklearnWrapper(xgb_cpu_model)

        import copy

        gpu_model = copy.deepcopy(xgb_cpu_model)
        gpu_model.set_params(device='cuda')
        wrapper_gpu = SklearnWrapper(gpu_model)

        X_np = np.array([[0.0, 0.0], [1.5, 1.5], [3.0, 3.0]])
        X_cpu = torch.tensor(X_np, dtype=torch.float32)
        X_cuda = X_cpu.cuda()

        result_cpu = wrapper_cpu.predict(X_cpu)
        result_gpu = wrapper_gpu.predict(X_cuda, device='cuda').cpu()

        torch.testing.assert_close(result_cpu, result_gpu, atol=1e-5, rtol=1e-5)

    def test_xgbrf_detection(self):
        """XGBRFClassifier should also be detected for GPU."""
        xgb = pytest.importorskip("xgboost")
        X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        y = np.array([0, 0, 1, 1])
        model = xgb.XGBRFClassifier(device='cpu', n_estimators=10, use_label_encoder=False)
        model.fit(X, y)
        wrapper = SklearnWrapper(model)
        assert wrapper._xgb_gpu_enabled is False  # CPU model
        # Verify detection mechanism works for RF too
        assert hasattr(wrapper.model, 'device')

    def test_setstate_backward_compat(self, xgb_cpu_model):
        """Test __setstate__ handles missing GPU fields."""
        wrapper = SklearnWrapper(xgb_cpu_model)
        state = wrapper.__dict__.copy()
        del state['_xgb_gpu_enabled']
        del state['_gpu_inference_warned']
        wrapper2 = SklearnWrapper.__new__(SklearnWrapper)
        wrapper2.__setstate__(state)
        assert hasattr(wrapper2, '_xgb_gpu_enabled')
        assert hasattr(wrapper2, '_gpu_inference_warned')
