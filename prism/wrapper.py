"""
Wrapper classes to provide a unified predict_proba interface for sklearn-style models.

This module provides SklearnWrapper which wraps sklearn-compatible models
(XGBoost, RandomForest, etc.) to match the PyTorch-style predict_proba API used
by MaskedMLP and other PRiSM models.

GPU Optimization
----------------
For XGBoost models trained with device='cuda', SklearnWrapper uses XGBoost's
inplace_predict with cupy arrays to avoid unnecessary GPU↔CPU data transfers.
This provides significant speedup for partial response calculations.
"""

import json
import logging

import torch

logger = logging.getLogger(__name__)

# Try to import cupy for GPU-accelerated inference
try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


class SklearnWrapper:
    """
    Wrapper class to match sklearn models to the MaskedMLP PyTorch API.

    This wrapper provides a unified predict_proba interface that returns PyTorch tensors,
    enabling sklearn-compatible models (XGBoost, RandomForest, etc.) to be used
    interchangeably with PyTorch models in the PRiSM pipeline.

    GPU Optimization
    ----------------
    For XGBoost models with GPU enabled (device='cuda' or 'gpu'), this wrapper
    uses XGBoost's inplace_predict with cupy arrays to keep data on the GPU,
    avoiding the CPU roundtrip that would otherwise occur with predict_proba().

    Parameters
    ----------
    model : sklearn-compatible model
        A model with predict_proba() method (e.g., XGBClassifier, RandomForest)

    Examples
    --------
    >>> import xgboost as xgb
    >>> from prism.wrapper import SklearnWrapper
    >>>
    >>> # Train an XGBoost model with GPU
    >>> xgb_model = xgb.XGBClassifier(device='cuda')
    >>> xgb_model.fit(X_train, y_train)
    >>>
    >>> # Wrap for use with PRiSM
    >>> model = SklearnWrapper(xgb_model)
    >>>
    >>> # Predict with CUDA tensors - stays on GPU
    >>> X_cuda = torch.tensor(X_test, device='cuda')
    >>> probs = model.predict_proba(X_cuda, device='cuda')
    """

    def __init__(self, model):
        # Check if we're wrapping an already wrapped model
        self.model = model.model if hasattr(model, 'model') else model

        # Detect GPU capability for XGBoost models
        self._xgb_gpu_enabled = self._detect_xgb_gpu()
        self._gpu_inference_warned = False

        if self._xgb_gpu_enabled:
            if CUPY_AVAILABLE:
                logger.debug("GPU-accelerated inference enabled for XGBoost model")
            else:
                logger.debug(
                    "XGBoost GPU model detected but cupy not available. "
                    "Install cupy-cuda12x for optimized GPU inference."
                )

    def __setstate__(self, state):
        """Support loading old pickled SklearnWrapper instances."""
        self.__dict__.update(state)
        if '_xgb_gpu_enabled' not in state:
            self._xgb_gpu_enabled = self._detect_xgb_gpu()
        if '_gpu_inference_warned' not in state:
            self._gpu_inference_warned = False

    def _detect_xgb_gpu(self) -> bool:
        """
        Check if this is a GPU-enabled XGBoost model.

        Returns True if:
        - Model has device='cuda' or device='gpu' attribute
        - Model's booster config indicates GPU device
        """
        # Check for XGBClassifier/XGBRegressor with cuda/gpu device
        if hasattr(self.model, 'device'):
            device = str(self.model.device).lower()
            if 'cuda' in device or 'gpu' in device:
                return True

        # Check booster config for device info
        if hasattr(self.model, 'get_booster'):
            try:
                booster = self.model.get_booster()
                config_str = booster.save_config()
                config = json.loads(config_str)
                device = config.get('learner', {}).get('generic_param', {}).get('device', '')
                if 'cuda' in device.lower() or 'gpu' in device.lower():
                    return True
            except Exception:
                pass

        return False

    def refresh_gpu_detection(self) -> bool:
        """Re-detect GPU capability after model device changes.

        Call after modifying the inner model's device parameter
        (e.g., via set_params(device='cuda')).

        Returns True if GPU inference is now enabled.
        """
        old_state = self._xgb_gpu_enabled
        self._xgb_gpu_enabled = self._detect_xgb_gpu()
        self._gpu_inference_warned = False

        if self._xgb_gpu_enabled != old_state:
            if self._xgb_gpu_enabled:
                logger.info("GPU-accelerated inference enabled after device refresh")
            else:
                logger.info("GPU-accelerated inference disabled after device refresh")

        return self._xgb_gpu_enabled

    def _predict_gpu_direct(self, X: torch.Tensor, target_device: torch.device) -> torch.Tensor:
        """
        Direct GPU inference for XGBoost without CPU roundtrip.

        Uses cupy for zero-copy conversion between PyTorch CUDA tensors and
        XGBoost's inplace_predict. Returns probability of positive class.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor on CUDA device
        target_device : torch.device
            Device for the output tensor

        Returns
        -------
        torch.Tensor
            Probability predictions on target_device
        """
        # Ensure tensor is contiguous and float32 for XGBoost
        if not X.is_contiguous():
            X = X.contiguous()
        if X.dtype != torch.float32:
            X = X.to(torch.float32)

        # Zero-copy conversion: PyTorch CUDA tensor -> CuPy array via DLPack
        cupy_array = cp.from_dlpack(X.detach())

        # Use predict_proba which respects the classifier's device setting.
        # This avoids booster-level device mismatch issues in XGBoost 3.x.
        probs = self.model.predict_proba(cupy_array)[:, 1]

        # Convert result back to torch tensor (handles both cupy and numpy returns)
        if hasattr(probs, '__dlpack__'):
            result = torch.from_dlpack(probs)
        else:
            result = torch.tensor(probs, dtype=torch.float32)

        return result.to(dtype=torch.float32, device=target_device)

    def predict_proba(self, X, device=None):
        """
        Probability of the positive class as a PyTorch tensor.

        For GPU-enabled XGBoost models with CUDA input tensors, this method
        uses optimized GPU inference to avoid CPU data transfers.

        Note: this intentionally SHADOWS the wrapped sklearn model's own
        predict_proba. The inner model returns an (n, 2) numpy array; this
        wrapper returns a torch tensor with only the positive-class column,
        matching the PRiSM model convention. The inner model's original
        method remains reachable via ``wrapper.model.predict_proba``.

        Parameters
        ----------
        X : torch.Tensor or numpy.ndarray
            Input features
        device : str or torch.device, optional
            Target device for output tensor. Defaults to 'cpu'.

        Returns
        -------
        torch.Tensor
            Probability of positive class for each sample, shape (n_samples,)
        """
        if device is None:
            device = torch.device('cpu')
        elif isinstance(device, str):
            device = torch.device(device)

        # Fast path: GPU XGBoost with CUDA tensor input and cupy available
        if self._xgb_gpu_enabled and CUPY_AVAILABLE and isinstance(X, torch.Tensor) and X.is_cuda:
            if not hasattr(self, '_gpu_path_logged'):
                logger.info("Using GPU fast path for XGBoost inference (cupy + predict_proba)")
                self._gpu_path_logged = True
            try:
                return self._predict_gpu_direct(X, device)
            except Exception as e:
                # Fall back to CPU path if GPU inference fails
                if not self._gpu_inference_warned:
                    logger.warning(f"GPU inference failed, falling back to CPU: {e}")
                    self._gpu_inference_warned = True

        # Standard path: Convert to numpy and use predict_proba
        if isinstance(X, torch.Tensor):
            if not hasattr(self, '_cpu_path_logged') and X.is_cuda:
                logger.info(
                    "Using CPU fallback for prediction " "(xgb_gpu=%s, cupy=%s)",
                    self._xgb_gpu_enabled,
                    CUPY_AVAILABLE,
                )
                self._cpu_path_logged = True
            X = X.cpu().numpy()

        result = self.model.predict_proba(X)[:, 1]
        return torch.tensor(result, dtype=torch.float32, device=device)

    def predict(self, X, device=None, threshold=0.5):
        """
        Binary class labels: (predict_proba(X, device) >= threshold) as a long tensor.

        Use predict_proba for the underlying probabilities. The wrapped sklearn
        model's own label predict remains reachable via ``wrapper.model.predict``.
        """
        return (self.predict_proba(X, device) >= threshold).long()

    def __call__(self, X):
        """
        Callable interface for predictions.

        Parameters
        ----------
        X : torch.Tensor or numpy.ndarray
            Input features

        Returns
        -------
        torch.Tensor
            Probability of positive class (on CPU by default, or same device as input)
        """
        # Determine output device from input
        if isinstance(X, torch.Tensor):
            output_device = X.device
        else:
            output_device = torch.device('cpu')

        return self.predict_proba(X, device=output_device)
