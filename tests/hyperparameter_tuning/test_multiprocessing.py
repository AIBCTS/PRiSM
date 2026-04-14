"""Tests for multiprocessing support in hyperparameter tuning."""

import multiprocessing

import numpy as np
import pytest
import torch

from prism.hyperparameter_tuning.config import TuningConfig
from prism.hyperparameter_tuning.tuning import (
    _convert_data_for_multiprocessing,
    _get_spawn_context,
)


class TestMultiprocessingConfig:
    """Test multiprocessing configuration options."""

    def test_use_multiprocessing_default_false(self):
        """use_multiprocessing defaults to False."""
        config = TuningConfig(enabled=True, n_trials=10)
        assert config.use_multiprocessing is False

    def test_use_multiprocessing_can_be_enabled(self):
        """use_multiprocessing can be explicitly enabled."""
        config = TuningConfig(enabled=True, n_trials=10, use_multiprocessing=True)
        assert config.use_multiprocessing is True

    def test_use_multiprocessing_with_n_jobs(self):
        """use_multiprocessing can be combined with n_jobs."""
        config = TuningConfig(enabled=True, n_trials=20, n_jobs=4, use_multiprocessing=True)
        assert config.use_multiprocessing is True
        assert config.n_jobs == 4


class TestDataPickling:
    """Test data conversion for multiprocessing pickling."""

    def test_cpu_tensor_to_numpy(self):
        """CPU tensors are converted to numpy arrays."""
        tensor = torch.randn(10, 5)
        result = _convert_data_for_multiprocessing(tensor)

        assert isinstance(result, np.ndarray)
        assert result.shape == (10, 5)
        np.testing.assert_array_almost_equal(result, tensor.numpy())

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_tensor_to_cpu_numpy(self):
        """GPU tensors are converted to CPU numpy arrays."""
        tensor = torch.randn(10, 5).cuda()
        result = _convert_data_for_multiprocessing(tensor)

        assert isinstance(result, np.ndarray)
        assert result.shape == (10, 5)
        # Verify it's on CPU (can be converted to numpy without error)
        assert result.dtype == np.float32

    def test_numpy_array_passthrough(self):
        """Numpy arrays pass through unchanged."""
        arr = np.random.randn(10, 5).astype(np.float32)
        result = _convert_data_for_multiprocessing(arr)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)

    def test_pandas_dataframe_to_numpy(self):
        """Pandas DataFrames are converted to numpy arrays."""
        import pandas as pd

        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        result = _convert_data_for_multiprocessing(df)

        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 2)

    def test_pandas_series_to_numpy(self):
        """Pandas Series are converted to numpy arrays."""
        import pandas as pd

        series = pd.Series([1, 2, 3, 4, 5])
        result = _convert_data_for_multiprocessing(series)

        assert isinstance(result, np.ndarray)
        assert result.shape == (5,)

    def test_list_to_numpy(self):
        """Lists are converted to numpy arrays."""
        lst = [1, 2, 3, 4, 5]
        result = _convert_data_for_multiprocessing(lst)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array(lst))


class TestSpawnMethodSetup:
    """Test multiprocessing spawn method setup."""

    def test_get_spawn_context_returns_spawn_context(self):
        """_get_spawn_context returns a spawn multiprocessing context."""
        ctx = _get_spawn_context()

        # Verify it's a context object with Process
        assert hasattr(ctx, 'Process')
        # The context should use 'spawn' method
        assert ctx.get_start_method() == 'spawn'

    def test_get_spawn_context_is_idempotent(self):
        """_get_spawn_context can be called multiple times safely."""
        ctx1 = _get_spawn_context()
        ctx2 = _get_spawn_context()

        # Both should return valid spawn contexts
        assert ctx1.get_start_method() == 'spawn'
        assert ctx2.get_start_method() == 'spawn'

    def test_spawn_context_independent_of_default_method(self):
        """_get_spawn_context works regardless of global default method."""
        # Get current default method
        current_default = multiprocessing.get_start_method(allow_none=True)

        # Our function should return spawn context regardless of default
        ctx = _get_spawn_context()
        assert ctx.get_start_method() == 'spawn'

        # Default should be unchanged
        assert multiprocessing.get_start_method(allow_none=True) == current_default
