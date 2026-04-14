"""Tests for device management tools."""

from unittest import mock

import pytest
import torch

from prism.device_tools import (
    _free_all_gpu_caches,
    cleanup_gpu_memory,
    device_empty_cache,
    get_available_gpus,
    get_device,
    to_xgb_device,
)


class TestGetDevice:
    """Tests for get_device function."""

    def test_get_device_returns_valid_device(self):
        """Test that get_device returns a valid torch device."""
        device = get_device()

        assert isinstance(device, torch.device)
        # Should be one of: cuda, mps, or cpu
        assert device.type in ["cuda", "mps", "cpu"]

    def test_get_device_consistency(self):
        """Test that get_device returns consistent results."""
        device1 = get_device()
        device2 = get_device()

        assert device1.type == device2.type

    @pytest.mark.skipif(
        not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available(),
        reason="MPS not available",
    )
    def test_get_device_mps_available(self):
        """Test device selection when MPS is available (Apple Silicon)."""
        # This test assumes MPS is available
        device = get_device()

        # Should use MPS or CUDA depending on priority
        assert device.type in ["mps", "cuda"]


class TestGetAvailableGPUs:
    """Tests for get_available_gpus function."""

    def test_get_available_gpus_returns_list(self):
        """Test that get_available_gpus returns a list."""
        gpus = get_available_gpus()

        assert isinstance(gpus, list)

    def test_get_available_gpus_count(self):
        """Test GPU count matches platform capabilities (CUDA, MPS, or CPU-only)."""
        gpus = get_available_gpus()

        if torch.cuda.is_available():
            # Should detect CUDA GPUs
            assert len(gpus) == torch.cuda.device_count()
        else:
            # Should return empty list if no CUDA and no MPS, or 1 if MPS is available
            mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            assert len(gpus) == (1 if mps_available else 0)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_available_gpus_with_cuda(self):
        """Test GPU detection when CUDA is available."""
        gpus = get_available_gpus()

        assert len(gpus) > 0
        # GPU list contains torch.device objects, not ints
        assert all(isinstance(gpu, torch.device) for gpu in gpus)

    def test_get_available_gpus_consistency(self):
        """Test that get_available_gpus returns consistent results."""
        gpus1 = get_available_gpus()
        gpus2 = get_available_gpus()

        assert gpus1 == gpus2


@pytest.mark.integration
class TestDeviceToolsIntegration:
    """Integration tests for device tools."""

    def test_tensor_creation_on_device(self):
        """Test creating tensor on detected device."""
        device = get_device()
        tensor = torch.randn(10, 10, device=device)

        assert tensor.device.type == device.type

    def test_device_transfer(self):
        """Test transferring tensor to detected device."""
        device = get_device()
        tensor_cpu = torch.randn(10, 10)

        tensor_device = tensor_cpu.to(device)

        assert tensor_device.device.type == device.type

    def test_computation_on_device(self):
        """Test performing computation on detected device."""
        device = get_device()

        a = torch.randn(5, 5, device=device)
        b = torch.randn(5, 5, device=device)

        c = torch.matmul(a, b)

        assert c.device.type == device.type
        assert c.shape == (5, 5)


class TestToXgbDevice:
    """Tests for to_xgb_device function."""

    def test_cuda_string_returns_cuda(self):
        """Test that CUDA device string returns 'cuda'."""
        assert to_xgb_device('cuda') == 'cuda'

    def test_cuda_with_index_returns_cuda(self):
        """Test that CUDA device with index returns 'cuda'."""
        assert to_xgb_device('cuda:0') == 'cuda'
        assert to_xgb_device('cuda:1') == 'cuda'

    def test_torch_cuda_device_returns_cuda(self):
        """Test that torch.device('cuda') returns 'cuda'."""
        assert to_xgb_device(torch.device('cuda')) == 'cuda'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_torch_cuda_device_with_index_returns_cuda(self):
        """Test that torch.device('cuda:0') returns 'cuda'."""
        assert to_xgb_device(torch.device('cuda:0')) == 'cuda'

    def test_cpu_string_returns_cpu(self):
        """Test that CPU device string returns 'cpu'."""
        assert to_xgb_device('cpu') == 'cpu'

    def test_torch_cpu_device_returns_cpu(self):
        """Test that torch.device('cpu') returns 'cpu'."""
        assert to_xgb_device(torch.device('cpu')) == 'cpu'

    def test_mps_string_returns_cpu(self):
        """Test that MPS device string returns 'cpu' (XGBoost doesn't support MPS)."""
        assert to_xgb_device('mps') == 'cpu'

    def test_current_device_returns_valid(self):
        """Test that current device returns valid XGBoost device."""
        device = get_device()
        xgb_device = to_xgb_device(device)

        assert xgb_device in ['cuda', 'cpu']


class TestFreeAllGpuCaches:
    """Mock-based tests for _free_all_gpu_caches (runs on CPU CI)."""

    def test_cuda_path_calls_empty_cache_and_cupy(self):
        """CUDA available: calls empty_cache + cupy cleanup."""
        with (
            mock.patch('prism.device_tools.torch') as mock_torch,
            mock.patch('prism.device_tools._cleanup_cupy_memory') as mock_cupy,
        ):
            mock_torch.cuda.is_available.return_value = True
            mock_torch.backends = mock.MagicMock()

            _free_all_gpu_caches()

            mock_torch.cuda.empty_cache.assert_called_once()
            mock_cupy.assert_called_once()
            # Should NOT call MPS methods
            mock_torch.mps.synchronize.assert_not_called()

    def test_mps_path_calls_sync_and_empty(self):
        """MPS available (no CUDA): calls synchronize + empty_cache."""
        with mock.patch('prism.device_tools.torch') as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            mock_torch.backends.mps.is_available.return_value = True

            _free_all_gpu_caches()

            mock_torch.mps.synchronize.assert_called_once()
            mock_torch.mps.empty_cache.assert_called_once()

    def test_cpu_is_noop(self):
        """CPU-only: no cleanup calls."""
        with mock.patch('prism.device_tools.torch') as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            mock_torch.backends.mps.is_available.return_value = False

            _free_all_gpu_caches()

            mock_torch.cuda.empty_cache.assert_not_called()
            mock_torch.mps.synchronize.assert_not_called()
            mock_torch.mps.empty_cache.assert_not_called()


class TestCleanupGpuMemory:
    """Mock-based tests for cleanup_gpu_memory."""

    def test_gc_collect_is_called(self):
        """gc.collect is always called regardless of device."""
        with (
            mock.patch('prism.device_tools.gc') as mock_gc,
            mock.patch('prism.device_tools.torch') as mock_torch,
        ):
            mock_torch.cuda.is_available.return_value = False
            mock_torch.backends.mps.is_available.return_value = False

            cleanup_gpu_memory()

            mock_gc.collect.assert_called_once()

    def test_cuda_device_gets_sync_empty_cupy(self):
        """Explicit CUDA device: sync + empty_cache + cupy cleanup."""
        with (
            mock.patch('prism.device_tools.gc'),
            mock.patch('prism.device_tools._cleanup_cupy_memory') as mock_cupy,
            mock.patch('prism.device_tools.torch') as mock_torch,
        ):
            # Avoid the device=None branch
            mock_torch.cuda.is_available.return_value = True

            device = mock.MagicMock()
            device.type = 'cuda'

            cleanup_gpu_memory(device)

            mock_torch.cuda.synchronize.assert_called_once()
            mock_torch.cuda.empty_cache.assert_called_once()
            mock_cupy.assert_called_once()

    def test_mps_device_gets_sync_empty(self):
        """Explicit MPS device: sync + empty_cache."""
        with (
            mock.patch('prism.device_tools.gc'),
            mock.patch('prism.device_tools.torch') as mock_torch,
        ):
            device = mock.MagicMock()
            device.type = 'mps'

            cleanup_gpu_memory(device)

            mock_torch.mps.synchronize.assert_called_once()
            mock_torch.mps.empty_cache.assert_called_once()

    def test_device_none_cleans_all_backends(self):
        """device=None cleans all available backends."""
        with (
            mock.patch('prism.device_tools.gc'),
            mock.patch('prism.device_tools._cleanup_cupy_memory') as mock_cupy,
            mock.patch('prism.device_tools.torch') as mock_torch,
        ):
            mock_torch.cuda.is_available.return_value = True
            mock_torch.backends.mps.is_available.return_value = True

            cleanup_gpu_memory(None)

            # Both CUDA and MPS cleaned
            mock_torch.cuda.synchronize.assert_called_once()
            mock_torch.cuda.empty_cache.assert_called_once()
            mock_cupy.assert_called_once()
            mock_torch.mps.synchronize.assert_called_once()
            mock_torch.mps.empty_cache.assert_called_once()


class TestDeviceEmptyCache:
    """Mock-based tests for device_empty_cache context manager."""

    def test_cuda_cleanup_on_exit(self):
        """CUDA device: sync + empty_cache + cupy on context exit."""
        with (
            mock.patch('prism.device_tools.torch') as mock_torch,
            mock.patch('prism.device_tools._cleanup_cupy_memory') as mock_cupy,
        ):
            device = mock.MagicMock()
            device.type = 'cuda'

            with device_empty_cache(device):
                pass

            mock_torch.cuda.synchronize.assert_called_once()
            mock_torch.cuda.empty_cache.assert_called_once()
            mock_cupy.assert_called_once()

    def test_mps_cleanup_on_exit(self):
        """MPS device: sync + empty_cache on context exit."""
        with mock.patch('prism.device_tools.torch') as mock_torch:
            device = mock.MagicMock()
            device.type = 'mps'

            with device_empty_cache(device):
                pass

            mock_torch.mps.synchronize.assert_called_once()
            mock_torch.mps.empty_cache.assert_called_once()

    def test_cpu_is_noop(self):
        """CPU device: no cleanup calls."""
        with mock.patch('prism.device_tools.torch') as mock_torch:
            device = mock.MagicMock()
            device.type = 'cpu'

            with device_empty_cache(device):
                pass

            mock_torch.cuda.synchronize.assert_not_called()
            mock_torch.mps.synchronize.assert_not_called()

    def test_cleanup_happens_on_exception(self):
        """Cleanup runs even if body raises an exception."""
        with (
            mock.patch('prism.device_tools.torch') as mock_torch,
            mock.patch('prism.device_tools._cleanup_cupy_memory'),
        ):
            device = mock.MagicMock()
            device.type = 'cuda'

            with pytest.raises(ValueError):
                with device_empty_cache(device):
                    raise ValueError("test error")

            mock_torch.cuda.synchronize.assert_called_once()
            mock_torch.cuda.empty_cache.assert_called_once()
