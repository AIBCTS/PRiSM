import gc
import inspect
import logging
from contextlib import contextmanager
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)


def _cleanup_cupy_memory():
    """Free cupy's separate memory pools (default + pinned).

    Safe no-op when cupy is not installed (e.g. CPU-only, MPS, or CUDA without cupy).
    """
    try:
        import cupy as cp

        pool = cp.get_default_memory_pool()
        pinned_pool = cp.get_default_pinned_memory_pool()
        pool.free_all_blocks()
        pinned_pool.free_all_blocks()
        logger.debug(
            "cupy memory pools freed (default: %d bytes, pinned: %d bytes remaining)",
            pool.used_bytes(),
            pinned_pool.n_free_blocks(),
        )
    except ImportError:
        pass


def _free_all_gpu_caches():
    """Free GPU caches for the active backend.

    Lightweight helper for use inside hot loops (e.g. lebesgue batching).
    CUDA: empty_cache only (no synchronize, no gc.collect) for speed.
    MPS: synchronize + empty_cache (MPS requires sync before cache release;
         acceptable cost since MPS targets dev machines, not GPU clusters).
    Safe no-op on CPU-only systems.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        _cleanup_cupy_memory()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.synchronize()
        torch.mps.empty_cache()


def cleanup_gpu_memory(device: Optional[torch.device] = None):
    """Explicit GPU memory cleanup between notebook stages.

    Runs gc.collect() then frees GPU caches for the specified device.
    When device is None, cleans all available backends.

    Parameters
    ----------
    device : torch.device or None
        Target device to clean. None cleans all available backends.
    """
    gc.collect()

    if device is None:
        # Clean all available backends
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            _cleanup_cupy_memory()
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.synchronize()
            torch.mps.empty_cache()
    elif device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        _cleanup_cupy_memory()
    elif device.type == 'mps':
        torch.mps.synchronize()
        torch.mps.empty_cache()
    # CPU: nothing to clean


@contextmanager
def device_empty_cache(device: torch.device):
    try:
        yield
    finally:
        if device.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            _cleanup_cupy_memory()
        elif device.type == 'mps':
            torch.mps.synchronize()
            torch.mps.empty_cache()


def print_tensor_info():
    # Get the calling frame
    frame = inspect.currentframe().f_back

    # Get all variables in the calling frame
    variables = frame.f_locals

    # Reverse dictionary to map object ids to variable names
    id_to_name = {id(v): k for k, v in variables.items()}

    print("Tensors currently in memory:")
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                # Try to get the variable name
                name = id_to_name.get(id(obj), "Unknown")

                # Get tensor properties
                size = obj.size() if hasattr(obj, 'size') else "N/A"
                device = obj.device if hasattr(obj, 'device') else "N/A"
                dtype = obj.dtype if hasattr(obj, 'dtype') else "N/A"

                print(
                    f"Name: {name}, Type: {type(obj).__name__}, Size: {size}, Device: {device}, Dtype: {dtype}"
                )
        except Exception:
            pass  # Skip any objects that cause errors

    print("\nDevice Memory Usage:")
    print_device_memory_usage()


def print_device_memory_usage():
    if torch.cuda.is_available():
        print("CUDA (GPU):")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Cached:    {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("\nMPS (Apple Silicon):")
        print(f"Allocated: {torch.mps.current_allocated_memory() / 1e9:.2f} GB")
        print(f"Cached:    {torch.mps.driver_allocated_memory() / 1e9:.2f} GB")

    # Generic approach for other devices
    for device in get_available_devices():
        if device.type not in ['cuda', 'mps', 'cpu']:
            print(f"\n{device.type.upper()}:")
            print("Memory usage information not directly available.")
            print(
                "Consider using device-specific APIs or profiling tools for detailed memory usage."
            )


def get_available_devices():
    available_devices = []
    if torch.cuda.is_available():
        available_devices.append(torch.device("cuda"))
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        available_devices.append(torch.device("mps"))

    # Always add CPU as a fallback
    available_devices.append(torch.device("cpu"))

    return available_devices


def get_device(preferred: Optional[str] = None):
    """Resolve the compute device.

    Priority: `preferred` argument > PRISM_DEVICE environment variable >
    auto-detect (CUDA, then MPS, then CPU).

    Parameters
    ----------
    preferred : str, optional
        Requested device: 'auto', 'cpu', 'cuda', 'cuda:N', or 'mps'.
        Typically passed from the YAML config's `device` key. None or
        'auto' selects the best available device.

    Returns
    -------
    torch.device
        The resolved device. If the requested device is unavailable, a
        warning is logged and auto-detection is used instead.
    """
    import os

    requested = preferred if preferred is not None else os.environ.get('PRISM_DEVICE')
    if requested:
        requested = str(requested).strip().lower()
        if requested != 'auto':
            if requested == 'cpu':
                return torch.device('cpu')
            if requested.startswith('cuda') and torch.cuda.is_available():
                return torch.device(requested)
            if (
                requested == 'mps'
                and hasattr(torch.backends, 'mps')
                and torch.backends.mps.is_available()
            ):
                return torch.device('mps')
            logger.warning(
                f"Requested device '{requested}' is not available; falling back to auto-detection"
            )
    available_devices = get_available_devices()
    return available_devices[0]


def get_available_gpus() -> List[torch.device]:
    """
    Get a list of all available GPU devices (CUDA and MPS).

    Returns:
        List[torch.device]: List of available GPU devices
    """
    gpus = []
    if torch.cuda.is_available():
        gpus.extend([torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())])
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        gpus.append(torch.device('mps'))
    return gpus


def get_num_cpu_workers() -> int:
    """
    Get the number of CPU workers to use for parallel processing (total number of cores - 1).

    Returns:
        int: Number of CPU workers
    """
    return max(1, torch.multiprocessing.cpu_count() - 1)  # Leave one CPU core free


def to_xgb_device(device) -> str:
    """
    Convert a PyTorch device to an XGBoost-compatible device string.

    XGBoost supports 'cuda' and 'cpu'. MPS (Apple Silicon) is not supported
    and falls back to 'cpu'.

    Args:
        device: PyTorch device (torch.device, str like 'cuda', 'cuda:0', 'mps', 'cpu')

    Returns:
        str: 'cuda' if CUDA device, otherwise 'cpu'

    Examples:
        >>> to_xgb_device(torch.device('cuda:0'))
        'cuda'
        >>> to_xgb_device('mps')
        'cpu'
        >>> to_xgb_device('cpu')
        'cpu'
    """
    device_str = str(device)
    if device_str.startswith('cuda'):
        return 'cuda'
    # XGBoost doesn't support MPS, fall back to CPU
    return 'cpu'
