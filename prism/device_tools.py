import gc
import inspect
import torch
from contextlib import contextmanager

@contextmanager
def device_empty_cache(device: torch.device):
    try:
        yield
    finally:
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        elif device.type == 'mps':
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
                
                print(f"Name: {name}, Type: {type(obj).__name__}, Size: {size}, Device: {device}, Dtype: {dtype}")
        except:
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
            print("Consider using device-specific APIs or profiling tools for detailed memory usage.")

def get_available_devices():
    available_devices = []
    if torch.cuda.is_available():
        available_devices.append(torch.device("cuda"))
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        available_devices.append(torch.device("mps"))
    
    # Always add CPU as a fallback
    available_devices.append(torch.device("cpu"))
    
    return available_devices

def get_device():
    """Return the first (usually best) available device: CUDA, MPS, or CPU."""
    available_devices = get_available_devices()
    return available_devices[0]