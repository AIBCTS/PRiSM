import numpy as np
import pandas as pd
import torch
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from prism.obsolete.partial_responses_old import partialResponses
from prism.config import MODELS_DIR

def save_partial_responses(x_train, x_test, model, method="dirac", device="cpu", filename=MODELS_DIR.joinpath("partial_responses_data.pkl")):
    """
    Calculate partial responses using the current implementation and save all relevant data.
    
    Args:
    x_train (pd.DataFrame): Training data
    x_test (pd.DataFrame): Test data
    model: Trained model
    method (str): Method for partial responses calculation
    device (str): Device to use for computation
    filename (str): Name of the file to save the results
    """
    partial_responses_train, partial_responses_test, bivariate_inputs = partialResponses(x_train, x_test, model, method=method, device=device)
    
    data_to_save = {
        'x_train': x_train,
        'x_test': x_test,
        'model': model,
        'partial_responses_train': partial_responses_train,
        'partial_responses_test': partial_responses_test,
        'method': method,
        'device': device
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(data_to_save, f)
    
    print(f"Data saved to {filename}")
    return partial_responses_train, partial_responses_test, bivariate_inputs

def save_partial_responses_results(x_train, x_test, model, partial_responses_train, partial_responses_test, bivariate_inputs, method, device, filename=MODELS_DIR.joinpath("partial_responses_results.pkl")):
    """
    Save the results and arguments after running partial responses calculation.
    
    Args:
    x_train (torch.Tensor): Training data
    x_test (torch.Tensor): Test data
    model: Trained model
    partial_responses_train (torch.Tensor or np.ndarray): Partial responses for training data
    partial_responses_test (torch.Tensor or np.ndarray): Partial responses for test data
    bivariate_inputs (List[Tuple[int, int]]): List of bivariate input pairs
    method (str): Method used for partial responses calculation
    device (str): Device used for computation
    filename (str): Name of the file to save the results
    """
    # Convert torch tensors to numpy arrays
    x_train_np = x_train.cpu().numpy()
    x_test_np = x_test.cpu().numpy()
    
    if isinstance(partial_responses_train, torch.Tensor):
        partial_responses_train = partial_responses_train.cpu().numpy()
    if isinstance(partial_responses_test, torch.Tensor):
        partial_responses_test = partial_responses_test.cpu().numpy()

    data_to_save = {
        'x_train': x_train_np,
        'x_test': x_test_np,
        'model': model,
        'partial_responses_train': partial_responses_train,
        'partial_responses_test': partial_responses_test,
        'bivariate_inputs': bivariate_inputs,
        'method': method,
        'device': str(device)
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(data_to_save, f)
    
    print(f"Results and arguments saved to {filename}")

def test_refactored_partial_responses(refactored_function, filename=MODELS_DIR.joinpath("partial_responses_data.pkl"), tensor_input=False, batch_size: int = None):
    """
    Test the refactored partialResponses function against saved data.
    
    Args:
    refactored_function: The refactored partialResponses function
    filename (str): Name of the file containing saved data
    
    Returns:
    bool: True if the refactored function produces similar results, False otherwise
    """
    # Load saved data
    with open(filename, 'rb') as f:
        saved_data = pickle.load(f)
    
    x_train = saved_data['x_train']
    x_test = saved_data['x_test']
    model = saved_data['model']
    saved_partial_responses_train = saved_data['partial_responses_train']
    saved_partial_responses_test = saved_data['partial_responses_test']
    method = saved_data['method']
    device = saved_data['device']
    
    # Calculate partial responses using the refactored function
    if tensor_input:
        x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32)
        x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32)
        if batch_size is not None:
            partial_responses_train_, partial_responses_test_, _ = refactored_function(x_train_tensor, x_test_tensor, model, method=method, device=device, batch_size=batch_size)
        else:
            partial_responses_train_, partial_responses_test_, _ = refactored_function(x_train_tensor, x_test_tensor, model, method=method, device=device)

        partial_responses_train = partial_responses_train_.numpy()
        partial_responses_test = partial_responses_test_.numpy()
    else:
        partial_responses_train, partial_responses_test, _ = refactored_function(x_train, x_test, model, method=method, device=device)

    
    # Compare results
    rtol=1e-5
    atol=1e-7
    train_similar = np.allclose(partial_responses_train, saved_partial_responses_train, rtol=rtol, atol=atol)
    test_similar = np.allclose(partial_responses_test, saved_partial_responses_test, rtol=rtol, atol=atol)
    
    if train_similar and test_similar:
        print("Refactored function produces similar results to the original implementation.")
        return True
    else:
        print(f"Refactored function produces different results from the original implementation. atol = {atol}, rtol = {rtol}")
        if not train_similar:
            print("Differences found in partial_responses_train")
            print("Max absolute difference:", np.max(np.abs(partial_responses_train - saved_partial_responses_train)))
            print("Max relative difference:", np.max(np.abs(partial_responses_train - saved_partial_responses_train) / (np.abs(saved_partial_responses_train) + atol/rtol)))
        if not test_similar:
            print("Differences found in partial_responses_test")
            print("Max absolute difference:", np.max(np.abs(partial_responses_test - saved_partial_responses_test)))
            print("Max relative difference:", np.max(np.abs(partial_responses_test - saved_partial_responses_test) / (np.abs(saved_partial_responses_test) + atol/rtol)))
        return False
    
def load_partial_responses(filename=MODELS_DIR.joinpath("partial_responses_data.pkl")):
    """
    Load partial responses and related data from a saved file.
    
    Args:
    filename (str): Name of the file containing saved data
    
    Returns:
    dict: A dictionary containing the loaded data
    """
    try:
        with open(filename, 'rb') as f:
            loaded_data = pickle.load(f)
        
        required_keys = ['x_train', 'x_test', 'model', 'partial_responses_train', 
                         'partial_responses_test', 'method', 'device']
        
        if not all(key in loaded_data for key in required_keys):
            raise KeyError("The loaded data does not contain all required keys.")
        
        print(f"Data successfully loaded from {filename}")
        print(f"Loaded data contains the following keys: {', '.join(loaded_data.keys())}")
        
        # Verify shapes        
        print(f"x_train shape: {loaded_data['x_train'].shape}")
        print(f"x_test shape: {loaded_data['x_test'].shape}")
        print(f"partial_responses_train shape: {loaded_data['partial_responses_train'].shape}")
        print(f"partial_responses_test shape: {loaded_data['partial_responses_test'].shape}")
        
        return loaded_data
    
    except FileNotFoundError:
        print(f"Error: The file {filename} was not found.")
        return None
    except pickle.UnpicklingError:
        print(f"Error: The file {filename} is not a valid pickle file.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return None
    
def find_large_deviations(torch_tensor, numpy_array, threshold=0.1):
    diff = np.abs(torch_tensor.numpy() - numpy_array)
    large_dev_indices = np.argwhere(diff > threshold)
    return large_dev_indices

def visualize_large_deviations(torch_tensor, numpy_array, large_dev_indices, num_samples=5):
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    for i, (row, col) in enumerate(large_dev_indices[:num_samples]):
        window_size = 5
        row_start, row_end = max(0, row-window_size), min(torch_tensor.shape[0], row+window_size+1)
        col_start, col_end = max(0, col-window_size), min(torch_tensor.shape[1], col+window_size+1)
        
        torch_subset = torch_tensor[row_start:row_end, col_start:col_end].numpy()
        numpy_subset = numpy_array[row_start:row_end, col_start:col_end]
        diff_subset = np.abs(torch_subset - numpy_subset)
        
        vmin = min(torch_subset.min(), numpy_subset.min())
        vmax = max(torch_subset.max(), numpy_subset.max())
        
        axes[i, 0].imshow(torch_subset, cmap='coolwarm', vmin=vmin, vmax=vmax)
        axes[i, 0].set_title(f'PyTorch (Row: {row}, Col: {col})')
        axes[i, 1].imshow(numpy_subset, cmap='coolwarm', vmin=vmin, vmax=vmax)
        axes[i, 1].set_title(f'NumPy (Row: {row}, Col: {col})')
        im = axes[i, 2].imshow(diff_subset, cmap='viridis')
        axes[i, 2].set_title('Absolute Difference')
        plt.colorbar(im, ax=axes[i, 2])
        
    plt.tight_layout()
    plt.show()

def scatter_plot_large_deviations(large_dev_indices, shape):
    plt.figure(figsize=(12, 8))
    plt.scatter(large_dev_indices[:, 0], large_dev_indices[:, 1], alpha=0.5)
    plt.xlabel('Row Index')
    plt.ylabel('Column Index')
    plt.title('Distribution of Large Deviations')
    plt.xlim(0, shape[0])
    plt.ylim(0, shape[1])
    plt.gca().invert_yaxis()  # Invert y-axis to match matrix coordinates
    plt.colorbar(label='Density')
    plt.show()

def compare_and_visualize(torch_tensor, numpy_array):
    print("Data Types:")
    print(f"PyTorch Tensor: {torch_tensor.dtype}")
    print(f"NumPy Array: {numpy_array.dtype}")
    print()

    print("Shape:")
    print(f"PyTorch Tensor: {torch_tensor.shape}")
    print(f"NumPy Array: {numpy_array.shape}")
    print()

    print("Basic Statistics:")
    print("Mean:")
    print(f"PyTorch Tensor: {torch_tensor.mean().item()}")
    print(f"NumPy Array: {numpy_array.mean()}")
    print()

    print("Standard Deviation:")
    print(f"PyTorch Tensor: {torch_tensor.std().item()}")
    print(f"NumPy Array: {numpy_array.std()}")
    print()

    print("Min:")
    print(f"PyTorch Tensor: {torch_tensor.min().item()}")
    print(f"NumPy Array: {numpy_array.min()}")
    print()

    print("Max:")
    print(f"PyTorch Tensor: {torch_tensor.max().item()}")
    print(f"NumPy Array: {numpy_array.max()}")
    print()

    # Overall distribution plot
    plt.figure(figsize=(12, 6))
    sns.histplot(torch_tensor.numpy().flatten(), kde=True, color='blue', alpha=0.5, label='PyTorch')
    sns.histplot(numpy_array.flatten(), kde=True, color='red', alpha=0.5, label='NumPy')
    plt.title('Distribution of Values')
    plt.legend()
    plt.show()

    # Correlation plot
    plt.figure(figsize=(10, 8))
    plt.scatter(torch_tensor.numpy().flatten(), numpy_array.flatten(), alpha=0.1)
    plt.xlabel('PyTorch Tensor Values')
    plt.ylabel('NumPy Array Values')
    plt.title('Correlation between PyTorch Tensor and NumPy Array')
    plt.plot([numpy_array.min(), numpy_array.max()], [numpy_array.min(), numpy_array.max()], 'r--', lw=2)
    plt.show()

    # Find and visualize large deviations
    large_dev_indices = find_large_deviations(torch_tensor, numpy_array)
    visualize_large_deviations(torch_tensor, numpy_array, large_dev_indices)

    # Scatter plot of large deviation indices
    scatter_plot_large_deviations(large_dev_indices, torch_tensor.shape)

    # Print summary of large deviations
    print(f"Number of elements with large deviations: {len(large_dev_indices)}")
    print(f"Percentage of elements with large deviations: {len(large_dev_indices) / torch_tensor.numel() * 100:.2f}%")