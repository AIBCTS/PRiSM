import numpy as np
import pandas as pd
import torch
import pickle
from prism.partial_responses_old import partialResponses
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

def test_refactored_partial_responses(refactored_function, filename=MODELS_DIR.joinpath("partial_responses_data.pkl"),tensor_input=False):
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