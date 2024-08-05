import torch
import pickle
from prism.maskedmlp import MaskedMLP
from collections import OrderedDict
from sklearn.linear_model import LogisticRegression

def load_mlp(filename, models_dir):
    """
    Load an MLP model from a file, deriving the structure from the state dict.
    
    Args:
    filename (str): Name of the file to load the model from (without extension)
    models_dir (str): Directory where the model is saved
    
    Returns:
    tuple: (model, params, metrics)
    """
    # Load model info
    with open(models_dir.joinpath(f"{filename}_info.pkl"), 'rb') as f:
        info = pickle.load(f)
    
    params = info['params']
    metrics = info['metrics']
    
    # Load the model state
    state_dict = torch.load(models_dir.joinpath(f"{filename}.pth"))
    
    # Derive model structure from state dict
    input_dim = state_dict['0.weight'].shape[1]
    hidden_units = state_dict['0.weight'].shape[0]
    output_dim = state_dict['2.weight'].shape[0]
    
    # Reconstruct the model
    model = MaskedMLP(input_dim, hidden_units, output_dim)
    
    # Create a new state dict with the correct keys
    new_state_dict = OrderedDict()
    new_state_dict['fc1.weight'] = state_dict['0.weight']
    new_state_dict['fc1.bias'] = state_dict['0.bias']
    new_state_dict['fc2.weight'] = state_dict['2.weight']
    new_state_dict['fc2.bias'] = state_dict['2.bias']
    
    # Load the corrected state dict
    model.load_state_dict(new_state_dict)
    
    # Update params with the derived structure
    params['input_dim'] = input_dim
    params['hidden_units'] = hidden_units
    params['output_dim'] = output_dim
    
    return model, params, metrics

def load_prn(filename, models_dir):
    """
    Load a Partial Response Network model from a file.
    
    Args:
    filename (str): Name of the file to load the model from (without extension)
    models_dir (str): Directory where the model is saved
    
    Returns:
    tuple: (model, params, metrics)
    """
    # Load model info
    with open(models_dir.joinpath(f"{filename}_info.pkl"), 'rb') as f:
        info = pickle.load(f)
    
    params = info['params']
    metrics = info['metrics']
    
    # Reconstruct the model
    input_dim = params['n_hidden']
    hidden_units = params['n_hidden']
    mask = params['mask']
    subnet_nodes = params['subnet_nodes']
    
    model = MaskedMLP(input_dim, hidden_units, 1, mask)
    
    # Load the model state
    model.load_state_dict(torch.load(models_dir.joinpath(f"{filename}.pth")))
    
    return model, params, metrics

def load_lasso(filename, models_dir):
    """
    Load a LASSO model from a file.
    
    Args:
    filename (str): Name of the file to load the model from (without extension)
    models_dir (str): Directory where the model is saved
    
    Returns:
    tuple: (model, params, metrics)
    """
    # Load the entire model
    with open(models_dir.joinpath(f"{filename}.pkl"), 'rb') as f:
        model = pickle.load(f)
    
    # Load model info
    with open(models_dir.joinpath(f"{filename}_info.pkl"), 'rb') as f:
        info = pickle.load(f)
    
    params = info['params']
    metrics = info['metrics']
    
    return model, params, metrics