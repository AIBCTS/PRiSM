import torch
import pickle
from datetime import datetime
import os

def save_mlp(model, params, metrics, save_dir):
    """
    Save the MLP model, its parameters, and metrics.
    
    Args:
    model (torch.nn.Module): The trained MLP model
    params (dict): Dictionary of model parameters
    metrics (dict): Dictionary of model metrics
    save_dir (str): Directory to save the model
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"mlp_model_{timestamp}"
    
    # Save the model state
    torch.save(model.state_dict(), os.path.join(save_dir, f"{model_name}.pth"))
    
    # Save parameters and metrics
    with open(os.path.join(save_dir, f"{model_name}_info.pkl"), 'wb') as f:
        pickle.dump({'params': params, 'metrics': metrics}, f)
    
    print(f"MLP model saved as {model_name}")

def save_prn(model, params, metrics, save_dir):
    """
    Save the Partial Response Network model, its parameters, and metrics.
    
    Args:
    model (torch.nn.Module): The trained PRN model
    params (dict): Dictionary of model parameters (including mask)
    metrics (dict): Dictionary of model metrics
    save_dir (str): Directory to save the model
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"prn_model_{timestamp}"
    
    # Save the model state
    torch.save(model.state_dict(), os.path.join(save_dir, f"{model_name}.pth"))
    
    # Save parameters and metrics
    with open(os.path.join(save_dir, f"{model_name}_info.pkl"), 'wb') as f:
        pickle.dump({'params': params, 'metrics': metrics}, f)
    
    print(f"PRN model saved as {model_name}")

def save_lasso(model, params, metrics, save_dir):
    """
    Save the LASSO model, its parameters, and metrics.
    
    Args:
    model: The trained LASSO model
    params (dict): Dictionary of model parameters
    metrics (dict): Dictionary of model metrics
    save_dir (str): Directory to save the model
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"lasso_model_{timestamp}"
    
    # Save the entire model
    with open(os.path.join(save_dir, f"{model_name}.pkl"), 'wb') as f:
        pickle.dump(model, f)
    
    # Save parameters and metrics
    with open(os.path.join(save_dir, f"{model_name}_info.pkl"), 'wb') as f:
        pickle.dump({'params': params, 'metrics': metrics}, f)
    
    print(f"LASSO model saved as {model_name}")