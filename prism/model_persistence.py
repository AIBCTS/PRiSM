import os
import pickle
from datetime import datetime
from pathlib import Path

import torch


def save_model(
    model,
    model_type,
    hyperparameters,
    train_results,
    save_dir,
    partial_responses=None,
    scaler=None,
    filename=None,
):
    """
    Save a model along with its hyperparameters, training results, partial responses, and scaler.

    Parameters
    ----------
    model : object
        The model to save (MaskedMLP or scikit-learn model).
    model_type : str
        Type of the model ('mlp', 'prn', or 'lasso').
    hyperparameters : dict
        Dictionary of model hyperparameters.
    train_results : dict
        Dictionary of training results (e.g., metrics, LASSO results).
    save_dir : str or Path
        Directory to save the model.
    partial_responses : dict, optional
        Dictionary containing partial responses for train and test sets.
    scaler : object, optional
        The scaler object used for data normalization (PRiSMScaler, sklearn scalers, etc.).
    filename : str, optional
        Custom filename for the model (with extension). If not provided,
        generates a timestamped filename based on model_type.

    Returns
    -------
    None    Notes
    -----
    This function saves the model, its hyperparameters, training results,
    partial responses (if provided), and scaler object (if provided) to files.
    If filename is not provided, the filenames include a timestamp for uniqueness.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Determine filename - use provided filename or generate timestamped one
    if filename is not None:
        # Extract base filename without extension for partial responses and scaler
        base_filename = Path(filename).stem
        model_filename = filename
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{model_type}_{timestamp}"
        # Extension will be added based on model type
        model_filename = None

    if model_type in ['mlp', 'prn']:
        # For MaskedMLP models, extract input_dim, hidden_dim, and output_dim from the model
        if hasattr(model, 'fc1') and hasattr(model, 'fc2'):
            hyperparameters['input_dim'] = model.fc1.in_features
            hyperparameters['hidden_dim'] = model.fc1.out_features
            hyperparameters['output_dim'] = model.fc2.out_features
        else:
            raise ValueError("Model structure is not as expected for MLP or PRN")

        # Determine the actual filename to save
        if model_filename is not None:
            save_path = save_dir / model_filename
        else:
            save_path = save_dir / f"{base_filename}.pt"

        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'model_class': type(model),
                'hyperparameters': hyperparameters,
                'train_results': train_results,
            },
            save_path,
        )
    elif model_type == 'lasso':
        # For LASSO models
        # Determine the actual filename to save
        if model_filename is not None:
            save_path = save_dir / model_filename
        else:
            save_path = save_dir / f"{base_filename}.pkl"

        with open(save_path, 'wb') as f:
            pickle.dump(
                {
                    'model': model,
                    'hyperparameters': hyperparameters,
                    'train_results': train_results,
                },
                f,
            )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Save partial responses if provided
    if partial_responses is not None:
        save_partial_responses(partial_responses, save_dir, base_filename)

    # Save scaler if provided
    if scaler is not None:
        save_scaler(scaler, save_dir, base_filename)

    print(f"{model_type.upper()} model saved as {base_filename}")


def load_model(filename, save_dir, load_partial_responses=False, load_scaler=False):
    """
    Load a saved model along with its hyperparameters, training results, and optionally partial responses and scaler.

    Parameters
    ----------
    filename : str
        Filename of the saved model (with extension).
    save_dir : str or Path
        Directory where the model is saved.
    load_partial_responses : bool, optional
        Whether to load partial responses if they exist (default is False).
    load_scaler : bool, optional
        Whether to load scaler if it exists (default is False).

    Returns
    -------
    model : object
        The loaded model.
    hyperparameters : dict
        Dictionary of model hyperparameters.
    train_results : dict
        Dictionary of training results.
    partial_responses : dict or None
        Dictionary of partial responses if load_partial_responses is True and the file exists, else None.
    scaler : object or None
        The scaler object if load_scaler is True and the file exists, else None.

    Raises
    ------
    ValueError
        If the file format is not supported.

    Notes
    -----
    This function can load both PyTorch models (.pt files) and
    scikit-learn models (.pkl files). It will also load partial responses
    and scaler if requested and if they exist.
    """
    import warnings

    save_dir = Path(save_dir)
    file_path = save_dir / filename

    if filename.endswith('.pt'):
        # Load PyTorch model
        checkpoint = torch.load(file_path)
        model_class = checkpoint['model_class']
        hyperparameters = checkpoint['hyperparameters']
        model = model_class(
            input_dim=hyperparameters['input_dim'],
            hidden_units=hyperparameters['hidden_dim'],
            output_dim=hyperparameters['output_dim'],
            mask=hyperparameters.get('mask'),
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        train_results = checkpoint['train_results']
    elif filename.endswith('.pkl'):
        # Load scikit-learn model
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        model = data['model']
        hyperparameters = data['hyperparameters']
        train_results = data['train_results']
    else:
        raise ValueError("Unsupported file format")

    partial_responses = None
    if load_partial_responses:
        pr_filename = file_path.stem + "_partial_responses.pt"
        pr_path = save_dir / pr_filename
        if pr_path.exists():
            partial_responses = torch.load(pr_path)
            print(f"Partial responses loaded from {pr_filename}")
        else:
            print(f"No partial responses file found for {filename}")

    scaler = None
    if load_scaler:
        scaler_filename = file_path.stem + "_scaler.pkl"
        scaler_path = save_dir / scaler_filename
        if scaler_path.exists():
            scaler = load_scaler_from_file(scaler_filename, save_dir)
        else:
            warnings.warn(f"No scaler file found for {filename}", UserWarning)

    if load_scaler:
        return model, hyperparameters, train_results, partial_responses, scaler
    else:
        return model, hyperparameters, train_results, partial_responses


def get_latest_file(directory, prefix):
    """
    Get the latest file in the specified directory with the given prefix,
    excluding partial response files.

    Parameters
    ----------
    directory : str or Path
        The directory to search for files.
    prefix : str
        The prefix of the files to consider.

    Returns
    -------
    str
        The filename of the latest file (excluding partial response files).

    Raises
    ------
    ValueError
        If no matching files are found.
    """
    directory = Path(directory)
    files = [
        f
        for f in os.listdir(directory)
        if f.startswith(prefix) and not f.endswith("_partial_responses.pt")
    ]

    if not files:
        raise ValueError(f"No files found with prefix '{prefix}' in directory: {directory}")

    return max(files, key=lambda f: os.path.getctime(directory / f))


def save_partial_responses(partial_responses, save_dir, base_filename):
    """
    Save partial responses to a file.

    Parameters
    ----------
    partial_responses : dict
        Dictionary containing partial responses for train and test sets.
    save_dir : Path
        Directory to save the partial responses.
    base_filename : str
        Base filename to use for the saved file.

    Returns
    -------
    None
    """
    torch.save(partial_responses, save_dir / f"{base_filename}_partial_responses.pt")
    print(f"Partial responses saved as {base_filename}_partial_responses.pt")


def save_scaler(scaler, save_dir, base_filename):
    """
    Save scaler object to a file.

    Parameters
    ----------
    scaler : object
        The scaler object to save (PRiSMScaler, sklearn scalers, etc.).
    save_dir : Path
        Directory to save the scaler.
    base_filename : str
        Base filename to use for the saved file.

    Returns
    -------
    None
    """
    scaler_path = save_dir / f"{base_filename}_scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved as {base_filename}_scaler.pkl")


def load_scaler_from_file(filename, save_dir):
    """
    Load saved scaler from a file.

    Parameters
    ----------
    filename : str
        Filename of the saved scaler.
    save_dir : str or Path
        Directory where the scaler is saved.

    Returns
    -------
    object
        The loaded scaler object.
    """
    save_dir = Path(save_dir)
    file_path = save_dir / filename
    with open(file_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"Scaler loaded from {filename}")
    return scaler


def load_partial_responses(filename, save_dir):
    """
    Load saved partial responses from a file.

    Parameters
    ----------
    filename : str
        Filename of the saved partial responses.
    save_dir : str or Path
        Directory where the partial responses are saved.

    Returns
    -------
    dict
        Dictionary containing the loaded partial responses.
    """
    save_dir = Path(save_dir)
    file_path = save_dir / filename
    partial_responses = torch.load(file_path)
    print(f"Partial responses loaded from {filename}")
    return partial_responses
