"""
Masked Multi-Layer Perceptron (MLP) Implementation

This module implements a Masked MLP, which is a neural network architecture that allows
for selective connectivity between layers through the use of a binary mask.

Key Components
--------------
1. MaskedMLP : Class
    A neural network model that supports masked connections between the input and hidden layer.

2. train_mlp : Function
    Main entry point for training a Masked or regular MLP.

3. train_model : Function
    Handles the training loop, including early stopping and loss plotting.

4. apply_mask : Function
    Applies the mask to the weights after each optimization step.

The masking process works as follows:
1. A binary mask is created based on the results of a LASSO feature selection process.
2. The mask is incorporated into the MaskedMLP model during initialization.
3. After each optimization step, the mask is reapplied to ensure the desired connectivity is maintained.

This implementation allows a specific structure to be enforced by based on feature importance, as determined
by the LASSO process.

Key Features
------------
- Support for both masked and regular MLPs
- Flexible input formats (numpy arrays, pandas DataFrames, PyTorch tensors)
- Early stopping to prevent overfitting
- Optional loss plotting for visualization of training progress
- Batch training support for handling large datasets

Notes
-----
- The mask is typically generated using the `get_mask()` method of the `LassoResultsManager`
  class (defined in `lasso_results.py`).
- This implementation uses PyTorch for efficient computation and GPU support.
- The architecture is designed for binary classification tasks, using sigmoid
  activation in the output layer.

Example usage
-------------
>>> lasso_results = lasso(...)  # Assume this is already performed
>>> mask = lasso_results.get_mask()
>>> model = train_mlp(x_train, y_train, x_test, y_test, n_hidden=50, mask=mask)
>>> probabilities = model.predict_proba(x_new)

See individual function and class docstrings for more detailed information.
"""

from typing import Optional, Union

from torch.utils.data import DataLoader, TensorDataset

try:
    from IPython.display import clear_output
except ImportError:
    clear_output = None

import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from sklearn.metrics import roc_auc_score

# Setup logging for training progress
logger = logging.getLogger(__name__)


class MaskedMLP(nn.Module):
    """
    A Masked Multi-Layer Perceptron (MLP) neural network. The class extends the base neural network class from PyTorch, nn.Module, so it fits the structure expected by PyTorch.

    This clas implements a feedforward neural network with one hidden layer and optional masking.
    The network uses Tanh activation for the hidden layer and Sigmoid activtion for the output layer (for binary classification tasks).

    Parameters
    ----------
    input_dim : int
        The number of input features.
    hidden_units : int
        The number of neurons in the hidden layer.
    output_dim : int
        The number of output neurons (typically 1 for binary classification).
    mask : np.ndarray or torch.Tensor, optional
        A binary mask to apply to the weights of the first layer. If provided,
        it should have shape (hidden_units, input_dim). Default is None.

    Attributes
    ----------
    fc1 : torch.nn.Linear
        The first (input to hidden) fully connected layer.
    activation : torch.nn.Tanh
        The activation function for the hidden layer.
    fc2 : torch.nn.Linear
        The second (hidden to output) fully connected layer.
    mask_tensor : torch.Tensor
        The mask applied to the weights of fc1, if a mask is provided.

    Notes
    -----
    - The mask, if provided, is applied after each optimization step to enforce the desired connectivity.
    - The network uses Xavier uniform initialization for weights and zero initialization for biases.
    """

    def __init__(self, input_dim, hidden_units, output_dim, mask=None):
        super(MaskedMLP, self).__init__()

        # Define the structure of the neural network
        self.fc1 = nn.Linear(input_dim, hidden_units)
        self.activation = nn.Tanh()
        self.fc2 = nn.Linear(hidden_units, output_dim)

        # Initialize the weights using best practices for tanh and sigmoid activations
        self._initialize_weights()

        if mask is not None:
            # Convert mask to a tensor if it's not already
            self.register_buffer('mask_tensor', torch.tensor(mask, dtype=torch.float32))

            # Check if the mask needs to be transposed
            if self.mask_tensor.shape == (input_dim, hidden_units):
                warnings.warn(
                    "Input mask was transposed to match the expected shape (hidden_units, input_dim)."
                )
                self.mask_tensor = self.mask_tensor.T
            elif self.mask_tensor.shape != (hidden_units, input_dim):
                raise ValueError(
                    f"Mask shape {self.mask_tensor.shape} does not match expected shape (hidden_units, input_dim) = ({hidden_units}, {input_dim}) or (input_dim, hidden_units) = ({input_dim}, {hidden_units})"
                )

    def _initialize_weights(self):
        """
        Initialize the weights of the neural network.

        This method uses Xavier uniform initialization for the weights and zeros for the biases.
        Xavier initialization helps in maintaining the scale of gradients roughly the same in all layers.
        """
        init.xavier_uniform_(self.fc1.weight, gain=init.calculate_gain('tanh'))
        init.zeros_(self.fc1.bias)
        init.xavier_uniform_(self.fc2.weight, gain=init.calculate_gain('sigmoid'))
        init.zeros_(self.fc2.bias)

    def forward(self, x):
        """
        Perform a forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output of the network after passing through all layers and activations.
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return torch.sigmoid(x)

    @torch.no_grad()
    def predict_proba(
        self, x: Union[np.ndarray, pd.DataFrame, torch.Tensor], device: Optional[str] = None
    ) -> torch.Tensor:
        """
        Probability of the positive class for each sample.

        This method sets the model to evaluation mode and performs a forward pass
        without computing gradients.

        Parameters
        ----------
        x : Union[np.ndarray, pd.DataFrame, torch.Tensor]
            The input data for prediction. Can be a NumPy array, Pandas DataFrame, or PyTorch tensor.
        device : str, optional
            The PyTorch device to use for computation. If None, uses the current model's device.

        Returns
        -------
        torch.Tensor
            P(y=1) per sample, shape (n_samples, 1).

        Notes
        -----
        - The input is automatically converted to a PyTorch tensor if it isn't already.
        - The output is returned on the same device as the input.
        - Unlike sklearn's predict_proba, this returns a torch tensor with only the
          positive-class column (the pipeline is torch-first and device-aware).
        """
        self.eval()

        # Determine the device to use
        if device is None:
            device = next(self.parameters()).device
        else:
            device = torch.device(device)

        # Move the model to the specified device
        self.to(device)

        # Convert input to tensor if necessary
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        elif isinstance(x, pd.DataFrame):
            x = torch.from_numpy(x.values)
        elif not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a numpy array, pandas DataFrame, or PyTorch tensor")

        # Ensure input is on the correct device and dtype
        x = x.to(device=device, dtype=torch.float32)

        # Perform prediction
        outputs = self(x)

        return outputs  # Return on the same device as input

    def predict(
        self,
        x: Union[np.ndarray, pd.DataFrame, torch.Tensor],
        device: Optional[str] = None,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """
        Binary class labels: (predict_proba(x, device) >= threshold).

        Use predict_proba for the underlying probabilities.

        Returns
        -------
        torch.Tensor
            Labels in {0, 1} with dtype long, same shape as predict_proba output.
        """
        return (self.predict_proba(x, device) >= threshold).long()

    @torch.no_grad()
    def predict_proba_numpy(
        self, x: Union[np.ndarray, pd.DataFrame], device: Optional[str] = None
    ) -> np.ndarray:
        """
        This method is a wrapper around the `predict_proba` method that automatically
        converts the output to a NumPy array.
        """
        return self.predict_proba(x, device).cpu().numpy()


def apply_mask(model):
    """
    Apply the mask to the weights of the first layer of a MaskedMLP model.

    This function is typically called after each optimization step to ensure
    that the mask is properly applied to the weights, effectively zeroing out
    the connections that should be inactive according to the mask.

    Parameters
    ----------
    model : MaskedMLP
        The MaskedMLP model to which the mask should be applied.

    Returns
    -------
    None

    Notes
    -----
    - Assumes that the model has a 'mask_tensor' attribute.
    - Applied element-wise to the weights of the first fully connected layer (fc1).
    - Performed in-place and does not compute gradients.
    - Applying the mask after each optimization step is done to maintain
      the desired network structure throughout training.
    """
    with torch.no_grad():
        model.fc1.weight *= model.mask_tensor


def train_model(
    x_tr,
    y_tr,
    x_ts,
    y_ts,
    model,
    criterion,
    optimizer,
    epochs,
    patience,
    tolerance,
    device,
    plot_loss=False,
    plot_update_epochs=10,
    batch_size=None,
    verbose=True,
):
    """
    Train a neural network model with early stopping and optional loss plotting.

    This function handles the training loop, including batch processing, loss calculation,
    backpropagation, and model evaluation. It also implements early stopping to prevent
    overfitting and can optionally plot the training progress.

    Parameters
    ----------
    x_tr : torch.Tensor
        Training feature data.
    y_tr : torch.Tensor
        Training target data.
    x_ts : torch.Tensor
        Test feature data for validation.
    y_ts : torch.Tensor
        Test target data for validation.
    model : torch.nn.Module
        The neural network model to be trained.
    criterion : torch.nn.Module
        The loss function to be used (e.g., nn.BCELoss()).
    optimizer : torch.optim.Optimizer
        The optimization algorithm (e.g., torch.optim.Adam).
    epochs : int
        Maximum number of training epochs.
    patience : int
        Number of epochs to wait for improvement before early stopping.
    tolerance : float
        Minimum improvement in validation loss to be considered significant.
    device : torch.device
        The PyTorch device to run the training on.
    plot_loss : bool, optional
        Whether to plot the training and validation loss. Default is False.
    plot_update_epochs : int, optional
        Number of epochs between plot updates. Default is 10.
    batch_size : int, optional
        Batch size for training. If None, full-batch training is used.
    verbose : bool, optional
        Whether to print training progress to console. Default is True.
        Set to False during hyperparameter tuning to avoid cluttering output.
        Training progress is always logged to the logger regardless of this setting.

    Returns
    -------
    torch.nn.Module
        The trained model.

    Notes
    -----
    - The function uses DataLoader for efficient batch processing.
    - Early stopping is implemented to stop training when validation loss stops improving.
    - If a mask is present in the model, it's reapplied after each optimization step.
    - The best model (based on validation loss) is saved and returned at the end of training.
    """
    # Convert data to tensors if they aren't already
    x_tr = (
        torch.tensor(x_tr, dtype=torch.float32, device=device)
        if not isinstance(x_tr, torch.Tensor)
        else x_tr
    )
    y_tr = (
        torch.tensor(y_tr, dtype=torch.float32, device=device)
        if not isinstance(y_tr, torch.Tensor)
        else y_tr
    )
    x_ts = (
        torch.tensor(x_ts, dtype=torch.float32, device=device)
        if not isinstance(x_ts, torch.Tensor)
        else x_ts
    )
    y_ts = (
        torch.tensor(y_ts, dtype=torch.float32, device=device)
        if not isinstance(y_ts, torch.Tensor)
        else y_ts
    )

    # Ensure y tensors have shape (n_samples, 1)
    y_tr = y_tr.view(-1, 1) if y_tr.dim() == 1 else y_tr
    y_ts = y_ts.view(-1, 1) if y_ts.dim() == 1 else y_ts

    # Create DataLoaders for efficient batch processing
    train_dataset = TensorDataset(x_tr, y_tr)
    test_dataset = TensorDataset(x_ts, y_ts)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size if batch_size else len(train_dataset), shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size if batch_size else len(test_dataset), shuffle=False
    )

    # Initialize variables for early stopping and loss tracking
    best_loss = float('inf')
    trigger_times = 0
    best_epoch = None
    best_model_state = None
    early_stop_epoch = None
    train_losses = []
    test_losses = []
    train_aurocs = []
    test_aurocs = []
    plot_epochs = []

    for epoch in range(epochs):
        model.train()  # Set model to training mode
        total_train_loss = 0
        total_train_samples = 0

        # Training loop
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()  # Clear previous gradients
            output = model(x_batch)  # Forward pass
            loss = criterion(output, y_batch)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            if hasattr(model, 'mask_tensor'):
                apply_mask(model)  # Reapply mask if present
            total_train_loss += loss.item() * x_batch.size(0)
            total_train_samples += x_batch.size(0)

        avg_train_loss = total_train_loss / total_train_samples

        # Validation loop
        model.eval()  # Set model to evaluation mode
        total_val_loss = 0
        total_val_samples = 0

        with torch.no_grad():  # Disable gradient computation
            for x_batch, y_batch in test_loader:
                val_output = model(x_batch)
                val_loss = criterion(val_output, y_batch)
                total_val_loss += val_loss.item() * x_batch.size(0)
                total_val_samples += x_batch.size(0)

        avg_val_loss = total_val_loss / total_val_samples

        # Calculate AUROC (model is already in eval mode, no gradients)
        with torch.no_grad():
            train_pred = model(x_tr).cpu().numpy()
            test_pred = model(x_ts).cpu().numpy()
        train_auroc = roc_auc_score(y_tr.cpu().numpy(), train_pred)
        test_auroc = roc_auc_score(y_ts.cpu().numpy(), test_pred)

        # Record losses and AUROCs
        train_losses.append(avg_train_loss)
        test_losses.append(avg_val_loss)
        train_aurocs.append(train_auroc)
        test_aurocs.append(test_auroc)
        plot_epochs.append(epoch)

        # Plot loss and AUROC if requested
        if plot_loss and (epoch % plot_update_epochs == 0 or trigger_times >= patience):
            if clear_output is not None:
                clear_output(wait=True)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

            # Loss subplot (top)
            ax1.plot(
                plot_epochs,
                train_losses,
                label='Train Loss',
                marker='o',
                linestyle='-',
                markersize=2,
            )
            ax1.plot(
                plot_epochs,
                test_losses,
                label='Test Loss',
                marker='s',
                linestyle='-',
                markersize=2,
            )
            if early_stop_epoch is not None:
                ax1.axvline(
                    x=early_stop_epoch,
                    color='r',
                    linestyle='--',
                    label=f'Early Stop ({early_stop_epoch})',
                )
            if best_epoch is not None:
                ax1.axvline(
                    x=best_epoch, color='g', linestyle='--', label=f'Best Epoch ({best_epoch})'
                )
            ax1.set_ylabel('Loss')
            ax1.set_title('Training and Test Loss')
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.7)

            # AUROC subplot (bottom)
            ax2.plot(
                plot_epochs,
                train_aurocs,
                label='Train AUROC',
                marker='o',
                linestyle='-',
                markersize=2,
            )
            ax2.plot(
                plot_epochs,
                test_aurocs,
                label='Test AUROC',
                marker='s',
                linestyle='-',
                markersize=2,
            )
            if early_stop_epoch is not None:
                ax2.axvline(x=early_stop_epoch, color='r', linestyle='--')
            if best_epoch is not None:
                ax2.axvline(x=best_epoch, color='g', linestyle='--')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('AUROC')
            ax2.set_title('Training and Test AUROC')
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.7)

            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.1)
        elif epoch % 10 == 0 and not plot_loss:
            msg = (
                f'Epoch {epoch}, Train Loss {avg_train_loss:.6f}, Test Loss {avg_val_loss:.6f}, '
                f'Train AUROC {train_auroc:.4f}, Test AUROC {test_auroc:.4f}'
            )
            logger.debug(msg)
            if verbose:
                print(msg)

        # Early stopping check
        if avg_val_loss < best_loss - tolerance:
            best_loss = avg_val_loss
            trigger_times = 0
            best_model_state = model.state_dict()
            best_epoch = epoch
        else:
            trigger_times += 1
            if trigger_times >= patience:
                early_stop_epoch = epoch
                logger.info(f"Early stopping at epoch {epoch}")
                if verbose:
                    print(f"Early stopping! Epoch {epoch}")
                if best_model_state is not None:
                    logger.info(f"Best model at epoch {best_epoch}")
                    if verbose:
                        print(f"Best model at epoch {best_epoch}")
                    model.load_state_dict(best_model_state)

                # Final plot update if early stopping occurs
                if plot_loss:
                    if clear_output is not None:
                        clear_output(wait=True)
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

                    # Loss subplot (top)
                    ax1.plot(
                        plot_epochs,
                        train_losses,
                        label='Train Loss',
                        marker='o',
                        linestyle='-',
                        markersize=2,
                    )
                    ax1.plot(
                        plot_epochs,
                        test_losses,
                        label='Test Loss',
                        marker='s',
                        linestyle='-',
                        markersize=2,
                    )
                    ax1.axvline(
                        x=early_stop_epoch,
                        color='r',
                        linestyle='--',
                        label=f'Early Stop ({early_stop_epoch})',
                    )
                    ax1.axvline(
                        x=best_epoch, color='g', linestyle='--', label=f'Best Epoch ({best_epoch})'
                    )
                    ax1.set_ylabel('Loss')
                    ax1.set_title('Training and Test Loss')
                    ax1.legend()
                    ax1.grid(True, linestyle='--', alpha=0.7)

                    # AUROC subplot (bottom)
                    ax2.plot(
                        plot_epochs,
                        train_aurocs,
                        label='Train AUROC',
                        marker='o',
                        linestyle='-',
                        markersize=2,
                    )
                    ax2.plot(
                        plot_epochs,
                        test_aurocs,
                        label='Test AUROC',
                        marker='s',
                        linestyle='-',
                        markersize=2,
                    )
                    ax2.axvline(x=early_stop_epoch, color='r', linestyle='--')
                    ax2.axvline(x=best_epoch, color='g', linestyle='--')
                    ax2.set_xlabel('Epoch')
                    ax2.set_ylabel('AUROC')
                    ax2.set_title('Training and Test AUROC')
                    ax2.legend()
                    ax2.grid(True, linestyle='--', alpha=0.7)

                    plt.tight_layout()
                    plt.show(block=False)
                    plt.pause(0.1)

                model.training_history_ = {
                    'train_loss': train_losses,
                    'test_loss': test_losses,
                    'train_auroc': train_aurocs,
                    'test_auroc': test_aurocs,
                    'best_epoch': best_epoch,
                    'early_stop_epoch': early_stop_epoch,
                }
                return model

    # Load best model if early stopping didn't occur
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.training_history_ = {
        'train_loss': train_losses,
        'test_loss': test_losses,
        'train_auroc': train_aurocs,
        'test_auroc': test_aurocs,
        'best_epoch': best_epoch,
        'early_stop_epoch': early_stop_epoch,
    }
    return model


def scale_learning_rate(base_lr, batch_size, base_batch_size=1024):
    """
    Scale the learning rate linearly with the batch size.

    Args:
        base_lr (float): The base learning rate for the base batch size
        batch_size (int): The current batch size
        base_batch_size (int): The base batch size (default is 1024)

    Returns:
        float: The scaled learning rate
    """
    return base_lr * (batch_size / base_batch_size)


def train_mlp(
    x_tr: Union[np.ndarray, pd.DataFrame, pd.Series, torch.Tensor],
    y_tr: Union[np.ndarray, pd.DataFrame, pd.Series, torch.Tensor],
    x_ts: Union[np.ndarray, pd.DataFrame, pd.Series, torch.Tensor],
    y_ts: Union[np.ndarray, pd.DataFrame, pd.Series, torch.Tensor],
    n_hidden: int,
    mask: Optional[np.ndarray] = None,
    subnet_nodes: int = 1,
    lr: float = 0.001,
    weight_decay: float = 0.00001,
    tolerance: float = 0.001,
    patience: int = 10,
    max_iter: int = 4000,
    batch_size: Optional[int] = None,
    scale_lr: bool = True,
    device: str = 'cpu',
    seed: int = 257,
    plot_loss: bool = True,
    plot_update_epochs: int = 10,
    verbose: bool = True,
) -> MaskedMLP:
    """
    Train a Masked or regular Multi-Layer Perceptron (MLP) with optional batching.

    This function sets up and trains either a masked MLP (if a mask is provided) or a regular MLP.
    It handles model initialization, and training loop management.

    Parameters:
    -----------
    x_tr : Union[np.ndarray, pd.DataFrame, pd.Series, torch.Tensor]
        Training features. Can be in various formats, which will be converted to PyTorch tensors.
    y_tr : Union[np.ndarray, pd.DataFrame, pd.Series, torch.Tensor]
        Training labels. Will be converted to PyTorch tensors.
    x_ts : Union[np.ndarray, pd.DataFrame, pd.Series, torch.Tensor]
        Test features. Used for validation during training.
    y_ts : Union[np.ndarray, pd.DataFrame, pd.Series, torch.Tensor]
        Test labels. Used for validation during training.
    n_hidden : int
        Number of hidden units in the MLP. For masked MLPs, this is multiplied by subnet_nodes.
    mask : Optional[np.ndarray], default=None
        Binary mask for the MLP. If provided, a masked MLP is trained; otherwise, a regular MLP is used.
    subnet_nodes : int, default=1
        Number of nodes per subnet in a masked MLP. Only used if a mask is provided.
    lr : float, default=0.001
        Initial learning rate for the optimizer.
    weight_decay : float, default=0.00001
        L2 regularization factor. Helps prevent overfitting.
    tolerance : float, default=0.001
        Minimum improvement in validation loss to reset early stopping counter.
    patience : int, default=10
        Number of epochs with no improvement after which training will be stopped.
    max_iter : int, default=4000
        Maximum number of training epochs.
    batch_size : Optional[int], default=None
        Batch size for training. If None, full-batch training is used.
    scale_lr : bool, default=True
        Whether to scale the learning rate based on batch size. When True, LR is scaled
        by (batch_size / 1024). Set to False for MLP training to match tuning behavior.
    device : str, default='cpu'
        The PyTorch device to use for training.
    seed : int, default=257
        Random seed for reproducibility.
    plot_loss : bool, default=True
        Whether to plot the loss during training.
    plot_update_epochs : int, default=10
        Number of epochs between plot updates.
    verbose : bool, default=True
        Whether to print training progress to console. Set to False during
        hyperparameter tuning to avoid cluttering output. Training progress
        is always logged to the logger regardless of this setting.

    Returns:
    --------
    MaskedMLP
        Trained MLP model (either masked or regular).

    Notes:
    ------
    - The function uses Adam optimizer and Binary Cross-Entropy Loss.
    - Early stopping is implemented to prevent overfitting.
    - If a mask is provided, the number of hidden units is adjusted based on the mask structure.
    """
    # Set the random seed for reproducibility
    torch.manual_seed(seed)
    device = torch.device(device)

    # Helper function to convert various input types to PyTorch tensors. TODO: move out and use in train_model() too.
    def to_tensor(x: Union[np.ndarray, pd.DataFrame, pd.Series, torch.Tensor]) -> torch.Tensor:
        if isinstance(x, (pd.DataFrame, pd.Series)):
            return torch.tensor(x.values, dtype=torch.float32, device=device)
        elif isinstance(x, np.ndarray):
            return torch.tensor(x, dtype=torch.float32, device=device)
        elif isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=torch.float32)
        else:
            return torch.tensor(x, dtype=torch.float32, device=device)

    # Convert inputs to tensors and ensure correct shape
    x_tr = to_tensor(x_tr)
    y_tr = to_tensor(y_tr).view(-1, 1)  # Ensure y is a column vector
    x_ts = to_tensor(x_ts)
    y_ts = to_tensor(y_ts).view(-1, 1)

    # Determine the input dimension and number of hidden units
    input_dim = x_tr.shape[1]
    hidden_units = n_hidden * subnet_nodes if mask is not None else n_hidden

    # Initialize the MaskedMLP model
    model = MaskedMLP(input_dim, hidden_units, 1, mask).to(device)

    # Set up the loss function (Binary Cross-Entropy)
    criterion = nn.BCELoss()

    # Adjust learning rate if batch training is used and scale_lr is True
    if batch_size is not None and scale_lr:
        lr = scale_learning_rate(lr, batch_size)

    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Train the model
    trained_model = train_model(
        x_tr,
        y_tr,
        x_ts,
        y_ts,
        model,
        criterion,
        optimizer,
        epochs=max_iter,
        patience=patience,
        tolerance=tolerance,
        device=device,
        plot_loss=plot_loss,
        plot_update_epochs=plot_update_epochs,
        batch_size=batch_size,
        verbose=verbose,
    )

    return trained_model
