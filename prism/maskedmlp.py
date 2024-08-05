from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class MaskedMLP(nn.Module):
    def __init__(self, input_dim, hidden_units, output_dim, mask=None):
        super(MaskedMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_units)
        self.activation = nn.Tanh()
        self.fc2 = nn.Linear(hidden_units, output_dim)

        self._initialize_weights()

        if mask is not None:
            # Transpose mask to match the dimensions (hidden_units, input_dim)
            mask = mask.T
            # Ensure mask is a tensor, correctly shaped, and on the right device.
            self.register_buffer('mask_tensor', torch.tensor(mask, dtype=torch.float32))
            if self.mask_tensor.shape != (hidden_units, input_dim):
                raise ValueError("Mask dimension does not match the weights dimension between input and hidden layers.")

    def _initialize_weights(self):
        init.xavier_uniform_(self.fc1.weight, gain=init.calculate_gain('tanh'))
        init.zeros_(self.fc1.bias)
        init.xavier_uniform_(self.fc2.weight, gain=init.calculate_gain('sigmoid'))
        init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return torch.sigmoid(x)
    
    def predict(self, x, device='cpu'):
        self.eval()
        outputnp = False

        # Check if x is a NumPy ndarray or pandas DataFrame and convert it to a PyTorch tensor
        if isinstance(x, np.ndarray):
            x = torch.tensor(x)
            outputnp = True
        elif isinstance(x, pd.DataFrame):
            x = torch.tensor(x.values)
            outputnp = True

        # Move tensor to the specified device
        x = x.to(device).to(torch.float32)

        with torch.no_grad():
            outputs = self(x)

        # Convert outputs to a ndarray if input was a ndarray or DataFrame
        if outputnp:
            if device != 'cpu':
                outputs = outputs.cpu()
            outputs = outputs.numpy()

        return outputs

def apply_mask(model):
    with torch.no_grad():
        model.fc1.weight *= model.mask_tensor


def train_model(x_tr, y_tr, x_ts, y_ts, model, criterion, optimizer, epochs, patience, tolerance, device):
    # If the data is not already tensors, convert them directly from numpy arrays
    if not isinstance(x_tr, torch.Tensor):
        x_tr = torch.from_numpy(x_tr.values.astype(np.float32))
    if not isinstance(y_tr, torch.Tensor):
        y_tr = torch.from_numpy(y_tr.values.astype(np.float32)).view(-1, 1)
    if not isinstance(x_ts, torch.Tensor):
        x_ts = torch.from_numpy(x_ts.values.astype(np.float32))
    if not isinstance(y_ts, torch.Tensor):
        y_ts = torch.from_numpy(y_ts.values.astype(np.float32)).view(-1, 1)

    # Ensure tensors are on the correct device
    x_tr, y_tr, x_ts, y_ts = [x.to(device) for x in [x_tr, y_tr, x_ts, y_ts]]

    # Setup DataLoader for training
    train_dataset = TensorDataset(x_tr, y_tr)
    train_loader = DataLoader(
        train_dataset, batch_size=len(train_dataset), shuffle=False)

    best_loss = np.inf
    trigger_times = 0
    best_epoch = None
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            apply_mask(model)  # Assuming apply_mask is defined elsewhere

        # Early stopping and validation loss check
        model.eval()
        with torch.no_grad():
            val_output = model(x_ts.to(device))
            val_loss = criterion(val_output, y_ts.to(device))
            print(
                f'Epoch {epoch}, Training loss {loss.item()}, Validation loss {val_loss.item()}')

            if val_loss < best_loss - tolerance:
                best_loss = val_loss
                trigger_times = 0
                best_model_state = model.state_dict()  # Save the best model state
                best_epoch = epoch
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    print(f"Early stopping! Epoch {epoch}")
                    if best_model_state is not None:
                        # Restore the best model state
                        print(f"Best model at epoch {best_epoch}")
                        model.load_state_dict(best_model_state)
                    return model

    if best_model_state is not None:
        model.load_state_dict(best_model_state)  # Restore the best model state
    return model


def mlpmask_pytorch(x_tr: pd.DataFrame, y_tr: pd.DataFrame, x_ts: pd.DataFrame, y_ts: pd.DataFrame, n_hidden: int, mask: Optional[np.ndarray] = None, subnet_nodes: int = 1, lr: float = 0.001, weight_decay: float = 0.00001, tolerance: float = 0.001, patience: int = 10, iter: int = 10000, device: str = "cpu", seed: int = 257) -> Any:
    torch.manual_seed(seed)
    device = torch.device(device)

    # Data preprocessing
    x_tr = torch.from_numpy(x_tr.values.astype(np.float32))
    y_tr = torch.from_numpy(y_tr.values.astype(np.float32)).view(-1, 1)
    x_ts = torch.from_numpy(x_ts.values.astype(np.float32))
    y_ts = torch.from_numpy(y_ts.values.astype(np.float32)).view(-1, 1)

    # Model setup
    input_dim = x_tr.shape[1]
    hidden_units = n_hidden * subnet_nodes
    model = MaskedMLP(input_dim, hidden_units, 1, mask).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)

    # Training
    model = train_model(x_tr, y_tr, x_ts, y_ts, model, criterion,
                        optimizer, iter, patience, tolerance, device)
    return model

def generate_mask(betas: np.ndarray, userLambda: int, x_train: pd.DataFrame, bivariate_inputs: np.ndarray, subnet_nodes: int = 5, bivariate_only_if_univariate: bool = False, include_bivariate_as_univariate: bool = True, verbose: bool = True) -> np.ndarray:
    num_features = x_train.shape[1]
    active_indices = np.where(abs(betas[:, userLambda]) > 0.1)[0]
    univ_active = [idx for idx in active_indices if idx < num_features]
    pr_names = x_train.columns[univ_active].tolist()

    # Check for active bivariate features based on enforcement setting
    biv_active_pairs = []
    for idx in active_indices:
        if idx >= num_features:
            first, second = bivariate_inputs[idx - num_features]
            if bivariate_only_if_univariate:
                if first in univ_active and second in univ_active:
                    biv_active_pairs.append((first, second))
                    pr_names.append(
                        f"{x_train.columns[int(first)]} : {x_train.columns[int(second)]}")
            elif include_bivariate_as_univariate:
                univ_active.extend(feature for feature in [first, second] if feature not in univ_active)
                biv_active_pairs.append((first, second))
                pr_names.append(
                    f"{x_train.columns[int(first)]} : {x_train.columns[int(second)]}")
            else:
                biv_active_pairs.append((first, second))
                pr_names.append(
                    f"{x_train.columns[int(first)]} : {x_train.columns[int(second)]}")

    # Create the mask
    univ_active.sort()
    nUniv = len(univ_active)
    nBiv = len(biv_active_pairs)
    mask = np.zeros((num_features, subnet_nodes * (nUniv + nBiv)))

    # Fill mask for univariate features
    for i, idx in enumerate(univ_active):
        mask[int(idx), i * subnet_nodes:(i + 1) * subnet_nodes] = 1

    # Fill mask for bivariate features
    biv_start = nUniv * subnet_nodes
    for i, (first, second) in enumerate(biv_active_pairs):
        start_col = biv_start + i * subnet_nodes
        end_col = start_col + subnet_nodes
        mask[int(first), start_col:end_col] = 1
        mask[int(second), start_col:end_col] = 1

    if verbose:
        print(pr_names)
        fig, ax = plt.subplots()
        heatmap = sns.heatmap(mask, ax=ax)

        # Set the x and y axis labels
        heatmap.set_xlabel('subnet index')
        heatmap.set_ylabel('input features')
        heatmap.set_title('input mask')

        # Set the y-tick labels
        ax.set_yticklabels(x_train.columns.tolist(), rotation=0)
        plt.show()

    return mask, nUniv + nBiv

def get_model_weights_with_biases(model: nn.Module):
    """
    Extracts the weight matrices [W1, W2] from a trained MaskedMLP model,
    including biases concatenated into the weight matrices as the last row.

    Parameters
    ----------
    model : nn.Module
        The trained MaskedMLP model from which to extract the weights.

    Returns
    -------
    list[torch.Tensor]
        A list containing the weight matrices [W1, W2], with biases included.
    """
    # Check if the model is an instance of MaskedMLP for safety
    if not isinstance(model, MaskedMLP):
        raise ValueError("Provided model is not an instance of MaskedMLP.")

    # Retrieve weights and biases
    W1 = model.fc1.weight.data.clone()
    # Reshape bias to a column vector to match hidden units
    b1 = model.fc1.bias.data.clone().view(-1, 1)

    W2 = model.fc2.weight.data.clone()
    # Reshape bias to a column vector to match output units
    b2 = model.fc2.bias.data.clone().view(-1, 1)

    # Check and apply the mask to W1 if it exists
    if hasattr(model, 'mask_tensor'):
        W1 *= model.mask_tensor

    # Concatenate the biases to the weight matrices as the last column
    # Concatenate bias column at the end of W1
    W1_with_bias = torch.cat((W1, b1), dim=1)
    # Concatenate bias column at the end of W2
    W2_with_bias = torch.cat((W2, b2), dim=1)

    return [W1_with_bias.t(), W2_with_bias.t()]

def train_mlp_batched(x_tr, y_tr, x_ts, y_ts, n_hidden, lr=0.001, weight_decay=0.00001, tolerance=0.001, patience=10, max_iter=10000, batch_size=32, device='cpu', seed=257):
    torch.manual_seed(seed)
    device = torch.device(device)

    # Create TensorDatasets and DataLoaders
    train_dataset = TensorDataset(torch.tensor(x_tr.values, dtype=torch.float32), torch.tensor(y_tr.values, dtype=torch.float32).unsqueeze(1))
    test_dataset = TensorDataset(torch.tensor(x_ts.values, dtype=torch.float32), torch.tensor(y_ts.values, dtype=torch.float32).unsqueeze(1))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define the MLP model using MaskedMLP
    model = MaskedMLP(input_dim=x_tr.shape[1], hidden_units=n_hidden, output_dim=1).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Variables for early stopping
    min_loss = float('inf')
    patience_counter = 0
    best_epoch = 0

    # Training loop
    for epoch in range(max_iter):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()

        val_loss /= len(test_loader)
        train_loss /= len(train_loader)

        # Early stopping conditions
        if val_loss < min_loss - tolerance:
            min_loss = val_loss
            best_model_wts = model.state_dict()  # Save the best model
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Stopping early at epoch {best_epoch}")
            model.load_state_dict(best_model_wts)
            break

        if epoch % 1 == 0:
            print(f"Epoch {epoch}: Train loss {train_loss:.4f}, Val loss {val_loss:.4f}")

    return model