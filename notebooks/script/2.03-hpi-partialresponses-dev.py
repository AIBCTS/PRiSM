#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import numpy as np
import pandas as pd
import torch
from prism.config import PROCESSED_DATA_DIR, MODELS_DIR
from prism.legacy import normalise
from prism.load_models import load_mlp
from prism.pr_save_test import save_partial_responses_results, test_refactored_partial_responses, load_partial_responses, compare_and_visualize
from prism.partial_responses import partial_responses, PartialResponseCalculator
from prism.lasso import lasso
from prism.nomogram import nomogram


def print_first_responses(x_train, x_test, rows=5, columns=5, n_univariate=11):
    print(f"First few univariate responses (train):\n{x_train[:rows, :columns]}")
    print(f"First few bivariate responses (train):\n{x_train[:rows, n_univariate:(n_univariate+columns)]}")
    print(f"First few univariate responses (test):\n{x_test[:rows, :columns]}")
    print(f"First few bivariate responses (test):\n{x_test[:rows, n_univariate:(n_univariate+columns)]}")


import matplotlib.pyplot as plt
import seaborn as sns

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


# # Load and setup

get_ipython().run_line_magic('reload_ext', 'autoreload')


seed = 257
np.random.seed(seed)
torch.manual_seed(seed)


import matplotlib.pyplot as plt

def plot_histograms(pr_train_benchmark, x_train, num_subplot_cols=3):
    # Univariate responses only
    m = x_train.shape[1]

    # Calculate number of rows needed for the subplots
    num_subplot_rows = (m + num_subplot_cols - 1) // num_subplot_cols

    # Create a figure with subplots arranged in 3 columns
    fig, axes = plt.subplots(num_subplot_rows, num_subplot_cols, figsize=(12, 4 * num_subplot_rows))
    fig.suptitle(f'Histograms of Benchmark Training Partial Responses')

    # Plot histogram for each column
    for i in range(m):
        row = i // num_subplot_cols
        col = i % num_subplot_cols
        ax = axes[row, col] if num_subplot_rows > 1 else axes[col]
        ax.hist(pr_train_benchmark[:, i], bins=15, edgecolor='black')
        ax.set_title(x_train.columns[i])
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')

    # Remove any unused subplots
    for i in range(m, num_subplot_rows * num_subplot_cols):
        row = i // num_subplot_cols
        col = i % num_subplot_cols
        fig.delaxes(axes[row, col] if num_subplot_rows > 1 else axes[col])

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


# ## Load and preprocess data

data_train = pd.read_csv(PROCESSED_DATA_DIR.joinpath('imputed_dataset1_train.csv'))
data_test = pd.read_csv(PROCESSED_DATA_DIR.joinpath('imputed_dataset1_test.csv'))
data_val = pd.read_csv(PROCESSED_DATA_DIR.joinpath('imputed_dataset1_val.csv'))

data_train_test = pd.concat([data_train,data_test]) # used for axis annotation in some plotting functions

# drop id column
data_train.drop('trr_id_code',axis=1,inplace=True)
data_test.drop('trr_id_code',axis=1,inplace=True)
data_val.drop('trr_id_code',axis=1,inplace=True)

target_col = 'oneyearmort'

x_train0 = data_train.drop(target_col,axis=1)
y_train = data_train[target_col]

x_test0 = data_test.drop(target_col,axis=1)
y_test = data_test[target_col]

x_val0 = data_val.drop(target_col,axis=1)
y_val = data_val[target_col]

[x_train,x_test] = normalise(x_train0,x_test0)
x_val = normalise(x_val0)


x_train.hist(bins=20,figsize=(12,10));


feature_names = [
    'don age',
    'don isch time min',
    'rec age yr',
    'rec creat',
    'rec infect 2wk',
    'rec vent',
    'rec sex',
    'tx year',
    'ICM',
    'NICM',
    'prior tx'
]


# ## Load MLP

filename_mlp = 'mlp_model_20240705_135534'
try:
    mlp, mlp_params, mlp_metrics = load_mlp(filename_mlp, MODELS_DIR)
    print("Model loaded successfully")
    print("Derived Model Structure:")
    print("\nModel Parameters:")
    print(mlp_params)
    print("\nModel Metrics:")
    print(mlp_metrics)
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Extract method and device from mlp_params, with defaults
method = mlp_params.get('method', 'dirac')
device = mlp_params.get('device', 'cpu')


# ## Load benchmark partial responses

pr_load = load_partial_responses()
pr_train_benchmark = pr_load['partial_responses_train']
pr_test_benchmark = pr_load['partial_responses_test']


plot_histograms(pr_train_benchmark,x_train)


# # Test Dirac

x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32, device=device)
x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32,device=device)


pr = PartialResponseCalculator(mlp, method, device, input_dim=x_train_tensor.shape[1])


univariate_train, bivariate_train, bivariate_inputs = pr.calculate(x_train_tensor)


pr_train, pr_test, bivariate_inputs = partial_responses(x_train_tensor,x_test_tensor,mlp,method='dirac',device=device)


print_first_responses(pr_train,pr_test)


print_first_responses(pr_train_benchmark, pr_test_benchmark)


# ## compare dirac results

## test dirac
test_refactored_partial_responses(partial_responses,tensor_input=True)


# # Test Lebesgue CPU 500

# Load results from old method
pr_load_leb= load_partial_responses(MODELS_DIR.joinpath("partial_responses_data_leb500.pkl"))
pr_train_benchmark_leb = pr_load_leb['partial_responses_train']
pr_test_benchmark_leb = pr_load_leb['partial_responses_test']


print_first_responses(pr_train_benchmark_leb, pr_test_benchmark_leb)


# View new lebesgue results
get_ipython().run_line_magic('reload_ext', 'autoreload')

pr_train_leb, pr_test_leb, bivariate_inputs_leb = partial_responses(x_train_tensor[:500,:],x_test_tensor[:500,:],mlp,method='lebesgue',device='cpu')


print_first_responses(pr_train_leb, pr_test_leb)


# Test lebesgue
get_ipython().run_line_magic('reload_ext', 'autoreload')
test_refactored_partial_responses(partial_responses,filename=MODELS_DIR.joinpath("partial_responses_data_leb500.pkl"), tensor_input=True)


# # Test lebesgue CPU 2000

import psutil
import os

def get_system_resources():
    # CPU information
    cpu_count = os.cpu_count()
    cpu_usage = psutil.cpu_percent(interval=1)

    # Memory information
    mem = psutil.virtual_memory()
    available_memory = mem.available / (1024 * 1024)  # Convert to MB

    print(f"CPU Cores: {cpu_count}")
    print(f"CPU Usage: {cpu_usage}%")
    print(f"Available Memory: {available_memory:.2f} MB")

    return {
        "cpu_count": cpu_count,
        "cpu_usage": cpu_usage,
        "available_memory": available_memory,
    }

resources = get_system_resources()


# Load results from old method
pr_load_leb= load_partial_responses(MODELS_DIR.joinpath("pr_cpu_lebesgue_20240816_170150.pkl"))
pr_train_benchmark_leb_2k = pr_load_leb['partial_responses_train']
pr_test_benchmark_leb_2k = pr_load_leb['partial_responses_test']


get_ipython().run_line_magic('reload_ext', 'autoreload')

device = 'cpu'

x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32, device=device)
x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32,device=device)

pr_train_leb_2k, pr_test_leb_2k, bivariate_inputs_leb_2k = partial_responses(
    x_train_tensor[:2000,:],
    x_test_tensor[:2000:],
    mlp,
    method='lebesgue',
    device=device,
    batch_size=2000);


print_first_responses(pr_train_leb_2k, pr_test_leb_2k)


print_first_responses(pr_train_benchmark_leb_2k, pr_test_benchmark_leb_2k)


# To debug differences, check for deviations between benchmark (numpy) and new (tensor) partial responses
compare_and_visualize(pr_train_leb_2k, pr_train_benchmark_leb_2k)


get_ipython().run_line_magic('reload_ext', 'autoreload')
test_refactored_partial_responses(partial_responses,tensor_input=True,filename=MODELS_DIR.joinpath("pr_cpu_lebesgue_20240816_170150.pkl"))


# # Test lebesgue GPU

print(torch.cuda.is_available())


# View new lebesgue results
get_ipython().run_line_magic('reload_ext', 'autoreload')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mlp = mlp.to(device)

x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32, device=device)
x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32,device=device)

pr_train_leb, pr_test_leb, bivariate_inputs_leb = partial_responses(x_train_tensor[:500,:],x_test_tensor[:500,:],mlp,method='lebesgue',device=device)


print(f"First few univariate responses (train): {pr_train_leb[:5, :5]}")
print(f"First few bivariate responses (train): {pr_train_leb[:5, 11:16]}")
print(f"First few univariate responses (test): {pr_test_leb[:5, :5]}")
print(f"First few bivariate responses (test): {pr_test_leb[:5, 11:16]}")


print(f"First few univariate responses (train): {pr_train_benchmark_leb[:5, :5]}")
print(f"First few bivariate responses (train): {pr_train_benchmark_leb[:5, 11:16]}")
print(f"First few univariate responses (test): {pr_test_benchmark_leb[:5, :5]}")
print(f"First few bivariate responses (test): {pr_test_benchmark_leb[:5, 11:16]}")


# ## Test Lebesgue GPU full dataset (batching)

import torch

print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")


# View new lebesgue results
get_ipython().run_line_magic('reload_ext', 'autoreload')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mlp = mlp.to(device)

x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32, device=device)
x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32,device=device)

pr_train_leb_full, pr_test_leb_full, bivariate_inputs_leb_full = partial_responses(x_train_tensor,x_test_tensor,mlp,method='lebesgue',device=device)


pr_train_leb_full.shape


print(f"First few univariate responses (train): {pr_train_leb_full[:5, :5]}")
print(f"First few bivariate responses (train): {pr_train_leb_full[:5, 11:16]}")
print(f"First few univariate responses (test): {pr_test_leb_full[:5, :5]}")
print(f"First few bivariate responses (test): {pr_test_leb_full[:5, 11:16]}")


# Saving
filename_pr_leb_full = MODELS_DIR.joinpath('partial_responses_data_leb_all.pkl')

save_partial_responses_results(x_train_tensor, x_test_tensor, mlp, pr_train_leb_full, pr_test_leb_full, bivariate_inputs_leb_full, 'lebesgue', device, filename=filename_pr_leb_full)


# ## Lebesgue results nomogram

lasso_leb_full = lasso_results = lasso(
    pr_train_leb_full, 
    pr_test_leb_full, 
    y_train, 
    y_test, 
    bivariate_inputs_leb_full,
    feature_names=feature_names,
    nlambda=25, 
    min_lambda=0.1,
    max_lambda=100,
    tol=1e-4,
    batch_size=2
)


lasso_leb_full.select_lambda(8)
lasso_leb_full.plot_lambda_path()
lasso_leb_full.plot_feature_importance()

