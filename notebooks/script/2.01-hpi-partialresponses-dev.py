#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import numpy as np
import pandas as pd
import torch
import pickle
from prism.config import PROCESSED_DATA_DIR, MODELS_DIR
from prism.PRiSM_functions import partialResponses, normalise, prPlots
from prism.load_models import load_mlp
from prism.pr_save_test import save_partial_responses, test_refactored_partial_responses, load_partial_responses
from prism.partial_responses_func import partial_responses, compute_partial_responses, predict


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

# 

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


# ## Load MLP

filename_mlp = 'mlp_model_20240705_135534'
try:
    mlp, mlp_params, mlp_metrics = load_mlp(filename_mlp, MODELS_DIR)
    print("Model loaded successfully")
    print("Derived Model Structure:")
    print(f"Input dimension: {mlp_params['input_dim']}")
    print(f"Hidden units: {mlp_params['hidden_units']}")
    print(f"Output dimension: {mlp_params['output_dim']}")
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


# ## Test new implementation `partial_responses`

x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32)


t1,t2 = compute_partial_responses(x_train_tensor,mlp)
t1.shape,t2.shape


x_combined = torch.cat([x_train_tensor, x_test_tensor], dim=0)
t3,t4 = compute_partial_responses(x_combined,mlp)


pr_train_benchmark.shape


print(f"First few univariate responses (train): {pr_train_benchmark[:5, :5]}")
print(f"First few bivariate responses (train): {pr_train_benchmark[:5, 11:16]}")
print(f"First few univariate responses (test): {pr_test_benchmark[:5, :5]}")
print(f"First few bivariate responses (test): {pr_test_benchmark[:5, 11:16]}")


## test dirac
test_refactored_partial_responses(partial_responses,tensor_input=True)


# ## Lebesgue test

# Save results from old method
pr_train_leb_old, pr_test_leb_old, bivariate_inputs_leb_old = save_partial_responses(x_train.iloc[:500,:],x_test.iloc[:500,:],mlp,method='lebesgue',device=device, filename="partial_responses_data_leb500.pkl")


# View new lebesgue results
pr_train, pr_test, bivariate_inputs = partial_responses(x_train_tensor[:500,:],x_test_tensor[:500,:],mlp,method='lebesgue',device=device)


# Test lebesgue
test_refactored_partial_responses(partial_responses,filename="partial_responses_data_leb500.pkl", tensor_input=True)




