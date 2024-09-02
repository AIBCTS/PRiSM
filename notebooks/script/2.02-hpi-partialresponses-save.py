#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import numpy as np
import pandas as pd
from datetime import datetime

from prism.config import PROCESSED_DATA_DIR, MODELS_DIR
from prism.load_models import load_mlp
from prism.pr_save_test import save_partial_responses
from prism.legacy import normalise
from prism.obsolete.partial_responses_old import partialResponses
from prism.obsolete.prplots_old import prPlots


get_ipython().run_line_magic('reload_ext', 'autoreload')


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


# ## Load MLP

filename_mlp = 'mlp_model_20240705_135534'
try:
    model, params, metrics = load_mlp(filename_mlp, MODELS_DIR)
    print("Model loaded successfully")
    print("Derived Model Structure:")
    print("\nModel Parameters:")
    print(params)
    print("\nModel Metrics:")
    print(metrics)
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Extract method and device from params, with defaults
method = params.get('method', 'dirac')
device = params.get('device', 'cpu')


# ## Calculate and save partial responses (existing benchmark implementation)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
pr_train_benchmark, pr_test_benchmark, bivariate_inputs = save_partial_responses(x_train, x_test, model, method=method, device=device, filename="partial_responses_data_{timestamp}.pkl")

print(f"Partial responses calculated using method '{method}' on device '{device}' and saved.")


pr_train_benchmark.shape


x_train.columns[2]


import matplotlib.pyplot as plt

# Univariate responses only
n = pr_train_benchmark.shape[0]
m = 11 

# Calculate number of rows needed for the subplots
num_subplot_cols = 3
num_subplot_rows = (m + num_subplot_cols - 1) // num_subplot_cols

# Create a figure with subplots arranged in 3 columns
fig, axes = plt.subplots(num_subplot_rows, num_subplot_cols, figsize=(12, 4 * num_subplot_rows))
fig.suptitle(f'Histograms of Benchmark Training Partial Responses')

# Plot histogram for each column
for i in range(m):
    row = i // num_subplot_cols
    col = i % num_subplot_cols
    ax = axes[row, col]
    ax.hist(pr_train_benchmark[:, i], bins=15, edgecolor='black')
    ax.set_title(x_train.columns[i])
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')

# Adjust layout and show the plot
plt.tight_layout()
plt.show()


# ## Plot partial responses

pr_test_benchmark.shape, x_test.shape


betas_univariate = np.zeros([pr_train_benchmark.shape[1],1])
betas_univariate[:x_train.shape[1],0] = 1


prPlots(betas_univariate, 0, x_train0, x_train, data_train_test, model, bivariate_inputs, n_steps = 15, sd_scale=2, method=method, device=device)


# ## Test the saved partial responses to verify test function

test_result = test_refactored_partial_responses(partialResponses, filename="partial_responses_data.pkl")

if test_result:
    print("Test passed.")
else:
    print("Test failed.")


# # Save Lebesgue CPU 2000

import psutil

# Get memory information
mem = psutil.virtual_memory()

# Process-specific memory info
process = psutil.Process()
process_mem = process.memory_info()

print(f"Total memory: {mem.total / (1024 * 1024):.2f} MB")
print(f"Available memory: {mem.available / (1024 * 1024):.2f} MB")
print(f"Used memory: {mem.used / (1024 * 1024):.2f} MB")
print(f"Memory usage percentage: {mem.percent:.2f}%")

# Check for Linux-specific attributes
if hasattr(mem, 'cached'):
    print(f"Cached memory: {mem.cached / (1024 * 1024):.2f} MB")
if hasattr(mem, 'buffers'):
    print(f"Buffered memory: {mem.buffers / (1024 * 1024):.2f} MB")

print("\nCurrent process memory usage:")
print(f"RSS (Resident Set Size): {process_mem.rss / (1024 * 1024):.2f} MB")
print(f"VMS (Virtual Memory Size): {process_mem.vms / (1024 * 1024):.2f} MB")

# Check for platform-specific attributes
if hasattr(process_mem, 'shared'):
    print(f"Shared memory: {process_mem.shared / (1024 * 1024):.2f} MB")


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
device = 'cpu'
method = 'lebesgue'
pr_train_benchmark, pr_test_benchmark, bivariate_inputs = save_partial_responses(x_train.iloc[:2000], x_test.iloc[:2000], model, method=method, device=device, filename=f"pr_{device}_{method}_{timestamp}.pkl")

print(f"Partial responses calculated using method '{method}' on device '{device}' and saved.")




