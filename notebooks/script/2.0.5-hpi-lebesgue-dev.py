#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import cProfile
import pstats
import io
import os
from datetime import datetime
import tracemalloc
import torch
import pandas as pd
import psutil
from memory_profiler import profile
import time

from prism.config import PROCESSED_DATA_DIR, MODELS_DIR
from prism.PRiSM_functions import normalise
from prism.partial_responses import PartialResponseCalculator
from prism.load_models import load_mlp


# Load and preprocess data
data_train = pd.read_csv(PROCESSED_DATA_DIR.joinpath('imputed_dataset1_train.csv'))
data_test = pd.read_csv(PROCESSED_DATA_DIR.joinpath('imputed_dataset1_test.csv'))

# drop id column and target column
target_col = 'oneyearmort'
data_train.drop(['trr_id_code', target_col], axis=1, inplace=True)
data_test.drop(['trr_id_code', target_col], axis=1, inplace=True)

# Normalize data
x_train, x_test = normalise(data_train, data_test)

# Load MLP model
filename_mlp = 'mlp_model_20240705_135534'
try:
    mlp, mlp_params, _ = load_mlp(filename_mlp, MODELS_DIR)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Extract method and device from mlp_params, with defaults
method = mlp_params.get('method', 'dirac')
device = mlp_params.get('device', 'cpu')


@profile
def _calculate_lebesgue_profiled(calculator, x):
    return calculator._calculate_lebesgue(x)

def run_profiling(n_samples=500):
    # Use the first n_samples from x_train
    x = torch.tensor(x_train.iloc[:n_samples].values, dtype=torch.float32).to(device)
    
    # Initialize PartialResponseCalculator with the loaded MLP model
    calculator = PartialResponseCalculator(mlp, method='lebesgue', device=device, input_dim=x.shape[1], x_train=x)

    # Start memory tracing
    tracemalloc.start()

    # Profile the _calculate_lebesgue method
    pr = cProfile.Profile()
    pr.enable()
    
    # Run the method
    _calculate_lebesgue_profiled(calculator, x)
    
    pr.disable()

    # Get memory usage
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Get CPU usage
    cpu_percent = psutil.cpu_percent()

    # Get GPU usage if available
    gpu_percent = None
    if device == 'cuda' and torch.cuda.is_available():
        gpu_percent = torch.cuda.utilization()

    # Create a string buffer to capture the output
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats()
    
    # Get the profiling results as a string
    profiling_results = s.getvalue()
    
    # Add memory and CPU/GPU usage to the results
    profiling_results += f"\nMemory usage: current={current/1e6:.1f}MB, peak={peak/1e6:.1f}MB\n"
    profiling_results += f"CPU usage: {cpu_percent}%\n"
    if gpu_percent is not None:
        profiling_results += f"GPU usage: {gpu_percent}%\n"
    
    # Print the results to console
    print(profiling_results)
    
    # Save the results to a file and return the filepath
    return save_results(profiling_results, n_samples, device)

def save_results(results, n_samples, device):
    # Create a 'profiling_results' directory within MODELS_DIR if it doesn't exist
    profiling_dir = os.path.join(MODELS_DIR, 'profiling_results')
    if not os.path.exists(profiling_dir):
        os.makedirs(profiling_dir)
    
    # Generate a filename with timestamp, sample size, and device
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"lebesgue_profile_{timestamp}_samples{n_samples}_{device}.txt"
    filepath = os.path.join(profiling_dir, filename)
    
    # Save the results to the file
    with open(filepath, 'w') as f:
        f.write(results)
    
    print(f"Profiling results saved to: {filepath}")
    return filepath


# torch.no_grad() in _calculate_lebesgue
get_ipython().run_line_magic('reload_ext', 'autoreload')
filepath = run_profiling(n_samples=500)


torch.tensor(x_train.iloc[:500].values, dtype=torch.float32).to(device).shape

