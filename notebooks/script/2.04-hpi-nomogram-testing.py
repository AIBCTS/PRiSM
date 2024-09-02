#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import numpy as np
import pandas as pd
import torch
import pickle
from prism.config import PROCESSED_DATA_DIR, MODELS_DIR
from prism.legacy import normalise
from prism.load_models import load_mlp, load_prn
from prism.nomogram import nomogram


get_ipython().run_line_magic('reload_ext', 'autoreload')


seed = 257
np.random.seed(seed)
torch.manual_seed(seed)


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

x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32)


# ## MLP nomogram

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


x_train0_median = torch.tensor(x_train0.median().values, dtype=torch.float32, device=device)
x_train0_std = torch.tensor(x_train0.std().values, dtype=torch.float32,device=device)


device='cpu'
method='dirac'
betas_univariate = torch.zeros([66,1], dtype=torch.float32,device=device)
betas_univariate[:x_train.shape[1],0] = 1


univariate_responses, bivariate_responses, x_univariate, x_bivariate = nomogram(
    betas_univariate, 0, x_train_tensor, x_train0_median, x_train0_std, mlp, 
    n_steps=15, sd_scale=2, method="dirac", device="cpu", categorical_threshold=15
)


# ## PRN nomogram

filename_prn = 'prn_model_20240705_140928'
try:
    prn, prn_params, prn_metrics = load_prn(filename_prn, MODELS_DIR)
    print("Model architecture:")
    print(prn)

    print("\nModel parameters:")
    for key, value in prn_params.items():
        print(f"{key}: {value}")

    print("\nModel metrics:")
    for key, value in prn_metrics.items():
        print(f"{key}: {value}")

except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Extract method and device from prn_params, with defaults
method = prn_params.get('method', 'dirac')
device = prn_params.get('device', 'cpu')


univariate_responses_senn, bivariate_responses_senn, x_univariate_senn, x_bivariate_senn = nomogram(
    betas_univariate, 0, x_train_tensor, x_train0_median, x_train0_std, prn, 
    n_steps=15, sd_scale=2, method=method, device=device, categorical_threshold=15
)


n_univ = x_train.shape[1]
n_biv = n_univ * (n_univ - 1) // 2
n_features = n_univ + n_biv
betas_all = torch.ones([n_features,1], dtype=torch.float32,device=device)


univariate_responses_senn, bivariate_responses_senn, x_univariate_senn, x_bivariate_senn = nomogram(
    betas_all, 0, x_train_tensor, x_train0_median, x_train0_std, prn, 
    n_steps=15, sd_scale=2, method=method, device=device, categorical_threshold=15
)

