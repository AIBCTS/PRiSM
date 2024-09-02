#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import pandas as pd
import numpy as np
import torch
# import seaborn as sns
# from sklearn.linear_model import LogisticRegression

from prism.config import PROCESSED_DATA_DIR, MODELS_DIR
from prism.legacy import normalise, modelStats
from prism.maskedmlp import train_mlp_batched, mlpmask_pytorch
from prism.save_models import save_mlp
from prism.lasso import lasso
from prism.partial_responses import partial_responses
from prism.nomogram import nomogram


get_ipython().run_line_magic('reload_ext', 'autoreload')


# Parameters
device = 'cpu'
method = 'dirac'
SAVE_MODELS = False
seed = 257

np.random.seed(seed)
torch.manual_seed(seed)


# # Import data
# 

data_train = pd.read_csv(PROCESSED_DATA_DIR.joinpath('imputed_dataset1_train.csv'))
data_test = pd.read_csv(PROCESSED_DATA_DIR.joinpath('imputed_dataset1_test.csv'))
data_val = pd.read_csv(PROCESSED_DATA_DIR.joinpath('imputed_dataset1_val.csv'))

data_train_test = pd.concat([data_train, data_test])

# Drop id column
data_train.drop('trr_id_code', axis=1, inplace=True)
data_test.drop('trr_id_code', axis=1, inplace=True)
data_val.drop('trr_id_code', axis=1, inplace=True)

target_col = 'oneyearmort'

x_train0 = data_train.drop(target_col, axis=1)
y_train = data_train[target_col]

x_test0 = data_test.drop(target_col, axis=1)
y_test = data_test[target_col]

x_val0 = data_val.drop(target_col, axis=1)
y_val = data_val[target_col]

[x_train, x_test] = normalise(x_train0, x_test0)
x_val = normalise(x_val0)

x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32, device=device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32, device=device)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
x_val_tensor = torch.tensor(x_val.values, dtype=torch.float32, device=device)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)


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


# # Train MLP
# 

mlp_params = {
    'n_hidden': 10,
    'weight_decay': 1e-5,
    'lr': 0.001,
    'patience': 50,
    'tolerance': 0.0001,
    'batch_size': 1024,
    'device': device,
    'seed': seed
}

mlp_batched = train_mlp_batched(x_train, y_train, x_test, y_test, **mlp_params)


# # Evaluate MLP

y_test_blackbox = mlp_batched.predict(x_test_tensor).cpu().numpy()
y_val_blackbox = mlp_batched.predict(x_val_tensor).cpu().numpy()

mlp_metrics_test = modelStats(y_test_blackbox, y_test, y_train, ROC=True, mdlCalibration=True, metricNames=True, auc_ci=True)
modelStats(y_val_blackbox, y_val, y_train, ROC=True, mdlCalibration=True, metricNames=True, auc_ci=True);


# Save MLP
if SAVE_MODELS:
    save_mlp(mlp_batched, mlp_params, mlp_metrics_test, MODELS_DIR)


# # MLP LASSO

partial_responses_train, partial_responses_test, bivariate_inputs = partial_responses(
    x_train_tensor,
    x_test_tensor,
    mlp_batched,
    method=method,
    device=device
)


# # Run LASSO on MLP Partial Responses

get_ipython().run_line_magic('reload_ext', 'autoreload')
lasso_results = lasso(
    partial_responses_train, 
    partial_responses_test, 
    y_train, 
    y_test, 
    bivariate_inputs,
    feature_names=feature_names,
    nlambda=25, 
    min_lambda=0.1,
    max_lambda=100,
    tol=1e-4,
    batch_size=2
)


lasso_results.plot_lambda_path()


lasso_results.select_lambda(7)
lasso_results.plot_feature_importance()


get_ipython().run_line_magic('reload_ext', 'autoreload')
# Note: The prPlots function is not yet available in the new implementation.
# We'll use the new nomogram function instead.
nomogram_results = nomogram(
    lasso_results,
    x_train_tensor,
    x_train0.median().values,
    x_train0.std().values,
    mlp_batched,
    n_steps=15,
    sd_scale=2,
    method=method,
    device=device,
    categorical_threshold=15,
    subtract_univariate=False
);


#  test with more features
lasso_results.select_lambda(24)


get_ipython().run_line_magic('reload_ext', 'autoreload')
nomogram_results = nomogram(
    lasso_results,
    x_train_tensor,
    x_train0.median().values,
    x_train0.std().values,
    mlp_batched,
    n_steps=15,
    sd_scale=2,
    method=method,
    device=device,
    categorical_threshold=15,
    subtract_univariate=False
)


# return to previously selected lambda
lasso_results.select_lambda(7)


# # Train the Partial Response Network

mask, n_features = lasso_results.get_mask()


prn_params = {
    'n_hidden': n_features,
    'mask': mask,
    'subnet_nodes': 5,
    'iter': 10000,
    'lr': 0.05,
    'weight_decay': 0.00001,
    'tolerance': 0.0001,
    'patience': 100,
    'device': device,
    'seed': seed
}

prn = mlpmask_pytorch(x_train, y_train, x_test, y_test, **prn_params)


# # Evaluate the Partial Response Network

y_test_prn_pytorch = prn.predict(x_test_tensor, device=device).cpu().numpy()
y_val_prn_pytorch = prn.predict(x_val_tensor, device=device).cpu().numpy()

prn_metrics_test = modelStats(y_test_prn_pytorch, y_test, y_train, auc_ci=True)
modelStats(y_val_prn_pytorch, y_val, y_train, auc_ci=True)


# # LASSO on the Partial Response Network

# Use the new partial_responses function for PRN
partial_responses_train_prn, partial_responses_test_prn, bivariate_inputs_prn = partial_responses(
    x_train_tensor,
    x_test_tensor,
    prn,
    method=method,
    device=device
)


get_ipython().run_line_magic('reload_ext', 'autoreload')
lasso_results_prn = lasso(
    partial_responses_train_prn, 
    partial_responses_test_prn, 
    y_train, 
    y_test, 
    bivariate_inputs_prn,
    feature_names=feature_names,
    nlambda=75, 
    min_lambda=0.1,
    max_lambda=100,
    tol=1e-4,
    batch_size=2
)


lasso_results_prn.plot_lambda_path()


lasso_results_prn.select_lambda(40)
lasso_results_prn.plot_feature_importance()


# # Validation inference

prn_lasso = lasso_results_prn.get_selected_model()


_, partial_responses_val_prn, bivariate_inputs_prn = partial_responses(
    x_train_tensor,
    x_val_tensor,
    prn,
    method=method,
    device=device
)


y_pred_test_prn_lasso = prn_lasso.predict_proba(partial_responses_test_prn)[:, 1]
y_pred_val_prn_lasso = prn_lasso.predict_proba(partial_responses_val_prn)[:, 1]
lasso_metrics_test = modelStats(y_pred_test_prn_lasso, y_test, y_train, auc_ci=True)
modelStats(y_pred_val_prn_lasso, y_val, y_train, auc_ci=True)


# # Partial Response Network Nomogram

nomogram(
    lasso_results_prn,
    x_train_tensor,
    x_train0.median().values,
    x_train0.std().values,
    prn,
    n_steps=15,
    sd_scale=2,
    method=method,
    device=device,
    categorical_threshold=15
);


# ## Nomogram with more 
# 
# TODO: modify lasso_results to include possibility to select all features, all univariate, all bivariate (e.g. for plotting).

# test with more features
# todo add all features method to lasso_results object.
lasso_results_prn.select_lambda(74)
nomogram(
    lasso_results_prn,
    x_train_tensor,
    x_train0.median().values,
    x_train0.std().values,
    prn,
    n_steps=15,
    sd_scale=2,
    method=method,
    device=device,
    categorical_threshold=15
);




