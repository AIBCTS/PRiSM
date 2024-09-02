#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import pandas as pd
import numpy as np
import torch
import seaborn as sns
from sklearn.linear_model import LogisticRegression

from prism.config import PROCESSED_DATA_DIR, MODELS_DIR
from prism.legacy import normalise, modelStats
from prism.maskedmlp import train_mlp_batched, mlpmask_pytorch, generate_mask
from prism.save_models import save_mlp, save_lasso, save_prn
# from prism.prlasso import prLASSO

from prism.partial_responses import partial_responses
from prism.nomogram import nomogram


get_ipython().run_line_magic('reload_ext', 'autoreload')


# Parameters
device = 'cuda'
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

y_test_blackbox = mlp_batched.predict_numpy(x_test_tensor)
y_val_blackbox = mlp_batched.predict_numpy(x_val_tensor)


mlp_metrics_test = modelStats(y_test_blackbox, y_test, y_train, ROC=True, mdlCalibration=True, metricNames=True, auc_ci=True)
modelStats(y_val_blackbox, y_val, y_train, ROC=True, mdlCalibration=True, metricNames=True, auc_ci=True)


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
num_lambdas = 25
verbose = True
lambdas, betas, trainAUC, testAUC, trainDev, testDev, glmPred = prLASSO(partial_responses_train.cpu().numpy(), partial_responses_test.cpu().numpy(), y_train, y_test, num_lambdas=num_lambdas, log_min_lambda=-1, verbose=verbose)


# Note: The selectLambda function is not available in the new implementation.
# We'll need to manually select a lambda value or implement a new selection method.
userLambda = 7  # This is now manually set
print(f"Selected lambda: {lambdas[userLambda]}")


# # Plots of the MLP partial responses for the selected lambda

get_ipython().run_line_magic('reload_ext', 'autoreload')
# Note: The prPlots function is not yet available in the new implementation.
# We'll use the new nomogram function instead.
nomogram(
    betas,
    userLambda,
    x_train_tensor,
    x_train0.median(),
    x_train0.std(),
    mlp_batched,
    n_steps=15,
    sd_scale=2,
    method=method,
    device=device,
    categorical_threshold=15
)


# # Train the Partial Response Network

mask, nPr = generate_mask(betas, userLambda, x_train, bivariate_inputs, include_bivariate_as_univariate=True)

prn_params = {
    'n_hidden': nPr,
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


if SAVE_MODELS:
    save_prn(prn, prn_params, prn_metrics_test, MODELS_DIR)


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
lambdas_prn, betas_prn, trainAUC_prn, testAUC_prn, trainDev_prn, testDev_prn, glmPred_prn = prLASSO(partial_responses_train_prn, partial_responses_test_prn, y_train, y_test, num_lambdas=75, log_max_lambda=2, log_min_lambda=-1, verbose=True)


# Note: userLambda_prn selection is not available yet in the new implementation.
# We'll need to manually select a lambda value.
userLambda_prn = 40  # This is now manually set
print(f"Selected PRN lambda: {lambdas_prn[userLambda_prn]}")


ax = sns.heatmap(betas_prn, xticklabels=lambdas_prn)
ax.set_xticklabels(["{:.2g}".format(x) for x in lambdas_prn], rotation=90)
ax.set_xlabel('Lambda')
ax.set_ylabel('Feature index')

# reduce number of xtick labels
ax.set_xticks(ax.get_xticks()[::3]);


# # Validation inference

partial_responses_train_prn, partial_responses_val_prn, _ = partial_responses(
    torch.tensor(x_train.values, dtype=torch.float32),
    torch.tensor(x_val.values, dtype=torch.float32),
    prn,
    method=method,
    device=device
)

lasso_params = {
    'C': 1/lambdas_prn[userLambda_prn],
    'penalty': 'l1',
    'solver': 'saga',
    'max_iter': 10000
}

prn_lasso = LogisticRegression(**lasso_params)
prn_lasso.fit(partial_responses_train_prn.cpu().numpy(), y_train.to_numpy().ravel())


y_pred_test_prn_lasso = prn_lasso.predict_proba(partial_responses_test_prn.cpu().numpy())[:, 1]
y_pred_val_prn_lasso = prn_lasso.predict_proba(partial_responses_val_prn.cpu().numpy())[:, 1]
lasso_metrics_test = modelStats(y_pred_test_prn_lasso, y_test, y_train, auc_ci=True)
modelStats(y_pred_val_prn_lasso, y_val, y_train, auc_ci=True)


if SAVE_MODELS:
    save_lasso(prn_lasso, lasso_params, lasso_metrics_test, MODELS_DIR)


# # Partial Response Network Nomogram

nomogram(
    betas_prn,
    userLambda_prn,
    x_train_tensor,
    x_train0.median().values,
    x_train0.std().values,
    prn,
    n_steps=15,
    sd_scale=2,
    method=method,
    device=device,
    categorical_threshold=15
)


# ## All nomograms

n_univ = x_train.shape[1]
n_biv = n_univ * (n_univ - 1) // 2
n_features = n_univ + n_biv
betas_all = np.ones([n_features,1])


get_ipython().run_line_magic('reload_ext', 'autoreload')

nomogram(
    betas_all,
    0,
    x_train_tensor,
    x_train0.median().values,
    x_train0.std().values,
    prn,
    n_steps=15,
    sd_scale=2,
    method=method,
    device=device,
    categorical_threshold=15
)


# ## All nomograms Lebesgue

get_ipython().run_line_magic('reload_ext', 'autoreload')

nomogram(
    betas_all,
    0,
    x_train_tensor,
    x_train0.median().values,
    x_train0.std().values,
    prn,
    n_steps=15,
    sd_scale=2,
    method='lebesgue',
    device=device,
    categorical_threshold=15
)

