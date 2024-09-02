#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
 
from prism.config import PROCESSED_DATA_DIR, MODELS_DIR
from prism.PRiSM_functions import (
    normalise, 
    train_mlp_batched, 
    modelStats, 
    extract_weights, 
    partialResponses, 
    prPlots
)
from prism.prlasso import prLASSO
from prism.prnomogram import prnomogram_refactor
from prism.maskedmlp import (
    mlpmask_pytorch, 
    generate_mask, 
    get_model_weights_with_biases
)
from prism.save_models import (
    save_mlp,
    save_lasso,
    save_prn
)

import pandas as pd
import numpy as np
import torch
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression


# ## Parameters

device = 'cpu'
method = 'dirac'
SAVE_MODELS = False

seed = 257

np.random.seed(seed)
torch.manual_seed(seed)


# Print gpu hardware info
if torch.cuda.is_available():
    print("Number of CUDA devices:", torch.cuda.device_count())
    
    for i in range(torch.cuda.device_count()):
        print("CUDA Device #{}: {}".format(i, torch.cuda.get_device_name(i)))
    
    for i in range(torch.cuda.device_count()):
        print("Total memory of CUDA Device #{}: {:.2f} GB".format(
            i, torch.cuda.get_device_properties(i).total_memory / 1e9))
    
    for i in range(torch.cuda.device_count()):
        device_properties = torch.cuda.get_device_properties(i)
        print("CUDA Capability of Device #{}: {}.{}".format(
            i, device_properties.major, device_properties.minor))
    
    print("Default data type for CUDA tensors:", torch.get_default_dtype())
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("CUDA is not available.")


# ## Import data
# 
# Preprocessed and split into train, test, and validate.
# 
# Target: `oneyearmort`

data_train = pd.read_csv(PROCESSED_DATA_DIR.joinpath('imputed_dataset1_train.csv'))
data_test = pd.read_csv(PROCESSED_DATA_DIR.joinpath('imputed_dataset1_test.csv'))
data_val = pd.read_csv(PROCESSED_DATA_DIR.joinpath('imputed_dataset1_val.csv'))

data_train_test = pd.concat([data_train,data_test]) # used for axis annotation in some plotting functions

# drop id column
data_train.drop('trr_id_code',axis=1,inplace=True)
data_test.drop('trr_id_code',axis=1,inplace=True)
data_val.drop('trr_id_code',axis=1,inplace=True)

target_col = 'oneyearmort'

data_train.hist(bins=20,figsize=(12,10));


x_train0 = data_train.drop(target_col,axis=1)
y_train = data_train[target_col]

x_test0 = data_test.drop(target_col,axis=1)
y_test = data_test[target_col]

x_val0 = data_val.drop(target_col,axis=1)
y_val = data_val[target_col]

[x_train,x_test] = normalise(x_train0,x_test0)
x_val = normalise(x_val0)

# check result
x_train.hist(bins=20,figsize=(12,10));


# ## Train MLP

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


# ## Evaluate MLP

x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32,device=device)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
x_val_tensor = torch.tensor(x_val.values, dtype=torch.float32,device=device)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)


# Get model inference
mlp_batched.eval()
with torch.no_grad():
    y_test_blackbox = mlp_batched(x_test_tensor).to('cpu').numpy()
    y_val_blackbox = mlp_batched(x_val_tensor).to('cpu').numpy()

mlp_metrics_test = modelStats(y_test_blackbox, y_test, y_train, ROC=True, mdlCalibration=True, metricNames=True, auc_ci=True)
modelStats(y_val_blackbox, y_val, y_train, ROC=True, mdlCalibration=True, metricNames=True, auc_ci=True)


# ## Save MLP

if SAVE_MODELS:
    save_mlp(mlp_batched, mlp_params, mlp_metrics_test, MODELS_DIR)


# ## MLP LASSO

blackbox_weights = extract_weights(mlp_batched)

F_covariates_train, F_covariates_test, bivariate_inputs = partialResponses(x_train, x_test, blackbox_weights, method=method,device=device)


# Plot all univariate partial responses, for initial eval of MLP training results.

betas_univariate = np.zeros([F_covariates_train.shape[1],1])
betas_univariate[:x_train.shape[1],0] = 1


prPlots(betas_univariate, 0, x_train0, x_train, data_train_test, blackbox_weights, bivariate_inputs, n_steps = 15, sd_scale=2, method=method, device=device)


# ## Run LASSO on MLP Partial Responses
# 
# **Note** losses shown are normalized to loss at epoch 0, so not for comparing between hyperparameter sets.

num_lambdas = 25
verbose = True
lambdas, betas, trainAUC, testAUC, trainDev, testDev, glmPred = prLASSO(F_covariates_train, F_covariates_test, y_train, y_test, num_lambdas=num_lambdas, log_min_lambda=-1, verbose=verbose)


# userLambda = selectLambda(lambdas, betas, testAUC, testDev, x_train,bivariate_inputs)
userLambda = 7
lambdas[userLambda]


# ## Plots of the MLP partial responses for the selected lambda

prPlots(betas, userLambda, x_train0, x_train, data_train_test, blackbox_weights, bivariate_inputs, n_steps = 15, sd_scale=2, method=method, device=device)


# ## Train the Partial Response Network using the responses from the selected lambda

mask, nPr = generate_mask(betas, userLambda, x_train, bivariate_inputs,include_bivariate_as_univariate=True)


# Note, that changing the patience from 10 to 100 caused the predicted probability histogram to more closely match previous iterations. However, inference seemed unusually slow.
# 
# After experimenting with hyperparameters, I changed from the default initialization (Kaiming, suitable for relu), to specifically Xavier (Glorot) with gains set according to the activation functions of each layer.
# 
# With this and the "default" params, training stopped after 227 (rather than 300) epochs.

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


# ## Evaluate the Partial Response Network

# Get model inference
y_test_prn_pytorch = prn.predict(x_test_tensor, device=device).to('cpu').numpy()
y_val_prn_pytorch = prn.predict(x_val_tensor, device=device).to('cpu').numpy()


prn_metrics_test = modelStats(y_test_prn_pytorch, y_test, y_train, auc_ci=True)
modelStats(y_val_prn_pytorch,y_val,y_train, auc_ci=True)


if SAVE_MODELS:
    save_prn(prn, prn_params, prn_metrics_test, MODELS_DIR)


# Quick prPlots check of all univariate features using , before proceeding with LASSO on the Partial Response Network.

prPlots(betas_univariate, 0, x_train0, x_train, data_train_test, prn, bivariate_inputs, n_steps = 15, sd_scale=2, method=method,device=device)


# ## LASSO on the Partial Response Network

prn_weights = get_model_weights_with_biases(prn)

F_covariates_train_prn, F_covariates_test_prn, bivariate_inputs_prn = partialResponses(x_train, x_test, prn_weights, residual=False, method = method,device=device);


lambdas_prn, betas_prn, trainAUC_prn, testAUC_prn, trainDev_prn, testDev_prn, glmPred_prn = prLASSO(F_covariates_train_prn, F_covariates_test_prn,y_train, y_test, num_lambdas=75, log_max_lambda=2,log_min_lambda=-1,verbose=True)


# userLambda_prn = selectLambda(lambdas_prn, betas_prn, testAUC_prn, testDev_prn, x_train, bivariate_inputs)
userLambda_prn = 40
# userLambda_prn = 20
lambdas_prn[userLambda_prn]


ax = sns.heatmap(betas_prn, xticklabels=lambdas_prn)
ax.set_xticklabels(["{:.2g}".format(x) for x in lambdas_prn], rotation=90)
ax.set_xlabel('Lambda')
ax.set_ylabel('Feature index')

# reduce number of xtick labels
ax.set_xticks(ax.get_xticks()[::3]);


modelStats(glmPred_prn[:,userLambda_prn], y_test, y_train, auc_ci=True)


# ## Validation inference

F_covariates_train_prn, F_covariates_val_prn, bivariate_inputs_prn = partialResponses(x_train, x_val, prn_weights, residual=False, method = method,device=device);

lasso_params = {
    'C': 1/lambdas_prn[userLambda_prn],
    'penalty': 'l1',
    'solver': 'saga',
    'max_iter': 10000
}

prn_lasso = LogisticRegression(**lasso_params)
prn_lasso.fit(F_covariates_train_prn, y_train.to_numpy().ravel())

y_pred_test_prn_lasso = prn_lasso.predict_proba(F_covariates_test_prn)[:, 1]
y_pred_val_prn_lasso = prn_lasso.predict_proba(F_covariates_val_prn)[:, 1]
lasso_metrics_test = modelStats(y_pred_test_prn_lasso, y_test, y_train, auc_ci=True)
modelStats(y_pred_val_prn_lasso, y_val, y_train, auc_ci=True)


if SAVE_MODELS:
    save_lasso(prn_lasso, lasso_params, lasso_metrics_test, MODELS_DIR)


# ## User selected Partial Response plots from the Partial Response Network

prPlots(betas_prn, userLambda_prn, x_train0, x_train, data_train_test, prn, bivariate_inputs_prn, n_steps = 15, sd_scale=2, method=method,device=device)


# ## Partial Response Network Nomogram

prnomogram_refactor(betas_prn, userLambda_prn, x_train0, x_train, data_train_test, prn_weights, bivariate_inputs_prn,device=device,method=method)

