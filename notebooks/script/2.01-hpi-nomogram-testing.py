#!/usr/bin/env python
# coding: utf-8

import pickle
import numpy as np


from importlib import reload
import prnomogram
reload(prnomogram)
from prnomogram import prnomogram_refactor


# ## Import nomogram arguments

filename = "nomogram_arguments_20240628_112200.pkl"

with open(filename, "rb") as f:
    prn_nomogram_args = pickle.load(f)

betas_senn, userLambda_senn, x_train0, x_train, data_train_test, senn_weights, bivariate_inputs_senn = prn_nomogram_args


# Get number of features
n_univ = x_train.shape[1]
n_biv = n_univ * (n_univ - 1) // 2
n_features = n_univ + n_biv

# feature_index = [12, 13, *range(60, 65)]  #
# feature_index = [*range(0, 66)]  # everything
# betas_all = np.zeros([n_features,1])
# betas_all[feature_index,0] = 1
betas_all = np.ones([n_features,1])


device='cpu'
method='dirac'


prnomogram_refactor(betas_all, 0, x_train0, x_train, data_train_test, senn_weights, bivariate_inputs_senn,device=device,method=method)
# prnomogram_refactor(betas_senn, userLambda_senn, x_train0, x_train, data_train_test, senn_weights, bivariate_inputs_senn,device=device,method=method)


# Reload prnomogram function after changes:

from importlib import reload
import prnomogram
reload(prnomogram)
from prnomogram import prnomogram_refactor

