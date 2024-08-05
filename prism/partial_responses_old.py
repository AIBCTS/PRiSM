import torch
import numpy as np
import pandas as pd
from typing import Any, Tuple, Optional, List

def partialResponses(x_train: pd.DataFrame, x_test: pd.DataFrame, model: Any, residual: bool = False, method: str = "dirac", device: str = "cpu") -> np.ndarray:
  """
  Generate partial responses for the model inputs.

  Parameters
  ----------
  x_train : pd.DataFrame
      Training dataset.
  x_test : pd.DataFrame
      Testing dataset.
  model : Any
      The trained model.
  residual : bool, optional
      Whether to calculate residuals instead of predictions.
  method : str, optional
      Method used for generating partial responses ('dirac' or 'lebesgue').
  device : str, optional
      Device to run the model computation ('cpu' or 'cuda').

  Returns
  -------
  np.ndarray
      The first two outputs, F_covariates_train, F_covariates_test, are arrays of partial response values. Each contain a number of rows equal to the number of observations in the original training and test data, and a number of columns equal n(n+1)/2 where n is the number of columns in the input data. Each column contains the predictions representing each partial response. The third output, bivariate_inputs, contains the index of the corresponding input columns for each bivariate response.
  """
  # # Initialize variables to avoid UnboundLocalError
  # F_covariates_train = np.array([])
  # F_covariates_test = np.array([])
  # bivariate_inputs = np.array([])

  # Check if method parameter is valid
  if method.lower() not in ["dirac", "lebesgue"]:
    print(f"Invalid method: {method}")
    raise ValueError("Method must be either 'dirac' or 'lebesgue'")

  all_x = pd.concat([x_train,x_test])

  if method.lower() == "dirac":
    x0 = np.zeros((1,all_x.shape[1]))

    if isinstance(model,list) == True:
      y0 = mlpmask_pred(x0,model,device=device)
    elif hasattr(model,'predict_proba'):
      y0 = model.predict_proba(x0)[:,1]
    else:
      y0 = model.predict(x0)

    logit_y0 = np.log(y0/(1-y0))

    univariate_y = np.zeros(all_x.shape)
    print("Univariate Responses:")
    # calculates univariate parital responses
    for i in range(0,all_x.shape[1]):
      x_temp = np.zeros(all_x.shape)
      x_temp[:,i] = all_x.iloc[:,i]
      print("Column {}:".format(i))

      if isinstance(model,list) == True:
        y_temp = mlpmask_pred(x_temp,model,device=device)
      elif hasattr(model,'predict_proba'):
        y_temp = model.predict_proba(x_temp)[:,1]
      else:
        y_temp = model.predict(x_temp)

      univariate_y[:,i] = np.log(np.divide(y_temp,(1-y_temp))).reshape(y_temp.shape[0])-logit_y0

    bivlen = int(all_x.shape[1]*(all_x.shape[1]-1)/2)
    bivariate_y = np.zeros([all_x.shape[0],bivlen])
    bivariate_inputs = np.zeros([bivlen,2])

    count = 0

    print("Bivariate Responses:")
    for i in range (0,all_x.shape[1]):
      for j in range(i+1,all_x.shape[1]):
        x_temp = np.zeros(all_x.shape)
        x_temp[:,[i,j]] = all_x.iloc[:,[i,j]]
        print("Columns {i} & {j}:".format(i=i, j=j))
        if isinstance(model,list) == True:
          y_temp = mlpmask_pred(x_temp,model,device=device)
        elif hasattr(model,'predict_proba'):
          y_temp = model.predict_proba(x_temp)[:,1]
        else:
          y_temp = model.predict(x_temp)
        bivariate_y[:,count] = np.log(np.divide(y_temp,(1-y_temp))).reshape(y_temp.shape[0])-univariate_y[:,i]-univariate_y[:,j]-logit_y0
        bivariate_inputs[count,:] = [i,j]
        count += 1

    print("Residual")
    if isinstance(model,list) == True:
      y_pred_train = mlpmask_pred(x_train,model,device=device)
    elif hasattr(model,'predict_proba'):
      y_pred_train = model.predict_proba(x_train)[:,1]
    else:
      y_pred_train = model.predict(x_train)

    logit_y_train = np.log(np.divide(y_pred_train,1-y_pred_train)).reshape(y_pred_train.shape[0])
    train_approx = logit_y0 + np.sum(univariate_y[range(0,len(x_train)),:],axis=1) + np.sum(bivariate_y[range(0,len(x_train)),:],axis=1)
    res_logit_train = logit_y_train - train_approx

    if isinstance(model,list) == True:
      y_test_pred = mlpmask_pred(x_test,model,device=device)
    elif hasattr(model,'predict_proba'):
      y_test_pred = model.predict_proba(x_test)[:,1]
    else:
      y_test_pred = model.predict(x_test)

    logit_y_test = np.log(np.divide(y_test_pred,1-y_test_pred)).reshape(y_test_pred.shape[0])
    test_approx = np.sum(univariate_y[range(all_x.shape[0]-len(x_test),all_x.shape[0]),:],axis=1) + np.sum(bivariate_y[range(all_x.shape[0]-len(x_test),all_x.shape[0]),:],axis=1)
    res_logit_test = logit_y_test - test_approx

    res_temp = res_logit_train
    res_logit_train = (res_temp - np.mean(res_temp))/np.std(res_temp)
    res_logit_test = (res_logit_test - np.mean(res_temp))/np.std(res_temp)

    if residual == True:
      F_covariates_train = np.concatenate((univariate_y[0:x_train.shape[0],:],bivariate_y[0:x_train.shape[0],:],res_logit_train.reshape(len(res_logit_train),1)),axis=1)
      F_covariates_test = np.concatenate((univariate_y[x_train.shape[0]:all_x.shape[0],:],bivariate_y[x_train.shape[0]:all_x.shape[0],:],res_logit_test.reshape(len(res_logit_test),1)),axis=1)
    else:
      F_covariates_train = np.concatenate((univariate_y[0:x_train.shape[0],:],bivariate_y[0:x_train.shape[0],:]),axis=1)
      F_covariates_test = np.concatenate((univariate_y[x_train.shape[0]:all_x.shape[0],:],bivariate_y[x_train.shape[0]:all_x.shape[0],:]),axis=1)

  if method.lower() == "lebesgue": # -------------------------------------------------------------- EDIT CODE: DO NOT CONCATENATE

    if isinstance(model,list) == True:
      y0 = mlpmask_pred(x_train,model,device=device)
    elif hasattr(model,'predict_proba'):
      y0 = model.predict_proba(x_train)[:,1]
    else:
      y0 = model.predict(x_train)

    logit_y0 = np.mean(np.log(y0/(1-y0)))

    univariate_y_train = np.zeros(x_train.shape)
    univariate_y_test = np.zeros(x_test.shape)

    print("Univariate Responses:")
    # calculates univariate parital responses for the training data
    for k in range(0,x_train.shape[0]):
      if k%100 == 0:
        print(f"Train pred: iterating over all features at x sample {k}")
      for i in range(0,x_train.shape[1]):
        x_temp = x_train.to_numpy() # assums x_train is a DF! TODO: make more type agnostic
        x_temp[:,i] = x_temp[k,i]

        if isinstance(model,list) == True:
          y_temp = mlpmask_pred(x_temp,model,device=device)
        elif hasattr(model,'predict_proba'):
          y_temp = model.predict_proba(x_temp)[:,1]
        else:
          y_temp = model.predict(x_temp)

        univariate_y_train[k,i] = np.mean(np.log(np.divide(y_temp,(1-y_temp))).reshape(y_temp.shape[0]))-logit_y0

    # calculates univariate parital responses for the test data
    for k in range(0,x_test.shape[0]):
      if k%100 == 0:
        print(f"Test pred: iterating over all features at x sample {k}")
      for i in range(0,x_test.shape[1]):
        # Assign all values of column i that of i at row k
        x_temp = x_test.to_numpy() # assums x_train is a DF! TODO: make more type agnostic
        x_temp[:, i] = x_temp[k, i]

        if isinstance(model,list) == True:
          y_temp = mlpmask_pred(x_temp,model,device=device)
        elif hasattr(model,'predict_proba'):
          y_temp = model.predict_proba(x_temp)[:,1]
        else:
          y_temp = model.predict(x_temp)

        univariate_y_test[k,i] = np.mean(np.log(np.divide(y_temp,(1-y_temp))).reshape(y_temp.shape[0]))-logit_y0

    bivlen = int(all_x.shape[1]*(all_x.shape[1]-1)/2)
    bivariate_y_train = np.zeros([x_train.shape[0],bivlen])
    bivariate_y_test = np.zeros([x_test.shape[0],bivlen])
    bivariate_inputs = np.zeros([bivlen,2])
    count=0
    for i in range (0,x_train.shape[1]):
      for j in range(i+1,x_train.shape[1]):
        bivariate_inputs[count,:] = [i,j]
        count += 1

    print("Bivariate Responses:")
    for k in range(0,x_train.shape[0]):
      count = 0
      for i in range (0,x_train.shape[1]):
        for j in range(i+1,x_train.shape[1]):
          x_temp = x_train.to_numpy()
          x_temp[:,[i,j]] = x_temp[k,[i,j]]

          if isinstance(model,list) == True:
            y_temp = mlpmask_pred(x_temp,model,device=device)
          elif hasattr(model,'predict_proba'):
            y_temp = model.predict_proba(x_temp)[:,1]
          else:
            y_temp = model.predict(x_temp)
          bivariate_y_train[k,count] = np.mean(np.log(np.divide(y_temp,(1-y_temp))).reshape(y_temp.shape[0])-univariate_y_train[k,i]-univariate_y_train[k,j]-logit_y0)
          count += 1

    for k in range(0,x_test.shape[0]):
      count=0
      for i in range (0,x_test.shape[1]):
        for j in range(i+1,x_test.shape[1]):
          x_temp = x_test.to_numpy()
          x_temp[:, [i, j]] = x_temp[k, [i, j]]
          if isinstance(model,list) == True:
            y_temp = mlpmask_pred(x_temp,model,device=device)
          elif hasattr(model,'predict_proba'):
            y_temp = model.predict_proba(x_temp)[:,1]
          else:
            y_temp = model.predict(x_temp)
          bivariate_y_test[k,count] = np.mean(np.log(np.divide(y_temp,(1-y_temp))).reshape(y_temp.shape[0])-univariate_y_test[k,i]-univariate_y_test[k,j]-logit_y0)
          count+=1
    print("Residual")
    if isinstance(model,list) == True:
      y_pred_train = mlpmask_pred(x_train,model,device=device)
    elif hasattr(model,'predict_proba'):
      y_pred_train = model.predict_proba(x_train)[:,1]
    else:
      y_pred_train = model.predict(x_train)

    logit_y_train = np.log(np.divide(y_pred_train,1-y_pred_train)).reshape(y_pred_train.shape[0])
    train_approx = logit_y0 + np.sum(univariate_y_train,axis=1) + np.sum(bivariate_y_train,axis=1)
    res_logit_train = logit_y_train - train_approx

    if isinstance(model,list) == True:
      y_test_pred = mlpmask_pred(x_test,model,device=device)
    elif hasattr(model,'predict_proba'):
      y_test_pred = model.predict_proba(x_test)[:,1]
    else:
      y_test_pred = model.predict(x_test)

    logit_y_test = np.log(np.divide(y_test_pred,1-y_test_pred)).reshape(y_test_pred.shape[0])
    test_approx = np.sum(univariate_y_test,axis=1) + np.sum(bivariate_y_test,axis=1)
    res_logit_test = logit_y_test - test_approx

    res_temp = res_logit_train
    res_logit_train = (res_temp - np.mean(res_temp))/np.std(res_temp)
    res_logit_test = (res_logit_test - np.mean(res_temp))/np.std(res_temp)

    if residual == True:
      F_covariates_train = np.concatenate((univariate_y_train,bivariate_y_train,res_logit_train.reshape(len(res_logit_train),1)),axis=1)
      F_covariates_test = np.concatenate((univariate_y_test,bivariate_y_test,res_logit_test.reshape(len(res_logit_test),1)),axis=1)
    else:
      F_covariates_train = np.concatenate((univariate_y_train,bivariate_y_train),axis=1)
      F_covariates_test = np.concatenate((univariate_y_test,bivariate_y_test),axis=1)


  return F_covariates_train, F_covariates_test, bivariate_inputs