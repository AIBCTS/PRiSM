# mac:
# create environment: python3 -m virtualenv /path/.venv -p="/path/python3.11" 
# activate it: source .venv/bin/activate  
# install needed package: pip install pandas numpy torch matplotlib scikit-learn ipykernel
#
# windows/linux:
# create environment: py -m virtualenv .venv -p="python3.11"
# activate it (Windows cmd): .venv\Scripts\activate
# or
# activate it (Jupyter in vscode): "Python: Select Notebook Kernel" command. Select .venv (python3.11).
# install needed package: pip install pandas numpy torch matplotlib scikit-learn ipykernel

# tested on python version 3.11

from typing import Tuple, List, Optional, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

from sklearn import metrics
from sklearn import calibration
from sklearn.impute import SimpleImputer
from sklearn.utils import resample

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from prism.maskedmlp import MaskedMLP

Seed = 257

np.random.seed(Seed)
torch.manual_seed(Seed)

# %% [markdown]
# ### Normalise

# %%
def normalise(data: pd.DataFrame, test: Optional[pd.DataFrame] = None, sd_scale: int = 2) -> Tuple[pd.DataFrame, pd.DataFrame]:
  """
  Normalise the dataset to have 0 median and a standard deviation scaled by `sd_scale`.

  Parameters
  ----------
  data : pd.DataFrame
      The dataset to normalise.
  test : pd.DataFrame, optional
      The test dataset to normalise using the same parameters as `data`.
  sd_scale : int, optional
      The scaling factor for the standard deviation of the data.

  Returns
  -------
  Tuple[pd.DataFrame, pd.DataFrame]
      Normalised training and testing datasets.
  """
  med_train = data.median()
  sd_train = data.std()

  x_train = (data-med_train)/(sd_scale*sd_train)

  if isinstance(test,type(None)) == False:
    x_test = (test-med_train)/(sd_scale*sd_train)
    return [x_train,x_test]
  else:
    return x_train

def bootstrap_auc_ci(pred, target, n_bootstraps=100, alpha=0.05):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pred = torch.tensor(pred, device=device)
    target = torch.tensor(target, device=device)
    n_samples = len(target)

    # Generate bootstrap indices
    bootstrap_indices = torch.randint(0, n_samples, (n_bootstraps, n_samples), dtype=torch.int64, device=device)

    # Calculate AUC scores for each bootstrap sample
    bootstrapped_scores = torch.zeros(n_bootstraps, device=device)
    valid_mask = torch.zeros(n_bootstraps, dtype=torch.bool, device=device)
    for i in range(n_bootstraps):
        indices = bootstrap_indices[i]
        if len(torch.unique(target[indices])) >= 2:
            bootstrapped_scores[i] = metrics.roc_auc_score(target[indices].cpu().numpy(), pred[indices].cpu().numpy())
            valid_mask[i] = True

    # Filter out invalid bootstrap samples
    valid_scores = bootstrapped_scores[valid_mask]

    # Calculate confidence intervals
    sorted_scores = torch.sort(valid_scores).values
    lower_bound = np.percentile(sorted_scores.cpu().numpy(), (alpha / 2) * 100)
    upper_bound = np.percentile(sorted_scores.cpu().numpy(), (1 - alpha / 2) * 100)

    return lower_bound, upper_bound

def modelStats(pred: np.ndarray, target: np.ndarray, train_target: np.ndarray, ROC: bool = True, mdlCalibration: bool = True, metricNames: bool = True, auc_ci : bool = False) -> dict:
  """
  Calculate and return statistics for the model predictions.

  Parameters
  ----------
  pred : np.ndarray
      Predicted values.
  target : np.ndarray
      Actual target values.
  train_target : np.ndarray
      Target values used during trainig of the model, for calculating prevalence which sets the classification threshold.
  ROC : bool, optional
      Whether to include ROC curve metrics.
  mdlCalibration : bool, optional
      Whether to include model calibration statistics.
  metricNames : bool, optional
      Whether to include names of metrics in the output.
  auc_ci : bool, optional
      Whether to include AUC 95% confidance interval in the output (slower).

  Returns
  -------
  dict
      Dictionary containing statistical metrics of the model performance.
  """
  if ROC == True and mdlCalibration == True:
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4))
    fpr, tpr, thresholds = metrics.roc_curve(target, pred)
    roc_auc = metrics.auc(fpr, tpr)
    ax1.plot(fpr,tpr)
    ax1.plot([0,1],[0,1])
    ax1.title.set_text("ROC Curve")
    ax1.set_xlabel('False positive rate')
    ax1.set_ylabel('True positive rate')
    ax1.legend(["AUC = {}".format(round(roc_auc,3))],loc="lower right")

    prob_train, prob_pred = calibration.calibration_curve(target,pred,n_bins=10, strategy="uniform")
    disp = calibration.CalibrationDisplay(prob_train, prob_pred, pred)
    ax22 = ax2.twinx()
    ax22.hist(pred, histtype="bar", edgecolor="k", range=(0,1),facecolor="None")
    disp.plot(ax=ax2,color="tab:orange")
    ax2.title.set_text("Calibration Curve")
    plt.show()

  # ROC Curve
  elif ROC == True and mdlCalibration == False:
    fpr, tpr, thresholds = metrics.roc_curve(target, pred)
    plt.plot(fpr,tpr)
    plt.plot([0,1],[0,1])
    plt.title("ROC Curve")
    plt.set_xlabel('False positive rate')
    plt.set_ylabel('True positive rate')
    plt.legend(["AUC = {}".format(round(roc_auc,3))],loc="lower right")
    plt.show()

  # Calibration Curve
  elif mdlCalibration == True and ROC == False:
    prob_train, prob_pred = calibration.calibration_curve(target,pred,n_bins=10, strategy="uniform")
    disp = calibration.CalibrationDisplay(prob_train, prob_pred, pred)
    fig, ax1 = plt.subplots()
    disp.plot(ax=ax1,color="tab:orange")
    ax1.title.set_text("Calibration Curve - Lasso")
    ax2 = ax1.twinx()
    ax2.hist(pred, histtype="bar", edgecolor="k", range=(0,1),facecolor="None")
    plt.show()

  mList = ["prevalence","sensitivity","specificity","accuracy","ppv","auc score"]

  if metricNames:
    metricNames =  ["prevalence","sensitivity","specificity","accuracy","ppv","auc score"]

    # Metrics
    prevalence = len(train_target[train_target==1])/len(train_target)
    pred_class = pred.copy(); pred_class[pred_class>prevalence] = 1; pred_class[pred_class<=prevalence] = 0
    [tn, fp], [fn, tp] = metrics.confusion_matrix(target,pred_class)
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    ppv = tp/(tp+fp);

    mVals = [round(prevalence,3),round(sensitivity,3),round(specificity,3),round(accuracy, 3),round(ppv,3),round(roc_auc,3)]
    mdlMetrics = {}

    print("\n-------- Metrics --------")
    for i in range(0,len(metricNames)):
      if metricNames[i].lower() in mList:
        print("{}: {}".format(mList[mList.index(metricNames[i].lower())],mVals[mList.index(metricNames[i].lower())]))
        mdlMetrics[mList[mList.index(metricNames[i].lower())]] = mVals[mList.index(metricNames[i].lower())]
      if metricNames[i].lower() == "auc score" and auc_ci:
        ci_lower, ci_upper = bootstrap_auc_ci(pred,target,alpha=0.05)
        mdlMetrics["auc lower ci"] = format(round(ci_lower, 3))
        mdlMetrics["auc upper ci"] = format(round(ci_upper, 3))
        print("{}: {}".format("auc lower ci", round(ci_lower, 3)))
        print("{}: {}".format("auc upper ci", round(ci_upper, 3)))
    print("-------------------------")

    return mdlMetrics

# %%
def selectLambda(lambdas: List[float], betas: List[float], testAUC: List[float], testDev: List[float], x_train: pd.DataFrame, bivariate_inputs: np.ndarray) -> float:
  """
  Select a lambda value from a set based on model performance metrics. Requires the user to enter the index of the desired lambda value, based on the provided metrics. Prints the selected responses (fatures) based on the selection. Number of features shown as |beta|>0.1, as used in `generateMask`.

  Parameters
  ----------
  lambdas : List[float]
      Array of lambda values considered.
  betas : List[float]
      Coefficients for each lambda.
  testAUC : List[float]
      Array of AUC (Area Under the Curve) scores for the test set.
  testDev : List[float]
      Array of deviance scores for the test set.
  x_train : pd.DataFrame
      Training data used for model fitting.
  bivariate_inputs: np.ndarray
      Contains the index of the corresponding input columns for each bivariate response.
  Returns
  -------
  float
      The selected lambda value.
  """
  minPlus = np.argmin(abs(testDev-((np.min(testDev)+np.std(testDev[np.where(np.sum(betas,0)!=0)])))))
  maxMinus = np.argmin(abs(testAUC-((np.max(testAUC)-np.std(testAUC[np.where(np.sum(betas,0)!=0)])))))
  for i in range (0,len(lambdas)):
    print("----------------------------")
    print(i,"- Lambda:",lambdas[i])
    #print("Test Deviance:",testDev[i])
    if lambdas[i] == lambdas[np.argmin(testDev)]:
      print("Test Deviance:",testDev[i]," <- Minimum Deviance")
    elif i == minPlus:
      print("Test Deviance:",testDev[i]," <- Minimum Deviance + 1 sd")
    else:
      print("Test Deviance:",testDev[i])
    #print("Test AUC:", testAUC[i])
    if lambdas[i] == lambdas[np.argmax(testAUC)]:
      print("Test AUC:", testAUC[i]," <- Maximum AUC")
    elif i == maxMinus:
      print("Test AUC:",testAUC[i]," <- Maxmimum AUC - 1 sd")
    else:
      print("Test AUC:", testAUC[i])
    print("Number of features:",len(np.where(abs(betas[:,i])>0.1)[0]))
    print("----------------------------")

  print("Minimum Deviance:  Lambda",np.argmin(testDev),"- N features:",len(np.where(abs(betas[:,np.argmin(testDev)])>0.1)[0]))
  print("Min Deviance +1sd: Lambda",minPlus,"- N features:",len(np.where(abs(betas[:,minPlus])>0.1)[0]))
  print("Maximum AUC:       Lambda",np.argmax(testAUC),"- N features:",len(np.where(abs(betas[:,np.argmax(testAUC)])>0.1)[0]))
  print("Max AUC -1sd:      Lambda",maxMinus,"- N features:",len(np.where(abs(betas[:,maxMinus])>0.1)[0]))

  #allows the user to select a lambda from the information displayed
  userLambda = int(input("Enter the Index of the chosen lambda: "))

  all_res = np.where(abs(betas[:,userLambda])>0.1)[0]
  u_res = list(x_train.columns[all_res[all_res < x_train.shape[1]]])

  b_res = []

  for i in all_res[all_res>=x_train.shape[1]]-x_train.shape[1]:
    b_res.append("{}/{}".format(x_train.columns[int(bivariate_inputs[i,0])],x_train.columns[int(bivariate_inputs[i,1])]))

  print("\n Selected Responses:")
  print("-----Univariate Responses-----")
  for i in range(0,len(u_res)):
    print(u_res[i])

  print("-----Bivariate Responses-----")
  for i in range(0,len(b_res)):
    print(b_res[i])
  return userLambda

# %% [markdown]
# ### generateMask

# %%
def generateMask(betas: List[float], userLambda: float, x_train: pd.DataFrame, bivariate_inputs: List[int], subnet_nodes: int = 5) -> np.ndarray:
  """
  Generate a mask for the input features (partial responses) based on selected lambda and coefficients.

  Parameters
  ----------
  betas : List[float]
      Model paritial response coefficients.
  userLambda : float
      Selected lambda value.
  x_train : pd.DataFrame
      Training dataset.
  bivariate_inputs : List[int]
      Indices of features to be used for bivariate analysis.
  subnet_nodes : int, optional
      Number of subnet nodes for each input feature. Should match the subnet_nodes parameter provided when calling mlpmask.

  Returns
  -------
  np.ndarray
      Mask array for input features.
  """
  prNames = []
  univList = []
  bivList = []
  prList = []


  for i in range(0,len(np.where(abs(betas[:,userLambda])>0.1)[0])):
    prPosition = np.where(abs(betas[:,userLambda])>0.1)[0][i]
    prList.append(prPosition)
    if prPosition < x_train.shape[1]:
      prNames.append(x_train.columns[prPosition])
      univList.append(prPosition)
    else:
      prNames.append("{} : {}".format(x_train.columns[int(bivariate_inputs[prPosition-x_train.shape[1],0])],x_train.columns[int(bivariate_inputs[prPosition-x_train.shape[1],1])]))
      bivList.append(prPosition-x_train.shape[1])
      if int(bivariate_inputs[prPosition-x_train.shape[1],0]) not in univList:
        univList.append(int(bivariate_inputs[prPosition-x_train.shape[1],0]))
      if int(bivariate_inputs[prPosition-x_train.shape[1],1]) not in univList:
        univList.append(int(bivariate_inputs[prPosition-x_train.shape[1],1]))

  print(f"univList:{univList}")
  print(f"bivList:{bivList}")
  nUniv = len(univList)
  nBiv = len(bivList)
  nPr = nUniv + nBiv

  #mask = np.zeros([nUniv,subnet_nodes*(nUniv+nBiv)])
  mask = np.zeros([x_train.shape[1],subnet_nodes*(nUniv+nBiv)])
  print(f"mask.shape: {mask.shape}")
  print(f"prList: {prList}")
  print(prNames)

  for i in list(range(0,nUniv)):
    mask[univList[i],i*subnet_nodes:(i*subnet_nodes)+subnet_nodes] = 1

  for i in range(0,nBiv):
    mask[int(bivariate_inputs[bivList[i],0]),(nUniv*subnet_nodes)+(i*subnet_nodes):(nUniv*subnet_nodes)+(i*subnet_nodes)+subnet_nodes] = 1
    mask[int(bivariate_inputs[bivList[i],1]),(nUniv*subnet_nodes)+(i*subnet_nodes):(nUniv*subnet_nodes)+(i*subnet_nodes)+subnet_nodes] = 1
  return mask, nPr
