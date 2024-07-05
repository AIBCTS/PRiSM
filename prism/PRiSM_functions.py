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

def preprocess(data, target, missing_val=np.nan, test=None, cols=None, test_split=None, seed=None):
  """
  Preprocess the dataset by handling missing values, splitting the dataset, and selecting specific columns.

  Parameters
  ----------
  data : DataFrame
    The dataset to preprocess.
  target : str
    The name of the target variable.
  missing_val : float, optional
    The value to use for missing data. Default is np.nan.
  test : DataFrame, optional
    The test dataset. If None, the test set is created from `data`.
  cols : list of str, optional
    List of column names to retain in the dataset.
  test_split : float, optional
    The proportion of the dataset to include in the test split.
  seed : int, optional
    The seed number for random operations.

  Returns
  -------
  DataFrame, DataFrame
    Preprocessed training and testing datasets.
  """
  if isinstance(test,type(None)) == False and isinstance(test_split,type(None)) == False:
    raise Exception("Trying to create test split when test data already exists")

  if isinstance(test,type(None)) == False:
    trainLen = data.shape[0]
    data = pd.concat([data, test])

  # binarise target
  if type(target) == int:
    target = data.columns[target]

  data[target] = data[target].astype('category')
  data[target] = data[target].cat.codes

  if len(pd.unique(data[target])) != 2:
    raise ValueError("Target variable must be binary")

  # convert given missing value signifier to NaN
  data = data.fillna(value = np.nan)
  if missing_val != np.nan: data = data.replace(missing_val, np.nan)

  if cols != None:
    tempTarg = data[target]
    if all([isinstance(item, int) for item in cols]) == True:
      data = data.iloc[:,cols] # selects columns as specified in cols given as integers
    elif all([isinstance(item, str) for item in cols]) == True:
      data = data[cols] # selects columns as specified in cols given as strings
    else: raise ValueError()
    data.insert(len(data.columns),target,tempTarg)

  # check for static columns

  # encode non-integer/non-continuous variables
  for i in data.columns:
    if(data.loc[:,i].dtype == 'object'):
      data.loc[:,i] = data.loc[:,i].astype('category')
      data.loc[:,i] = data.loc[:,i].cat.codes

  # impute median into missing values
  medianImputer = SimpleImputer(missing_values = np.nan, strategy = 'median')
  data = pd.DataFrame(medianImputer.fit_transform(data), columns = data.columns)

  if isinstance(test,type(None)) == False:
    train_data = data.iloc[0:trainLen,:]
    test_data = data.iloc[trainLen:data.shape[0],:]
  else: train_data = data

  if test_split != None:
    # if integer sample
    if type(test_split) == int:
      test_data = data.sample(n=test_split, random_state=np.random.seed(seed))

    # if float determine size and sample
    if type(test_split) == float:
      test_data = data.sample(frac=test_split, random_state=np.random.seed(seed))

    # if list select given rows
    if type(test_split) == list:
      test_data = data.iloc[test_split,:]

    # all rows not in test_data added to dataframe train_data
    train_data = data[~data.index.isin(test_data.index)]

  # separate target variable
  y_train = train_data.loc[:,target]
  x_train0 = train_data.drop(columns = target)

  y_test = test_data[target]
  x_test0 = test_data.drop(columns = target)

  return [x_train0, y_train, x_test0, y_test]

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

# %% [markdown]
# ### mlpmask

# %%
def train_mlp_batched(x_tr, y_tr, x_ts, y_ts, n_hidden, lr=0.001, weight_decay=0.00001, tolerance=0.001, patience=10, max_iter=10000, batch_size=32, device='cpu', seed=257):
    """
    Train a multi-layer perceptron model on the training data and validate it on the test data.

    Parameters
    ----------
    x_tr : pd.DataFrame
        Training data features.
    y_tr : pd.DataFrame
        Training data targets.
    x_ts : pd.DataFrame
        Test data features.
    y_ts : pd.DataFrame
        Test data targets.
    n_hidden : int
        Number of neurons in the hidden layer.
    lr : float, optional
        Learning rate for the optimizer.
    weight_decay : float, optional
        Weight decay for regularization.
    tolerance : float, optional
        Tolerance for early stopping.
    patience : int, optional
        Number of epochs to wait before early stop if no progress on the validation set.
    iter : int, optional
        Number of iterations for training.
    batch_size : int, optional
        Train/test data batch size.
    device : str, optional
        Device to run the model computation ('cpu' or 'cuda').
    seed : int, optional
        Seed for random number generation.

    Returns
    -------
    The PyTorch model.
    """
    torch.manual_seed(seed)
    device = torch.device(device)

    # Create TensorDatasets and DataLoaders
    train_dataset = TensorDataset(torch.tensor(x_tr.values, dtype=torch.float32), torch.tensor(y_tr.values, dtype=torch.float32).unsqueeze(1))
    test_dataset = TensorDataset(torch.tensor(x_ts.values, dtype=torch.float32), torch.tensor(y_ts.values, dtype=torch.float32).unsqueeze(1))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define the MLP model
    model = nn.Sequential(
        nn.Linear(x_tr.shape[1], n_hidden),
        nn.Tanh(),
        nn.Linear(n_hidden, 1),
        nn.Sigmoid()
    ).to(device)

    init.xavier_uniform_(model[0].weight, gain=init.calculate_gain('tanh'))
    init.zeros_(model[0].bias)
    init.xavier_uniform_(model[2].weight, gain=init.calculate_gain('sigmoid'))
    init.zeros_(model[2].bias)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Variables for early stopping
    min_loss = float('inf')
    patience_counter = 0
    best_epoch = 0

    # Training loop
    for epoch in range(max_iter):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()

        val_loss /= len(test_loader)
        train_loss /= len(train_loader)

        # Early stopping conditions
        if val_loss < min_loss - tolerance:
            min_loss = val_loss
            best_model_wts = model.state_dict()  # Save the best model
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Stopping early at epoch {best_epoch}")
            model.load_state_dict(best_model_wts)
            break

        if epoch % 1 == 0:
            print(f"Epoch {epoch}: Train loss {train_loss:.4f}, Val loss {val_loss:.4f}")

    return model

def extract_weights(model, combined=True):
    """
    Extracts weights and biases from a PyTorch model with a single hidden layer,
    optionally combining them into single matrices as in `mlpmask` and `mlpmask_pred`.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model from which weights are to be extracted.
    combined : bool, optional
        If True, combine weights and biases into the same matrix. Defaults to True.

    Returns
    -------
    list of torch.Tensor
        If combined is True, returns [W1, W2] where `W1` and `W2` are augmented weight matrices
        with dimensions [(input features + 1, hidden nodes), (hidden nodes + 1, 1)] respectively.
        If combined is False, returns [W1, B1, W2, B2] with separate weight and bias matrices.
    """
    layers = list(model.children())
    W1 = layers[0].weight.data
    B1 = layers[0].bias.data
    W2 = layers[2].weight.data
    B2 = layers[2].bias.data

    if combined:
        # Concatenate biases to weights to form the augmented weight matrices
        # W1 should be [input_features + 1, hidden_nodes]
        # W2 should be [hidden_nodes + 1, 1]
        W1_combined = torch.cat((W1, B1.unsqueeze(1)), 1).t()
        W2_combined = torch.cat((W2, B2.unsqueeze(1)), 1).t()
        return [W1_combined, W2_combined]
    else:
        return [W1, B1, W2, B2]
# %%
def mlpmask(x_tr: pd.DataFrame, y_tr: pd.DataFrame, x_ts: pd.DataFrame, y_ts: pd.DataFrame, n_hidden: int, mask: Optional[np.ndarray] = None, subnet_nodes: int = 1, lr: float = 0.001, weight_decay: float = 0.00001, tolerance: float = 0.001, patience: int = 10, iter: int = 10000, device: str = "cpu", seed: int = 257) -> Any:
  """
  Train a masked multi-layer perceptron model on the training data and validate it on the test data.

  Parameters
  ----------
  x_tr : pd.DataFrame
      Training data features.
  y_tr : pd.DataFrame
      Training data target.
  x_ts : pd.DataFrame
      Test data features.
  y_ts : pd.DataFrame
      Test data target.
  n_hidden : int
      Number of subnets in the hidden layer (one neuron each, by default).
  mask : np.ndarray, optional
      Mask to apply on the input layer.
  subnet_nodes : int, optional
      Number of nodes in each subnet. If called with a mask, this should match the subnet_nodes parameter provided when creating the mask by calling mlpmask.
  lr : float, optional
      Learning rate for the optimizer.
  weight_decay : float, optional
      Weight decay for regularization.
  tolerance : float, optional
      Tolerance for early stopping.
  patience : int, optional
      Number of epochs to wait before early stop if no progress on the validation set.
  iter : int, optional
      Number of iterations for training.
  device : str, optional
      Device to run the model computation ('cpu' or 'cuda').
  seed : int, optional
      seed for random number generation.

  Returns
  -------
  Any
      The trained neural network model.
  """

  torch.manual_seed(seed)

  #dtype = torch.double
  x_tr=x_tr.astype("float32")
  y_tr=y_tr.astype("float32")
  x_ts=x_ts.astype("float32")
  y_ts=y_ts.astype("float32")
  dtype = torch.float32
  torch_device = torch.device("cpu")
  x_tr = torch.tensor(x_tr.values,dtype=dtype)
  x_tra = torch.cat((x_tr,torch.tensor(np.ones([x_tr.shape[0],1]),dtype=dtype)),1)
  y_tr = torch.reshape(torch.tensor(y_tr.values,dtype=dtype),[len(y_tr),1])
  x_ts = torch.tensor(x_ts.values,dtype=dtype)
  x_tsa = torch.cat((x_ts,torch.tensor(np.ones([x_ts.shape[0],1]),dtype=dtype)),1)
  y_ts = torch.reshape(torch.tensor(y_ts.values,dtype=dtype),[len(y_ts),1])
  if device.lower() != "cpu":
    if device.lower() == "cuda":
      torch_device = torch.device("cuda:0")
    if device.lower() == "mps":
      torch_device = torch.device("mps")
    x_tr = x_tr.to(torch_device)
    x_tra = x_tra.to(torch_device)
    y_tr = y_tr.to(torch_device)
    x_ts = x_ts.to(torch_device)
    x_tsa = x_tsa.to(torch_device)
    y_ts = y_ts.to(torch_device)

  batch_size = x_tra.shape[0]
  input_shape = x_tra.shape[1]
  hidden = subnet_nodes*n_hidden
  learning_rate = lr/np.sqrt(batch_size)

  W1 = torch.randn(input_shape,hidden, device=torch_device, dtype=dtype)
  W2 = torch.randn(hidden+1,1, device=torch_device, dtype=dtype)
  if device.lower() != "cpu":
    W1 = W1.to(torch_device)
    W2 = W2.to(torch_device)
  print(W1.dtype)

  if isinstance(mask,type(None)) == False:
    if mask.shape[1] != hidden:
      warnings.warn(
        f"The shape of the mask ({mask.shape}) doesn't match the the hidden layer dimension ({hidden}). Check that the mask matches `n_hidden` and `subnet_nodes`.")
    mask_a = np.concatenate((mask,np.ones([1,mask.shape[1]])),0)

  losses_tr,losses_ts = list(),list()

  for t in range(iter):
    # forward pass
    z1 = x_tra.mm(W1)
    # h1 = 1/(1+torch.exp(-z1))
    h1 = (torch.exp(z1)-torch.exp(-z1))/(torch.exp(z1)+torch.exp(-z1)) # tanh
    if device.lower() == "cpu":
      h1a = torch.cat((h1,torch.tensor(np.ones([x_tr.shape[0],1]),dtype=dtype)),1)
    elif device.lower() != "cpu":
      h1a = torch.cat((h1,torch.tensor(np.ones([x_tr.shape[0],1]),dtype=dtype).to(torch_device)),1)
    y_pred_tr = 1/(1+torch.exp(-h1a.mm(W2)))

    # backprop
    grad_y_pred = (y_pred_tr - y_tr)
    grad_w2 = h1a.t().mm(grad_y_pred)
    grad_h1 = grad_y_pred.mm(W2[0:W2.shape[0]-1,:].t())
    # grad_z1 = grad_h1*h1*(1-h1)
    grad_z1 = grad_h1*(1-h1*h1)
    grad_w1 = x_tra.t().mm(grad_z1)

    # update weights and apply mask
    W1 -= learning_rate * grad_w1 + (weight_decay*W1)
    W1 = W1.to(torch_device)
    if isinstance(mask,type(None)) == False:
      if device.lower() != "cpu":
        W1 = W1 * torch.tensor(mask_a).to(torch_device)
        W1 = W1.to(device=torch_device,dtype=dtype)
    W2 -= learning_rate * grad_w2 + (weight_decay*W2)

    #recalculate training feedforward for loss
    z1 = x_tra.mm(W1)
    # h1 = 1/(1+torch.exp(-z1))
    h1 = (torch.exp(z1)-torch.exp(-z1))/(torch.exp(z1)+torch.exp(-z1)) # tanh
    if device.lower() == "cpu":
      h1a = torch.cat((h1,torch.tensor(np.ones([x_tr.shape[0],1]),dtype=dtype)),1)
    elif device.lower() != "cpu":
      h1a = torch.cat((h1,torch.tensor(np.ones([x_tr.shape[0],1]),dtype=dtype).to(torch_device)),1)
    y_pred_tr = 1/(1+torch.exp(-h1a.mm(W2)))
    loss = (torch.mul(-y_tr,torch.nan_to_num(torch.log(y_pred_tr)))-torch.mul((1-y_tr),torch.nan_to_num(torch.log(1-y_pred_tr)))).sum().item()

    # forward pass on test data
    z1 = x_tsa.mm(W1)
    # h1 = 1/(1+torch.exp(-z1))
    h1 = (torch.exp(z1)-torch.exp(-z1))/(torch.exp(z1)+torch.exp(-z1)) # tanh
    if device.lower() == "cpu":
      h1a = torch.cat((h1,torch.tensor(np.ones([x_ts.shape[0],1]),dtype=dtype)),1)
    elif device.lower() != "cpu":
      h1a = torch.cat((h1,torch.tensor(np.ones([x_ts.shape[0],1]),dtype=dtype).to(torch_device)),1)
    y_pred_ts = 1/(1+torch.exp(-h1a.mm(W2)))
    loss_ts = (torch.mul(-y_ts,torch.nan_to_num(torch.log(y_pred_ts)))-torch.mul((1-y_ts),torch.nan_to_num(torch.log(1-y_pred_ts)))).sum().item()

    losses_tr.append(loss)
    losses_ts.append(loss_ts)

    if t == 0:
      best_loss_tr = loss
      best_loss_ts = loss_ts
      best_epoch = t
      best_W1 = W1
      best_W2 = W2

    if best_loss_ts - loss_ts > tolerance:
      best_loss_tr = loss
      best_loss_ts = loss_ts
      best_epoch = t
      best_W1 = W1
      best_W2 = W2

    if t%100 == 0:
      print("Epoch {}: ".format(t), "Loss - {},".format(loss), "Test Loss - {}".format(loss_ts))
    if t - best_epoch >= patience:
      print("----------------------- Stopped at {} epochs -----------------------".format(t))
      print("                       ----  Best Epoch  ----                       ")
      print("Epoch {}: ".format(best_epoch), "Loss - {},".format(best_loss_tr), "Test Loss - {}".format(best_loss_ts))
      break

    W1 = best_W1
    W2 = best_W2

  #return [W1, W2], losses_tr, losses_ts, y_pred_tr.numpy(), y_pred_ts.numpy()
  return [W1, W2]

# %% [markdown]
# ### mlpmask_pred

# %%
def mlpmask_pred(x_ts: pd.DataFrame, model: Any, device: str = "cpu") -> np.ndarray:
  """
  Predict outcomes using the trained multi-layer perceptron model.

  Parameters
  ----------
  x_ts : pd.DataFrame
      Testing data features.
  model : Any
      The trained MLP model.
  device : str, optional
      Device to run the model computation ('cpu' or 'cuda').

  Returns
  -------
  np.ndarray
      Predicted values.
  """
  # initialize dtype and device config
  dtype = torch.float32
  if device.lower() == "cuda":
    torch_device = "cuda"
  if device.lower() == "mps":
    torch_device = "mps"
  else:
    torch_device = "cpu"

  # convert input and ensure it's on the right device
  if isinstance(x_ts, np.ndarray):
      x_ts = torch.from_numpy(x_ts).float()
  else:
      x_ts = torch.tensor(x_ts.values, dtype=dtype)

  x_ts = x_ts.to(torch_device)

  # input data append ones for bias
  ones = torch.ones(x_ts.shape[0], 1, dtype=dtype, device=torch_device)
  x_tsa = torch.cat((x_ts, ones), 1)

  # hidden layer forward pass
  z1 = x_tsa.mm(model[0].to(torch_device))
  # h1 = torch.tanh(z1)
  h1 = (torch.exp(z1)-torch.exp(-z1))/(torch.exp(z1)+torch.exp(-z1)) # tanh

  # output layer append ones for bias
  ones_h1 = torch.ones(h1.shape[0], 1, dtype=dtype, device=torch_device)
  h1a = torch.cat((h1, ones_h1), 1)

  # output layer forward pass and detach from device to return as numpy array
  # y_pred = torch.sigmoid(h1a.mm(model[1].to(torch_device)))
  y_pred = 1/(1+torch.exp(-h1a.mm(model[1].to(torch_device)))) # sigmoid
  y_pred = y_pred.detach().cpu().numpy()

  return y_pred

# %% [markdown]
# ### modelStats

# %%

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

# %% [markdown]
# ### partialResponses

# %%
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

# %% [markdown]
# ### selectLambda

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
# ### prPlots

# %%
def prPlots(betas: List[float], userLambda: float, x_train0: pd.DataFrame, x_train: pd.DataFrame, data: pd.DataFrame, model: Any, bivariate_inputs: List[int], n_steps: int = 15, sd_scale: int = 2, method: str = "dirac", device: str = "cpu") -> None:
  """
  Generate partial response plots based on the selected lambda and model coefficients.

  Parameters
  ----------
  betas : List[float]
      Model coefficients.
  userLambda : float
      Selected lambda value.
  x_train0 : pd.DataFrame
      Original training dataset before any transformations.
  x_train : pd.DataFrame
      Transformed training dataset.
  data : pd.DataFrame
      Combined dataset used for training/testing.
  model : Any
      The trained model.
  bivariate_inputs : List[int]
      Indices of features to be used for bivariate analysis.
  n_steps : int, optional
      Number of steps to use for generating plots.
  sd_scale : int, optional
      Scaling factor for standard deviation in the data normalization.
  method : str, optional
      Method used to compute the partial responses ('dirac' or 'lebesgue').
  device : str, optional
      Device to run the model computation ('cpu' or 'cuda').

  Returns
  -------
  None
  """
  if method.lower() == "dirac":

    x0 = np.zeros((1,x_train.shape[1]))
    if isinstance(model,list) == True:
      y0 = mlpmask_pred(x0,model,device=device)
    elif hasattr(model,'predict_proba'):
      y0 = model.predict_proba(x0)[:,1]
    else:
      y0 = model.predict(x0,device=device)

    logit_y0 = np.log(y0/(1-y0))

    # plot all selected partial responses
    for pr in np.where(abs(betas[:,userLambda])>0.1)[0]:

      # univariate partial response plots ----------------------------------------------------------------------------------------------------------------- univariate plots
      if pr < x_train.shape[1]:
        print(pr,"- univ")

        x_step0 = np.linspace(min(x_train0.iloc[:,pr]),max(x_train0.iloc[:,pr]),n_steps)
        x_step = np.linspace(min(x_train.iloc[:,pr]),max(x_train.iloc[:,pr]),n_steps)

        x_in = np.zeros([n_steps,x_train.shape[1]])
        x_in[:,pr] = x_step
        if isinstance(model,list) == True:
          pred_y_xi = mlpmask_pred(x_in,model,device=device)
        elif hasattr(model,'predict_proba'):
          pred_y_xi = model.predict_proba(x_in)[:,1]
        else:
          pred_y_xi = model.predict(x_in, device=device)

        y_xi = np.log(pred_y_xi/(1-pred_y_xi))-logit_y0

        fig, ax1 = plt.subplots()
        ax1.set_title("Partial Response Plot")
        ax1.set_xlabel(x_train0.columns[pr])
        ax1.set_ylabel("Contribution to logit",color="red")
        ax2 = ax1.twinx()
        ax2.set_ylabel("Frequency")

        # pr plot for discrete data
        if len(x_train0.iloc[:,pr].unique()) < n_steps:
          ax2.bar(np.sort(x_train0.iloc[:,pr].unique()),x_train0.iloc[:,pr].value_counts().sort_index(),facecolor="none",ec="black")
        # pr plot for continuous data
        else:
          ax2.hist(x_train0.iloc[:,pr],histtype="bar",facecolor="none",ec="black",bins=n_steps)
        ax1.plot((x_step*(x_train0.iloc[:,pr].std()*2))+x_train0.iloc[:,pr].median(),y_xi,color="red")
        plt.show()

      # bivariate partial response plots ----------------------------------------------------------------------------------------------------------------------- bivariate plots
      else:
        pr_i = int(bivariate_inputs[pr-x_train.shape[1],0])
        pr_j = int(bivariate_inputs[pr-x_train.shape[1],1])

        if (len(x_train.iloc[:,pr_i].unique()) < n_steps) != (len(x_train.iloc[:,pr_j].unique()) < n_steps):
          if len(x_train.iloc[:,pr_i].unique()) < n_steps:
            x_step0_i = np.sort(x_train0.iloc[:,pr_i].unique())
            x_step0_j = np.linspace(min(x_train0.iloc[:,pr_j]),max(x_train0.iloc[:,pr_j]),n_steps)
          else:
            x_step0_i = np.linspace(min(x_train0.iloc[:,pr_i]),max(x_train0.iloc[:,pr_i]),n_steps)
            x_step0_j = np.sort(x_train0.iloc[:,pr_j].unique())

        elif len(x_train.iloc[:,pr_i].unique()) < n_steps and len(x_train.iloc[:,pr_j].unique()) < n_steps:
          x_step0_i = np.sort(x_train0.iloc[:,pr_i].unique())
          x_step0_j = np.sort(x_train0.iloc[:,pr_j].unique())

        else:
          x_step0_i = np.linspace(min(x_train0.iloc[:,pr_i]),max(x_train0.iloc[:,pr_i]),n_steps)
          x_step0_j = np.linspace(min(x_train0.iloc[:,pr_j]),max(x_train0.iloc[:,pr_j]),n_steps)

        y_xij = np.zeros([len(x_step0_j),len(x_step0_i)])

        for i in range(0,len(x_step0_i)):
          for j in range(0,len(x_step0_j)):
            x_in = np.zeros(x_train.shape[1])
            x_in[pr_i] = (x_step0_i[i]-x_train0.iloc[:,pr_i].median())/(x_train0.iloc[:,pr_i].std()*2)
            x_in[pr_j] = (x_step0_j[j]-x_train0.iloc[:,pr_j].median())/(x_train0.iloc[:,pr_j].std()*2)
            if isinstance(model,list) == True:
              pred_y_xij = mlpmask_pred(x_in.reshape(1,len(x_in)),model,device=device)
            elif hasattr(model,'predict_proba'):
              pred_y_xij = model.predict_proba(x_in.reshape(1,len(x_in)))[:,1]
            else:
              # pred_y_xij = model.predict(x_in, device=device.reshape(1,len(x_in)))
              pred_y_xij = model.predict(x_in.reshape(1,len(x_in)), device=device)

            y_xij[j,i] = np.log(pred_y_xij[0][0]/(1-pred_y_xij[0][0]))-logit_y0

# ------------------------------------------------------------------------------------------------------------------------------------------------ mixed responses
        if (len(x_train.iloc[:,pr_i].unique()) < n_steps) != (len(x_train.iloc[:,pr_j].unique()) < n_steps):

          fig, ax1 = plt.subplots()
          ax1.set_title("Partial Response Plot")

          ax1.set_ylabel("Contribution to logit",color="red")
          ax2 = ax1.twinx()
          ax2.set_ylabel("Frequency")
          colourmap = plt.get_cmap('seismic_r')
          if len(x_train.iloc[:,pr_i].unique()) < n_steps:
            for i in range(0,len(x_train.iloc[:,pr_i].unique())):
              ax1.set_xlabel(x_train0.columns[pr_j])
              ax1.plot(x_step0_j,y_xij[:,i],label=np.sort(x_train0.iloc[:,pr_i].unique())[i])
              ax1.legend(title=x_train0.columns[pr_i])
          else:
            for j in range(0,len(x_train.iloc[:,pr_j].unique())):
              ax1.set_xlabel(x_train0.columns[pr_i])
              ax1.plot(x_step0_i,y_xij[j,:],label=np.sort(x_train0.iloc[:,pr_j].unique())[j])
              ax1.legend(title=x_train0.columns[pr_j])

          plt.show()

        else:
          fig = plt.figure()
          ax = plt.axes()
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------- categorical/categorical
          if len(x_train.iloc[:,pr_i].unique()) < n_steps and len(x_train.iloc[:,pr_j].unique()) < n_steps:
            heatmap = ax.imshow(y_xij,cmap="viridis", aspect="auto")
            print(x_step0_i)
            ax.set_xticks(list(range(0,len(x_step0_i))),labels=np.round(x_step0_i,2))
            ax.set_yticks(list(range(0,len(x_step0_j))),labels=np.round(x_step0_j,2))
            boxSettings = dict(boxstyle='round', facecolor='grey', alpha=0.5)
            for i in range(0,len(x_step0_i)):
              for j in range(0,len(x_step0_j)):
                ax.text(i, j, np.round(y_xij[j, i],2),ha="center", va="center", color="w",bbox=boxSettings)
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------- continuous/continuous
          else:
            X, Y = np.meshgrid(x_step0_i.reshape(len(x_step0_i),1), x_step0_j.reshape(len(x_step0_j),1))
            contour_heatmap = ax.contourf(X,Y,y_xij)
            c2 = plt.contour(X,Y,y_xij, cmap='Greys')
            ax.clabel(c2, inline=True, fontsize=10)
            ax.set_xticks(x_step0_i,labels=np.round(x_step0_i,2),rotation=45)
            ax.set_yticks(x_step0_j,labels=np.round(x_step0_j,2))
            fig.colorbar(contour_heatmap, orientation='vertical')
          ax.set_xlabel(x_train.columns[pr_i])
          ax.set_ylabel(x_train.columns[pr_j])

          plt.show()

#---------------------------------------------------------------------------------------------------------LEBESGUE MEASURE-------------------------------------------------------------------------------------------------------

  if method.lower() == "lebesgue":

    if isinstance(model,list) == True:
      y0 = mlpmask_pred(x_train,model,device=device)
    elif hasattr(model,'predict_proba'):
      y0 = model.predict_proba(x_train)[:,1]
    else:
      y0 = model.predict(x_train, device=device)

    logit_y0 = np.mean(np.log(y0/(1-y0)))

    # plot all selected partial responses
    for pr in np.where(abs(betas[:,userLambda])>0.1)[0]:

      # univariate parital response plots ----------------------------------------------------------------------------------------------------------------- univariate plots
      if pr < x_train.shape[1]:

        x_step0 = np.linspace(min(x_train0.iloc[:,pr]),max(x_train0.iloc[:,pr]),n_steps)
        x_step = np.linspace(min(x_train.iloc[:,pr]),max(x_train.iloc[:,pr]),n_steps)
        y_xi = np.zeros(len(x_step))

        for k in range(0,len(x_step)):

          x_in = x_train.copy()
          x_in.iloc[:,pr] = x_step[k]

          if isinstance(model,list) == True:
            pred_y_xi = mlpmask_pred(x_in,model,device=device)
          elif hasattr(model,'predict_proba'):
            pred_y_xi = model.predict_proba(x_in)[:,1]
          else:
            pred_y_xi = model.predict(x_in, device=device)

          y_xi[k] = np.mean(np.log(pred_y_xi/(1-pred_y_xi)))-logit_y0

        fig, ax1 = plt.subplots()
        ax1.set_title("Partial Response Plot")
        ax1.set_xlabel(x_train0.columns[pr])
        ax1.set_ylabel("Contribution to logit",color="red")
        ax2 = ax1.twinx()
        ax2.set_ylabel("Frequency")

        # pr plot for discrete data
        if len(x_train0.iloc[:,pr].unique()) < n_steps:
          ax2.bar(np.sort(x_train0.iloc[:,pr].unique()),x_train0.iloc[:,pr].value_counts().sort_index(),facecolor="none",ec="black")
        # pr plot for continuous data
        else:
          ax2.hist(x_train0.iloc[:,pr],histtype="bar",facecolor="none",ec="black",bins=n_steps)
        ax1.plot((x_step*(x_train0.iloc[:,pr].std()*2))+x_train0.iloc[:,pr].median(),y_xi,color="red")
        plt.show()

      # bivariate partial response plots ----------------------------------------------------------------------------------------------------------------------- bivariate plots
      else:
        pr_i = int(bivariate_inputs[pr-x_train.shape[1],0])
        pr_j = int(bivariate_inputs[pr-x_train.shape[1],1])

        if (len(x_train.iloc[:,pr_i].unique()) < n_steps) != (len(x_train.iloc[:,pr_j].unique()) < n_steps):
          if len(x_train.iloc[:,pr_i].unique()) < n_steps:
            x_step0_i = np.sort(x_train0.iloc[:,pr_i].unique())
            x_step0_j = np.linspace(min(x_train0.iloc[:,pr_j]),max(x_train0.iloc[:,pr_j]),n_steps)
          else:
            x_step0_i = np.linspace(min(x_train0.iloc[:,pr_i]),max(x_train0.iloc[:,pr_i]),n_steps)
            x_step0_j = np.sort(x_train0.iloc[:,pr_j].unique())

        elif len(x_train.iloc[:,pr_i].unique()) < n_steps and len(x_train.iloc[:,pr_j].unique()) < n_steps:
          x_step0_i = np.sort(x_train0.iloc[:,pr_i].unique())
          x_step0_j = np.sort(x_train0.iloc[:,pr_j].unique())

        else:
          x_step0_i = np.linspace(min(x_train0.iloc[:,pr_i]),max(x_train0.iloc[:,pr_i]),n_steps)
          x_step0_j = np.linspace(min(x_train0.iloc[:,pr_j]),max(x_train0.iloc[:,pr_j]),n_steps)


        y_xij = np.zeros([len(x_step0_j),len(x_step0_i)])

        for i in range(0,len(x_step0_i)):
          for j in range(0,len(x_step0_j)):
            x_in = x_train.copy()
            x_in.iloc[:,pr_i] = (x_step0_i[i]-x_train0.iloc[:,pr_i].median())/(x_train0.iloc[:,pr_i].std()*2)
            x_in.iloc[:,pr_j] = (x_step0_j[j]-x_train0.iloc[:,pr_j].median())/(x_train0.iloc[:,pr_j].std()*2)

            if isinstance(model,list) == True:
              pred_y_xij = mlpmask_pred(x_in,model,device=device)
            elif hasattr(model,'predict_proba'):
              pred_y_xij = model.predict_proba(x_in)[:,1]
            else:
              pred_y_xij = model.predict(x_in, device=device)

            y_xij[j,i] = np.log(np.mean(pred_y_xij)/(1-np.mean(pred_y_xij)))-logit_y0


# ------------------------------------------------------------------------------------------------------------------------------------------------ mixed responses
        if (len(x_train.iloc[:,pr_i].unique()) < n_steps) != (len(x_train.iloc[:,pr_j].unique()) < n_steps):

          fig, ax1 = plt.subplots()
          ax1.set_title("Partial Response Plot")

          ax1.set_ylabel("Contribution to logit",color="red")
          ax2 = ax1.twinx()
          ax2.set_ylabel("Frequency")
          colourmap = plt.get_cmap('seismic_r')
          if len(x_train.iloc[:,pr_i].unique()) < n_steps:
            for i in range(0,len(x_train.iloc[:,pr_i].unique())):
              ax1.set_xlabel(x_train0.columns[pr_j])
              ax1.plot(x_step0_j,y_xij[:,i],label=np.sort(x_train0.iloc[:,pr_i].unique())[i])
              ax1.legend(title=x_train0.columns[pr_i])
          else:
            for j in range(0,len(x_train.iloc[:,pr_j].unique())):
              ax1.set_xlabel(x_train0.columns[pr_i])
              ax1.plot(x_step0_i,y_xij[j,:],label=np.sort(x_train0.iloc[:,pr_j].unique())[j])
              ax1.legend(title=x_train0.columns[pr_j])

          plt.show()

        else:
          fig = plt.figure()
          ax = plt.axes()
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------- categorical/categorical
          if len(x_train.iloc[:,pr_i].unique()) < n_steps and len(x_train.iloc[:,pr_j].unique()) < n_steps:
            heatmap = ax.imshow(y_xij,cmap="viridis", aspect="auto")
            ax.set_xticks(list(range(0,len(x_step0_i))),labels=np.round(x_step0_i,2))
            ax.set_yticks(list(range(0,len(x_step0_j))),labels=np.round(x_step0_j,2))
            boxSettings = dict(boxstyle='round', facecolor='grey', alpha=0.5)
            for i in range(0,len(x_step0_i)):
              for j in range(0,len(x_step0_j)):
                ax.text(i, j, np.round(y_xij[j, i],2),ha="center", va="center", color="w",bbox=boxSettings)
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------- continuous/continuous
          else:

            X, Y = np.meshgrid(x_step0_i.reshape(len(x_step0_i),1), x_step0_j.reshape(len(x_step0_j),1))
            contour_heatmap = ax.contourf(X,Y,y_xij)
            c2 = plt.contour(X,Y,y_xij, cmap='Greys')
            ax.clabel(c2, inline=True, fontsize=10)
            ax.set_xticks(x_step0_i,labels=np.round(x_step0_i,2),rotation=45)
            ax.set_yticks(x_step0_j,labels=np.round(x_step0_j,2))
            fig.colorbar(contour_heatmap, orientation='vertical')
          ax.set_xlabel(x_train.columns[pr_i])
          ax.set_ylabel(x_train.columns[pr_j])
          plt.show()


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

# %% [markdown]
# ### prNomogram

# %%
def prNomogram(betas: List[float], userLambda: float, x_train0: pd.DataFrame, x_train: pd.DataFrame, data: pd.DataFrame, model: Any, bivariate_inputs: List[int], n_steps: int = 15, sd_scale: int = 2, method: str = "dirac", device: str = "cpu") -> None:
  """
  Generate a nomogram for visualizing the effect of each predictor variable on the outcome.

  Parameters
  ----------
  betas : List[float]
      Model coefficients.
  userLambda : float
      Selected lambda value.
  x_train0 : pd.DataFrame
      Original training dataset before any transformations.
  x_train : pd.DataFrame
      Transformed training dataset.
  data : pd.DataFrame
      Combined dataset used for training/testing.
  model : Any
      The trained model.
  bivariate_inputs : List[int]
      Indices of features to be used for bivariate analysis.
  n_steps : int, optional
      Number of steps to use for generating the nomogram.
  sd_scale : int, optional
      Scaling factor for standard deviation in the data normalization.
  method : str, optional
      Method used to compute the partial responses ('dirac' or 'lebesgue').
  device : str, optional
      Device to run the model computation ('cpu' or 'cuda').

  Returns
  -------
  None
  """
  nUniv = (np.where(abs(betas[:,userLambda])>0.1)[0])

  m_pr = list()

  for pr in np.where(abs(betas[:,userLambda])>0.1)[0]:

    if pr >= x_train.shape[1]:
      pr_i = int(bivariate_inputs[pr-x_train.shape[1]][0])
      pr_j = int(bivariate_inputs[pr-x_train.shape[1]][1])
      if (len(x_train.iloc[:,pr_i].unique()) < 15) != (len(x_train.iloc[:,pr_j].unique()) < 15):
        m_pr.append(pr)

  # Set up nomogram figure
  nomo = plt.figure(figsize=(6,len(np.where(nUniv<x_train.shape[1])[0])+len(m_pr)))
  ax0 = nomo.add_subplot(len(nUniv[nUniv<x_train.shape[1]])+len(m_pr),1,1)
  ax0.xaxis.set_ticks_position('top')
  ax0.xaxis.set_label_position('top')
  ax0.spines['top'].set_position(('outward', 10))
  ax0.set_xlabel("log OR")

  if method.lower() == "dirac":

    x0 = np.zeros((1,x_train.shape[1]))
    if isinstance(model,list) == True:
      y0 = mlpmask_pred(x0,model,device=device)
    elif hasattr(model,'predict_proba'):
      y0 = model.predict_proba(x0)[:,1]
    else:
      y0 = model.predict(x0)
    logit_y0 = np.log(y0/(1-y0))

    if np.where(abs(betas[:,userLambda])>0.1)[0][0] < x_train.shape[1]:
      response = np.where(abs(betas[:,userLambda])>0.1)[0][0]
      # Initial subplot for categorical case
      if len(x_train.iloc[:,response].unique()) < n_steps:

        catvals = x_train.iloc[:,response].unique()
        x_in = np.zeros([len(catvals),x_train.shape[1]])
        x_in[:,response] = catvals

        if isinstance(model,list) == True:
          pred_y_xi = mlpmask_pred(x_in,model,device=device)
        elif hasattr(model,'predict_proba'):
          pred_y_xi = model.predict_proba(x_in)[:,1]
        else:
          pred_y_xi = model.predict(x_in)
        logit_y = np.reshape(np.log(pred_y_xi/(1-pred_y_xi))-logit_y0,[len(pred_y_xi),])

        ax0.axvline(0,color="black")
        ax0.set_ylabel(x_train.columns[response],rotation=0,loc="bottom",labelpad=-1)
        ax0.set_yticks([])
        ax0.plot(logit_y,np.full(len(logit_y),0.5))
        ax0.scatter(logit_y,np.full(len(logit_y),0.5),marker="|")
        for i in range(0,len(logit_y)):
          ax0.annotate(data[x_train.columns[response]].unique()[i],(logit_y[i],0.5),xytext=(logit_y[i],0.505))

      # Initial subplot for continuous case
      else:
        x_step0 = np.linspace(min(x_train0.iloc[:,0]),max(x_train0.iloc[:,0]),n_steps)
        x_step = np.linspace(min(x_train.iloc[:,0]),max(x_train.iloc[:,0]),n_steps)
        x_step0_small = np.linspace(min(x_train0.iloc[:,0]),max(x_train0.iloc[:,0]),5)


        x_in = np.zeros([n_steps,x_train.shape[1]])
        x_in[:,response] = x_step
        if isinstance(model,list) == True:
          pred_y_xi = mlpmask_pred(x_in,model,device=device)
        elif hasattr(model,'predict_proba'):
          pred_y_xi = model.predict_proba(x_in)[:,1]
        else:
          pred_y_xi = model.predict(x_in)

        logit_y = np.reshape(np.log(pred_y_xi/(1-pred_y_xi))-logit_y0,[len(pred_y_xi),])

        ax0.plot(logit_y,x_step0)

        line_marks0 = ax0.get_yticks()
        line_marks0 = line_marks0[np.where(line_marks0<=max(x_step0))]
        if min(x_step0) < 0:
          line_marks0 = line_marks0[np.where(line_marks0>=min(x_step0))]
        else:
          line_marks0 = line_marks0[np.where(line_marks0>=0)]
        if len(line_marks0) == 2:
          line_marks0 = np.linspace(line_marks0[0],line_marks0[1],3)


        ax0.set_yticks(line_marks0)

        line_marks = (line_marks0 - x_train0.iloc[:,pr].median())/(x_train0.iloc[:,pr].std()*2)

        x_in = np.zeros([len(line_marks),x_train.shape[1]])
        x_in[:,pr] = line_marks
        if isinstance(model,list) == True:
          pred_y_xi = mlpmask_pred(x_in,model,device=device)
        elif hasattr(model,'predict_proba'):
          pred_y_xi = model.predict_proba(x_in)[:,1]
        else:
          pred_y_xi = model.predict(x_in)

        logit_line_marks = np.reshape(np.log(pred_y_xi/(1-pred_y_xi))-logit_y0,[len(pred_y_xi),])

        ax0.scatter(logit_line_marks,line_marks0,marker="|")
        for i in range(0,len(line_marks0)):
          ax0.annotate(line_marks0[i],(logit_line_marks[i],line_marks0[i]),xytext=(logit_line_marks[i]-0.1,line_marks0[i]+1))

        #ax.scatter(logit_y[range(0,len(logit_y),3)],x_step0[range(0,len(logit_y),3)],marker="|")
        #for i in range(0,len(logit_y),3):
          #ax.annotate(np.round(x_step0[i],2),(logit_y[i],x_step0[i]),xytext=(logit_y[i]-0.1,x_step0[i]+1.5))
        ax0.set_ylabel(x_train.columns[response],rotation=0,loc="bottom",labelpad=-1)
        ax0.yaxis.tick_right()
        ax0.axvline(0,color="black")

      # Loop creates subplot for each other univariate response
      subplot = 1
      for pr in np.where(abs(betas[:,userLambda])>0.1)[0]:
        if pr < x_train.shape[1]:

          # categorical case
          if len(x_train.iloc[:,pr].unique()) < n_steps:

            catvals = x_train.iloc[:,pr].unique()

            x_in = np.zeros([len(catvals),x_train.shape[1]])
            x_in[:,pr] = catvals

            if isinstance(model,list) == True:
              pred_y_xi = mlpmask_pred(x_in,model,device=device)
            elif hasattr(model,'predict_proba'):
              pred_y_xi = model.predict_proba(x_in)[:,1]
            else:
              pred_y_xi = model.predict(x_in)

            logit_y = np.reshape(np.log(pred_y_xi/(1-pred_y_xi))-logit_y0,[len(pred_y_xi),])

            ax = nomo.add_subplot(len(nUniv[nUniv<x_train.shape[1]])+len(m_pr),1,subplot,sharex=ax0)
            ax.axvline(0,color="black")
            ax.set_yticks([])
            ax.set_ylabel(x_train.columns[pr],rotation=0,loc="bottom",labelpad=-1)
            ax.get_xaxis().set_visible(False)
            ax.plot(logit_y,np.full(len(logit_y),0.5))
            ax.scatter(logit_y,np.full(len(logit_y),0.5),marker="|")
            for i in range(0,len(logit_y)):
              ax.annotate(data[x_train.columns[pr]].unique()[i],(logit_y[i],0.5),xytext=(logit_y[i],0.505))


          # continuous case
          else:
            x_step0 = np.linspace(min(x_train0.iloc[:,pr]),max(x_train0.iloc[:,pr]),n_steps)
            x_step = np.linspace(min(x_train.iloc[:,pr]),max(x_train.iloc[:,pr]),n_steps)

            x_in = np.zeros([n_steps,x_train.shape[1]])
            x_in[:,pr] = x_step
            if isinstance(model,list) == True:
              pred_y_xi = mlpmask_pred(x_in,model,device=device)
            elif hasattr(model,'predict_proba'):
              pred_y_xi = model.predict_proba(x_in)[:,1]
            else:
              pred_y_xi = model.predict(x_in)

            logit_y = np.reshape(np.log(pred_y_xi/(1-pred_y_xi))-logit_y0,[len(pred_y_xi),])

            ax = nomo.add_subplot(len(nUniv[nUniv<x_train.shape[1]])+len(m_pr),1,subplot,sharex=ax0)
            ax.plot(logit_y,x_step0)

            line_marks0 = ax.get_yticks()
            line_marks0 = line_marks0[np.where(line_marks0<=max(x_step0))]
            if min(x_step0) < 0:
              line_marks0 = line_marks0[np.where(line_marks0>=min(x_step0))]
            else:
              line_marks0 = line_marks0[np.where(line_marks0>=0)]
            if len(line_marks0) == 2:
              line_marks0 = np.linspace(line_marks0[0],line_marks0[1],3)


            ax.set_yticks(line_marks0)

            line_marks = (line_marks0 - x_train0.iloc[:,pr].median())/(x_train0.iloc[:,pr].std()*2)

            x_in = np.zeros([len(line_marks),x_train.shape[1]])
            x_in[:,pr] = line_marks
            if isinstance(model,list) == True:
              pred_y_xi = mlpmask_pred(x_in,model,device=device)
            elif hasattr(model,'predict_proba'):
              pred_y_xi = model.predict_proba(x_in)[:,1]
            else:
              pred_y_xi = model.predict(x_in)

            logit_line_marks = np.reshape(np.log(pred_y_xi/(1-pred_y_xi))-logit_y0,[len(pred_y_xi),])

            ax.scatter(logit_line_marks,line_marks0,marker="|")
            for i in range(0,len(line_marks0)):
              ax.annotate(line_marks0[i],(logit_line_marks[i],line_marks0[i]),xytext=(logit_line_marks[i]-0.1,line_marks0[i]+1))

            #ax.scatter(logit_y[range(0,len(logit_y),3)],x_step0[range(0,len(logit_y),3)],marker="|")
            #for i in range(0,len(logit_y),3):
              #ax.annotate(np.round(x_step0[i],2),(logit_y[i],x_step0[i]),xytext=(logit_y[i]-0.1,x_step0[i]+1.5),fontsize=8)
            ax.axvline(0,color="black")
            ax.set_ylabel(x_train.columns[pr],rotation=0,loc="bottom",labelpad=-1)
            ax.yaxis.tick_right()
            ax.get_xaxis().set_visible(False)
          subplot+=1

        elif pr >= x_train.shape[1]:
          pr_i = int(bivariate_inputs[pr-x_train.shape[1],0])
          pr_j = int(bivariate_inputs[pr-x_train.shape[1],1])
          if (len(x_train.iloc[:,pr_i].unique()) < 15) != (len(x_train.iloc[:,pr_j].unique()) < 15):
            "mixed response"
            if len(x_train.iloc[:,pr_i].unique()) < n_steps:
              x_step0_i = np.sort(x_train0.iloc[:,pr_i].unique())
              x_step0_j = np.linspace(min(x_train0.iloc[:,pr_j]),max(x_train0.iloc[:,pr_j]),n_steps)
            else:
              x_step0_i = np.linspace(min(x_train0.iloc[:,pr_i]),max(x_train0.iloc[:,pr_i]),n_steps)
              x_step0_j = np.sort(x_train0.iloc[:,pr_j].unique())

            y_xij = np.zeros([len(x_step0_j),len(x_step0_i)])

            for i in range(0,len(x_step0_i)):
              for j in range(0,len(x_step0_j)):
                x_in = np.zeros(x_train.shape[1])
                x_in[pr_i] = (x_step0_i[i]-x_train0.iloc[:,pr_i].median())/(x_train0.iloc[:,pr_i].std()*2)
                x_in[pr_j] = (x_step0_j[j]-x_train0.iloc[:,pr_j].median())/(x_train0.iloc[:,pr_j].std()*2)
                if isinstance(model,list) == True:
                  pred_y_xij = mlpmask_pred(x_in.reshape(1,len(x_in)),model,device=device)
                elif hasattr(model,'predict_proba'):
                 pred_y_xij = model.predict_proba(x_in.reshape(1,len(x_in)))[:,1]
                else:
                  pred_y_xij = model.predict(x_in.reshape(1,len(x_in)))

                y_xij[j,i] = np.log(pred_y_xij[0][0]/(1-pred_y_xij[0][0]))-logit_y0

            colourmap = plt.get_cmap('seismic_r')
            ax = nomo.add_subplot(len(nUniv[nUniv<x_train.shape[1]])+len(m_pr),1,subplot,sharex=ax0)
            if len(x_train.iloc[:,pr_i].unique()) < n_steps:
              for i in range(0,len(x_train.iloc[:,pr_i].unique())):
                ax.set_ylabel(x_train0.columns[pr_j],rotation=0,loc="bottom",labelpad=-1)
                ax.plot(y_xij[:,i],x_step0_j,label=np.sort(x_train0.iloc[:,pr_i].unique())[i])
                ax.scatter(y_xij[:,i],x_step0_j,marker="|")
                ax.legend(title=x_train0.columns[pr_i],fontsize=6,ncol=2)
            else:
              for j in range(0,len(x_train.iloc[:,pr_j].unique())):
                ax.set_ylabel(x_train0.columns[pr_i],rotation=0,loc="bottom",labelpad=-1)
                ax.plot(y_xij[j,:],x_step0_i,label=np.sort(x_train0.iloc[:,pr_j].unique())[j])
                ax.scatter(y_xij[j,:],x_step0_i,marker="|")
                ax.legend(title=x_train0.columns[pr_j],fontsize=6,ncol=2)
            ax.axvline(0,color="black")
            ax.yaxis.tick_right()
            ax.get_xaxis().set_visible(False)
            subplot+=1


    prscale = np.array([0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95])
    prscaleMinor = np.array([0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85])
    logit_prscale = np.log(prscale/(1-prscale))
    logit_prscaleMinor = np.log(prscaleMinor/(1-prscaleMinor))

    logitAx = ax.twiny()
    logitAx.set_xlim(-3,3)
    logitAx.xaxis.set_ticks_position('bottom')
    logitAx.xaxis.set_label_position('bottom')
    logitAx.spines['bottom'].set_position(('outward', 10))
    logitAx.set_xlabel("Log OR Sum")

    prAx = ax.twiny()
    prAx.xaxis.set_ticks_position('bottom')
    prAx.xaxis.set_label_position('bottom')
    prAx.set_xlim(min(logit_prscale),max(logit_prscale))
    prAx.set_xticks(logit_prscale,labels=prscale)
    prAx.set_xticks(logit_prscaleMinor,minor=True)
    prAx.spines['bottom'].set_position(('outward', 46))
    prAx.set_xlabel("pr")

    nomo.suptitle("Nomogram of univariate partial responses")
    nomo.tight_layout()
    nomo.subplots_adjust(hspace=0)
    nomo.show()
    #-------------------------------------------------------------------------------------------------------------------------------------------------------------------Bivariate
    if len(nUniv[nUniv>=x_train.shape[1]])-len(m_pr) > 0:
      binomo, axes = plt.subplots(ncols = 1, nrows = len(nUniv[nUniv>=x_train.shape[1]])-len(m_pr),figsize=(6,10*len(m_pr)))

      subplot=0
      for pr in np.where(abs(betas[:,userLambda])>0.1)[0]:
        if pr >= x_train.shape[1]:
          pr_i = int(bivariate_inputs[pr-x_train.shape[1],0])
          pr_j = int(bivariate_inputs[pr-x_train.shape[1],1])

          if not (len(x_train.iloc[:,pr_i].unique()) < n_steps) != (len(x_train.iloc[:,pr_j].unique()) < n_steps):

            if len(x_train.iloc[:,pr_i].unique()) < n_steps and len(x_train.iloc[:,pr_j].unique()) < n_steps:
              x_step0_i = np.sort(x_train0.iloc[:,pr_i].unique())
              x_step0_j = np.sort(x_train0.iloc[:,pr_j].unique())

            if len(x_train.iloc[:,pr_i].unique()) >= n_steps and len(x_train.iloc[:,pr_j].unique()) >= n_steps:
              x_step0_i = np.linspace(min(x_train0.iloc[:,pr_i]),max(x_train0.iloc[:,pr_i]),n_steps)
              x_step0_j = np.linspace(min(x_train0.iloc[:,pr_j]),max(x_train0.iloc[:,pr_j]),n_steps)

            y_xij = np.zeros([len(x_step0_j),len(x_step0_i)])

            for i in range(0,len(x_step0_i)):
              for j in range(0,len(x_step0_j)):
                x_in = np.zeros(x_train.shape[1])
                x_in[pr_i] = (x_step0_i[i]-x_train0.iloc[:,pr_i].median())/(x_train0.iloc[:,pr_i].std()*2)
                x_in[pr_j] = (x_step0_j[j]-x_train0.iloc[:,pr_j].median())/(x_train0.iloc[:,pr_j].std()*2)
                if isinstance(model,list) == True:
                  pred_y_xij = mlpmask_pred(x_in.reshape(1,len(x_in)),model,device=device)
                elif hasattr(model,'predict_proba'):
                  pred_y_xij = model.predict_proba(x_in.reshape(1,len(x_in)))[:,1]
                else:
                  pred_y_xij = model.predict(x_in.reshape(1,len(x_in)))

                y_xij[j,i] = np.log(pred_y_xij[0][0]/(1-pred_y_xij[0][0]))-logit_y0

          if len(x_train.iloc[:,pr_i].unique()) >= n_steps and len(x_train.iloc[:,pr_j].unique()) >= n_steps:
            X, Y = np.meshgrid(x_step0_i.reshape(len(x_step0_i),1), x_step0_j.reshape(len(x_step0_j),1))
            contour_heatmap = axes[subplot].contourf(X,Y,y_xij)
            c2 = axes[subplot].contour(X,Y,y_xij, cmap='Greys')
            axes[subplot].clabel(c2, inline=True)
            axes[subplot].set_xlabel(x_train.columns[pr_i])
            axes[subplot].set_ylabel(x_train.columns[pr_j])
            axes[subplot].set_xticks(x_step0_i,labels=np.round(x_step0_i,2),rotation=30)
            axes[subplot].set_yticks(x_step0_j,labels=np.round(x_step0_j,2))
            subplot+=1

          if len(x_train.iloc[:,pr_i].unique()) < n_steps and len(x_train.iloc[:,pr_j].unique()) < n_steps:
            heatmap = axes[subplot].imshow(y_xij,cmap="viridis", aspect="auto")
            axes[subplot].set_xlabel(x_train.columns[pr_i])
            axes[subplot].set_ylabel(x_train.columns[pr_j])
            axes[subplot].set_xticks(list(range(0,len(x_step0_i))),labels=np.round(x_step0_i,2))
            axes[subplot].set_yticks(list(range(0,len(x_step0_j))),labels=np.round(x_step0_j,2))
            boxSettings = dict(boxstyle='round', facecolor='grey', alpha=0.5)
            for i in range(0,len(x_step0_i)):
              for j in range(0,len(x_step0_j)):
                axes[subplot].text(i, j, np.round(y_xij[j, i],2),ha="center", va="center", color="w",bbox=boxSettings)
            subplot+=1

      binomo.tight_layout()
      binomo.subplots_adjust(hspace=0.35)

#-----------------------------------------------------------------------------------------------------------------------------------------------LEBESGUE


  if method.lower() == "lebesgue":

    if isinstance(model,list) == True:
      y0 = mlpmask_pred(x_train,model,device=device)
    elif hasattr(model,'predict_proba'):
      y0 = model.predict_proba(x_train)[:,1]
    else:
      y0 = model.predict(x_train)
    logit_y0 = np.mean(np.log(y0/(1-y0)))

    if np.where(abs(betas[:,userLambda])>0.1)[0][0] < x_train.shape[1]:
      response = np.where(abs(betas[:,userLambda])>0.1)[0][0]
      if len(x_train.iloc[:,response].unique()) < n_steps:

        catvals = x_train.iloc[:,response].unique()
        logit_y = np.zeros(len(catvals))

        for k in range(0,len(catvals)):

          x_in = x_train.copy()
          x_in.iloc[:,response] = catvals[k]

          if isinstance(model,list) == True:
            pred_y_xi = mlpmask_pred(x_in,model,device=device)
          elif hasattr(model,'predict_proba'):
            pred_y_xi = model.predict_proba(x_in)[:,1]
          else:
            pred_y_xi = model.predict(x_in)

          logit_y[k] = np.mean(np.log(pred_y_xi/(1-pred_y_xi)))-logit_y0

        ax0.axvline(0,color="black")
        ax0.set_ylabel(x_train.columns[0],rotation=0,loc="bottom",labelpad=-1)
        ax0.set_yticks([])
        ax0.plot(logit_y,np.full(len(logit_y),0.5))
        ax0.scatter(logit_y,np.full(len(logit_y),0.5),marker="|")
        for i in range(0,len(logit_y)):
          ax0.annotate(data[x_train.columns[response]].unique()[i],(logit_y[i],0.5),xytext=(logit_y[i],0.505))

      # Initial subplot for continuous case
      else:
        x_step0 = np.linspace(min(x_train0.iloc[:,response]),max(x_train0.iloc[:,response]),n_steps)
        x_step = np.linspace(min(x_train.iloc[:,response]),max(x_train.iloc[:,response]),n_steps)

        for k in range(0,len(x_step)):
          x_in = x_train.copy()
          x_in.iloc[:,response] = x_step[k]

          if isinstance(model,list) == True:
            pred_y_xi = mlpmask_pred(x_in,model,device=device)
          elif hasattr(model,'predict_proba'):
            pred_y_xi = model.predict_proba(x_in)[:,1]
          else:
            pred_y_xi = model.predict(x_in)

          logit_y[k] = np.mean(np.log(pred_y_xi/(1-pred_y_xi)))-logit_y0

        ax0.plot(logit_y,x_step0)

        line_marks0 = ax0.get_yticks()
        line_marks0 = line_marks0[np.where(line_marks0<=max(x_step0))]
        if min(x_step0) < 0:
          line_marks0 = line_marks0[np.where(line_marks0>=min(x_step0))]
        else:
          line_marks0 = line_marks0[np.where(line_marks0>=0)]
        if len(line_marks0) == 2:
          line_marks0 = np.linspace(line_marks0[0],line_marks0[1],3)


        ax0.set_yticks(line_marks0)

        logit_line_marks = np.zeros(len(line_marks0))
        line_marks = (line_marks0 - x_train0.iloc[:,pr].median())/(x_train0.iloc[:,pr].std()*2)

        for k in range(0,len(line_marks)):
          x_in = x_train.copy()
          x_in.iloc[:,pr] = line_marks[k]
          if isinstance(model,list) == True:
            pred_y_xi = mlpmask_pred(x_in,model,device=device)
          elif hasattr(model,'predict_proba'):
            pred_y_xi = model.predict_proba(x_in)[:,1]
          else:
            pred_y_xi = model.predict(x_in)

          logit_line_marks[k] = np.mean(np.log(pred_y_xi/(1-pred_y_xi)))-logit_y0

        ax0.scatter(logit_line_marks,line_marks0,marker="|")
        for i in range(0,len(line_marks0)):
          ax0.annotate(line_marks0[i],(logit_line_marks[i],line_marks0[i]),xytext=(logit_line_marks[i]-0.1,line_marks0[i]+1))#

        #ax.scatter(logit_y[range(0,len(logit_y),3)],x_step0[range(0,len(logit_y),3)],marker="|")
        #for i in range(0,len(logit_y),3):
          #ax.annotate(np.round(x_step0[i],2),(logit_y[i],x_step0[i]),xytext=(logit_y[i]-0.1,x_step0[i]+1))
        ax0.set_ylabel(x_train.columns[response],rotation=0,loc="bottom",labelpad=-1)
        ax0.yaxis.tick_right()
        ax0.axvline(0,color="black")

      # Loop creates subplot for each other univariate response
      subplot = 1
      for pr in np.where(abs(betas[:,userLambda])>0.1)[0]:
        if pr < x_train.shape[1]:

          # categorical case
          if len(x_train.iloc[:,pr].unique()) < n_steps:

            catvals = x_train.iloc[:,pr].unique()
            logit_y = np.zeros(len(catvals))

            for k in range(0,len(catvals)):
              x_in = x_train.copy()
              x_in.iloc[:,pr] = catvals[k]

              if isinstance(model,list) == True:
                pred_y_xi = mlpmask_pred(x_in,model,device=device)
              elif hasattr(model,'predict_proba'):
                pred_y_xi = model.predict_proba(x_in)[:,1]
              else:
                pred_y_xi = model.predict(x_in)

              logit_y[k] = np.mean(np.log(pred_y_xi/(1-pred_y_xi))-logit_y0)

            ax = nomo.add_subplot(len(nUniv[nUniv<x_train.shape[1]])+len(m_pr),1,subplot,sharex=ax0)
            ax.axvline(0,color="black")
            ax.set_yticks([])
            ax.set_ylabel(x_train.columns[pr],rotation=0,loc="bottom",labelpad=-1)
            ax.get_xaxis().set_visible(False)
            ax.plot(logit_y,np.full(len(logit_y),0.5))
            ax.scatter(logit_y,np.full(len(logit_y),0.5),marker="|")
            for i in range(0,len(logit_y)):
              ax.annotate(data[x_train.columns[pr]].unique()[i],(logit_y[i],0.5),xytext=(logit_y[i],0.505))

          # continuous case
          else:
            x_step0 = np.linspace(min(x_train0.iloc[:,pr]),max(x_train0.iloc[:,pr]),n_steps)
            x_step = np.linspace(min(x_train.iloc[:,pr]),max(x_train.iloc[:,pr]),n_steps)
            logit_y = np.zeros(n_steps)

            for k in range(0,n_steps):
              x_in = x_train.copy()
              x_in.iloc[:,pr] = x_step[k]
              if isinstance(model,list) == True:
                pred_y_xi = mlpmask_pred(x_in,model,device=device)
              elif hasattr(model,'predict_proba'):
                pred_y_xi = model.predict_proba(x_in)[:,1]
              else:
                pred_y_xi = model.predict(x_in)

              logit_y[k] = np.mean(np.log(pred_y_xi/(1-pred_y_xi)))-logit_y0

            ax = nomo.add_subplot(len(nUniv[nUniv<x_train.shape[1]])+len(m_pr),1,subplot,sharex=ax0)
            ax.plot(logit_y,x_step0)

            line_marks0 = ax.get_yticks()
            line_marks0 = line_marks0[np.where(line_marks0<=max(x_step0))]
            if min(x_step0) < 0:
              line_marks0 = line_marks0[np.where(line_marks0>=min(x_step0))]
            else:
              line_marks0 = line_marks0[np.where(line_marks0>=0)]
            if len(line_marks0) == 2:
              line_marks0 = np.linspace(line_marks0[0],line_marks0[1],3)


            ax.set_yticks(line_marks0)

            logit_line_marks = np.zeros(len(line_marks0))
            line_marks = (line_marks0 - x_train0.iloc[:,pr].median())/(x_train0.iloc[:,pr].std()*2)

            for k in range(0,len(line_marks)):
              x_in = x_train.copy()
              x_in.iloc[:,pr] = line_marks[k]
              if isinstance(model,list) == True:
                pred_y_xi = mlpmask_pred(x_in,model,device=device)
              elif hasattr(model,'predict_proba'):
                pred_y_xi = model.predict_proba(x_in)[:,1]
              else:
                pred_y_xi = model.predict(x_in)

              logit_line_marks[k] = np.mean(np.log(pred_y_xi/(1-pred_y_xi)))-logit_y0

            ax.scatter(logit_line_marks,line_marks0,marker="|")
            for i in range(0,len(line_marks0)):
              ax.annotate(line_marks0[i],(logit_line_marks[i],line_marks0[i]),xytext=(logit_line_marks[i]-0.1,line_marks0[i]+1))

            #ax.scatter(logit_y[range(0,len(logit_y),3)],x_step0[range(0,len(logit_y),3)],marker="|")
            #for i in range(0,len(logit_y),3):
              #ax.annotate(np.round(x_step0[i],2),(logit_y[i],x_step0[i]),xytext=(logit_y[i]-0.1,x_step0[i]+1))

            ax.axvline(0,color="black")
            ax.set_ylabel(x_train.columns[pr],rotation=0,loc="bottom",labelpad=-1)
            ax.yaxis.tick_right()
            ax.get_xaxis().set_visible(False)
          subplot+=1

        elif pr >= x_train.shape[1]:
          pr_i = int(bivariate_inputs[pr-x_train.shape[1],0])
          pr_j = int(bivariate_inputs[pr-x_train.shape[1],1])
          if (len(x_train.iloc[:,pr_i].unique()) < 15) != (len(x_train.iloc[:,pr_j].unique()) < 15):
            "mixed response"
            if len(x_train.iloc[:,pr_i].unique()) < n_steps:
              x_step0_i = np.sort(x_train0.iloc[:,pr_i].unique())
              x_step0_j = np.linspace(min(x_train0.iloc[:,pr_j]),max(x_train0.iloc[:,pr_j]),n_steps)
            else:
              x_step0_i = np.linspace(min(x_train0.iloc[:,pr_i]),max(x_train0.iloc[:,pr_i]),n_steps)
              x_step0_j = np.sort(x_train0.iloc[:,pr_j].unique())

            y_xij = np.zeros([len(x_step0_j),len(x_step0_i)])

            for i in range(0,len(x_step0_i)):
              for j in range(0,len(x_step0_j)):
                x_in = x_train.copy()
                x_in.iloc[:,pr_i] = (x_step0_i[i]-x_train0.iloc[:,pr_i].median())/(x_train0.iloc[:,pr_i].std()*2)
                x_in.iloc[:,pr_j] = (x_step0_j[j]-x_train0.iloc[:,pr_j].median())/(x_train0.iloc[:,pr_j].std()*2)

                if isinstance(model,list) == True:
                  pred_y_xij = mlpmask_pred(x_in,model,device=device)
                elif hasattr(model,'predict_proba'):
                  pred_y_xij = model.predict_proba(x_in)[:,1]
                else:
                  pred_y_xij = model.predict(x_in)

                y_xij[j,i] = np.log(np.mean(pred_y_xij)/(1-np.mean(pred_y_xij)))-logit_y0

            colourmap = plt.get_cmap('seismic_r')
            ax = nomo.add_subplot(len(nUniv[nUniv<x_train.shape[1]])+len(m_pr),1,subplot,sharex=ax0)
            if len(x_train.iloc[:,pr_i].unique()) < n_steps:
              for i in range(0,len(x_train.iloc[:,pr_i].unique())):
                ax.set_ylabel(x_train0.columns[pr_j],rotation=0,loc="bottom",labelpad=-1)
                ax.plot(y_xij[:,i],x_step0_j,label=np.sort(x_train0.iloc[:,pr_i].unique())[i])
                ax.scatter(y_xij[:,i],x_step0_j,marker="|")
                ax.legend(title=x_train0.columns[pr_i],fontsize=6,ncol=2)
            else:
              for j in range(0,len(x_train.iloc[:,pr_j].unique())):
                ax.set_ylabel(x_train0.columns[pr_i],rotation=0,loc="bottom",labelpad=-1)
                ax.plot(y_xij[j,:],x_step0_i,label=np.sort(x_train0.iloc[:,pr_j].unique())[j])
                ax.scatter(y_xij[j,:],x_step0_i,marker="|")
                ax.legend(title=x_train0.columns[pr_j],fontsize=6,ncol=2)
            ax.axvline(0,color="black")
            ax.yaxis.tick_right()
            ax.get_xaxis().set_visible(False)
            subplot+=1

    prscale = np.array([0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95])
    prscaleMinor = np.array([0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85])
    logit_prscale = np.log(prscale/(1-prscale))
    logit_prscaleMinor = np.log(prscaleMinor/(1-prscaleMinor))

    logitAx = ax.twiny()
    logitAx.set_xlim(-3,3)
    logitAx.xaxis.set_ticks_position('bottom')
    logitAx.xaxis.set_label_position('bottom')
    logitAx.spines['bottom'].set_position(('outward', 10))
    logitAx.set_xlabel("Log OR Sum")

    prAx = ax.twiny()
    prAx.xaxis.set_ticks_position('bottom')
    prAx.xaxis.set_label_position('bottom')
    prAx.set_xlim(min(logit_prscale),max(logit_prscale))
    prAx.set_xticks(logit_prscale,labels=prscale)
    prAx.set_xticks(logit_prscaleMinor,minor=True)
    prAx.spines['bottom'].set_position(('outward', 46))
    prAx.set_xlabel("pr")

    nomo.suptitle("Nomogram of univariate partial responses")
    nomo.tight_layout()
    nomo.subplots_adjust(hspace=0)
    nomo.show()
    #--------------------------------------------------------------------------------------------------------------------------------------------bivariate
    if len(nUniv[nUniv>=x_train.shape[1]])-len(m_pr) > 0:
      binomo, axes = plt.subplots(ncols = 1, nrows = len(nUniv[nUniv>=x_train.shape[1]])-len(m_pr),figsize=(6,5*len(m_pr)))

      subplot=0
      for pr in np.where(abs(betas[:,userLambda])>0.1)[0]:
        if pr >= x_train.shape[1]:
          pr_i = int(bivariate_inputs[pr-x_train.shape[1],0])
          pr_j = int(bivariate_inputs[pr-x_train.shape[1],1])

          if not (len(x_train.iloc[:,pr_i].unique()) < n_steps) != (len(x_train.iloc[:,pr_j].unique()) < n_steps):
            if len(x_train.iloc[:,pr_i].unique()) < n_steps and len(x_train.iloc[:,pr_j].unique()) < n_steps:
              x_step0_i = np.sort(x_train0.iloc[:,pr_i].unique())
              x_step0_j = np.sort(x_train0.iloc[:,pr_j].unique())

            if len(x_train.iloc[:,pr_i].unique()) >= n_steps and len(x_train.iloc[:,pr_j].unique()) >= n_steps:
              x_step0_i = np.linspace(min(x_train0.iloc[:,pr_i]),max(x_train0.iloc[:,pr_i]),n_steps)
              x_step0_j = np.linspace(min(x_train0.iloc[:,pr_j]),max(x_train0.iloc[:,pr_j]),n_steps)

            y_xij = np.zeros([len(x_step0_j),len(x_step0_i)])

            for i in range(0,len(x_step0_i)):
              for j in range(0,len(x_step0_j)):
                x_in = x_train.copy()
                x_in.iloc[:,pr_i] = (x_step0_i[i]-x_train0.iloc[:,pr_i].median())/(x_train0.iloc[:,pr_i].std()*2)
                x_in.iloc[:,pr_j] = (x_step0_j[j]-x_train0.iloc[:,pr_j].median())/(x_train0.iloc[:,pr_j].std()*2)

                if isinstance(model,list) == True:
                  pred_y_xij = mlpmask_pred(x_in,model,device=device)
                elif hasattr(model,'predict_proba'):
                  pred_y_xij = model.predict_proba(x_in)[:,1]
                else:
                  pred_y_xij = model.predict(x_in)

                y_xij[j,i] = np.log(np.mean(pred_y_xij)/(1-np.mean(pred_y_xij)))-logit_y0

          if len(x_train.iloc[:,pr_i].unique()) >= n_steps and len(x_train.iloc[:,pr_j].unique()) >= n_steps:
            X, Y = np.meshgrid(x_step0_i.reshape(len(x_step0_i),1), x_step0_j.reshape(len(x_step0_j),1))
            contour_heatmap = axes[subplot].contourf(X,Y,y_xij)
            c2 = axes[subplot].contour(X,Y,y_xij, cmap='Greys')
            axes[subplot].clabel(c2, inline=True, fontsize=10)
            axes[subplot].set_xlabel(x_train.columns[pr_i])
            axes[subplot].set_ylabel(x_train.columns[pr_j])
            axes[subplot].set_xticks(x_step0_i,labels=np.round(x_step0_i,2),rotation=30)
            axes[subplot].set_yticks(x_step0_j,labels=np.round(x_step0_j,2))
            subplot+=1

          if len(x_train.iloc[:,pr_i].unique()) < n_steps and len(x_train.iloc[:,pr_j].unique()) < n_steps:
            heatmap = axes[subplot].imshow(y_xij,cmap="viridis", aspect="auto")
            axes[subplot].set_xlabel(x_train.columns[pr_i])
            axes[subplot].set_ylabel(x_train.columns[pr_j])
            axes[subplot].set_xticks(list(range(0,len(x_step0_i))),labels=np.round(x_step0_i,2))
            axes[subplot].set_yticks(list(range(0,len(x_step0_j))),labels=np.round(x_step0_j,2))
            boxSettings = dict(boxstyle='round', facecolor='grey', alpha=0.5)
            for i in range(0,len(x_step0_i)):
              for j in range(0,len(x_step0_j)):
                axes[subplot].text(i, j, np.round(y_xij[j, i],2),ha="center", va="center", color="w",bbox=boxSettings)
            subplot+=1

      binomo.tight_layout()
      binomo.subplots_adjust(hspace=0.35)
