import torch
from torch import nn, optim
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from IPython.display import clear_output
from typing import Tuple, List, Union


def live_plot(all_train_losses, all_val_losses, lambdas, figsize=(15, 10), title=''):
    clear_output(wait=True)
    fig, ax = plt.subplots(1, 2, figsize=figsize)

    norm = mcolors.Normalize(vmin=lambdas.min(), vmax=lambdas.max())
    cmap = cm.viridis

    min_loss = min(min(min(train_losses) for train_losses in all_train_losses),
                   min(min(val_losses) for val_losses in all_val_losses))
    max_loss = max(max(max(train_losses) for train_losses in all_train_losses),
                   max(max(val_losses) for val_losses in all_val_losses))

    if len(all_train_losses[0]) > 1:
        # Plot over epochs (pytorch)
        for i, (train_losses, val_losses) in enumerate(zip(all_train_losses, all_val_losses)):
            color = cmap(norm(lambdas[i]))
            ax[0].plot(
                range(len(train_losses)), train_losses, label=f'Lambda {lambdas[i]:.5f}', color=color, alpha=0.5, marker='o')
            ax[1].plot(
                range(len(val_losses)), val_losses, label=f'Lambda {lambdas[i]:.5f}', color=color, alpha=0.5, marker='o')

        ax[0].set_xlabel('Epoch')
        ax[1].set_xlabel('Epoch')

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical',
                            fraction=0.02, pad=0.04)
        cbar.set_label('Lambda')
    else:
        # Plot over lambdas (sklearn)
        for i, (train_losses, val_losses) in enumerate(zip(all_train_losses, all_val_losses)):
            ax[0].semilogx(
                lambdas[i], train_losses, marker='o', color='black')
            ax[1].semilogx(
                lambdas[i], val_losses, marker='o', color='black')

        ax[0].set_xlabel('Lambda')
        ax[1].set_xlabel('Lambda')
        
        ax[0].invert_xaxis()
        ax[1].invert_xaxis()


    ax[0].set_title('Train Loss')
    ax[0].grid(True)
    ax[0].set_ylabel('Loss')
    ax[0].set_ylim([min_loss, max_loss])

    ax[1].set_title('Test Loss')
    ax[1].grid(True)
    ax[1].set_ylabel('Loss')
    ax[1].set_ylim([min_loss, max_loss])

    plt.suptitle(title)
    plt.show()
    
class LogisticRegressionModel(nn.Module):
    def __init__(self, num_features):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(num_features, 1)
        # initialize weights and biases
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


def run_glm_pytorch(i: int, lambdas: np.ndarray, partial_responses_train: np.ndarray, y_train: Union[np.ndarray, pd.Series], partial_responses_test: np.ndarray, y_test: Union[np.ndarray, pd.Series], num_epochs: int = 100000, lr: float = 0.001, tolerance: float = 1e-4) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, float, float, float, float, List[float], List[float]]:
    device = 'cpu'
    model = LogisticRegressionModel(partial_responses_train.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-10)

    partial_responses_train = torch.tensor(
        partial_responses_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train.values if isinstance(y_train, pd.Series) else y_train, dtype=torch.float32).reshape(-1, 1).to(device)
    partial_responses_test = torch.tensor(
        partial_responses_test, dtype=torch.float32).to(device)
    y_test_np = y_test.values if isinstance(y_test, pd.Series) else y_test
    y_test_np = y_test_np.reshape(-1, 1)

    lambda_l1 = lambdas[i]
    initial_loss = None
    best_loss = float('inf')
    patience = 100
    patience_counter = 0
    best_model = None  # to store the state dict of the best model
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(partial_responses_train)
        l1_penalty = lambda_l1 * \
            torch.linalg.vector_norm(model.linear.weight, ord=1)
        loss = criterion(outputs, y_train) + l1_penalty
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(partial_responses_test)
            val_loss = criterion(val_outputs, torch.tensor(
                y_test_np, dtype=torch.float32).to(device))

        # Normalizing the loss to the initial loss to make it relative
        if initial_loss is None:
            initial_loss = val_loss.item()

        train_losses.append((loss.item() / initial_loss))

        normalized_loss = val_loss.item() / initial_loss
        val_losses.append(normalized_loss)

        if normalized_loss < best_loss - tolerance:
            best_loss = normalized_loss
            patience_counter = 0
            # Save the best model if validation improves
            best_model = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                # print(
                # f"{i:2d} - Early stopping at epoch {epoch} with normalized validation loss {best_loss}")
                break

    # Load the best model weights
    model.load_state_dict(best_model)

    # Final evaluation with the best model
    model.eval()
    with torch.no_grad():
        y_pred_train = model(partial_responses_train).cpu().numpy().ravel()
        y_pred_test = model(partial_responses_test).cpu().numpy().ravel()
        train_auc = roc_auc_score(y_train.cpu(), y_pred_train)
        test_auc = roc_auc_score(y_test_np, y_pred_test)

        # eps = 1e-8  # A small number to prevent log(0)
        # train_dev = -2 * np.sum(y_train.cpu().numpy() * np.log(np.clip(y_pred_train, eps, 1 - eps)) +
        #                         (1 - y_train.cpu().numpy()) * np.log(np.clip(1 - y_pred_train, eps, 1 - eps)))
        # test_dev = -2 * np.sum(y_test_np * np.log(np.clip(y_pred_test, eps, 1 - eps)) +
        #                        (1 - y_test_np) * np.log(np.clip(1 - y_pred_test, eps, 1 - eps)))
        train_dev = -2 * np.sum(y_train.cpu().numpy() * np.log(y_pred_train) +
                                (1 - y_train.cpu().numpy()) * np.log(1 - y_pred_train))
        test_dev = -2 * np.sum(y_test_np * np.log(y_pred_test) +
                               (1 - y_test_np) * np.log(1 - y_pred_test))

        beta = model.linear.weight.detach().cpu().numpy()

    return i, beta, y_pred_train, y_pred_test, train_dev, test_dev, train_auc, test_auc, train_losses, val_losses


def run_glm_sklearn(i: int, lambdas: np.ndarray, partial_responses_train: np.ndarray, y_train: Union[np.ndarray, pd.Series], partial_responses_test: np.ndarray, y_test: Union[np.ndarray, pd.Series], lr: float = 0.001) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, float, float, float, float, List[float], List[float]]:
    """Run logistic regresssion with L1 regularization using sklearn. Learning rate `lr` not used."""
    C_values = 1/lambdas

    lr = LogisticRegression(
        C=C_values[i], penalty='l1', solver='saga', max_iter=10000)
    lr.fit(partial_responses_train, y_train.values.ravel() if isinstance(y_train, pd.Series) else y_train.ravel())

    beta = lr.coef_
    y_pred_train = lr.predict_proba(partial_responses_train)[:, 1]
    y_pred_test = lr.predict_proba(partial_responses_test)[:, 1]
    train_auc = roc_auc_score(y_train, y_pred_train)
    test_auc = roc_auc_score(y_test, y_pred_test)

    # Deviance approximated as -2 * log-likelihood
    y_train_np = y_train.values if isinstance(y_train, pd.Series) else y_train
    y_test_np = y_test.values if isinstance(y_test, pd.Series) else y_test
    train_dev = -2 * np.sum(y_train_np * np.log(y_pred_train) +
                            (1 - y_train_np) * np.log(1 - y_pred_train))
    test_dev = -2 * np.sum(y_test_np * np.log(y_pred_test) +
                           (1 - y_test_np) * np.log(1 - y_pred_test))

    # Calculate BCELoss using sklearn's log_loss function
    train_bce_loss = log_loss(y_train, y_pred_train)
    test_bce_loss = log_loss(y_test, y_pred_test)

    return i, beta, y_pred_train, y_pred_test, train_dev, test_dev, train_auc, test_auc, [train_bce_loss], [test_bce_loss]


def prLASSO(
    partial_responses_train: Union[np.ndarray, torch.Tensor],
    partial_responses_test: Union[np.ndarray, torch.Tensor],
    y_train: Union[np.ndarray, pd.Series, torch.Tensor],
    y_test: Union[np.ndarray, pd.Series, torch.Tensor],
    num_lambdas: int = 25,
    verbose: bool = True,
    log_min_lambda: float = -3,
    log_max_lambda: float = 2,
    lasso_function: callable = run_glm_sklearn,
    lr: float = 0.001
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    # Convert inputs to numpy arrays if they're tensors
    if isinstance(partial_responses_train, torch.Tensor):
        partial_responses_train = partial_responses_train.cpu().numpy()
    if isinstance(partial_responses_test, torch.Tensor):
        partial_responses_test = partial_responses_test.cpu().numpy()
    if isinstance(y_train, torch.Tensor):
        y_train = y_train.cpu().numpy()
    if isinstance(y_test, torch.Tensor):
        y_test = y_test.cpu().numpy()

    num_features = partial_responses_train.shape[1]
    n_univ = int((np.sqrt(1 + 8 * num_features) - 1) / 2)
    n_biv = num_features - n_univ
    lambdas = np.logspace(log_max_lambda, log_min_lambda, num=num_lambdas)
    betas = np.zeros((partial_responses_train.shape[1], len(lambdas)))
    glmPred = np.zeros((partial_responses_test.shape[0], len(lambdas)))
    trainAUC = np.zeros(len(lambdas))
    trainDev = np.zeros(len(lambdas))
    testAUC = np.zeros(len(lambdas))
    testDev = np.zeros(len(lambdas))
    univ_count = np.zeros(len(lambdas), dtype=int)
    biv_count = np.zeros(len(lambdas), dtype=int)
    all_train_losses = []
    all_test_losses = []

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(lasso_function, i, lambdas, partial_responses_train,
                                   y_train, partial_responses_test, y_test, lr=lr) 
                   for i in range(len(lambdas))]
        for future in as_completed(futures):
            i, beta, y_pred_train, y_pred_test, train_dev, test_dev, train_auc, test_auc, train_losses, test_losses = future.result()
            betas[:, i] = beta.flatten()
            glmPred[:, i] = y_pred_test
            trainDev[i] = train_dev
            testDev[i] = test_dev
            trainAUC[i] = train_auc
            testAUC[i] = test_auc
            univ_count[i] = np.sum(np.abs(betas[:, i][:n_univ]) > 0.1)
            biv_count[i] = np.sum(np.abs(betas[:, i][n_univ:]) > 0.1)
            all_train_losses.append(train_losses)
            all_test_losses.append(test_losses)
            live_plot(all_train_losses, all_test_losses, lambdas,
                      title=f'BCE loss (lr={lr})')
            print(
                f"{i:2d} - Lambda {lambdas[i]:.5f}: Test AUC {testAUC[i]:.4f}, n_univ: {univ_count[i]}, n_biv: {biv_count[i]}")

    if verbose:
        min_dev, min_dev_sd = np.argmin(testDev), np.argmin(
            np.abs(testDev - (np.min(testDev) + np.std(testDev))))
        max_auc, max_auc_sd = np.argmax(testAUC), np.argmax(
            np.abs(testAUC - (np.max(testAUC) - np.std(testAUC))))
        for i in range(len(lambdas)):
            print(
                f"{i:2d} - Lambda {lambdas[i]:.5f}: Train AUC {trainAUC[i]:.4f}, Train Deviance {trainDev[i]:.4f}, Test AUC {testAUC[i]:.4f}, Test Deviance {testDev[i]:.4f}, n_univ: {univ_count[i]}, n_biv: {biv_count[i]}")
            if i == min_dev:
                print(f"    -> Minimum Test Deviance at this lambda")
            if i == min_dev_sd:
                print(f"    -> Within 1 SD of minimum Test Deviance")
            if i == max_auc:
                print(f"    -> Maximum Test AUC at this lambda")
            if i == max_auc_sd:
                print(f"    -> Within 1 SD of maximum Test AUC")

    # Plotting
    fig, ax = plt.subplots(2, 2, figsize=(12, 6), sharex=True)

    ax[0, 0].semilogx(lambdas, trainDev, label="Training Deviance")
    ax[0, 0].set_title("Training Deviance vs Lambda")
    ax[0, 0].set_ylabel("Deviance")

    ax[0, 1].semilogx(lambdas, trainAUC, label="Training AUC")
    ax[0, 1].set_title("Training AUC vs Lambda")
    ax[0, 1].set_ylabel("AUC")

    ax[1, 0].semilogx(lambdas, testDev, label="Test Deviance")
    ax[1, 0].set_title("Test Deviance vs Lambda")
    ax[1, 0].set_xlabel("Lambda")
    ax[1, 0].set_ylabel("Deviance")

    ax[1, 1].semilogx(lambdas, testAUC, label="Test AUC")
    ax[1, 1].set_title("Test AUC vs Lambda")
    ax[1, 1].set_xlabel("Lambda")
    ax[1, 1].set_ylabel("AUC")

    ax[0, 0].invert_xaxis()
    plt.tight_layout()
    plt.show()

    # Trace plot for coefficients
    plt.figure(figsize=(10, 6))
    for j in range(betas.shape[0]):
        plt.semilogx(lambdas, betas[j, :])
    plt.gca().invert_xaxis()
    plt.title("Trace plot of Coefficients")
    plt.xlabel("Lambda")
    plt.ylabel(r"$\beta$")
    plt.show()

    return lambdas, betas, trainAUC, testAUC, trainDev, testDev, glmPred