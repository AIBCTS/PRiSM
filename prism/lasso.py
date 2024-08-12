import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from typing import List, Optional, Tuple
from prism.lasso_results import LassoResultsManager
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, roc_auc_score
from IPython.display import display, clear_output, HTML

def _fit_lasso(X, y, lambda_val, max_iter, random_state, prev_coef=None):
    C = 1 / lambda_val
    model = LogisticRegression(C=C, penalty='l1', solver='saga', max_iter=max_iter, 
                               random_state=random_state, warm_start=True)
    if prev_coef is not None:
        model.classes_ = np.unique(y)
        model.coef_ = prev_coef.reshape(1, -1)
        model.intercept_ = np.array([0.0])
    model.fit(X, y)
    return model.coef_[0], model

def plot_lasso_path(lambda_values, train_losses, test_losses, train_aucs, test_aucs,train_devs, test_devs):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))
    
    ax1.semilogx(lambda_values, train_losses, marker='o', label='Train Loss')
    ax1.semilogx(lambda_values, test_losses, marker='o', label='Test Loss')
    ax1.set_xlabel('Lambda')
    ax1.set_ylabel('Log Loss')
    ax1.set_title('LASSO Path - Loss')
    ax1.legend()
    ax1.invert_xaxis()
    
    ax2.semilogx(lambda_values, train_aucs, marker='o', label='Train AUC')
    ax2.semilogx(lambda_values, test_aucs, marker='o', label='Test AUC')
    ax2.set_xlabel('Lambda')
    ax2.set_ylabel('AUC')
    ax2.set_title('LASSO Path - AUC')
    ax2.invert_xaxis()
    
    ax3.semilogx(lambda_values, train_devs, marker='o', label='Train Deviance')
    ax3.semilogx(lambda_values, test_devs, marker='o', label='Test Deviance')
    ax3.set_xlabel('Lambda')
    ax3.set_ylabel('Deviance')
    ax3.set_title('LASSO Path - Deviance')
    ax3.invert_xaxis()
    
    plt.tight_layout()
    return fig

def lasso(partial_responses_train: torch.Tensor, 
          partial_responses_test: torch.Tensor, 
          y_train: np.ndarray, 
          y_test: np.ndarray, 
          bivariate_inputs: List[Tuple[int, int]], 
          nlambda: int = 100, 
          feature_names: Optional[List[str]] = None,
          max_iter: int = 10000,
          min_lambda: float = 0.001,
          max_lambda: float = 1000,
          n_jobs: int = -1,
          tol: float = 1e-4,
          batch_size: int = 10,
          real_time_plot: bool = True) -> 'LassoResultsManager':
    """
    Perform LASSO regression on partial responses and return a LassoResultsManager object.

    Parameters:
    -----------
    partial_responses_train : torch.Tensor
        Partial responses for training data
    partial_responses_test : torch.Tensor
        Partial responses for test data
    y_train : np.ndarray
        Training target data, shape (n_samples_train,)
    y_test : np.ndarray
        Test target data, shape (n_samples_test,)
    bivariate_inputs : List[Tuple[int, int]]
        List of tuples containing indices for bivariate inputs
    nlambda : int, optional
        Number of lambda values to use (default is 100)
    feature_names : List[str], optional
        List of univariate feature names. If not provided, generic names will be created.
    max_iter : int, optional
        Maximum number of iterations for the LASSO solver (default is 10000)
    min_lambda : float, optional
        Minimum lambda value (default is 0.001)
    max_lambda : float, optional
        Maximum lambda value (default is 1000)

    Returns:
    --------
    LassoResultsManager
        An object containing LASSO results and utility methods
    """
    seed = 257

    # Convert torch tensors to numpy arrays
    partial_responses_train = partial_responses_train.cpu().numpy()
    partial_responses_test = partial_responses_test.cpu().numpy()

    num_features = partial_responses_train.shape[1]
    n_univ = int((np.sqrt(1 + 8 * num_features) - 1) / 2)
    n_biv = num_features - n_univ

    # Generate feature names if not provided
    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(n_univ)]
    elif len(feature_names) != n_univ:
        raise ValueError(f"Number of feature names ({len(feature_names)}) does not match number of univariate features ({n_univ})")

    # Calculate lambda values
    lambda_values = np.logspace(np.log10(max_lambda), np.log10(min_lambda), nlambda)

    # Initialize arrays to store results
    betas = np.zeros((num_features, nlambda))
    models = []
    train_losses = np.zeros(nlambda)
    test_losses = np.zeros(nlambda)
    train_aucs = np.zeros(nlambda)
    test_aucs = np.zeros(nlambda)
    train_devs = np.zeros(nlambda)
    test_devs = np.zeros(nlambda)
    beta_univ_counts = np.zeros(nlambda)
    beta_biv_counts = np.zeros(nlambda)

    # Convert y_train and y_test to numpy arrays if they're not already
    y_train = y_train.to_numpy() if hasattr(y_train, 'to_numpy') else y_train
    y_test = y_test.to_numpy() if hasattr(y_test, 'to_numpy') else y_test

    # Initialize output log with adjusted column widths
    output_log = ["Index Lambda     Train AUC  Test AUC   Train Dev   Test Dev    Train Loss Test Loss  β_univ>0.1 β_biv>0.1"]
    output_log.append("-" * 115)

    # Function to format value without asterisk
    def format_value(value, width=10):
        return f"{value:<{width}.4f}"

    # Process lambda values in batches
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        for batch_start in tqdm(range(0, nlambda, batch_size), desc="Processing lambda batches"):
            batch_end = min(batch_start + batch_size, nlambda)
            batch_lambdas = lambda_values[batch_start:batch_end]
            
            # Use joblib for parallel processing
            results = Parallel(n_jobs=n_jobs)(
                delayed(_fit_lasso)(
                    partial_responses_train, 
                    y_train, 
                    lambda_val, 
                    max_iter, 
                    seed,
                    betas[:, batch_start-1] if batch_start > 0 else None
                ) for lambda_val in batch_lambdas
            )

            for i, (beta, model) in enumerate(results):
                idx = batch_start + i
                betas[:, idx] = beta
                models.append(model)
                
                # Calculate predictions
                train_pred = model.predict_proba(partial_responses_train)[:, 1]
                test_pred = model.predict_proba(partial_responses_test)[:, 1]
                
                # Calculate losses
                train_losses[idx] = log_loss(y_train, train_pred)
                test_losses[idx] = log_loss(y_test, test_pred)
                
                # Calculate AUC
                train_aucs[idx] = roc_auc_score(y_train, train_pred)
                test_aucs[idx] = roc_auc_score(y_test, test_pred)
                
                # Calculate Deviance
                train_devs[idx] = -2 * np.sum(y_train * np.log(train_pred) + (1 - y_train) * np.log(1 - train_pred))
                test_devs[idx] = -2 * np.sum(y_test * np.log(test_pred) + (1 - y_test) * np.log(1 - test_pred))

                # Count betas < 0.1
                beta_univ_counts[idx] = int(np.sum(np.abs(beta[:n_univ]) > 0.1))
                beta_biv_counts[idx] = int(np.sum(np.abs(beta[n_univ:]) > 0.1))

            # Update output log with adjusted formatting
            for idx in range(batch_start, batch_end):
                output_log.append(f"{idx:<6d} {lambda_values[idx]:<10.5f} "
                                  f"{format_value(train_aucs[idx])}"
                                  f"{format_value(test_aucs[idx])}"
                                  f"{format_value(train_devs[idx], 12)}"
                                  f"{format_value(test_devs[idx], 12)}"
                                  f"{format_value(train_losses[idx])}"
                                  f"{format_value(test_losses[idx])}"
                                  f"{int(beta_univ_counts[idx]):<11d}"
                                  f"{int(beta_biv_counts[idx]):<10d}")

            # Update plot and display output
            if real_time_plot:
                clear_output(wait=True)
                fig = plot_lasso_path(lambda_values[:batch_end], train_losses[:batch_end], 
                                      test_losses[:batch_end], train_aucs[:batch_end], test_aucs[:batch_end],
                                      train_devs[:batch_end], test_devs[:batch_end])
                display(fig)
                plt.close(fig)
                display(HTML("<pre style='font-family: monospace;'>" + "\n".join(output_log) + "</pre>"))

            # Check for early stopping
            if batch_start > 0 and np.all(np.abs(betas[:, batch_end-1] - betas[:, batch_start-1]) < tol):
                print(f"Early stopping at lambda index {batch_end-1}")
                betas = betas[:, :batch_end]
                lambda_values = lambda_values[:batch_end]
                train_losses = train_losses[:batch_end]
                test_losses = test_losses[:batch_end]
                train_aucs = train_aucs[:batch_end]
                test_aucs = test_aucs[:batch_end]
                train_devs = train_devs[:batch_end]
                test_devs = test_devs[:batch_end]
                break

    # Find maximum/minimum values for each metric
    max_train_auc, max_test_auc = np.max(train_aucs), np.max(test_aucs)
    min_train_dev, min_test_dev = np.min(train_devs), np.min(test_devs)
    min_train_loss, min_test_loss = np.min(train_losses), np.min(test_losses)

    # Function to format value with asterisk if it's the maximum/minimum
    def format_with_asterisk(value, highlight_value, width=10, maximize=True):
        if (maximize and value == highlight_value) or (not maximize and value == highlight_value):
            return f"{value:<{width-1}.4f}*"
        return f"{value:<{width}.4f}"

    # Update the output log with asterisks for the final printout
    final_output_log = [output_log[0], output_log[1]]
    for idx in range(len(lambda_values)):
        final_output_log.append(f"{idx:<6d} {lambda_values[idx]:<10.5f} "
                                f"{format_with_asterisk(train_aucs[idx], max_train_auc, 10, True)}"
                                f"{format_with_asterisk(test_aucs[idx], max_test_auc, 10, True)}"
                                f"{format_with_asterisk(train_devs[idx], min_train_dev, 12, False)}"
                                f"{format_with_asterisk(test_devs[idx], min_test_dev, 12, False)}"
                                f"{format_with_asterisk(train_losses[idx], min_train_loss, 10, False)}"
                                f"{format_with_asterisk(test_losses[idx], min_test_loss, 10, False)}"
                                f"{int(beta_univ_counts[idx]):<11d}"
                                f"{int(beta_biv_counts[idx]):<10d}")

    # Display the final output with asterisks
    clear_output(wait=True)
    fig = plot_lasso_path(lambda_values, train_losses, test_losses, train_aucs, test_aucs, train_devs, test_devs)
    display(fig)
    plt.close(fig)
    display(HTML("<pre style='font-family: monospace;'>" + "\n".join(final_output_log) + "</pre>"))

    # Print warnings after the loop
    if len(w) > 0:
        print("\nWarnings encountered during LASSO regression:")
        for warning in w:
            print(warning.message)

    # Create and return LassoResultsManager
    return LassoResultsManager(lambda_values, betas, models, feature_names, bivariate_inputs, 
                               train_losses, test_losses, train_aucs, test_aucs, train_devs, test_devs)