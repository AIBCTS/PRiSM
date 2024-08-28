import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from typing import Dict, Any, Union
import warnings

def evaluate_model_performance(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    y_train: Union[np.ndarray, torch.Tensor] = None,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    n_bins: int = 10,
    show_results: bool = True,
    threshold: float = 0.5,
    title: str = None
) -> Dict[str, Any]:
    """
    Evaluate model performance with various metrics and plots.

    Parameters:
    -----------
    y_true : np.ndarray or torch.Tensor
        True labels or continuous values.
    y_pred : np.ndarray or torch.Tensor
        Predicted probabilities.
    y_train : np.ndarray or torch.Tensor, optional
        Training labels for prevalence calculation.
    n_bootstraps : int, optional
        Number of bootstrap samples for CI calculation.
    alpha : float, optional
        Significance level for CI calculation.
    n_bins : int, optional
        Number of bins for calibration curve.
    show_results : bool, optional
        Whether to display results and plots (default is True).
    threshold : float, optional
        Threshold for binarizing continuous y_true values (default is 0.5).
    title : str, optional
        Title for the overall figure (default is None).

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing performance metrics and figures.
    """
    # Convert inputs to NumPy arrays if they are PyTorch tensors
    y_true = y_true.numpy() if isinstance(y_true, torch.Tensor) else y_true
    y_pred = y_pred.numpy() if isinstance(y_pred, torch.Tensor) else y_pred
    y_train = y_train.numpy() if isinstance(y_train, torch.Tensor) else y_train
    
    if y_train is not None:
        # Warn if y_train is not binary
        unique_values = np.unique(y_train)
        if not np.array_equal(unique_values, [0, 1]) and not np.array_equal(unique_values, [0]) and not np.array_equal(unique_values, [1]):
            warnings.warn("y_train contains non-binary values. The prevalence (threshold) will be calculated as the mean of these values. "
                          "This may lead to unexpected results if y_train is not already a probability or decision function output.", 
                          UserWarning)
        
        threshold = np.mean(y_train) # Set threshold to prevalence

    # Binarize y_true if it's continuous
    if y_true.dtype in [np.float32, np.float64]:
        y_true_binary = (y_true > threshold).astype(float)
    else:
        y_true_binary = y_true

    # Binarize y_pred
    y_pred_binary = (y_pred > threshold).astype(float)

    # Calculate AUROC and confidence interval
    auc = metrics.roc_auc_score(y_true_binary, y_pred)
    ci_lower, ci_upper = bootstrap_auc_ci(y_pred, y_true_binary, n_bootstraps, alpha)
    
    # Calculate other metrics
    fpr, tpr, _ = metrics.roc_curve(y_true_binary, y_pred)
    
    accuracy = metrics.accuracy_score(y_true_binary, y_pred_binary)
    precision = metrics.precision_score(y_true_binary, y_pred_binary)
    recall = metrics.recall_score(y_true_binary, y_pred_binary)
    f1 = metrics.f1_score(y_true_binary, y_pred_binary)
    
    # Create ROC curve and Calibration curve plots side-by-side
    fig, (ax_roc, ax_cal) = plt.subplots(1, 2, figsize=(12, 5))

    # Set the overall figure title if provided
    if title:
        fig.suptitle(title, fontsize=16)

    # ROC curve
    ax_roc.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.3f}, 95% CI: {ci_lower:.3f}-{ci_upper:.3f})')
    ax_roc.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random classifier')
    
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax_roc.legend(loc='lower right')
    
    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_true_binary, y_pred, n_bins=n_bins)
    ax_cal.plot(prob_pred, prob_true, marker='o', label='Model')
    ax_cal.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
    ax_cal.set_xlabel('Mean predicted probability')
    ax_cal.set_ylabel('Fraction of positives')
    ax_cal.set_title('Calibration Curve')

    # Add histogram to calibration curve with visible edges
    ax_hist = ax_cal.twinx()
    ax_hist.hist(y_pred, bins=n_bins, alpha=0.3, range=(0, 1), 
                edgecolor='black', linewidth=1)
    ax_hist.set_ylabel('Count')

    # Adjust y-axis for density
    ax_hist.set_ylim(0, ax_hist.get_ylim()[1] * 1.2)

    # Mark the classification threshold
    ax_cal.axvline(x=threshold, color='green', linestyle=':', linewidth=2, label=f'Threshold ({threshold:.2f})')

    ax_cal.legend(loc='upper left')
    
    plt.tight_layout()
    
    # Compile results
    results = {
        'auroc': auc,
        'auroc_ci': (ci_lower, ci_upper),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'figure': fig
    }
    
    if show_results:
        print(f"\n---- Model Performance Metrics {title} ----")
        print(f"Threshold (prevalence) set to: {threshold:.3f}")
        print(f"AUROC:\t\t{auc:.3f} (95% CI: {ci_lower:.3f}-{ci_upper:.3f}) (Area Under the Receiver Operating Characteristic curve)")
        print(f"Accuracy:\t{accuracy:.3f} (Proportion of correct predictions)")
        print(f"Precision:\t{precision:.3f} (Proportion of true positives among positive predictions, aka PPV)")
        print(f"Recall:\t\t{recall:.3f} (Proportion of true positives among actual positives, aka sensitivity)")
        print(f"F1 Score:\t{f1:.3f} (Harmonic mean of precision and recall)")
        print("--------------------------------------------")
        
        plt.show()
    
    return results

def compare_model_performance(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred_1: Union[np.ndarray, torch.Tensor],
    y_pred_2: Union[np.ndarray, torch.Tensor],
    model_names: tuple = ("Model 1", "Model 2"),
    y_train: Union[np.ndarray, torch.Tensor] = None,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    n_bins: int = 10,
    show_results: bool = True,
    threshold: float = 0.5,
    title: str = None
) -> Dict[str, Any]:
    """
    Compare performance of two models with various metrics and plots.
    """
    # Convert inputs to NumPy arrays if they are PyTorch tensors and ensure they are flattened
    y_true = y_true.numpy().flatten() if isinstance(y_true, torch.Tensor) else np.array(y_true).flatten()
    y_pred_1 = y_pred_1.numpy().flatten() if isinstance(y_pred_1, torch.Tensor) else np.array(y_pred_1).flatten()
    y_pred_2 = y_pred_2.numpy().flatten() if isinstance(y_pred_2, torch.Tensor) else np.array(y_pred_2).flatten()
    y_train = y_train.numpy().flatten() if isinstance(y_train, torch.Tensor) else np.array(y_train).flatten() if y_train is not None else None
    
    if y_train is not None:
        # Warn if y_train is not binary
        unique_values = np.unique(y_train)
        if not np.array_equal(unique_values, [0, 1]) and not np.array_equal(unique_values, [0]) and not np.array_equal(unique_values, [1]):
            warnings.warn("y_train contains non-binary values. The prevalence (threshold) will be calculated as the mean of these values. "
                          "This may lead to unexpected results if y_train is not already a probability or decision function output.", 
                          UserWarning)
        
        threshold = np.mean(y_train) # Set threshold to prevalence

    # Binarize y_true if it's continuous
    if y_true.dtype in [np.float32, np.float64]:
        y_true_binary = (y_true > threshold).astype(float)
    else:
        y_true_binary = y_true

    # Binarize y_pred for both models
    y_pred_binary_1 = (y_pred_1 > threshold).astype(float)
    y_pred_binary_2 = (y_pred_2 > threshold).astype(float)

    # Calculate metrics for both models
    results = {}
    for i, (y_pred, y_pred_binary) in enumerate([(y_pred_1, y_pred_binary_1), (y_pred_2, y_pred_binary_2)]):
        model_name = model_names[i]
        auc = metrics.roc_auc_score(y_true_binary, y_pred)
        ci_lower, ci_upper = bootstrap_auc_ci(y_pred, y_true_binary, n_bootstraps, alpha)
        fpr, tpr, _ = metrics.roc_curve(y_true_binary, y_pred)
        
        accuracy = metrics.accuracy_score(y_true_binary, y_pred_binary)
        precision = metrics.precision_score(y_true_binary, y_pred_binary)
        recall = metrics.recall_score(y_true_binary, y_pred_binary)
        f1 = metrics.f1_score(y_true_binary, y_pred_binary)
        
        results[model_name] = {
            'y_pred': y_pred,
            'auroc': auc,
            'auroc_ci': (ci_lower, ci_upper),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'fpr': fpr,
            'tpr': tpr
        }

    # Create ROC curve and Calibration curve plots side-by-side
    fig, (ax_roc, ax_cal) = plt.subplots(1, 2, figsize=(12, 5))

    # Set the overall figure title if provided
    if title:
        fig.suptitle(title, fontsize=16)

    colors = ['blue', 'red']
    
    # ROC curve
    for i, model_name in enumerate(model_names):
        model_results = results[model_name]
        ax_roc.plot(model_results['fpr'], model_results['tpr'], color=colors[i],
                    label=f'{model_name} (AUC = {model_results["auroc"]:.3f}, 95% CI: {model_results["auroc_ci"][0]:.3f}-{model_results["auroc_ci"][1]:.3f})')
        
    ax_roc.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random classifier')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax_roc.legend(loc='lower right')
    
    # Calibration curve
    for i, model_name in enumerate(model_names):
        prob_true, prob_pred = calibration_curve(y_true_binary, results[model_name]['y_pred'], n_bins=n_bins)
        ax_cal.plot(prob_pred, prob_true, marker='o', color=colors[i], label=model_name)
    
    ax_cal.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
    ax_cal.set_xlabel('Mean predicted probability')
    ax_cal.set_ylabel('Fraction of positives')
    ax_cal.set_title('Calibration Curve')

    # Add histograms to calibration curve
    ax_hist = ax_cal.twinx()
    ax_hist.hist([results[model_name]['y_pred'] for model_name in model_names], 
                 bins=n_bins, alpha=0.3, range=(0, 1), 
                 color=colors, label=model_names)
    ax_hist.set_ylabel('Count')

    # Adjust y-axis for density
    ax_hist.set_ylim(0, ax_hist.get_ylim()[1] * 1.2)

    # Mark the classification threshold
    ax_cal.axvline(x=threshold, color='green', linestyle=':', linewidth=2, label=f'Threshold ({threshold:.2f})')

    ax_cal.legend(loc='upper left')
    ax_hist.legend(loc='upper right')
    
    plt.tight_layout()
    
    if show_results:
        print("\n-------- Model Performance Comparison --------")
        print(f"Threshold (prevalence) set to: {threshold:.3f}")
        for model_name in model_names:
            print(f"\n{model_name}:")
            print(f"AUROC:\t\t{results[model_name]['auroc']:.3f} (95% CI: {results[model_name]['auroc_ci'][0]:.3f}-{results[model_name]['auroc_ci'][1]:.3f})")
            print(f"Accuracy:\t{results[model_name]['accuracy']:.3f}")
            print(f"Precision:\t{results[model_name]['precision']:.3f}")
            print(f"Recall:\t\t{results[model_name]['recall']:.3f}")
            print(f"F1 Score:\t{results[model_name]['f1_score']:.3f}")
        print("------------------------------------------------")
        
        plt.show()
    
    results['figure'] = fig
    return results

def bootstrap_auc_ci(pred, target, n_bootstraps=1000, alpha=0.05):
    """
    Calculate confidence interval for AUROC using bootstrapping.
    
    Parameters:
    -----------
    pred : array-like or pandas Series
        Predicted probabilities.
    target : array-like or pandas Series
        True binary labels.
    n_bootstraps : int, optional
        Number of bootstrap samples (default is 1000).
    alpha : float, optional
        Significance level for CI calculation (default is 0.05).
    
    Returns:
    --------
    tuple
        Lower and upper bounds of the confidence interval.
    """
    if isinstance(pred, pd.Series):
        pred = pred.values
    if isinstance(target, pd.Series):
        target = target.values
    
    n_samples = len(target)
    bootstrapped_scores = []
    
    rng = np.random.default_rng()
    
    for _ in range(n_bootstraps):
        # Use choice instead of randint for resampling
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        if len(np.unique(target[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            continue
        score = metrics.roc_auc_score(target[indices], pred[indices])
        bootstrapped_scores.append(score)
    
    sorted_scores = np.sort(bootstrapped_scores)
    ci_lower = np.percentile(sorted_scores, alpha / 2 * 100)
    ci_upper = np.percentile(sorted_scores, (1 - alpha / 2) * 100)
    
    return ci_lower, ci_upper