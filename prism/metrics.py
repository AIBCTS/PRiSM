import warnings
from typing import Any, Dict, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import metrics
from sklearn.calibration import calibration_curve

from prism.plotting import PlotFormatter


def to_numpy(tensor_or_array):
    if isinstance(tensor_or_array, torch.Tensor):
        return tensor_or_array.cpu().numpy()
    return np.asarray(tensor_or_array)


def evaluate_model_performance(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    y_train: Union[np.ndarray, torch.Tensor] = None,
    n_bootstraps: int = 20,
    alpha: float = 0.05,
    n_bins: int = 10,
    show_results: bool = True,
    threshold: float = 0.5,
    title: str = None,
    min_bin_samples: int = 5,
    random_seed: int = None,
) -> Dict[str, Any]:
    """
    Evaluate model performance with various metrics and plots.

    Confidence intervals are computed using non-parametric bootstrap resampling
    (n_bootstraps samples with replacement). For ROC curves, pointwise confidence
    intervals are calculated at each false positive rate. For calibration curves,
    uniform binning is used with n_bins equal-width intervals from 0 to 1, and
    confidence intervals are calculated for each bin. Bins with insufficient samples
    (< min_bin_samples) are excluded from confidence interval calculations to avoid
    unstable estimates.

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
    min_bin_samples : int, optional
        Minimum number of samples required in a bin to calculate confidence intervals (default is 5).
    random_seed : int, optional
        Random seed for reproducible bootstrapping (default is None).

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing performance metrics and figures.
    """

    y_true = to_numpy(y_true)
    y_pred = to_numpy(y_pred)
    y_train = to_numpy(y_train) if y_train is not None else None

    if y_train is not None:
        # Warn if y_train is not binary
        unique_values = np.unique(y_train)
        if (
            not np.array_equal(unique_values, [0, 1])
            and not np.array_equal(unique_values, [0])
            and not np.array_equal(unique_values, [1])
        ):
            warnings.warn(
                "y_train contains non-binary values. The prevalence (threshold) will be calculated as the mean of these values."
                + " This may lead to unexpected results if y_train is not already a probability or decision function output.",
                UserWarning,
            )

        threshold = np.mean(y_train)  # Set threshold to prevalence

    # Binarize y_true if it's continuous
    if y_true.dtype in [np.float32, np.float64]:
        y_true_binary = (y_true > threshold).astype(float)
    else:
        y_true_binary = y_true

    # Binarize y_pred
    y_pred_binary = (y_pred > threshold).astype(float)

    # Calculate AUROC and confidence interval
    auc = metrics.roc_auc_score(y_true_binary, y_pred)
    ci_lower, ci_upper = bootstrap_auc_ci(y_pred, y_true_binary, n_bootstraps, alpha, random_seed)

    # Calculate other metrics
    fpr, tpr, _ = metrics.roc_curve(y_true_binary, y_pred)

    accuracy = metrics.accuracy_score(y_true_binary, y_pred_binary)
    precision = metrics.precision_score(y_true_binary, y_pred_binary)
    recall = metrics.recall_score(y_true_binary, y_pred_binary)
    f1 = metrics.f1_score(y_true_binary, y_pred_binary)

    # Create plot formatter instance for consistent styling
    plot_formatter = PlotFormatter()

    # Create ROC curve and Calibration curve plots side-by-side
    fig, (ax_roc, ax_cal) = plt.subplots(1, 2, figsize=(7, 3.5))

    # Prepare the histogram right y-axis for calibration curve
    ax_hist = ax_cal.twinx()

    # Set the overall figure title if provided with standardized font sizing
    if title:
        fig.suptitle(title, fontsize=plot_formatter.get_font_size('title'))

    # Apply full formatting defaults to each axis
    for ax in [ax_roc, ax_cal, ax_hist]:
        plot_formatter.apply_defaults(ax)
        ax.grid(False)

    # ROC curve with confidence intervals
    roc_ci = bootstrap_roc_curve_ci(
        y_true_binary, y_pred, n_bootstraps=n_bootstraps, alpha=alpha, random_seed=random_seed
    )

    # Plot original ROC curve
    ax_roc.plot(
        roc_ci['fpr'],
        roc_ci['tpr_orig'],
        label=f'ROC curve (AUC = {auc:.3f})',
        color='blue',
    )

    # Add confidence intervals as shaded area
    ax_roc.fill_between(
        roc_ci['fpr'],
        roc_ci['tpr_lower'],
        roc_ci['tpr_upper'],
        color='blue',
        alpha=0.25,
        label=f'{int((1-alpha)*100)}% CI (AUC: {ci_lower:.3f}-{ci_upper:.3f})',
    )

    ax_roc.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random classifier')

    ax_roc.set_xlabel('False positive rate', fontsize=plot_formatter.get_font_size('label'))
    ax_roc.set_ylabel('True positive rate', fontsize=plot_formatter.get_font_size('label'))
    ax_roc.set_title(
        'Receiver operating characteristic (ROC) curve',
        fontsize=plot_formatter.get_font_size('title'),
    )
    plot_formatter.style_legend(ax_roc)

    # Calibration curve with confidence intervals
    cal_ci = bootstrap_calibration_curve_ci(
        y_true_binary,
        y_pred,
        n_bootstraps=n_bootstraps,
        alpha=alpha,
        n_bins=n_bins,
        min_bin_samples=min_bin_samples,
        title=title,
        random_seed=random_seed,
    )

    # Check and warn about missing calibration points
    expected_points = n_bins
    actual_points = len(cal_ci['prob_true'])
    missing_points = expected_points - actual_points

    # Also check for bins with no samples
    zero_count_bins = np.where(cal_ci['counts'] == 0)[0]

    if missing_points > 0 or len(zero_count_bins) > 0:
        warning_msg = []
        if missing_points > 0:
            warning_msg.append(
                f"{missing_points} point(s) omitted: insufficient samples in probability range(s)"
            )
        if len(zero_count_bins) > 0:
            warning_msg.append(
                f"Points omitted for probability range(s) {zero_count_bins}: no predictions in these ranges"
            )

        warnings.warn(
            f"{title or 'Model'}: " + ". ".join(warning_msg),
            UserWarning,
        )

    # Plot calibration curve
    # Plot reference and threshold lines first with lower zorder
    ax_cal.plot(
        [0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated', zorder=1
    )
    # Mark the classification threshold
    ax_cal.axvline(
        x=threshold,
        color='green',
        linestyle=':',
        linewidth=2,
        label=f'Threshold ({threshold:.2f})',
        zorder=1,
    )

    # Plot model curve and confidence intervals with higher zorder
    ax_cal.plot(
        cal_ci['prob_pred'], cal_ci['prob_true'], marker='o', color='blue', label='Model', zorder=3
    )

    # Add confidence intervals as shaded area only for valid bins
    valid_mask = cal_ci['valid_mask']
    if np.any(valid_mask):
        # Make sure mask length matches array length
        if len(valid_mask) == len(cal_ci['prob_pred']):
            # Filter arrays using the valid mask
            valid_x = cal_ci['prob_pred'][valid_mask]
            valid_lower = cal_ci['prob_true_lower'][valid_mask]
            valid_upper = cal_ci['prob_true_upper'][valid_mask]

            # Plot confidence intervals only for valid regions
            ax_cal.fill_between(
                valid_x,
                valid_lower,
                valid_upper,
                color='blue',
                alpha=0.25,
                label=f'{int((1-alpha)*100)}% CI',
                zorder=2,
            )
        else:
            warnings.warn(
                f"Mask length ({len(valid_mask)}) doesn't match array length ({len(cal_ci['prob_pred'])}). "
                f"Skipping confidence interval display.",
                UserWarning,
            )
    ax_cal.set_xlabel('Mean predicted probability', fontsize=plot_formatter.get_font_size('label'))
    ax_cal.set_ylabel('Fraction of positives', fontsize=plot_formatter.get_font_size('label'))
    ax_cal.set_title('Calibration curve', fontsize=plot_formatter.get_font_size('title'))

    # Add histogram to calibration curve with visible edges
    ax_hist.hist(
        y_pred,
        bins=n_bins,
        alpha=0.3,
        range=(0, 1),
        color='gray',
        linewidth=1,
        edgecolor='white',
    )
    ax_hist.set_ylabel('Count', color='gray', fontsize=plot_formatter.get_font_size('label'))
    ax_hist.tick_params(axis='y', labelcolor='gray')

    # Adjust y-axis for density
    ax_hist.set_ylim(0, ax_hist.get_ylim()[1] * 1.2)

    # Mark the classification threshold
    ax_cal.axvline(
        x=threshold,
        color='green',
        linestyle=':',
        linewidth=2,
        label=f'Threshold ({threshold:.2f})',
    )

    plot_formatter.style_legend(ax_cal, loc='upper left')

    plt.tight_layout()

    # Compile results
    results = {
        'auroc': auc,
        'auroc_ci': (ci_lower, ci_upper),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'figure': fig,
    }

    if show_results:
        print(f"\n---- Model Performance Metrics {title} ----")
        print(f"Threshold (prevalence) set to: {threshold:.3f}")
        print(
            f"AUROC:\t\t{auc:.3f} (95% CI: {ci_lower:.3f}-{ci_upper:.3f}) (Area Under the Receiver Operating Characteristic curve)"
        )
        print(f"Accuracy:\t{accuracy:.3f} (Proportion of correct predictions)")
        print(
            f"Precision:\t{precision:.3f} (Proportion of true positives among positive predictions, aka PPV)"
        )
        print(
            f"Recall:\t\t{recall:.3f} (Proportion of true positives among actual positives, aka sensitivity)"
        )
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
    n_bootstraps: int = 20,
    alpha: float = 0.05,
    n_bins: int = 10,
    show_results: bool = True,
    threshold: float = 0.5,
    title: str = None,
    min_bin_samples: int = 5,
    random_seed: int = None,
) -> Dict[str, Any]:
    """
    Compare performance of two models with various metrics and plots.

    Confidence intervals are computed using non-parametric bootstrap resampling
    (n_bootstraps samples with replacement). For ROC curves, pointwise confidence
    intervals are calculated at each false positive rate. For calibration curves,
    uniform binning is used with n_bins equal-width intervals from 0 to 1, and
    confidence intervals are calculated for each bin. Bins with insufficient samples
    (< min_bin_samples) are excluded from confidence interval calculations. Note that
    overlapping confidence intervals between models do not necessarily indicate lack
    of statistical significance in the difference between model performances.

    Parameters:
    -----------
    y_true : np.ndarray or torch.Tensor
        True labels or continuous values.
    y_pred_1 : np.ndarray or torch.Tensor
        Predicted probabilities from first model.
    y_pred_2 : np.ndarray or torch.Tensor
        Predicted probabilities from second model.
    model_names : tuple, optional
        Names of the two models for display (default is ("Model 1", "Model 2")).
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
    min_bin_samples : int, optional
        Minimum number of samples required in a bin to calculate confidence intervals (default is 5).

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing performance metrics and figures for both models.
    """
    # Convert inputs to NumPy arrays and ensure they are flattened
    y_true = to_numpy(y_true).flatten()
    y_pred_1 = to_numpy(y_pred_1).flatten()
    y_pred_2 = to_numpy(y_pred_2).flatten()
    y_train = to_numpy(y_train).flatten() if y_train is not None else None

    if y_train is not None:
        # Warn if y_train is not binary
        unique_values = np.unique(y_train)
        if (
            not np.array_equal(unique_values, [0, 1])
            and not np.array_equal(unique_values, [0])
            and not np.array_equal(unique_values, [1])
        ):
            warnings.warn(
                "y_train contains non-binary values. The prevalence (threshold) will be calculated as the mean of these values."
                + " This may lead to unexpected results if y_train is not already a probability or decision function output.",
                UserWarning,
            )

        threshold = np.mean(y_train)  # Set threshold to prevalence

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
    for i, (y_pred, y_pred_binary) in enumerate(
        [(y_pred_1, y_pred_binary_1), (y_pred_2, y_pred_binary_2)]
    ):
        model_name = model_names[i]
        auc = metrics.roc_auc_score(y_true_binary, y_pred)
        ci_lower, ci_upper = bootstrap_auc_ci(
            y_pred, y_true_binary, n_bootstraps, alpha, random_seed
        )
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
            'tpr': tpr,
        }

    # Create ROC curve and Calibration curve plots side-by-side
    fig, (ax_roc, ax_cal) = plt.subplots(1, 2, figsize=(8, 4))

    # Create plot formatter instance for consistent styling
    plot_formatter = PlotFormatter()

    # Set the overall figure title if provided with standardized font sizing
    if title:
        fig.suptitle(title, fontsize=plot_formatter.get_font_size('title'))

    # Create second y-axis for histograms
    ax_hist = ax_cal.twinx()

    # Apply full formatting defaults to each axis
    for ax in [ax_roc, ax_cal, ax_hist]:
        plot_formatter.apply_defaults(ax)
        ax.grid(False)

    colors = ['blue', 'red']

    # ROC curve with confidence intervals
    for i, model_name in enumerate(model_names):
        model_results = results[model_name]

        # Calculate ROC curve confidence intervals
        roc_ci = bootstrap_roc_curve_ci(
            y_true_binary,
            model_results['y_pred'],
            n_bootstraps=n_bootstraps,
            alpha=alpha,
            random_seed=random_seed,
        )

        # Plot ROC curve
        ax_roc.plot(
            roc_ci['fpr'],
            roc_ci['tpr_orig'],
            color=colors[i],
            label=f'{model_name} (AUC = {model_results["auroc"]:.3f})',
        )

        # Add confidence interval as shaded area
        ax_roc.fill_between(
            roc_ci['fpr'],
            roc_ci['tpr_lower'],
            roc_ci['tpr_upper'],
            color=colors[i],
            alpha=0.3,
            label=f'{model_name} {int((1-alpha)*100)}% CI (AUC: {model_results["auroc_ci"][0]:.3f}-{model_results["auroc_ci"][1]:.3f})',
        )

    ax_roc.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random classifier')
    ax_roc.set_xlabel('False positive rate')
    ax_roc.set_ylabel('True positive rate')
    ax_roc.set_title(
        'Receiver operating characteristic (ROC) curve',
        fontsize=plot_formatter.get_font_size('title'),
    )
    plot_formatter.style_legend(ax_roc)

    # Ensure the calibration axis stays on top
    ax_cal.set_zorder(ax_hist.get_zorder() + 1)
    ax_cal.patch.set_visible(False)  # Make the calibration axis background transparent
    # Add histograms with denser hatched patterns
    hatches = ['//////', '\\\\\\\\\\\\']  # Denser hatch patterns
    for i, (model_name, hatch) in enumerate(zip(model_names, hatches)):
        ax_hist.hist(
            results[model_name]['y_pred'],
            bins=n_bins,
            alpha=0.3,
            range=(0, 1),
            color='lightgray',
            label=model_name,
            edgecolor=colors[i],
            linewidth=1,
            hatch=hatch,
            density=True,  # Normalize the histograms for better comparison
            zorder=-1,  # Ensure histograms are definitely in the background
            rwidth=0.8,  # Make bars slightly narrower to create spacing
        )
    ax_hist.set_ylabel('Count', color='gray', fontsize=plot_formatter.get_font_size('label'))
    ax_hist.tick_params(axis='y', labelcolor='gray')

    # Adjust y-axis for density
    ax_hist.set_ylim(0, ax_hist.get_ylim()[1] * 1.2)

    # Calibration curve with confidence intervals
    for i, model_name in enumerate(model_names):
        # Calculate calibration curve confidence intervals
        cal_ci = bootstrap_calibration_curve_ci(
            y_true_binary,
            results[model_name]['y_pred'],
            n_bootstraps=n_bootstraps,
            alpha=alpha,
            n_bins=n_bins,
            min_bin_samples=min_bin_samples,
            title=model_name,
            random_seed=random_seed,
        )

        # Check and warn about missing calibration points
        expected_points = n_bins
        actual_points = len(cal_ci['prob_true'])
        missing_points = expected_points - actual_points

        # Also check for bins with no samples
        zero_count_bins = np.where(cal_ci['counts'] == 0)[0]

        if missing_points > 0 or len(zero_count_bins) > 0:
            warning_msg = []
            if missing_points > 0:
                warning_msg.append(
                    f"{missing_points} point(s) omitted: insufficient samples in probability range(s)"
                )
            if len(zero_count_bins) > 0:
                warning_msg.append(
                    f"Points omitted for probability range(s) {zero_count_bins}: no predictions in these ranges"
                )

            warnings.warn(
                f"{model_name}: " + ". ".join(warning_msg),
                UserWarning,
            )

        # Plot calibration curve
        ax_cal.plot(
            cal_ci['prob_pred'],
            cal_ci['prob_true'],
            marker='o',
            color=colors[i],
            label=model_name,
            zorder=3,  # Ensure curves are on top
        )

        # Add confidence interval as shaded area only for valid bins
        valid_mask = cal_ci['valid_mask']
        if np.any(valid_mask):
            # Make sure mask length matches array length
            if len(valid_mask) == len(cal_ci['prob_pred']):
                # Filter arrays using the valid mask
                valid_x = cal_ci['prob_pred'][valid_mask]
                valid_lower = cal_ci['prob_true_lower'][valid_mask]
                valid_upper = cal_ci['prob_true_upper'][valid_mask]

                # Plot confidence intervals only for valid regions
                ax_cal.fill_between(
                    valid_x,
                    valid_lower,
                    valid_upper,
                    color=colors[i],
                    alpha=0.3,
                    label=f'{model_name} {int((1-alpha)*100)}% CI',
                    zorder=2,  # Ensure confidence intervals are above histograms but below curves
                )
            else:
                warnings.warn(
                    f"Mask length ({len(valid_mask)}) doesn't match array length ({len(cal_ci['prob_pred'])}). "
                    f"Skipping confidence interval display.",
                    UserWarning,
                )

    # Plot reference lines with lower zorder
    ax_cal.plot(
        [0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated', zorder=1
    )

    # Mark the classification threshold with lower zorder
    ax_cal.axvline(
        x=threshold,
        color='green',
        linestyle=':',
        linewidth=2,
        label=f'Threshold ({threshold:.2f})',
        zorder=1,
    )

    ax_cal.set_xlabel('Mean predicted probability')
    ax_cal.set_ylabel('Fraction of positives')
    ax_cal.set_title('Calibration curve', fontsize=plot_formatter.get_font_size('title'))

    plot_formatter.style_legend(ax_cal, loc='upper left')
    plot_formatter.style_legend(ax_hist, loc='lower right')
    ax_hist.get_legend().set_title('Count')

    plt.tight_layout()

    if show_results:
        print("\n-------- Model Performance Comparison --------")
        print(f"Threshold (prevalence) set to: {threshold:.3f}")
        for model_name in model_names:
            print(f"\n{model_name}:")
            print(
                f"AUROC:\t\t{results[model_name]['auroc']:.3f} (95% CI: {results[model_name]['auroc_ci'][0]:.3f}-{results[model_name]['auroc_ci'][1]:.3f})"
            )
            print(f"Accuracy:\t{results[model_name]['accuracy']:.3f}")
            print(f"Precision:\t{results[model_name]['precision']:.3f}")
            print(f"Recall:\t\t{results[model_name]['recall']:.3f}")
            print(f"F1 Score:\t{results[model_name]['f1_score']:.3f}")
        print("------------------------------------------------")

        plt.show()

    results['figure'] = fig
    return results


def bootstrap_auc_ci(pred, target, n_bootstraps=1000, alpha=0.05, random_seed=None):
    """
    Calculate confidence interval for Area Under the Receiver Operating Characteristic curve (AUROC)
    using non-parametric bootstrap resampling.

    This function performs case resampling with replacement to create multiple bootstrap samples
    from the original data, computes the AUROC for each sample, and derives confidence intervals
    from the empirical distribution of these values.

    Parameters:
    -----------
    pred : array-like or pandas Series
        Predicted probabilities or scores from the classification model.
    target : array-like or pandas Series
        True binary labels (ground truth).
    n_bootstraps : int, optional
        Number of bootstrap samples to generate (default is 1000).
        Higher values provide more precise interval estimation at increased computational cost.
    alpha : float, optional
        Significance level for confidence interval calculation (default is 0.05).
        The resulting interval will be a (1-alpha) confidence interval.
    random_seed : int, optional
        Seed for random number generator to ensure reproducibility.

    Returns:
    --------
    tuple
        Lower and upper bounds of the bootstrap confidence interval.

    Notes:
    ------
    Bootstrap samples that lack both positive and negative examples are skipped
    as AUROC cannot be calculated in these cases. This can occur with highly
    imbalanced datasets and may result in fewer than n_bootstraps effective samples.
    """
    pred = to_numpy(pred)
    target = to_numpy(target)

    n_samples = len(target)
    bootstrapped_scores = []

    rng = np.random.default_rng(random_seed)

    # Keep sampling until we get exactly n_bootstraps valid samples
    while len(bootstrapped_scores) < n_bootstraps:
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        if (
            len(np.unique(target[indices])) >= 2
        ):  # Only proceed if we have both positive and negative samples
            score = metrics.roc_auc_score(target[indices], pred[indices])
            bootstrapped_scores.append(score)

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()  # Sort in place to ensure deterministic behavior
    ci_lower = np.percentile(sorted_scores, alpha / 2 * 100)
    ci_upper = np.percentile(sorted_scores, (1 - alpha / 2) * 100)

    return ci_lower, ci_upper


def bootstrap_calibration_curve_ci(
    y_true,
    y_pred,
    n_bootstraps=1000,
    alpha=0.05,
    n_bins=10,
    strategy='uniform',
    min_bin_samples=5,
    title=None,
    random_seed=None,
):
    """
    Calculate pointwise confidence intervals for calibration curves using non-parametric bootstrap resampling.

    This function creates bootstrap samples by case resampling with replacement, generates
    calibration curves for each sample, and derives pointwise confidence intervals at each
    probability bin.

    Bins with insufficient samples (below min_bin_samples threshold) are excluded from
    confidence interval calculation to prevent misleading estimates from sparse data.

    Parameters:
    -----------
    y_true : array-like or pandas Series
        True binary labels (ground truth).
    y_pred : array-like or pandas Series
        Predicted probabilities or scores from the classification model.
    n_bootstraps : int, optional
        Number of bootstrap samples to generate (default is 1000).
        Higher values provide more precise interval estimation at increased computational cost.
    alpha : float, optional
        Significance level for confidence interval calculation (default is 0.05).
        The resulting intervals will be (1-alpha) confidence intervals.
    n_bins : int, optional
        Number of bins for discretizing predicted probabilities (default is 10).
        Affects smoothness of calibration curve and statistical stability.
    strategy : str, optional
        Binning strategy for calibration curve ('uniform' or 'quantile').
        'uniform' creates equal-width bins, while 'quantile' creates equal-population bins.
    min_bin_samples : int, optional
        Minimum number of samples required in a bin to calculate confidence intervals (default is 5).
        Protects against unstable estimates from sparse data.
    title : str, optional
        Title or model name to include in warning messages (default is None).
    random_seed : int, optional
        Seed for random number generator to ensure reproducibility.

    Returns:
    --------
    dict
        A dictionary containing:
        - 'prob_pred': Mean predicted probabilities for each bin
        - 'prob_true': Mean observed positive rate for each bin
        - 'prob_true_lower': Lower bound of pointwise confidence intervals
        - 'prob_true_upper': Upper bound of pointwise confidence intervals
        - 'counts': Counts of samples in each bin
        - 'valid_mask': Boolean mask indicating which bins have sufficient samples

    Notes:
    ------
    Bootstrap samples that lack both positive and negative examples are skipped
    as calibration curves cannot be meaningfully calculated in these cases.
    This function provides bin-specific warnings when confidence intervals are
    omitted due to insufficient data.
    """
    y_true = to_numpy(y_true)
    y_pred = to_numpy(y_pred)

    n_samples = len(y_true)
    bootstrapped_curves = []

    rng = np.random.default_rng(random_seed)

    # Keep sampling until we get exactly n_bootstraps valid samples
    while len(bootstrapped_curves) < n_bootstraps:
        # Bootstrap resample the data
        indices = rng.choice(n_samples, size=n_samples, replace=True)

        if len(np.unique(y_true[indices])) >= 2:
            # Only proceed if we have both positive and negative examples
            try:
                prob_true, prob_pred = calibration_curve(
                    y_true[indices], y_pred[indices], n_bins=n_bins, strategy=strategy
                )
                bootstrapped_curves.append((prob_true, prob_pred))
            except ValueError:
                # Skip if there's a ValueError (common with uniform binning when bins are empty)
                continue

    # Calculate the original calibration curve (non-bootstrapped)
    orig_prob_true, orig_prob_pred = calibration_curve(
        y_true, y_pred, n_bins=n_bins, strategy=strategy
    )

    # Count samples in each bin for the original data
    bin_counts = np.zeros(len(orig_prob_pred))  # Use actual length of prob_pred

    if strategy == 'uniform':
        bin_edges = np.linspace(0, 1, n_bins + 1)
        # Add small epsilon to include 1.0 in the last bin
        bin_edges[-1] += 1e-10
    else:  # quantile strategy
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.percentile(y_pred, quantiles * 100)
        bin_edges[-1] = 1.0 + 1e-10  # Ensure the last bin includes 1.0

    for i in range(len(bin_counts)):
        if i < len(bin_edges) - 1:
            if i == len(bin_edges) - 2:  # Last bin
                # Include both edges for last bin to catch predictions of exactly 1.0
                bin_counts[i] = np.sum((y_pred >= bin_edges[i]) & (y_pred <= bin_edges[i + 1]))
            else:
                # Regular bin counting for other bins
                bin_counts[i] = np.sum((y_pred >= bin_edges[i]) & (y_pred < bin_edges[i + 1]))

    # Create a mask for bins with sufficient samples
    valid_mask = bin_counts >= min_bin_samples

    # Warn about skipped bins
    skipped_bins = np.sum(~valid_mask)
    if skipped_bins > 0:
        bin_indices = np.where(~valid_mask)[0]
        warnings.warn(
            f"{title or 'Model'}: Skipped confidence intervals for {skipped_bins} bin(s) "
            f"due to insufficient samples (threshold: {min_bin_samples}). "
            f"Bin indices: {bin_indices}. "
            f"These bins had the following sample counts: {bin_counts[~valid_mask]}.",
            UserWarning,
        )

    # For each bin position, collect all bootstrap values
    prob_true_samples = [[] for _ in range(len(orig_prob_pred))]

    for prob_true, prob_pred in bootstrapped_curves:
        for i in range(len(prob_true)):
            # Only include bins that exist in this bootstrap sample
            if i < len(prob_true) and i < len(prob_true_samples):
                prob_true_samples[i].append(prob_true[i])

    # Calculate CI for each bin position
    prob_true_lower = np.full(len(orig_prob_pred), np.nan)  # Initialize with NaN
    prob_true_upper = np.full(len(orig_prob_pred), np.nan)  # Initialize with NaN

    for i in range(len(orig_prob_pred)):
        if i < len(valid_mask) and valid_mask[i] and len(prob_true_samples[i]) > 0:
            prob_true_lower[i] = np.percentile(prob_true_samples[i], alpha / 2 * 100)
            prob_true_upper[i] = np.percentile(prob_true_samples[i], (1 - alpha / 2) * 100)

    return {
        'prob_pred': orig_prob_pred,
        'prob_true': orig_prob_true,
        'prob_true_lower': prob_true_lower,
        'prob_true_upper': prob_true_upper,
        'counts': bin_counts,
        'valid_mask': valid_mask,
    }


def bootstrap_roc_curve_ci(
    y_true, y_pred, n_bootstraps=1000, alpha=0.05, n_points=100, random_seed=None
):
    """
    Calculate pointwise confidence intervals for ROC curves using non-parametric bootstrap resampling.

    This function generates bootstrap samples through case resampling with replacement,
    computes ROC curves for each sample, and derives pointwise confidence bands by
    interpolating to a common false positive rate (FPR) grid.

    Parameters:
    -----------
    y_true : array-like or pandas Series
        True binary labels (ground truth).
    y_pred : array-like or pandas Series
        Predicted probabilities or scores from the classification model.
    n_bootstraps : int, optional
        Number of bootstrap samples to generate (default is 1000).
        Higher values provide more precise interval estimation at increased computational cost.
    alpha : float, optional
        Significance level for confidence interval calculation (default is 0.05).
        The resulting intervals will be (1-alpha) confidence intervals.
    n_points : int, optional
        Number of interpolation points for the false positive rate grid (default is 100).
        Higher values provide smoother curves but may increase computational burden.
    random_seed : int, optional
        Seed for random number generator to ensure reproducibility.

    Returns:
    --------
    dict
        A dictionary containing:
        - 'fpr': Common false positive rate grid for interpolation
        - 'tpr': Mean true positive rate values across bootstrap samples
        - 'tpr_lower': Lower bound of the pointwise confidence interval for true positive rate
        - 'tpr_upper': Upper bound of the pointwise confidence interval for true positive rate
        - 'tpr_orig': True positive rate values from the original (non-bootstrapped) ROC curve
        - 'auc': Area under the ROC curve for the original data

    Notes:
    ------
    Bootstrap samples that lack both positive and negative examples are skipped
    as ROC curves cannot be calculated in these cases. The approach handles edge
    cases where ROC curves don't precisely start at (0,0) or end at (1,1) by adding
    these points to ensure proper interpolation across the full FPR range.
    """
    y_true = to_numpy(y_true)
    y_pred = to_numpy(y_pred)

    # Calculate the original ROC curve
    fpr_orig, tpr_orig, _ = metrics.roc_curve(y_true, y_pred)
    auc_orig = metrics.auc(fpr_orig, tpr_orig)

    # Create common FPR points for interpolation
    # Adding 0 and 1 ensures we have the endpoints
    fpr_grid = np.linspace(0, 1, n_points)

    # Interpolate the original TPR values onto the common grid
    tpr_interp_orig = np.interp(fpr_grid, fpr_orig, tpr_orig)

    # Arrays to store interpolated TPR values for each bootstrap
    tpr_bootstraps = []

    n_samples = len(y_true)
    rng = np.random.default_rng(random_seed)

    # Keep sampling until we get exactly n_bootstraps valid samples
    while len(tpr_bootstraps) < n_bootstraps:
        # Bootstrap resample the data
        indices = rng.choice(n_samples, size=n_samples, replace=True)

        # Only proceed if we have both positive and negative examples
        if len(np.unique(y_true[indices])) >= 2:
            # Calculate ROC curve for this bootstrap sample
            fpr_boot, tpr_boot, _ = metrics.roc_curve(y_true[indices], y_pred[indices])

            # Handle edge cases where ROC curve doesn't start at (0,0) or end at (1,1)
            if fpr_boot[0] != 0 or tpr_boot[0] != 0:
                fpr_boot = np.concatenate(([0], fpr_boot))
                tpr_boot = np.concatenate(([0], tpr_boot))

            if fpr_boot[-1] != 1 or tpr_boot[-1] != 1:
                fpr_boot = np.concatenate((fpr_boot, [1]))
                tpr_boot = np.concatenate((tpr_boot, [1]))

            # Interpolate bootstrap TPR onto common grid and store
            tpr_interp = np.interp(fpr_grid, fpr_boot, tpr_boot)
            tpr_bootstraps.append(tpr_interp)

    # Convert list of TPR arrays to 2D array for easier percentile calculation
    tpr_bootstraps = np.array(tpr_bootstraps)

    # Calculate percentiles across the bootstrap dimension (axis 0)
    tpr_lower = np.percentile(tpr_bootstraps, alpha / 2 * 100, axis=0)
    tpr_upper = np.percentile(tpr_bootstraps, (1 - alpha / 2) * 100, axis=0)

    # Calculate mean TPR across bootstrap samples
    tpr_mean = np.mean(tpr_bootstraps, axis=0)

    return {
        'fpr': fpr_grid,
        'tpr': tpr_mean,
        'tpr_lower': tpr_lower,
        'tpr_upper': tpr_upper,
        'tpr_orig': tpr_interp_orig,
        'auc': auc_orig,
    }
