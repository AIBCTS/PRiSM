import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Any, Tuple, Optional
from prism.lasso_results import LassoResultsManager
from prism.partial_responses import partial_responses_subset
from prism.nomogram import NomogramGenerator

def to_numpy(tensor_or_array):
    if isinstance(tensor_or_array, torch.Tensor):
        return tensor_or_array.cpu().numpy()
    return np.asarray(tensor_or_array)

def plot_continuous_response_with_histogram(ax, response, x_values, feature_name, nomogram_generator, feature_index, use_odds_ratio):
    """
    Plot continuous partial response with histogram using denormalized values.
    """
    # Denormalize x_values
    x_denormalized = nomogram_generator.denormalize(x_values, feature_index)
    
    # Convert to odds ratio if necessary
    if use_odds_ratio:
        response = np.exp(response)
    
    # Plot partial response
    ax.plot(x_denormalized, response, label='Partial Response')
    
    # Add histogram of original (denormalized) data
    hist_ax = ax.twinx()
    original_data = nomogram_generator.denormalize(nomogram_generator.x[:, feature_index], feature_index)
    hist_ax.hist(original_data, bins=nomogram_generator.n_steps, alpha=0.3, color='gray')
    hist_ax.set_ylabel('Count', color='gray')
    hist_ax.tick_params(axis='y', labelcolor='gray')
    
    # Set x-ticks
    x_ticks = np.linspace(x_denormalized.min(), x_denormalized.max(), num=5)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{val:.2f}" for val in x_ticks])
    
    ax.set_xlabel(feature_name)
    ax.set_ylabel('Odds Ratio' if use_odds_ratio else 'Log Odds Ratio')

    # Adjust y-axis for density
    ylim = hist_ax.get_ylim()
    hist_ax.set_ylim(0, ylim[1] * 1.2)

    # Add reference line
    if use_odds_ratio:
        ax.axhline(y=1, color='k', linewidth=1, alpha=0.8)
    else:
        ax.axhline(y=0, color='k', linewidth=1, alpha=0.8)

def plot_mixed_response_with_histogram(ax, response, x_values, feature_names, nomogram_generator, feature_indices, categorical_feature, use_odds_ratio):
    """
    Plot mixed (one categorical, one continuous) partial response with histogram using denormalized values.
    """
    cont_feature = 1 - categorical_feature
    cont_index = feature_indices[cont_feature]
    
    # Denormalize x_values
    x_denormalized = np.column_stack([
        nomogram_generator.denormalize(x_values[:, 0], feature_indices[0]),
        nomogram_generator.denormalize(x_values[:, 1], feature_indices[1])
    ])
    
    # Convert to odds ratio if necessary
    if use_odds_ratio:
        response = np.exp(response)
    
    cat_values = np.unique(x_denormalized[:, categorical_feature])
    cont_values = x_denormalized[:, cont_feature]
    
    # Plot lines for each categorical value
    for cat_val in cat_values:
        mask = x_denormalized[:, categorical_feature] == cat_val
        ax.plot(cont_values[mask], response[mask], label=f"{feature_names[categorical_feature]}={cat_val:.2g}")
    
    # Add histogram of original (denormalized) continuous data
    hist_ax = ax.twinx()
    original_cont_data = nomogram_generator.denormalize(nomogram_generator.x[:, cont_index], cont_index)
    hist_ax.hist(original_cont_data, bins=nomogram_generator.n_steps, alpha=0.3, color='gray')
    hist_ax.set_ylabel('Count', color='gray')
    hist_ax.tick_params(axis='y', labelcolor='gray')
    
    # Set x-ticks
    x_ticks = np.linspace(cont_values.min(), cont_values.max(), num=5)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{val:.2f}" for val in x_ticks])
    
    ax.set_xlabel(feature_names[cont_feature])
    ax.set_ylabel('Odds Ratio' if use_odds_ratio else 'Log Odds Ratio')
    ax.legend()

    # Adjust y-axis for density
    ylim = hist_ax.get_ylim()
    hist_ax.set_ylim(0, ylim[1] * 1.2)

    # Add reference line
    if use_odds_ratio:
        ax.axhline(y=1, color='k', linewidth=1, alpha=0.8)
    else:
        ax.axhline(y=0, color='k', linewidth=1, alpha=0.8)

def plot_categorical_response_with_histogram(ax, response, x_values, feature_name, nomogram_generator, feature_index, use_odds_ratio):
    """
    Plot categorical partial response with histogram.
    """
    # Denormalize x_values
    x_denormalized = nomogram_generator.denormalize(x_values, feature_index)
    
    # Convert to odds ratio if necessary
    if use_odds_ratio:
        response = np.exp(response)
    
    # Get unique categories and their counts from the full dataset
    original_data = nomogram_generator.denormalize(nomogram_generator.x[:, feature_index], feature_index)
    categories, counts = np.unique(original_data, return_counts=True)
    
    # Create scatter plot for log odds ratio or odds ratio
    scatter = ax.scatter(x_denormalized, response, s=50, zorder=3, marker='_', linewidth=3)
    ax.set_ylabel('Odds Ratio' if use_odds_ratio else 'Log Odds Ratio')

    # Create histogram
    hist_ax = ax.twinx()
    hist_ax.bar(categories, counts, alpha=0.3, color='gray', width=0.8)
    hist_ax.set_ylabel('Count', color='gray')
    hist_ax.tick_params(axis='y', labelcolor='gray')
    
    # Set x-ticks to category values
    ax.set_xticks(categories)
    ax.set_xticklabels([f"{val:.2g}" for val in categories])
    
    ax.set_xlabel(feature_name)

    # Adjust y-axis for count
    ylim = hist_ax.get_ylim()
    hist_ax.set_ylim(0, ylim[1] * 1.2)

    # Add reference line
    if use_odds_ratio:
        ax.axhline(y=1, color='k', linewidth=1, alpha=0.8)
    else:
        ax.axhline(y=0, color='k', linewidth=1, alpha=0.8)

def plot_partial_responses(lasso_results: LassoResultsManager, 
                           x: torch.Tensor, 
                           x0_median: np.ndarray, 
                           x0_std: np.ndarray, 
                           model: Any, 
                           n_steps: int = 15, 
                           sd_scale: float = 2, 
                           method: str = "dirac", 
                           device: str = "cpu", 
                           categorical_threshold: int = 15,
                           subtract_univariate: bool = True, 
                           figsize: Tuple[int, int] = (15, 20),
                           show_fig: bool = True,
                           return_fig: bool = False,
                           use_odds_ratio: bool = False) -> Optional[plt.Figure]:
    """
    Generate a grid of subplots showing partial responses for the selected lambda.
    """
    univariate_responses, bivariate_responses, x_univariate, x_bivariate = partial_responses_subset(
        x, model, method=method, device=device, n_steps=n_steps, 
        categorical_threshold=categorical_threshold,
        subtract_univariate=subtract_univariate
    )
    
    x_numpy = to_numpy(x)
    nomogram_generator = NomogramGenerator(lasso_results, x_numpy, x0_median, x0_std, n_steps, categorical_threshold, sd_scale, use_odds_ratio)
    
    selected_univariate_indices = lasso_results.get_selected_univariate_indices()
    selected_bivariate_indices = lasso_results.get_selected_bivariate_indices()
    selected_bivariate_index_pairs = lasso_results.get_selected_bivariate_index_pairs()
    
    n_plots = len(selected_univariate_indices) + len(selected_bivariate_indices)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    plot_index = 0
    
    # Plot univariate responses
    for i in selected_univariate_indices:
        ax = axes[plot_index]
        feature_name = lasso_results.univariate_feature_names[i]
        response = to_numpy(univariate_responses[i])
        x_values = to_numpy(x_univariate[i])
        if len(np.unique(x_values)) < categorical_threshold:
            plot_categorical_response_with_histogram(ax, response, x_values, feature_name, nomogram_generator, i, use_odds_ratio)
        else:
            plot_continuous_response_with_histogram(ax, response, x_values, feature_name, nomogram_generator, i, use_odds_ratio)
        plot_index += 1
    
    # Plot bivariate responses
    for biv_index, (i, j) in zip(selected_bivariate_indices, selected_bivariate_index_pairs):
        ax = axes[plot_index]
        feature1 = lasso_results.univariate_feature_names[i]
        feature2 = lasso_results.univariate_feature_names[j]
        response = to_numpy(bivariate_responses[biv_index])
        x_values = to_numpy(x_bivariate[biv_index])
        is_categorical1 = len(np.unique(x_numpy[:, i])) < categorical_threshold
        is_categorical2 = len(np.unique(x_numpy[:, j])) < categorical_threshold
        
        if is_categorical1 != is_categorical2:  # Mixed (one categorical, one continuous)
            categorical_feature = 0 if is_categorical1 else 1
            plot_mixed_response_with_histogram(ax, response, x_values, (feature1, feature2), 
                                               nomogram_generator, (i, j), categorical_feature, use_odds_ratio)
        else:  # Non-mixed (both categorical or both continuous)
            nomogram_generator._plot_bivariate_response(ax, response, x_values, (i, j))

        plot_index += 1
    
    # Remove any unused subplots
    for i in range(plot_index, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.suptitle(f"Partial Responses for Selected Features ({method.title()})", y=1.01)
    
    if show_fig:
        plt.show()
    
    if return_fig:
        return fig
    else:
        plt.close(fig)
        return None