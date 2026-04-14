from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def extract_shapley_like_values(
    partial_responses: np.ndarray,
    feature_names: Optional[List[str]] = None,
    lasso_results=None,
    threshold: float = 0.1,
) -> dict:
    """
    Extract Shapley-like values from partial responses, optionally weighted by LASSO coefficients.

    Parameters:
    -----------
    partial_responses : np.ndarray
        Array of shape (n_samples, n_features + n_bivariate)
        First n_features columns are univariate responses
    feature_names : List[str], optional
        Names of features for labeling
    lasso_results : LassoResultsManager, optional
        LASSO results manager to get beta coefficients and selected features
    threshold : float, optional
        Threshold for LASSO feature selection (default 0.1)

    Returns:
    --------
    dict with Shapley-like values, feature info, and LASSO weighting details
    """
    n_samples, total_features = partial_responses.shape

    if lasso_results is not None:
        # Use LASSO results to filter and weight features
        if lasso_results.selected_lambda_index is None:
            raise ValueError(
                "LASSO results must have a selected lambda. Call select_lambda() first."
            )

        # Get selected features and their coefficients
        beta = lasso_results.get_selected_beta()
        selected_indices = lasso_results.get_selected_feature_indicies(threshold=threshold)
        selected_feature_names = lasso_results.get_selected_feature_names_clean(
            threshold=threshold
        )

        # Filter partial responses to selected features only
        selected_responses = partial_responses[:, selected_indices]

        # Weight by corresponding beta coefficients
        selected_betas = beta[selected_indices]
        weighted_responses = selected_responses * selected_betas.reshape(1, -1)

        # Use LASSO feature names
        feature_names_final = selected_feature_names

        lasso_info = {
            'lambda_index': lasso_results.selected_lambda_index,
            'lambda_value': lasso_results.lambdas[lasso_results.selected_lambda_index],
            'selected_indices': selected_indices,
            'beta_coefficients': selected_betas,
            'n_selected': len(selected_indices),
            'weighted': True,
        }

        # Final responses are the weighted ones
        final_responses = weighted_responses

    else:
        # Original behavior - use all features without weighting
        n_features = int((-1 + np.sqrt(1 + 8 * total_features)) / 2)
        final_responses = partial_responses[:, :n_features]

        if feature_names is None:
            feature_names_final = [f'Feature_{i}' for i in range(n_features)]
        else:
            feature_names_final = feature_names[:n_features]

        lasso_info = {'weighted': False}

    # Calculate individual and mean Shapley-like values
    individual_shapley = final_responses
    mean_shapley = np.mean(np.abs(final_responses), axis=0)

    return {
        'individual_shapley': individual_shapley,
        'mean_shapley': mean_shapley,
        'feature_names': feature_names_final,
        'responses': final_responses,
        'lasso_info': lasso_info,
    }


def visualize_shapley_values(shapley_dict: dict, figsize=(15, 10)):
    """
    Create comprehensive visualizations of Shapley-like values with LASSO integration.
    """
    individual_shapley = shapley_dict['individual_shapley']
    mean_shapley = shapley_dict['mean_shapley']
    feature_names = shapley_dict['feature_names']
    lasso_info = shapley_dict['lasso_info']

    # Determine if we're using LASSO weighting
    is_weighted = lasso_info.get('weighted', False)

    # Dynamically adjust figure size based on number of features
    n_features = len(feature_names)
    # Scale width for readability with many features
    fig_width = max(15, min(30, n_features * 0.5))
    fig_height = max(10, min(20, n_features * 0.35))

    # Use consistent 2x2 layout
    fig, axes = plt.subplots(2, 2, figsize=(fig_width, fig_height))

    if is_weighted:
        title_suffix = (
            f" (LASSO λ={lasso_info['lambda_value']:.4f}, {lasso_info['n_selected']} features)"
        )
    else:
        title_suffix = ""

    fig.suptitle(f'Shapley-like Values Analysis{title_suffix}', fontsize=16)

    # 1. Box plot of individual contributions
    ax1 = axes[0, 0]

    # Truncate long feature names for better display
    display_names = [name[:20] + '...' if len(name) > 20 else name for name in feature_names]

    # Use matplotlib boxplot directly for better control
    box_data = [individual_shapley[:, i] for i in range(individual_shapley.shape[1])]
    ax1.boxplot(box_data, labels=display_names)

    ax1.set_title('Distribution of Feature Contributions\n(Box Plot)')
    ylabel = 'Weighted Partial Response' if is_weighted else 'Partial Response (Log-odds)'
    ax1.set_ylabel(ylabel)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)

    # Ensure evenly spaced, rotated labels
    ax1.tick_params(axis='x', rotation=45)

    # Improve label positioning
    plt.setp(ax1.get_xticklabels(), ha='right')

    # 2. Mean contributions bar plot
    ax2 = axes[0, 1]
    tick_positions = list(range(len(feature_names)))
    bars = ax2.bar(tick_positions, mean_shapley)
    ax2.set_title('Mean Absolute Feature Contributions')
    ylabel = 'Mean Absolute Weighted Response' if is_weighted else 'Mean Absolute Partial Response'
    ax2.set_ylabel(ylabel)
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(display_names, rotation=45, ha='right')
    # Remove the horizontal line at y=0 since all values are positive
    # ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)

    # Color bars by magnitude (all positive)
    for bar, val in zip(bars, mean_shapley):
        bar.set_color('blue')
        bar.set_alpha(0.7)

    # 3. Violin plot for distribution shape
    ax3 = axes[1, 0]
    positions = list(range(len(feature_names)))
    ax3.violinplot(
        [individual_shapley[:, i] for i in positions], positions=positions, showmeans=True
    )
    ax3.set_xticks(positions)
    ax3.set_xticklabels(display_names, rotation=45, ha='right')
    ax3.set_title('Distribution Shape of Contributions\n(Violin Plot)')
    ax3.set_ylabel(ylabel)
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)

    # 4. Heatmap of correlations between feature contributions
    ax4 = axes[1, 1]

    # Handle single feature case
    if len(feature_names) == 1:
        # Single feature: correlation with itself is 1.0
        corr_matrix = np.array([[1.0]])
        im = ax4.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax4.set_xticks([0])
        ax4.set_yticks([0])
        ax4.set_xticklabels([display_names[0]], rotation=45, ha='right')
        ax4.set_yticklabels([display_names[0]])
        ax4.set_title('Correlation Between\nFeature Contributions\n(Single Feature)')
        plt.colorbar(im, ax=ax4)
    else:
        # Multiple features: compute correlation matrix
        corr_matrix = np.corrcoef(individual_shapley.T)
        im = ax4.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        tick_positions_heatmap = list(range(len(feature_names)))
        ax4.set_xticks(tick_positions_heatmap)
        ax4.set_yticks(tick_positions_heatmap)
        ax4.set_xticklabels(display_names, rotation=45, ha='right')
        ax4.set_yticklabels(display_names)
        ax4.set_title('Correlation Between\nFeature Contributions')
        plt.colorbar(im, ax=ax4)

    plt.tight_layout()
    return fig


def analyze_bivariate_interactions(
    partial_responses: np.ndarray,
    n_features: int,
    feature_names: Optional[List[str]] = None,
    lasso_results=None,
    threshold: float = 0.1,
):
    """
    Analyze bivariate interaction effects with optional LASSO weighting.
    """
    if lasso_results is not None:
        # Get selected bivariate features and their coefficients
        beta = lasso_results.get_selected_beta()
        selected_indices = lasso_results.get_selected_feature_indicies(threshold=threshold)

        # Separate univariate and bivariate selections
        n_univ = len(lasso_results.univariate_feature_names_clean)
        bivariate_indices = [idx for idx in selected_indices if idx >= n_univ]

        if not bivariate_indices:
            print("No bivariate features selected by LASSO")
            return None

        # Extract bivariate responses and weight them
        bivariate_responses = partial_responses[:, bivariate_indices]
        bivariate_betas = beta[bivariate_indices]
        weighted_bivariate = bivariate_responses * bivariate_betas.reshape(1, -1)

        mean_interactions = np.mean(weighted_bivariate, axis=0)

        # Create interaction matrix for selected features only
        interaction_matrix = np.zeros((n_univ, n_univ))

        for i, global_idx in enumerate(bivariate_indices):
            local_bivar_idx = global_idx - n_univ
            pair_indices = lasso_results.bivariate_inputs[local_bivar_idx]
            feat_i, feat_j = pair_indices
            interaction_matrix[feat_i, feat_j] = mean_interactions[i]
            interaction_matrix[feat_j, feat_i] = mean_interactions[i]  # Symmetric

        title = f'Selected Bivariate Interaction Effects (LASSO λ={lasso_results.lambdas[lasso_results.selected_lambda_index]:.4f})'
        plot_feature_names = lasso_results.univariate_feature_names_clean

    else:
        # Original behavior
        # Ensure feature_names is initialized
        plot_feature_names: List[str] = (
            feature_names
            if feature_names is not None
            else [f'Feature_{i}' for i in range(n_features)]
        )

        bivariate_start = n_features
        bivariate_responses = partial_responses[:, bivariate_start:]
        mean_interactions = np.mean(bivariate_responses, axis=0)

        interaction_matrix = np.zeros((n_features, n_features))
        idx = 0
        for i in range(n_features):
            for j in range(i + 1, n_features):
                interaction_matrix[i, j] = mean_interactions[idx]
                interaction_matrix[j, i] = mean_interactions[idx]
                idx += 1

        title = 'Mean Bivariate Interaction Effects'

    # Visualize interaction heatmap
    plt.figure(figsize=(10, 8))
    mask = interaction_matrix == 0  # Mask zero interactions for clarity
    sns.heatmap(
        interaction_matrix,
        xticklabels=plot_feature_names,
        yticklabels=plot_feature_names,
        cmap='RdBu_r',
        center=0,
        annot=True,
        fmt='.3f',
        mask=mask if lasso_results is not None else None,
    )
    plt.title(title)
    plt.tight_layout()

    return interaction_matrix


def print_lasso_summary(shapley_dict: dict):
    """
    Print a summary of LASSO integration results.
    """
    lasso_info = shapley_dict['lasso_info']

    if not lasso_info.get('weighted', False):
        print("No LASSO weighting applied.")
        return

    print(f"\n{'='*50}")
    print("LASSO Integration Summary")
    print(f"{'='*50}")
    print(f"Selected Lambda Index: {lasso_info['lambda_index']}")
    print(f"Lambda Value: {lasso_info['lambda_value']:.6f}")
    print(f"Features Selected: {lasso_info['n_selected']}")
    print("\nSelected Features and Their Coefficients:")
    print(f"{'Feature':<30} {'beta Coefficient':<15} {'Mean |beta x phi|':<15}")
    print("-" * 60)

    feature_names = shapley_dict['feature_names']
    beta_coeffs = lasso_info['beta_coefficients']
    mean_contrib = shapley_dict['mean_shapley']

    # Sort by absolute coefficient value
    sorted_indices = np.argsort(np.abs(mean_contrib))[::-1]

    for idx in sorted_indices:
        name = feature_names[idx][:28]  # Truncate long names
        beta = beta_coeffs[idx]
        contrib = mean_contrib[idx]
        print(f"{name:<30} {beta:>+8.4f}      {contrib:>+8.4f}")


# Example usage with LASSO integration:
"""
# Assuming you have both partial responses and LASSO results
partial_responses = your_partial_response_array  # Shape: (n_samples, n_features + n_bivariate)
lasso_results = your_lasso_results_manager      # LassoResultsManager object

# Make sure LASSO has selected a lambda
lasso_results.select_lambda_max_test_auc()

# Extract weighted Shapley-like values using LASSO selection and coefficients
shapley_results = extract_shapley_like_values(
    partial_responses,
    feature_names=['Age', 'Income', 'Credit_Score'],  # Original feature names
    lasso_results=lasso_results,
    threshold=0.1  # LASSO selection threshold
)

# Print summary of LASSO integration
print_lasso_summary(shapley_results)

# Visualize weighted results (now shows only selected features, weighted by beta)
fig = visualize_shapley_values(shapley_results)
plt.show()

# Analyze selected bivariate interactions
interaction_matrix = analyze_bivariate_interactions(
    partial_responses,
    n_features=len(['Age', 'Income', 'Credit_Score']),
    lasso_results=lasso_results
)

# Without LASSO (original behavior):
shapley_results_orig = extract_shapley_like_values(partial_responses)
fig_orig = visualize_shapley_values(shapley_results_orig)
plt.show()
"""
