import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import numpy as np
from typing import Any, List, Tuple, Optional
from prism.lasso_results import LassoResultsManager
from prism.partial_responses import partial_responses_subset

class NomogramGenerator:
    def __init__(self, lasso_results: LassoResultsManager, x: np.ndarray, x0_median: np.ndarray, x0_std: np.ndarray, n_steps: int = 15, categorical_threshold: int = 15, sd_scale: float = 2, use_odds_ratio: bool = False):
        self.lasso_results = lasso_results
        self.x = x
        self.x0_median = x0_median
        self.x0_std = x0_std
        self.beta = lasso_results.get_selected_beta()
        self.n_steps = n_steps
        self.categorical_threshold = categorical_threshold
        self.sd_scale = sd_scale
        self.all_feature_names = lasso_results.all_feature_names
        self.use_odds_ratio = use_odds_ratio

    def denormalize(self, x: np.ndarray, feature: int) -> np.ndarray:
        return x * (self.x0_std[feature] * self.sd_scale) + self.x0_median[feature]

    def _get_log_tick_locations(self, min_x, max_x):
        # Calculate the orders of magnitude for min and max
        min_order = np.floor(np.log10(min_x))
        max_order = np.ceil(np.log10(max_x))
        
        # Generate tick locations
        tick_locations = [10**i for i in range(int(min_order), int(max_order)+1)]
        
        # Add intermediate ticks if range is small
        if max_order - min_order <= 2:
            tick_locations += [x * 5 for x in tick_locations[:-1]]
        
        # Filter tick locations to be within the data range
        tick_locations = [x for x in tick_locations if min_x <= x <= max_x]
        
        return sorted(tick_locations)

    def generate_main_nomogram(self, univariate_responses: List[np.ndarray], x_univariate: List[np.ndarray], 
                            bivariate_responses: List[np.ndarray], x_bivariate: List[np.ndarray]):
        univariate_features = self.lasso_results.get_selected_univariate_indices()
        bivariate_indices = self.lasso_results.get_selected_bivariate_indices()
        bivariate_index_pairs = self.lasso_results.get_selected_bivariate_index_pairs()

        # Count mixed bivariate plots
        mixed_bivariate_count = sum(
            1 for feature1, feature2 in bivariate_index_pairs
            if self.lasso_results.is_mixed_bivariate(feature1, feature2, self.x, self.categorical_threshold)
        )

        num_plots = len(univariate_features) + mixed_bivariate_count
        
        # Create the main figure
        subfig_height = 1.8
        fig_height = subfig_height * num_plots
        nomo = plt.figure(figsize=(8, fig_height))
        gs = gridspec.GridSpec(num_plots, 1, height_ratios=[subfig_height] * num_plots, hspace=0)

        all_x_values = []
        plot_index = 0

        # Plot univariate responses
        for feature in univariate_features:
            ax = nomo.add_subplot(gs[plot_index])
            self._plot_univariate_response(ax, feature, univariate_responses[feature], x_univariate[feature])
            all_x_values.extend(univariate_responses[feature])
            plot_index += 1

        # Plot mixed bivariate responses
        for biv_index, (feature1, feature2) in zip(bivariate_indices, bivariate_index_pairs):
            if self.lasso_results.is_mixed_bivariate(feature1, feature2, self.x, self.categorical_threshold):
                ax = nomo.add_subplot(gs[plot_index])
                response = bivariate_responses[biv_index]
                x_values = x_bivariate[biv_index]
                is_categorical1 = len(np.unique(self.x[:, feature1])) < self.categorical_threshold
                self._plot_mixed_response(ax, response, x_values[:, 0], x_values[:, 1], feature1, feature2, is_categorical1)
                all_x_values.extend(response)
                plot_index += 1

        # Set common x-axis limits and adjust x-axis presentation
        if all_x_values:
            min_x = min(all_x_values)
            max_x = max(all_x_values)
            if self.use_odds_ratio:
                x_padding_ratio = 0.1  # pad for log scale
                min_x = min_x / (1 + x_padding_ratio)
                max_x = max_x * (1 + x_padding_ratio)
                
                # Get appropriate tick locations
                tick_locations = self._get_log_tick_locations(min_x, max_x)
                
                for ax in nomo.axes:
                    ax.set_xlim(min_x, max_x)
                    ax.set_xscale('log')
                    ax.set_xticks(tick_locations)
                    ax.set_xticklabels([f"{x:.2g}" for x in tick_locations])
            else:
                x_padding_ratio = 0.05
                min_x = min_x - (x_padding_ratio * (max_x - min_x))
                max_x = max_x + (x_padding_ratio * (max_x - min_x))
                
                for ax in nomo.axes:
                    ax.set_xlim(min_x, max_x)
                    
        # Adjust the x-axis presentation
        for i, ax in enumerate(nomo.axes):
            if self.use_odds_ratio:
                ax.axvline(1, color="black", alpha=0.5)
            else:
                ax.axvline(0, color="black", alpha=0.5)
            if i == 0:
                ax.xaxis.tick_top()
                ax.xaxis.set_label_position('top')
                ax.set_xlabel("Odds Ratio" if self.use_odds_ratio else "Log Odds Ratio")
            elif i == num_plots-1:
                ax.xaxis.tick_bottom()
                ax.set_xlabel("Odds Ratio" if self.use_odds_ratio else "Log Odds Ratio")
            else:
                ax.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=False)
                ax.set_xlabel('')
                
        # Adjust the title positioning
        title_height = 1  # Height reserved for title in inches
        fig_top = 1 - (title_height / nomo.get_figheight())

        # Adjust layout
        nomo.tight_layout()
        nomo.subplots_adjust(top=fig_top, hspace=0)

        return nomo

    def _plot_univariate_response(self, ax, feature, response, x_values):
        is_categorical = len(np.unique(self.x[:, feature])) < self.categorical_threshold
        if is_categorical:
            self._plot_categorical_response(ax, feature, response, x_values)
        else:
            self._plot_continuous_response(ax, feature, response, x_values)

        if self.use_odds_ratio:
            ax.axvline(1, color="black", alpha=0.5)
        else:
            ax.axvline(0, color="black", alpha=0.5)
        ax.set_ylabel(self.all_feature_names[feature], rotation=90, loc="center", labelpad=5)
        ax.yaxis.tick_right()

    def _plot_categorical_response(self, ax, feature, response, x_values):
        denormalized_values = self.denormalize(x_values, feature)
        y_value = 0.5
        line, = ax.plot(response, np.full_like(response, y_value))
        line_color = line.get_color()

        ax.scatter(response, np.full_like(response, y_value), marker="|", color=line_color)
        for i, value in enumerate(denormalized_values):
            if (self.use_odds_ratio and response[i] != 1) or (not self.use_odds_ratio and response[i] != 0):
                ax.annotate(f"{value:.2g}", (response[i], y_value), 
                            xytext=(response[i], y_value + 0.003),
                            ha='center', va='bottom')
        ax.set_yticks([])

    def _plot_continuous_response(self, ax, feature, response, x_values):
        denormalized_range = self.denormalize(x_values, feature)
        line, = ax.plot(response, denormalized_range)
        line_color = line.get_color()

        # Set y-ticks
        y_ticks = np.linspace(denormalized_range.min(), denormalized_range.max(), num=5)
        ax.set_yticks(y_ticks)

        # Add scatter points and annotations
        for y_tick in y_ticks:
            if denormalized_range.min() <= y_tick <= denormalized_range.max():
                x_value = np.interp(y_tick, denormalized_range, response)
                ax.scatter(x_value, y_tick, marker="o", color=line_color)
                if np.isclose(y_tick, np.round(y_tick), atol=1e-8):
                    annotation = f"{int(y_tick)}"
                else:
                    annotation = f"{y_tick:.1f}".rstrip('0').rstrip('.')
                ax.annotate(annotation, (x_value, y_tick), 
                            xytext=(x_value - 0.1, y_tick),
                            ha='right', va='center', color='black')

    def _plot_mixed_response(self, ax, response, x1, x2, feature1, feature2, is_categorical1):
        cat_feature, cont_feature = (feature1, feature2) if is_categorical1 else (feature2, feature1)
        cat_values = np.unique(x1 if is_categorical1 else x2)
        cont_values = x2 if is_categorical1 else x1
        
        # Determine y-ticks (continuous feature values)
        y_ticks = np.linspace(cont_values.min(), cont_values.max(), num=5)
        
        # Plot lines for each categorical value
        for i, cat_val in enumerate(cat_values):
            if is_categorical1:
                x = response[x1 == cat_val]
                y = cont_values[x1 == cat_val]
            else:
                x = response[x2 == cat_val]
                y = cont_values[x2 == cat_val]
            line, = ax.plot(x, y, label=f"{self.denormalize(cat_val, cat_feature):.4g}")
            line_color = line.get_color()

        # Set y-ticks and labels
        ax.set_yticks(y_ticks)
        y_labels = [f"{self.denormalize(y, cont_feature):.4g}" for y in y_ticks]
        ax.set_yticklabels(y_labels)

        # Add scatter points and annotations
        reference_line = ax.get_lines()[0]  # Use the first line as reference for annotations
        x_data, y_data = reference_line.get_data()
        x_offset = -0.1
        y_offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01

        for line in ax.get_lines():
            line_x_data, line_y_data = line.get_data()
            line_color = line.get_color()
            for y_tick, y_label in zip(y_ticks, y_labels):
                if line_y_data.min() <= y_tick <= line_y_data.max():
                    x_value = np.interp(y_tick, line_y_data, line_x_data)
                    ax.scatter(x_value, y_tick, marker="o", color=line_color)

                    # Add annotation only for the reference line
                    if line is reference_line:
                        ax.annotate(y_label, (x_value, y_tick), 
                                    xytext=(x_value + x_offset, y_tick + y_offset),
                                    ha='right', va='center', color='black')

        ax.axvline(0, color="black", alpha=0.5)
        ax.set_xlabel("Log Odds Ratio")
        ax.set_ylabel(self.all_feature_names[cont_feature])
        
        # Move y-axis label to the left and keep ticks on the right
        ax.yaxis.set_label_position("left")
        ax.yaxis.tick_right()

        # Add legend
        ax.legend(title=self.all_feature_names[cat_feature], fontsize=8, title_fontsize=10,
                loc='lower left', bbox_to_anchor=(0.05, 0.05), borderaxespad=0.)

    def generate_non_mixed_bivariate_nomogram(self, bivariate_responses: List[np.ndarray], 
                                            x_bivariate: List[np.ndarray]):
        bivariate_indices = self.lasso_results.get_selected_bivariate_indices()
        bivariate_index_pairs = self.lasso_results.get_selected_bivariate_index_pairs()

        non_mixed_bivariate = []
        for biv_index, (f1, f2) in zip(bivariate_indices, bivariate_index_pairs):
            if not self.lasso_results.is_mixed_bivariate(f1, f2, self.x, self.categorical_threshold):
                non_mixed_bivariate.append(biv_index)

        if not non_mixed_bivariate:
            return None

        fig, axes = plt.subplots(len(non_mixed_bivariate), 1, figsize=(6, 4 * len(non_mixed_bivariate)))
        if len(non_mixed_bivariate) == 1:
            axes = [axes]

        for i, biv_index in enumerate(non_mixed_bivariate):
            feature1, feature2 = self.lasso_results.bivariate_inputs[biv_index]
            response = bivariate_responses[biv_index]
            x_values = x_bivariate[biv_index]
            
            self._plot_bivariate_response(axes[i], response, x_values, (feature1, feature2))

        plt.tight_layout()
        return fig

    def _plot_bivariate_response(self, ax, response, x_values, feature_pair):
        feature1, feature2 = feature_pair
        x1, x2 = x_values[:, 0], x_values[:, 1]
        
        is_categorical1 = len(np.unique(self.x[:, feature1])) < self.categorical_threshold
        is_categorical2 = len(np.unique(self.x[:, feature2])) < self.categorical_threshold

        if is_categorical1 and is_categorical2:
            self._plot_categorical_categorical(ax, response, x1, x2, feature1, feature2)
        elif not is_categorical1 and not is_categorical2:
            self._plot_continuous_continuous(ax, response, x1, x2, feature1, feature2)
        else:
            raise ValueError("Mixed categorical and continuous features should not be plotted here. Use `generate_main_nomogram` instead.")

    def _plot_categorical_categorical(self, ax, response, x1, x2, feature1, feature2):
        response_matrix = response.reshape(len(np.unique(x1)), len(np.unique(x2)))
        im = ax.imshow(response_matrix.T, cmap='viridis', aspect='auto', origin='lower')
        ax.set_xticks(range(len(np.unique(x1))))
        ax.set_yticks(range(len(np.unique(x2))))
        ax.set_xticklabels([f"{val:.2g}" for val in self.denormalize(np.unique(x1), feature1)])
        ax.set_yticklabels([f"{val:.2g}" for val in self.denormalize(np.unique(x2), feature2)])
        ax.set_xlabel(self.all_feature_names[feature1])
        ax.set_ylabel(self.all_feature_names[feature2])
        plt.colorbar(im, ax=ax, label='Odds Ratio' if self.use_odds_ratio else 'Log Odds Ratio')

        # Add text annotations
        for i in range(len(np.unique(x1))):
            for j in range(len(np.unique(x2))):
                text = ax.text(i, j, f"{response_matrix[i, j]:.2f}",
                            ha="center", va="center", color="white",
                            bbox=dict(boxstyle="round", facecolor="black", edgecolor="none", alpha=0.5))

    def _plot_continuous_continuous(self, ax, response, x1, x2, feature1, feature2):
        # Create 2D grids for plotting
        X = self.denormalize(x1, feature1).reshape(self.n_steps, self.n_steps)
        Y = self.denormalize(x2, feature2).reshape(self.n_steps, self.n_steps)
        Z = response.reshape(self.n_steps, self.n_steps)
        
        contour_heatmap = ax.contourf(X, Y, Z, cmap='viridis', levels=20)
        contour_lines = ax.contour(X, Y, Z, colors='white', alpha=0.5, levels=10)
        ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2f')
        
        ax.set_xlabel(self.all_feature_names[feature1])
        ax.set_ylabel(self.all_feature_names[feature2])
        plt.colorbar(contour_heatmap, ax=ax, label='Odds Ratio' if self.use_odds_ratio else 'Log Odds Ratio')

def nomogram(lasso_results: LassoResultsManager, x: torch.Tensor, 
             x0_median: np.ndarray, x0_std: np.ndarray, model: Any, 
             n_steps: int = 15, sd_scale: float = 2, 
             method: str = "dirac", device: str = "cpu", 
             categorical_threshold: int = 15,
             subtract_univariate: bool = True, show_fig: bool = True,
             return_fig: bool = False, use_odds_ratio: bool = False) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    

    # Calculate subset of partial responses
    univariate_responses, bivariate_responses, x_univariate, x_bivariate = partial_responses_subset(
        x, model, method=method, device=device, n_steps=n_steps, 
        categorical_threshold=categorical_threshold,
        subtract_univariate=subtract_univariate
    )

    # Convert to odds ratio if necessary
    if use_odds_ratio:
        univariate_responses = [np.exp(resp) for resp in univariate_responses]
        bivariate_responses = [np.exp(resp) for resp in bivariate_responses]

    # Ensure inputs are numpy arrays
    x = x.cpu().numpy()
    x0_median = np.asarray(x0_median)
    x0_std = np.asarray(x0_std)
    
    # Generate plots
    nomogram_generator = NomogramGenerator(lasso_results, x, x0_median, x0_std, n_steps, categorical_threshold, sd_scale, use_odds_ratio)
    
    fig_main = nomogram_generator.generate_main_nomogram(univariate_responses, x_univariate, bivariate_responses, x_bivariate)
    if fig_main:
        plt.figure(fig_main.number)
        plt.suptitle(f"Nomogram of univariate and mixed bivariate partial responses ({method.title()})", y=1.0)
        plt.tight_layout()
        if show_fig:
            plt.show()

    # Generate separate plot for non-mixed bivariate responses
    fig_non_mixed = nomogram_generator.generate_non_mixed_bivariate_nomogram(bivariate_responses, x_bivariate)
    if fig_non_mixed:
        plt.figure(fig_non_mixed.number)
        plt.suptitle(f"Nomogram of non-mixed bivariate partial responses ({method.title()})", y=1.0)
        plt.tight_layout()
        if show_fig:
            plt.show()

    # Return responses and optionally return figures
    if return_fig:
        return univariate_responses, bivariate_responses, x_univariate, x_bivariate, fig_main, fig_non_mixed
    else:
        return univariate_responses, bivariate_responses, x_univariate, x_bivariate

def align_and_plot(fig1, fig2, title1, title2, max_width, max_height):
    # Get the image data from the figures
    img1 = np.array(fig1.canvas.renderer.buffer_rgba())
    img2 = np.array(fig2.canvas.renderer.buffer_rgba())
    
    # Get the heights and widths of both images
    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape
    
    # Calculate the aspect ratios
    aspect1 = h1 / w1
    aspect2 = h2 / w2
    
    # Calculate the optimal figure size
    width = min(max_width, (w1 + w2) / 100)
    height = min(max_height, max(h1 / 100, h2 / 100))
    
    # Create a new figure
    fig = plt.figure(figsize=(width, height))
    
    # Calculate the width of each subplot (leaving some space for padding)
    subplot_width = 0.48
    
    # Calculate the heights based on the aspect ratios and subplot width
    height1 = subplot_width * width * aspect1 / height
    height2 = subplot_width * width * aspect2 / height
    
    # Add the subplots, aligning them at the top
    ax1 = fig.add_axes([0.01, 1 - height1, subplot_width, height1])
    ax2 = fig.add_axes([0.51, 1 - height2, subplot_width, height2])
    
    # Plot the images
    ax1.imshow(img1)
    ax2.imshow(img2)
    
    # Remove axes
    ax1.axis('off')
    ax2.axis('off')
    
    # Set titles
    ax1.set_title(title1)
    ax2.set_title(title2)
    
    # Show the plot
    plt.show()

def display_nomograms_side_by_side(fig_main1: Optional[plt.Figure], fig_non_mixed1: Optional[plt.Figure], 
                                   fig_main2: Optional[plt.Figure], fig_non_mixed2: Optional[plt.Figure],
                                   titles: List[str] = ["MLP", "PRN"],
                                   max_width: float = 20,
                                   max_height: float = 25):
    """
    Display nomograms side by side for comparison, with equal width, top-aligned, and preserved aspect ratios.
    
    Parameters:
    fig_main1: Main nomogram figure for model 1
    fig_non_mixed1: Non-mixed bivariate nomogram figure for model 1
    fig_main2: Main nomogram figure for model 2
    fig_non_mixed2: Non-mixed bivariate nomogram figure for model 2
    titles: List of titles for the two models being compared
    max_width: Maximum width of the figure in inches
    max_height: Maximum height of the figure in inches
    """
    
    # Display main nomograms side by side
    if fig_main1 and fig_main2:
        align_and_plot(fig_main1, fig_main2, 
                       f"{titles[0]}", 
                       f"{titles[1]}",
                       max_width, max_height)
    
    # Display non-mixed bivariate nomograms side by side
    if fig_non_mixed1 and fig_non_mixed2:
        align_and_plot(fig_non_mixed1, fig_non_mixed2, 
                       f"{titles[0]}", 
                       f"{titles[1]}",
                       max_width, max_height)