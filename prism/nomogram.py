import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import numpy as np
from typing import Any, List, Tuple

from prism.partial_responses import partial_responses_subset

class NomogramGenerator:
    def __init__(self, x: torch.Tensor, x0_median: torch.Tensor, x0_std: torch.Tensor, 
                 betas: torch.Tensor, userLambda: int, n_steps: int = 15, 
                 categorical_threshold: int = 15, sd_scale: float = 2):
        self.x = x
        self.x0_median = x0_median
        self.x0_std = x0_std
        self.betas = betas
        self.userLambda = userLambda
        self.n_steps = n_steps
        self.categorical_threshold = categorical_threshold
        self.sd_scale = sd_scale

    def denormalize(self, x: torch.Tensor, feature: int) -> torch.Tensor:
        return x * (self.x0_std[feature] * self.sd_scale) + self.x0_median[feature]

    def generate_main_nomogram(self, univariate_responses: List[torch.Tensor], x_univariate: List[torch.Tensor], 
                            bivariate_responses: List[torch.Tensor], x_bivariate: List[torch.Tensor], bivariate_inputs: List[Tuple[int, int]]):
        active_features = torch.where(torch.abs(self.betas[:, self.userLambda]) > 0.1)[0]
        univariate_features = active_features[active_features < self.x.shape[1]]
        bivariate_features = active_features[active_features >= self.x.shape[1]] - self.x.shape[1]

        # Count mixed bivariate plots
        mixed_bivariate_count = sum(
            1 for biv_index in bivariate_features
            if (len(torch.unique(self.x[:, bivariate_inputs[biv_index][0]])) < self.categorical_threshold) !=
            (len(torch.unique(self.x[:, bivariate_inputs[biv_index][1]])) < self.categorical_threshold)
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
        for i, feature in enumerate(univariate_features):
            ax = nomo.add_subplot(gs[plot_index])
            self._plot_univariate_response(ax, feature, univariate_responses[feature], x_univariate[feature])
            all_x_values.extend(univariate_responses[feature])
            plot_index += 1

        # Plot mixed bivariate responses
        for i, biv_index in enumerate(bivariate_features):
            feature1, feature2 = bivariate_inputs[biv_index]
            is_categorical1 = len(torch.unique(self.x[:, feature1])) < self.categorical_threshold
            is_categorical2 = len(torch.unique(self.x[:, feature2])) < self.categorical_threshold

            if is_categorical1 != is_categorical2:  # Mixed case
                ax = nomo.add_subplot(gs[plot_index])
                response = bivariate_responses[biv_index]
                x_values = x_bivariate[biv_index]
                self._plot_mixed_response(ax, response, x_values[:, 0], x_values[:, 1], feature1, feature2, is_categorical1)
                all_x_values.extend(response)
                plot_index += 1

        # Set common x-axis limits and adjust x-axis presentation
        if all_x_values:
            min_x = min(all_x_values)
            max_x = max(all_x_values)
            x_padding_ratio = 0.05
            for ax in nomo.axes:
                ax.set_xlim(min_x-(x_padding_ratio*(max_x-min_x)),
                            max_x+(x_padding_ratio*(max_x-min_x)))

        # Adjust the x-axis presentation
        for i, ax in enumerate(nomo.axes):
            ax.axvline(0, color="black", alpha=0.5)
            if i == 0:
                ax.xaxis.tick_top()
                ax.xaxis.set_label_position('top')
                ax.set_xlabel("Log Odds Ratio")
            else:
                ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
                ax.set_xlabel('')
                
        # Adjust the title positioning
        title_height = 1  # Height reserved for title in inches
        fig_top = 1 - (title_height / nomo.get_figheight())

        # Adjust layout
        nomo.tight_layout()
        nomo.subplots_adjust(top=fig_top, hspace=0)

        return nomo

    def _plot_univariate_response(self, ax, feature, response, x_values):
        is_categorical = len(torch.unique(self.x[:, feature])) < self.categorical_threshold
        if is_categorical:
            self._plot_categorical_response(ax, feature, response, x_values)
        else:
            self._plot_continuous_response(ax, feature, response, x_values)

        ax.axvline(0, color="black", alpha=0.5)
        ax.set_ylabel(f"Feature {feature}", rotation=90, loc="center", labelpad=5)
        ax.yaxis.tick_right()

    def _plot_categorical_response(self, ax, feature, response, x_values):
        denormalized_values = self.denormalize(x_values, feature)
        y_value = 0.5
        line, = ax.plot(response.cpu(), torch.full_like(response, y_value).cpu())
        line_color = line.get_color()

        ax.scatter(response.cpu(), torch.full_like(response, y_value).cpu(), marker="|", color=line_color)
        for i, value in enumerate(denormalized_values):
            if response[i] != 0:
                ax.annotate(f"{value.item():.2f}", (response[i].item(), y_value), 
                            xytext=(response[i].item(), y_value + 0.003),
                            ha='center', va='bottom')
        ax.set_yticks([])

    def _plot_continuous_response(self, ax, feature, response, x_values):
        denormalized_range = self.denormalize(x_values, feature)
        line, = ax.plot(response.cpu().numpy(), denormalized_range.cpu().numpy())
        line_color = line.get_color()

        # Set y-ticks
        y_ticks = torch.linspace(denormalized_range.min(), denormalized_range.max(), steps=5)
        ax.set_yticks(y_ticks.cpu().numpy())

        # Add scatter points and annotations
        for y_tick in y_ticks:
            if denormalized_range.min() <= y_tick <= denormalized_range.max():
                x_value = np.interp(y_tick.item(), denormalized_range.cpu().numpy(), response.cpu().numpy())
                ax.scatter(x_value, y_tick.item(), marker="o", color=line_color)
                if torch.isclose(y_tick, torch.round(y_tick), atol=1e-8):
                    annotation = f"{int(y_tick.item())}"
                else:
                    annotation = f"{y_tick.item():.1f}".rstrip('0').rstrip('.')
                ax.annotate(annotation, (x_value, y_tick.item()), 
                            xytext=(x_value - 0.1, y_tick.item()),
                            ha='right', va='center', color='black')

    def _plot_mixed_response(self, ax, response, x1, x2, feature1, feature2, is_categorical1):
        cat_feature, cont_feature = (feature1, feature2) if is_categorical1 else (feature2, feature1)
        cat_values = torch.unique(x1 if is_categorical1 else x2)
        cont_values = x2 if is_categorical1 else x1
        
        # Determine y-ticks (continuous feature values)
        y_ticks = torch.linspace(cont_values.min(), cont_values.max(), steps=5)
        
        # Plot lines for each categorical value
        for i, cat_val in enumerate(cat_values):
            if is_categorical1:
                x = response[x1 == cat_val]
                y = cont_values[x1 == cat_val]
            else:
                x = response[x2 == cat_val]
                y = cont_values[x2 == cat_val]
            line, = ax.plot(x.cpu(), y.cpu(), label=f"{self.denormalize(cat_val, cat_feature).item():.2f}")
            line_color = line.get_color()

        # Set y-ticks and labels
        ax.set_yticks(y_ticks.cpu().numpy())
        y_labels = [f"{self.denormalize(y, cont_feature).item():.2f}" for y in y_ticks]
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
                    if line == reference_line:
                        ax.annotate(y_label, (x_value, y_tick), 
                                    xytext=(x_value + x_offset, y_tick + y_offset),
                                    ha='right', va='center', color='black')

        ax.axvline(0, color="black", alpha=0.5)
        ax.set_xlabel("Log Odds Ratio")
        ax.set_ylabel(f"Feature {cont_feature}")
        
        # Move y-axis label to the left and keep ticks on the right
        ax.yaxis.set_label_position("left")
        ax.yaxis.tick_right()

        # Add legend
        ax.legend(title=f"Feature {cat_feature}", fontsize=8, title_fontsize=10,
                  loc='lower left', bbox_to_anchor=(0.05, 0.05), borderaxespad=0.)

    def generate_non_mixed_bivariate_nomogram(self, bivariate_responses: List[torch.Tensor], x_bivariate: List[torch.Tensor], bivariate_inputs: List[Tuple[int, int]]):
        non_mixed_bivariate = []
        for i, (f1, f2) in enumerate(bivariate_inputs):
            is_categorical1 = len(torch.unique(self.x[:, f1])) < self.categorical_threshold
            is_categorical2 = len(torch.unique(self.x[:, f2])) < self.categorical_threshold
            if (is_categorical1 == is_categorical2) and torch.abs(self.betas[self.x.shape[1] + i, self.userLambda]) > 0.1:
                non_mixed_bivariate.append(i)

        if not non_mixed_bivariate:
            return None

        fig, axes = plt.subplots(len(non_mixed_bivariate), 1, figsize=(10, 5 * len(non_mixed_bivariate)))
        if len(non_mixed_bivariate) == 1:
            axes = [axes]

        for i, biv_index in enumerate(non_mixed_bivariate):
            feature1, feature2 = bivariate_inputs[biv_index]
            response = bivariate_responses[biv_index]
            x_values = x_bivariate[biv_index]
            self._plot_bivariate_response(axes[i], response, x_values, (feature1, feature2))

        plt.tight_layout()
        return fig

    def _plot_bivariate_response(self, ax, response, x_values, feature_pair):
        feature1, feature2 = feature_pair
        x1, x2 = x_values[:, 0], x_values[:, 1]
        
        is_categorical1 = len(torch.unique(self.x[:, feature1])) < self.categorical_threshold
        is_categorical2 = len(torch.unique(self.x[:, feature2])) < self.categorical_threshold

        if is_categorical1 and is_categorical2:
            self._plot_categorical_categorical(ax, response, x1, x2, feature1, feature2)
        elif not is_categorical1 and not is_categorical2:
            self._plot_continuous_continuous(ax, response, x1, x2, feature1, feature2)
        else:
            raise ValueError("Mixed categorical and continuous features should not be plotted here. Use `generate_main_nomogram` instead.")

    def _plot_categorical_categorical(self, ax, response, x1, x2, feature1, feature2):
        response_matrix = response.view(len(torch.unique(x1)), len(torch.unique(x2)))
        im = ax.imshow(response_matrix.t(), cmap='viridis', aspect='auto', origin='lower')
        ax.set_xticks(range(len(torch.unique(x1))))
        ax.set_yticks(range(len(torch.unique(x2))))
        ax.set_xticklabels([f"{val:.2f}" for val in self.denormalize(torch.unique(x1), feature1)])
        ax.set_yticklabels([f"{val:.2f}" for val in self.denormalize(torch.unique(x2), feature2)])
        ax.set_xlabel(f"Feature {feature1}")
        ax.set_ylabel(f"Feature {feature2}")
        plt.colorbar(im, ax=ax, label='Log Odds Ratio')

        # Add text annotations
        for i in range(len(torch.unique(x1))):
            for j in range(len(torch.unique(x2))):
                text = ax.text(i, j, f"{response_matrix[i, j].item():.2f}",
                               ha="center", va="center", color="white",
                               bbox=dict(boxstyle="round", facecolor="black", edgecolor="none", alpha=0.5))

    def _plot_continuous_continuous(self, ax, response, x1, x2, feature1, feature2):
        x1_denorm = self.denormalize(x1, feature1).view(self.n_steps, self.n_steps)
        x2_denorm = self.denormalize(x2, feature2).view(self.n_steps, self.n_steps)
        response_matrix = response.view(self.n_steps, self.n_steps)
        
        X = x1_denorm.numpy()
        Y = x2_denorm.numpy()
        Z = response_matrix.numpy()
        
        contour_heatmap = ax.contourf(X, Y, Z, cmap='viridis', levels=20)
        contour_lines = ax.contour(X, Y, Z, colors='white', alpha=0.5, levels=10)
        ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2f')
        
        ax.set_xlabel(f"Feature {feature1}")
        ax.set_ylabel(f"Feature {feature2}")
        plt.colorbar(contour_heatmap, ax=ax, label='Log Odds Ratio')

def nomogram(betas: torch.Tensor, userLambda: int, x: torch.Tensor, x0_median: torch.Tensor, 
             x0_std: torch.Tensor, model: Any, n_steps: int = 15, sd_scale: float = 2, 
             method: str = "dirac", device: str = "cpu", categorical_threshold: int = 15,
             subtract_univariate: bool = False) -> None:
    
    # Calculate subset of partial responses
    univariate_responses, bivariate_responses, bivariate_inputs, x_univariate, x_bivariate = partial_responses_subset(
        x, model, method=method, device=device, n_steps=n_steps, categorical_threshold=categorical_threshold,
        subtract_univariate=subtract_univariate
    )

    # Generate plots
    nomogram_generator = NomogramGenerator(x, x0_median, x0_std, betas, userLambda, n_steps, categorical_threshold, sd_scale)
    
    fig_main = nomogram_generator.generate_main_nomogram(univariate_responses, x_univariate, 
                                                         bivariate_responses, x_bivariate, bivariate_inputs)
    if fig_main:
        plt.figure(fig_main.number)
        plt.suptitle("Nomogram of univariate and mixed bivariate partial responses", y=1.02)
        plt.tight_layout()
        plt.show()

    # Generate separate plot for non-mixed bivariate responses
    fig_non_mixed = nomogram_generator.generate_non_mixed_bivariate_nomogram(bivariate_responses, x_bivariate, bivariate_inputs)
    if fig_non_mixed:
        plt.figure(fig_non_mixed.number)
        plt.suptitle("Nomogram of non-mixed bivariate partial responses", y=1.02)
        plt.tight_layout()
        plt.show()

    # Return responses for potential future use
    return univariate_responses, bivariate_responses, x_univariate, x_bivariate