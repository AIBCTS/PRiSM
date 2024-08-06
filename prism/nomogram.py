import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Any, List, Tuple

from prism.partial_responses import partial_responses_subset

def denormalize(x: torch.Tensor, x0_median: torch.Tensor, x0_std: torch.Tensor, sd_scale: float = 2) -> torch.Tensor:
    return x * (x0_std * sd_scale) + x0_median

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

    def generate_univariate_plots(self, univariate_responses: List[torch.Tensor], x_univariate: List[torch.Tensor]):
        active_features = torch.where(torch.abs(self.betas[:, self.userLambda]) > 0.1)[0]
        active_features = active_features[active_features < self.x.shape[1]]

        fig, axes = plt.subplots(len(active_features), 1, figsize=(10, 5 * len(active_features)))
        if len(active_features) == 1:
            axes = [axes]

        for i, feature in enumerate(active_features):
            self._plot_univariate_response(axes[i], feature, univariate_responses[feature], x_univariate[feature])

        plt.tight_layout()
        return fig

    def _plot_univariate_response(self, ax, feature, response, x_values):
        is_categorical = len(torch.unique(self.x[:, feature])) < self.categorical_threshold
        if is_categorical:
            self._plot_categorical_response(ax, feature, response, x_values)
        else:
            self._plot_continuous_response(ax, feature, response, x_values)

    def _plot_categorical_response(self, ax, feature, response, x_values):
        denormalized_values = denormalize(x_values, self.x0_median[feature], self.x0_std[feature], self.sd_scale)
        ax.plot(response, np.full(len(response), 0.5))
        ax.scatter(response, np.full(len(response), 0.5), marker="|")
        for i, value in enumerate(denormalized_values):
            ax.annotate(f"{value.item():.2f}", (response[i], 0.5), xytext=(response[i], 0.505))
        ax.set_yticks([])
        ax.set_ylabel(f"Feature {feature}", rotation=0, ha='right')
        ax.axvline(0, color="black", alpha=0.5)

    def _plot_continuous_response(self, ax, feature, response, x_values):
        denormalized_range = denormalize(x_values, self.x0_median[feature], self.x0_std[feature], self.sd_scale)
        ax.plot(response, denormalized_range)
        ax.scatter(response, denormalized_range, marker="|")
        ax.set_ylabel(f"Feature {feature}", rotation=0, ha='right')
        ax.yaxis.set_label_position("right")
        ax.axvline(0, color="black", alpha=0.5)

    def generate_bivariate_plots(self, bivariate_responses: List[torch.Tensor], x_bivariate: List[torch.Tensor], bivariate_inputs: List[Tuple[int, int]]):
        active_bivariate = [i for i, (f1, f2) in enumerate(bivariate_inputs)
                            if torch.abs(self.betas[self.x.shape[1] + i, self.userLambda]) > 0.1]

        if not active_bivariate:
            return None

        fig, axes = plt.subplots(len(active_bivariate), 1, figsize=(10, 5 * len(active_bivariate)))
        if len(active_bivariate) == 1:
            axes = [axes]

        for i, biv_index in enumerate(active_bivariate):
            self._plot_bivariate_response(axes[i], bivariate_responses[biv_index], x_bivariate[biv_index], bivariate_inputs[biv_index])

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
            self._plot_mixed_response(ax, response, x1, x2, feature1, feature2, is_categorical1)

    def _plot_categorical_categorical(self, ax, response, x1, x2, feature1, feature2):
        response_matrix = response.view(len(torch.unique(x1)), len(torch.unique(x2)))
        im = ax.imshow(response_matrix.t(), cmap='viridis', aspect='auto', origin='lower')
        ax.set_xticks(range(len(torch.unique(x1))))
        ax.set_yticks(range(len(torch.unique(x2))))
        ax.set_xticklabels([f"{val:.2f}" for val in denormalize(torch.unique(x1), self.x0_median[feature1], self.x0_std[feature1], self.sd_scale)])
        ax.set_yticklabels([f"{val:.2f}" for val in denormalize(torch.unique(x2), self.x0_median[feature2], self.x0_std[feature2], self.sd_scale)])
        ax.set_xlabel(f"Feature {feature1}")
        ax.set_ylabel(f"Feature {feature2}")
        plt.colorbar(im, ax=ax, label='Log Odds Ratio')

    def _plot_continuous_continuous(self, ax, response, x1, x2, feature1, feature2):
        x1_denorm = denormalize(x1, self.x0_median[feature1], self.x0_std[feature1], self.sd_scale)
        x2_denorm = denormalize(x2, self.x0_median[feature2], self.x0_std[feature2], self.sd_scale)
        response_matrix = response.view(self.n_steps, self.n_steps)
        
        contour = ax.contourf(x1_denorm.view(self.n_steps, self.n_steps), 
                              x2_denorm.view(self.n_steps, self.n_steps), 
                              response_matrix, cmap='viridis', levels=20)
        ax.set_xlabel(f"Feature {feature1}")
        ax.set_ylabel(f"Feature {feature2}")
        plt.colorbar(contour, ax=ax, label='Log Odds Ratio')

    def _plot_mixed_response(self, ax, response, x1, x2, feature1, feature2, is_categorical1):
        cat_feature, cont_feature = (feature1, feature2) if is_categorical1 else (feature2, feature1)
        cat_values = torch.unique(x1 if is_categorical1 else x2)
        cont_values = x2 if is_categorical1 else x1
        
        for i, cat_val in enumerate(cat_values):
            if is_categorical1:
                y = response[x1 == cat_val]
                x = denormalize(cont_values[x1 == cat_val], self.x0_median[cont_feature], self.x0_std[cont_feature], self.sd_scale)
            else:
                y = response[x2 == cat_val]
                x = denormalize(cont_values[x2 == cat_val], self.x0_median[cont_feature], self.x0_std[cont_feature], self.sd_scale)
            ax.plot(x, y, label=f"{cat_feature}={cat_val.item():.2f}")
        
        ax.set_xlabel(f"Feature {cont_feature}")
        ax.set_ylabel("Log Odds Ratio")
        ax.legend(title=f"Feature {cat_feature}")

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
    
    fig_univariate = nomogram_generator.generate_univariate_plots(univariate_responses, x_univariate)
    if fig_univariate:
        plt.figure(fig_univariate.number)
        plt.show()

    fig_bivariate = nomogram_generator.generate_bivariate_plots(bivariate_responses, x_bivariate, bivariate_inputs)
    if fig_bivariate:
        plt.figure(fig_bivariate.number)
        plt.show()

    # Return responses for potential future use
    return univariate_responses, bivariate_responses, x_univariate, x_bivariate