import torch
import numpy as np
from typing import Any, Tuple, List, Optional

class PartialResponseCalculator:
    def __init__(self, model: Any, method: str = 'dirac', device: str = 'cpu', input_dim: int = 1, x_train: Optional[torch.Tensor] = None):
        self.model = model
        self.method = method
        self.device = device
        self.input_dim = input_dim
        self.logit_y0 = None
        
        if method == 'lebesgue':
            if x_train is None:
                raise ValueError("x_train must be provided for the Lebesgue method")
            self.calculate_baseline(x_train)
        
        # Check if the model is compatible
        self._check_model_compatibility(x_train)

    def _check_model_compatibility(self, x_train: Optional[torch.Tensor]):
        """Check if the model is compatible with the predict method."""
        try:
            if self.method == 'dirac':
                dummy_input = torch.zeros((1, self.input_dim), device=self.device)
            else:  # lebesgue
                dummy_input = x_train[:1].to(self.device)
            _ = self.predict(dummy_input)
        except Exception as e:
            raise ValueError(f"The provided model is not compatible with the predict method. Error: {str(e)}")

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make predictions using the given model."""
        if hasattr(self.model, 'predict_proba'): # sklearn
            return torch.tensor(self.model.predict_proba(x.cpu().numpy())[:, 1], device=self.device).squeeze()
        else: # torch
            return self.model.predict(x, device=self.device).squeeze()

    def calculate_baseline(self, x: torch.Tensor) -> None:
        if self.method == 'dirac':
            x0 = torch.zeros((1, self.input_dim), device=self.device)
            y0 = self.predict(x0)
            self.logit_y0 = torch.log(y0 / (1 - y0)).item()
        else:  # lebesgue
            y0 = self.predict(x)
            self.logit_y0 = torch.log(y0 / (1 - y0)).mean().item()

    def calculate(self, x: torch.Tensor, for_plotting: bool = False, n_steps: int = 15, categorical_threshold: int = 15) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
        if self.method == 'dirac':
            if self.logit_y0 is None:
                self.calculate_baseline(x)
            return self._calculate_dirac(x, for_plotting, n_steps, categorical_threshold)
        elif self.method == 'lebesgue':
            return self._calculate_lebesgue(x, for_plotting, n_steps, categorical_threshold)
        else:
            raise ValueError(f"Method {self.method} not implemented")

    def _calculate_dirac(self, x: torch.Tensor, for_plotting: bool, n_steps: int, categorical_threshold: int) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:

        n_features = x.shape[1]
        n_samples = x.shape[0] if not for_plotting else n_steps

        univariate_responses = torch.zeros((n_samples, n_features), device=self.device)
        for i in range(n_features):
            if for_plotting:
                x_subset = get_variable_range(x[:, i], n_steps, categorical_threshold)
            else:
                x_subset = x[:, i]
            
            x_i = torch.zeros((len(x_subset), n_features), device=self.device)
            x_i[:, i] = x_subset
            
            y_i = self.predict(x_i)
            univariate_responses[:, i] = torch.log(y_i / (1 - y_i)) - self.logit_y0

        bivariate_responses = []
        bivariate_inputs = []
        for i in range(n_features):
            for j in range(i+1, n_features):
                if for_plotting:
                    x_subset_i = get_variable_range(x[:, i], n_steps, categorical_threshold)
                    x_subset_j = get_variable_range(x[:, j], n_steps, categorical_threshold)
                    x_ij = torch.cartesian_prod(x_subset_i, x_subset_j)
                else:
                    x_ij = x[:, [i, j]]

                x_input = torch.zeros((len(x_ij), n_features), device=self.device)
                x_input[:, i] = x_ij[:, 0]
                x_input[:, j] = x_ij[:, 1]

                y_ij = self.predict(x_input)
                bivariate_response = (torch.log(y_ij / (1 - y_ij)) - self.logit_y0
                                      - univariate_responses[:len(x_ij), i]
                                      - univariate_responses[:len(x_ij), j])
                bivariate_responses.append(bivariate_response)
                bivariate_inputs.append((i, j))

                # Clear unnecessary tensors
                del x_input, y_ij
                if self.device == 'cuda':
                    torch.cuda.empty_cache()

        bivariate_responses = torch.stack(bivariate_responses, dim=1)

        return univariate_responses, bivariate_responses, bivariate_inputs

    def _calculate_lebesgue(self, x: torch.Tensor, for_plotting: bool, n_steps: int, categorical_threshold: int) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
        n_features = x.shape[1]
        x = x.to(self.device)

        univariate_responses = []
        for i in range(n_features):
            if for_plotting:
                x_subset = get_variable_range(x[:, i], n_steps, categorical_threshold)
            else:
                x_subset = x[:, i]
            
            response = torch.zeros(len(x_subset), device=self.device)
            for k, value in enumerate(x_subset):
                x_i = x.clone()
                x_i[:, i] = value
                y_i = self.predict(x_i)
                logit_y_i = torch.log(y_i / (1 - y_i)).mean()
                response[k] = logit_y_i - self.logit_y0
            univariate_responses.append(response)

        bivariate_responses = []
        bivariate_inputs = []
        for i in range(n_features):
            for j in range(i+1, n_features):
                if for_plotting:
                    x_subset_i = get_variable_range(x[:, i], n_steps, categorical_threshold)
                    x_subset_j = get_variable_range(x[:, j], n_steps, categorical_threshold)
                    x_ij = torch.cartesian_prod(x_subset_i, x_subset_j)
                else:
                    x_ij = x[:, [i, j]]

                bivariate_response = torch.zeros(len(x_ij), device=self.device)

                for k, (val_i, val_j) in enumerate(x_ij):
                    x_temp = x.clone()
                    x_temp[:, i] = val_i
                    x_temp[:, j] = val_j
                    y_ij = self.predict(x_temp)
                    logit_y_ij = torch.log(y_ij / (1 - y_ij)).mean()
                    bivariate_response[k] = (logit_y_ij - self.logit_y0
                                             - univariate_responses[i][x[:, i] == val_i].mean()
                                             - univariate_responses[j][x[:, j] == val_j].mean())

                bivariate_responses.append(bivariate_response)
                bivariate_inputs.append((i, j))

        # Pad univariate responses to have the same length
        max_length = max(len(r) for r in univariate_responses)
        univariate_responses_padded = [torch.nn.functional.pad(r, (0, max_length - len(r))) for r in univariate_responses]
        univariate_responses = torch.stack(univariate_responses_padded, dim=1)

        bivariate_responses = torch.stack(bivariate_responses, dim=1)

        return univariate_responses, bivariate_responses, bivariate_inputs
    
def get_variable_range(x: torch.Tensor, n_steps: int, categorical_threshold: int) -> torch.Tensor:
    if x.unique().shape[0] < categorical_threshold:
        return x.unique()
    else:
        return torch.linspace(x.min(), x.max(), steps=n_steps)

def partial_responses(x_train: torch.Tensor, x_test: torch.Tensor, model: Any, method: str = 'dirac', device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
    pr = PartialResponseCalculator(model, method, device, input_dim=x_train.shape[1],x_train=x_train)
    
    univariate_train, bivariate_train, bivariate_inputs = pr.calculate(x_train)
    univariate_test, bivariate_test, _ = pr.calculate(x_test)
    
    train_responses = torch.cat([univariate_train, bivariate_train], dim=1)
    test_responses = torch.cat([univariate_test, bivariate_test], dim=1)
    
    return train_responses, test_responses, bivariate_inputs