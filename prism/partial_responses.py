import torch
import numpy as np
from typing import Any, Tuple, List, Optional
from prism.device_tools import device_empty_cache

class PartialResponseCalculator:
    def __init__(self, model: Any, method: str = 'dirac', device: Optional[str] = None, input_dim: int = 1, x_train: Optional[torch.Tensor] = None):
        self.model = model
        self.method = method
        self.device = torch.device(device) if device is not None else next(model.parameters()).device if hasattr(model, 'parameters') else torch.device('cpu')
        self.input_dim = input_dim
        self.logit_y0 = None
        
        if method == 'lebesgue':
            if x_train is None:
                raise ValueError("x_train must be provided for the Lebesgue method")
            self.calculate_baseline(x_train)
        
        self._check_model_compatibility(x_train)

    def _check_model_compatibility(self, x_train: Optional[torch.Tensor]):
        """Check if the model is compatible with the predict method."""
        try:
            with torch.no_grad(), device_empty_cache(self.device):
                if self.method == 'dirac':
                    dummy_input = torch.zeros((1, self.input_dim), device=self.device)
                else:  # lebesgue
                    dummy_input = x_train[:1].to(self.device)
                _ = self.predict(dummy_input)
        except Exception as e:
            raise ValueError(f"The provided model is not compatible with the predict method. Error: {str(e)}")

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        if hasattr(self.model, 'predict_proba'): # sklearn
            return torch.tensor(self.model.predict_proba(x.cpu().numpy())[:, 1], device=self.device).squeeze()
        else: # torch
            return self.model.predict(x, device=self.device).squeeze()

    def calculate_baseline(self, x: torch.Tensor) -> None:
        with torch.no_grad(), device_empty_cache(self.device):
            if self.method == 'dirac':
                x0 = torch.zeros((1, self.input_dim), device=self.device)
                y0 = self.predict(x0)
                self.logit_y0 = torch.log(y0 / (1 - y0)).item()
            else:  # lebesgue
                y0 = self.predict(x)
                self.logit_y0 = torch.log(y0 / (1 - y0)).mean().item()

    def calculate(self, x: torch.Tensor, batch_size: int = 1024) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
        with device_empty_cache(self.device):
            if self.method == 'dirac':
                if self.logit_y0 is None:
                    self.calculate_baseline(x)
                return self._calculate_dirac(x)
            elif self.method == 'lebesgue':
                return self._calculate_lebesgue(x, batch_size=batch_size)
            else:
                raise ValueError(f"Method {self.method} not implemented. Choose 'dirac' or 'lebesgue'.")

    def _calculate_dirac(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
        n_features = x.shape[1]
        n_samples = x.shape[0]
        x = x.to(self.device)

        with torch.no_grad():
            univariate_responses = torch.zeros((n_samples, n_features), device=self.device)
            for i in range(n_features):
                x_i = torch.zeros((n_samples, n_features), device=self.device)
                x_i[:, i] = x[:, i]
                
                y_i = self.predict(x_i)
                univariate_responses[:, i] = torch.log(y_i / (1 - y_i)) - self.logit_y0

            bivariate_responses = []
            bivariate_inputs = []
            for i in range(n_features):
                for j in range(i+1, n_features):
                    x_input = torch.zeros((n_samples, n_features), device=self.device)
                    x_input[:, i] = x[:, i]
                    x_input[:, j] = x[:, j]

                    y_ij = self.predict(x_input)
                    bivariate_response = (torch.log(y_ij / (1 - y_ij)) - self.logit_y0
                                          - univariate_responses[:, i]
                                          - univariate_responses[:, j])
                    bivariate_responses.append(bivariate_response)
                    bivariate_inputs.append((i, j))

            bivariate_responses = torch.stack(bivariate_responses, dim=1)

        return univariate_responses, bivariate_responses, bivariate_inputs

    def _calculate_lebesgue(self, x: torch.Tensor, batch_size: int = 1024) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
        n_features = x.shape[1]
        n_samples = x.shape[0]
        x = x.to(self.device)
        
        with torch.no_grad():
            # Preallocate tensors
            univariate_responses = torch.zeros((n_samples, n_features), device=self.device)
            n_bivariate = n_features * (n_features - 1) // 2
            bivariate_responses = torch.zeros((n_samples, n_bivariate), device=self.device)
            bivariate_inputs = []

            for i in range(n_features):
                print(f"Univariate {i}")
                for batch_start in range(0, n_samples, batch_size):
                    batch_end = min(batch_start + batch_size, n_samples)
                    batch_size_current = batch_end - batch_start

                    # Repeat x by batch size along new dimension
                    x_i = x.unsqueeze(0).repeat(batch_size_current, 1, 1)

                    # Replace ith value with current batch of values
                    x_i[:, :, i] = x[batch_start:batch_end, i].unsqueeze(1).repeat(1, n_samples)

                    # Reshape to (batch_size_current * n_samples, n_features)
                    x_i = x_i.reshape(-1, n_features)
                    
                    y_i = self.predict(x_i)

                    # Reshape and calculate mean
                    logit_y_i = torch.log(y_i / (1 - y_i)).reshape(batch_size_current, n_samples).mean(dim=1)
                    
                    # Calculate univariate response for the current batch
                    univariate_responses[batch_start:batch_end, i] = logit_y_i - self.logit_y0

                    #del x_i, y_i, logit_y_i

            biv_idx = 0
            for i in range(n_features):
                for j in range(i+1, n_features):
                    print(f"Bivariate {i},{j}")
                    for batch_start in range(0, n_samples, batch_size):
                        batch_end = min(batch_start + batch_size, n_samples)
                        batch_size_current = batch_end - batch_start

                        # Create batch for features i and j
                        x_ij = x.unsqueeze(0).repeat(batch_size_current, 1, 1)
                        
                        # Set values for feature i
                        x_ij[:, :, i] = x[batch_start:batch_end, i].unsqueeze(1).repeat(1, n_samples)
                        
                        # Set values for feature j (corrected)
                        x_ij[:, :, j] = x[batch_start:batch_end, j].unsqueeze(1).repeat(1, n_samples)
                        
                        # Reshape to (batch_size_current * n_samples, n_features)
                        x_ij = x_ij.reshape(-1, n_features)

                        y_ij = self.predict(x_ij)
                        
                        # Reshape and calculate mean
                        logit_y_ij = torch.log(y_ij / (1 - y_ij)).reshape(batch_size_current, n_samples).mean(dim=1)
                        
                        # Calculate bivariate response for the current batch
                        bivariate_responses[batch_start:batch_end, biv_idx] = (
                            logit_y_ij - self.logit_y0
                            - univariate_responses[batch_start:batch_end, i]
                            - univariate_responses[batch_start:batch_end, j]
                        )

                        #del x_ij, y_ij, logit_y_ij

                    bivariate_inputs.append((i, j))
                    biv_idx += 1

        return univariate_responses, bivariate_responses, bivariate_inputs
    
    def calculate_subset(self, x: torch.Tensor, n_steps: int = 15, categorical_threshold: int = 15, subtract_univariate: bool = False) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Tuple[int, int]], List[torch.Tensor], List[torch.Tensor]]:
        if self.method == 'dirac':
            if self.logit_y0 is None:
                self.calculate_baseline(x)
            return self._calculate_dirac_subset(x, n_steps, categorical_threshold, subtract_univariate)
        elif self.method == 'lebesgue':
            return self._calculate_lebesgue_subset(x, n_steps, categorical_threshold, subtract_univariate)
        else:
            raise ValueError(f"Method {self.method} not implemented. Choose 'dirac' or 'lebesgue'.")
        
    def _calculate_dirac_subset(self, x: torch.Tensor, n_steps: int, categorical_threshold: int, subtract_univariate: bool) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Tuple[int, int]], List[torch.Tensor], List[torch.Tensor]]:
        n_features = x.shape[1]
        x = x.to(self.device)

        univariate_responses = []
        x_univariate = []
        for i in range(n_features):
            x_subset = get_variable_range(x[:, i], n_steps, categorical_threshold)
            x_i = torch.zeros((len(x_subset), n_features), device=self.device)
            x_i[:, i] = x_subset
            
            y_i = self.predict(x_i)
            response = torch.log(y_i / (1 - y_i)) - self.logit_y0
            univariate_responses.append(response)
            x_univariate.append(x_subset)

        bivariate_responses = []
        bivariate_inputs = []
        x_bivariate = []
        for i in range(n_features):
            for j in range(i+1, n_features):
                x_subset_i = get_variable_range(x[:, i], n_steps, categorical_threshold)
                x_subset_j = get_variable_range(x[:, j], n_steps, categorical_threshold)
                x_ij = torch.cartesian_prod(x_subset_i, x_subset_j)

                x_input = torch.zeros((len(x_ij), n_features), device=self.device)
                x_input[:, i] = x_ij[:, 0]
                x_input[:, j] = x_ij[:, 1]

                y_ij = self.predict(x_input)
                bivariate_response = torch.log(y_ij / (1 - y_ij)) - self.logit_y0
                
                if subtract_univariate:
                    bivariate_response -= (univariate_responses[i][torch.searchsorted(x_subset_i, x_ij[:, 0])] +
                                           univariate_responses[j][torch.searchsorted(x_subset_j, x_ij[:, 1])])
                
                bivariate_responses.append(bivariate_response)
                bivariate_inputs.append((i, j))
                x_bivariate.append(x_ij)

        return univariate_responses, bivariate_responses, bivariate_inputs, x_univariate, x_bivariate

    def _calculate_lebesgue_subset(self, x: torch.Tensor, n_steps: int, categorical_threshold: int, subtract_univariate: bool) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Tuple[int, int]], List[torch.Tensor], List[torch.Tensor]]:
        n_features = x.shape[1]
        x = x.to(self.device)

        univariate_responses = []
        x_univariate = []
        for i in range(n_features):
            x_subset = get_variable_range(x[:, i], n_steps, categorical_threshold)
            response = torch.zeros(len(x_subset), device=self.device)
            for k, value in enumerate(x_subset):
                x_i = x.clone()
                x_i[:, i] = value
                y_i = self.predict(x_i)
                logit_y_i = torch.log(y_i / (1 - y_i)).mean()
                response[k] = logit_y_i - self.logit_y0
            univariate_responses.append(response)
            x_univariate.append(x_subset)

        bivariate_responses = []
        bivariate_inputs = []
        x_bivariate = []
        for i in range(n_features):
            for j in range(i+1, n_features):
                x_subset_i = get_variable_range(x[:, i], n_steps, categorical_threshold)
                x_subset_j = get_variable_range(x[:, j], n_steps, categorical_threshold)
                x_ij = torch.cartesian_prod(x_subset_i, x_subset_j)

                bivariate_response = torch.zeros(len(x_ij), device=self.device)
                for k, (val_i, val_j) in enumerate(x_ij):
                    x_temp = x.clone()
                    x_temp[:, i] = val_i
                    x_temp[:, j] = val_j
                    y_ij = self.predict(x_temp)
                    logit_y_ij = torch.log(y_ij / (1 - y_ij)).mean()
                    bivariate_response[k] = logit_y_ij - self.logit_y0
                    
                    if subtract_univariate:
                        bivariate_response[k] -= (univariate_responses[i][torch.searchsorted(x_subset_i, val_i)] +
                                                  univariate_responses[j][torch.searchsorted(x_subset_j, val_j)])

                bivariate_responses.append(bivariate_response)
                bivariate_inputs.append((i, j))
                x_bivariate.append(x_ij)

        return univariate_responses, bivariate_responses, bivariate_inputs, x_univariate, x_bivariate

def get_variable_range(x: torch.Tensor, n_steps: int, categorical_threshold: int) -> torch.Tensor:
    if x.unique().shape[0] < categorical_threshold:
        return x.unique()
    else:
        return torch.linspace(x.min(), x.max(), steps=n_steps, device=x.device)

def partial_responses(x_train: torch.Tensor, x_test: torch.Tensor, model: Any, method: str = 'dirac', device: str = 'cpu', batch_size: int = 1024) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
    with device_empty_cache(torch.device(device)):
        pr = PartialResponseCalculator(model, method, device, input_dim=x_train.shape[1], x_train=x_train)
        
        univariate_train, bivariate_train, bivariate_inputs = pr.calculate(x_train, batch_size=batch_size)
        univariate_test, bivariate_test, _ = pr.calculate(x_test, batch_size=batch_size)
        
        train_responses = torch.cat([univariate_train, bivariate_train], dim=1)
        test_responses = torch.cat([univariate_test, bivariate_test], dim=1)
        
        # Move results to CPU before returning
        train_responses = train_responses.cpu()
        test_responses = test_responses.cpu()

    return train_responses, test_responses, bivariate_inputs

def partial_responses_subset(x: torch.Tensor, model: Any, method: str = 'dirac', device: str = 'cpu', n_steps: int = 15, categorical_threshold: int = 15, subtract_univariate: bool = True) -> Tuple[List[np.ndarray], List[np.ndarray], List[Tuple[int, int]], List[np.ndarray], List[np.ndarray]]:
    pr = PartialResponseCalculator(model, method, device, input_dim=x.shape[1], x_train=x)
    univariate_responses, bivariate_responses, bivariate_inputs, x_univariate, x_bivariate = pr.calculate_subset(x, n_steps, categorical_threshold, subtract_univariate)
    
    # Convert PyTorch tensors to NumPy arrays
    univariate_responses_np = [response.cpu().numpy() for response in univariate_responses]
    bivariate_responses_np = [response.cpu().numpy() for response in bivariate_responses]
    x_univariate_np = [x.cpu().numpy() for x in x_univariate]
    x_bivariate_np = [x.cpu().numpy() for x in x_bivariate]
    
    return univariate_responses_np, bivariate_responses_np, bivariate_inputs, x_univariate_np, x_bivariate_np