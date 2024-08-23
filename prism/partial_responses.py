import torch
import numpy as np
import copy
from typing import Any, Tuple, List, Optional
from prism.device_tools import device_empty_cache, get_available_gpus, get_num_cpu_workers
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed

class PartialResponseCalculator:
    def __init__(self, model: Any, method: str = 'dirac', device: Optional[str] = None, input_dim: int = 1, x_train: Optional[torch.Tensor] = None):
        self.original_model = model
        self.method = method
        self.device = torch.device(device) if device is not None else next(model.parameters()).device if hasattr(model, 'parameters') else torch.device('cpu')
        self.input_dim = input_dim
        self.logit_y0 = None
        self.models = {}  # Dictionary to store model copies for each GPU
        
        if method == 'lebesgue':
            if x_train is None:
                raise ValueError("the x_train: Optional[torch.Tensor] argument must be provided for the Lebesgue method.")
            self.calculate_baseline(x_train)
        
        self._check_model_compatibility(x_train)

    @torch.no_grad()
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
        
    def _prepare_multi_gpu(self, gpus: List[torch.device]):
        for gpu in gpus:
            self.models[gpu] = copy.deepcopy(self.original_model).to(gpu)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        if x.device in self.models:
            return self.models[x.device].predict(x).squeeze()
        else:
            return self.original_model.predict(x, device=self.device).squeeze()


    @torch.no_grad()
    def calculate_baseline(self, x: torch.Tensor) -> None:
        if self.method == 'dirac':
            x0 = torch.zeros((1, self.input_dim), device=self.device)
            y0 = self.predict(x0)
            self.logit_y0 = torch.log(y0 / (1 - y0)).item()
        else:  # lebesgue
            y0 = self.predict(x)
            self.logit_y0 = torch.log(y0 / (1 - y0)).mean().item()

    @torch.no_grad()
    def calculate(self, x: torch.Tensor, batch_size: int = 1024, max_workers: int = 1) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
        with device_empty_cache(self.device):
            if self.method == 'dirac':
                if self.logit_y0 is None:
                    self.calculate_baseline(x)
                return self._calculate_dirac(x)
            elif self.method == 'lebesgue':
                return self._calculate_lebesgue(x, batch_size=batch_size, max_workers=max_workers)
            else:
                raise ValueError(f"Method {self.method} not implemented. Choose 'dirac' or 'lebesgue'.")

    def _calculate_dirac(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
        n_features = x.shape[1]
        n_samples = x.shape[0]
        x = x.to(self.device)

        univariate_responses = torch.zeros((n_samples, n_features), device=self.device)
        for i in range(n_features):
            x_i = torch.zeros((n_samples, n_features), device=self.device)
            x_i[:, i] = x[:, i]
            
            y_i = self.predict(x_i)
            univariate_responses[:, i] = torch.log(y_i / (1 - y_i)) - self.logit_y0

        bivariate_responses = []
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

        bivariate_responses = torch.stack(bivariate_responses, dim=1)

        return univariate_responses, bivariate_responses

    def _calculate_lebesgue(self, x: torch.Tensor, batch_size: int = 1024,max_workers: int = 1, multi_gpus: bool = False) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
        n_features = x.shape[1]
        n_samples = x.shape[0]
        x = x.to(self.device)
        main_device = self.device
        gpus = get_available_gpus() if multi_gpus else []
        
        # Preallocate tensors
        univariate_responses = torch.zeros((n_samples, n_features), device=main_device)
        n_bivariate = n_features * (n_features - 1) // 2
        bivariate_responses = torch.zeros((n_samples, n_bivariate), device=main_device)

        try:
            if multi_gpus:
                self._prepare_multi_gpu(gpus)
                # Pre-transfer x to all GPUs
                x_on_gpus = {gpu: x.to(gpu) for gpu in gpus}
            else:
                x_on_gpus = {main_device: x}
            
            def process_univariate_batch(i, batch_start):
                device = gpus[i % len(gpus)] if gpus else main_device
                if batch_start==0:
                    print(f"Univariate {i},\t({device})")
                
                batch_end = min(batch_start + batch_size, n_samples)
                batch_size_current = batch_end - batch_start

                # Repeat x by batch size along new dimension
                x_i = x_on_gpus[device].unsqueeze(0).repeat(batch_size_current, 1, 1)

                # Replace ith value with current batch of values
                x_i[:, :, i] = x_on_gpus[device][batch_start:batch_end, i].unsqueeze(1).repeat(1, n_samples)

                # Reshape to (batch_size_current * n_samples, n_features)
                x_i = x_i.reshape(-1, n_features)
                
                y_i = self.predict(x_i)

                # Reshape and calculate mean
                logit_y_i = torch.log(y_i / (1 - y_i)).reshape(batch_size_current, n_samples).mean(dim=1)

                # Calculate univariate response for the current batch
                response = logit_y_i - self.logit_y0

                return i, batch_start, response.to(main_device)
            
            def process_bivariate_batch(ij, batch_start):
                i, j = ij
                device = gpus[(i * n_features + j) % len(gpus)] if gpus else main_device
                if batch_start==0:
                    print(f"Bivariate {i},{j},\t({device})")
                
                batch_end = min(batch_start + batch_size, n_samples)
                batch_size_current = batch_end - batch_start

                # Create batch for features i and j
                x_ij = x_on_gpus[device].unsqueeze(0).repeat(batch_size_current, 1, 1)

                # Set values for feature i
                x_ij[:, :, i] = x_on_gpus[device][batch_start:batch_end, i].unsqueeze(1).repeat(1, n_samples)
                
                # Set values for feature j
                x_ij[:, :, j] = x_on_gpus[device][batch_start:batch_end, j].unsqueeze(1).repeat(1, n_samples)
                
                # Reshape to (batch_size_current * n_samples, n_features)
                x_ij = x_ij.reshape(-1, n_features)

                y_ij = self.predict(x_ij)
                
                # Reshape and calculate mean
                logit_y_ij = torch.log(y_ij / (1 - y_ij)).reshape(batch_size_current, n_samples).mean(dim=1)
                
                # Calculate bivariate response for the current batch
                response = (
                    logit_y_ij - self.logit_y0
                    - univariate_responses[batch_start:batch_end, i].to(device)
                    - univariate_responses[batch_start:batch_end, j].to(device)
                )

                return (i, j), batch_start, response.to(main_device)

            # Print resource status
            print(f"Main compute device: {main_device}")
            if gpus:
                print("Workload spread to GPUs: ", ", ".join(gpus))
            print(f"Max threads: {max_workers}")
            print(f"Batch size: {batch_size}")

            # Process univariate responses
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for i in range(n_features):
                    for batch_start in range(0, n_samples, batch_size):
                        futures.append(executor.submit(process_univariate_batch, i, batch_start))
                
                for future in as_completed(futures):
                    i, batch_start, response = future.result()
                    batch_end = min(batch_start + batch_size, n_samples)
                    univariate_responses[batch_start:batch_end, i] = response

            # Process bivariate responses
            # bivariate_inputs = list(combinations(range(n_features), 2))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for ij in combinations(range(n_features), 2):
                    for batch_start in range(0, n_samples, batch_size):
                        futures.append(executor.submit(process_bivariate_batch, ij, batch_start))
                
                for future in as_completed(futures):
                    (i, j), batch_start, response = future.result()
                    batch_end = min(batch_start + batch_size, n_samples)
                    idx = i * n_features + j - ((i + 2) * (i + 1)) // 2
                    bivariate_responses[batch_start:batch_end, idx] = response

        finally:
            # Clean up GPU memory
            if multi_gpus:
                for gpu in gpus:
                    if gpu in self.models:
                        del self.models[gpu]
                    if gpu in x_on_gpus:
                        del x_on_gpus[gpu]
                self.models.clear()
            x_on_gpus.clear()

        return univariate_responses, bivariate_responses
    
    @torch.no_grad()
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
                x_bivariate.append(x_ij)

        return univariate_responses, bivariate_responses, x_univariate, x_bivariate

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
                x_bivariate.append(x_ij)

        return univariate_responses, bivariate_responses, x_univariate, x_bivariate

def get_variable_range(x: torch.Tensor, n_steps: int, categorical_threshold: int) -> torch.Tensor:
    if x.unique().shape[0] < categorical_threshold:
        return x.unique()
    else:
        return torch.linspace(x.min(), x.max(), steps=n_steps, device=x.device)

def partial_responses(x: torch.Tensor, model: Any, x_train: Optional[torch.Tensor] = None, method: str = 'dirac', device: str = 'cpu', batch_size: int = 1024, max_workers = get_num_cpu_workers()) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
    with device_empty_cache(torch.device(device)):
        pr = PartialResponseCalculator(model, method, device, input_dim=x.shape[1], x_train=x_train)
        
        univariate_train, bivariate_train= pr.calculate(x, batch_size=batch_size, max_workers=max_workers)
        
        responses = torch.cat([univariate_train, bivariate_train], dim=1)
        
        responses = responses.cpu()

    return responses

def partial_responses_subset(x: torch.Tensor, model: Any, method: str = 'dirac', device: str = 'cpu', n_steps: int = 15, categorical_threshold: int = 15, subtract_univariate: bool = True) -> Tuple[List[np.ndarray], List[np.ndarray], List[Tuple[int, int]], List[np.ndarray], List[np.ndarray]]:
    pr = PartialResponseCalculator(model, method, device, input_dim=x.shape[1], x_train=x)
    univariate_responses, bivariate_responses, x_univariate, x_bivariate = pr.calculate_subset(x, n_steps, categorical_threshold, subtract_univariate)
    
    # Convert PyTorch tensors to NumPy arrays
    univariate_responses_np = [response.cpu().numpy() for response in univariate_responses]
    bivariate_responses_np = [response.cpu().numpy() for response in bivariate_responses]
    x_univariate_np = [x.cpu().numpy() for x in x_univariate]
    x_bivariate_np = [x.cpu().numpy() for x in x_bivariate]
    
    return univariate_responses_np, bivariate_responses_np, x_univariate_np, x_bivariate_np