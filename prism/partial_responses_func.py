import torch
from typing import Any, Tuple, Optional, List

def predict(x: torch.Tensor, model: Any) -> torch.Tensor:
    """Make predictions using the given model."""
    if isinstance(model,list) == True:
        return torch.tensor(mlpmask_pred(x.cpu().numpy(),model,device=device), device=x.device).squeeze()
    elif hasattr(model, 'predict_proba'):
        return torch.tensor(model.predict_proba(x.cpu().numpy())[:, 1], device=x.device).squeeze()
    else:
        return model.predict(x, device=x.device).squeeze()

def compute_partial_responses(x: torch.Tensor, model: Any, n_steps: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    device = x.device
    n_samples, n_features = x.shape
    
    # Compute baseline prediction
    x0 = torch.zeros((1, n_features), device=device)
    y0 = predict(x0, model)
    logit_y0 = torch.log(y0 / (1 - y0)).item()  # Convert to scalar

    # Compute univariate responses
    univariate_responses = torch.zeros((n_samples, n_features), device=device)
    for i in range(n_features):
        x_i = x0.repeat(n_samples, 1)
        x_i[:, i] = x[:, i]
        y_i = predict(x_i, model)
        univariate_responses[:, i] = torch.log(y_i / (1 - y_i)) - logit_y0

    # Compute bivariate responses
    n_bivariate = n_features * (n_features - 1) // 2
    bivariate_responses = torch.zeros((n_samples, n_bivariate), device=device)
    idx = 0
    for i in range(n_features):
        for j in range(i+1, n_features):
            x_ij = x0.repeat(n_samples, 1)
            x_ij[:, i] = x[:, i]
            x_ij[:, j] = x[:, j]
            y_ij = predict(x_ij, model)
            bivariate_responses[:, idx] = torch.log(y_ij / (1 - y_ij)) - logit_y0 - univariate_responses[:, i] - univariate_responses[:, j]
            idx += 1

    print(f"logit_y0: {logit_y0}")
    print(f"First few univariate responses: {univariate_responses[:5, :5]}")
    print(f"First few bivariate responses: {bivariate_responses[:5, :5]}")

    return univariate_responses, bivariate_responses

def lebesgue_partial_responses(x: torch.Tensor, model: Any, x_train: Optional[torch.Tensor] = None, n_steps: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    device = x.device
    n_samples, n_features = x.shape
    
    # Compute baseline prediction
    if x_train != None:
        y0 = predict(x_train, model)
    else:
        y0 = predict(x, model)
    logit_y0 = torch.log(y0 / (1 - y0)).mean().item()

    # Compute univariate responses
    univariate_responses = torch.zeros((n_samples, n_features), device=device)
    for i in range(n_features):
        for k in range(n_samples):
            x_i = x.clone()
            x_i[:, i] = x[k, i]
            y_i = predict(x_i, model)
            univariate_responses[k, i] = torch.log(y_i / (1 - y_i)).mean() - logit_y0

    # Compute bivariate responses
    n_bivariate = n_features * (n_features - 1) // 2
    bivariate_responses = torch.zeros((n_samples, n_bivariate), device=device)
    idx = 0
    for i in range(n_features):
        for j in range(i+1, n_features):
            for k in range(n_samples):
                x_ij = x.clone()
                x_ij[:, i] = x[k, i]
                x_ij[:, j] = x[k, j]
                y_ij = predict(x_ij, model)
                bivariate_responses[k, idx] = (torch.log(y_ij / (1 - y_ij)).mean() - logit_y0 
                                               - univariate_responses[k, i] - univariate_responses[k, j])
            idx += 1

    print(f"logit_y0: {logit_y0}")
    print(f"First few univariate responses: {univariate_responses[:5, :5]}")
    print(f"First few bivariate responses: {bivariate_responses[:5, :5]}")

    return univariate_responses, bivariate_responses

def partial_responses(x_train: torch.Tensor, x_test: torch.Tensor, model: Any, residual: bool = False, method: str = "dirac", device: str = "cuda" if torch.cuda.is_available() else "cpu") -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
    """
    Generate partial responses for the model inputs.
    
    Args:
    x_train (torch.Tensor): Training data
    x_test (torch.Tensor): Test data
    model (Any): The trained model
    residual (bool): Whether to include residuals
    method (str): 'dirac' or 'lebesgue'
    device (str): Device to use for computations
    
    Returns:
    Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]]]: 
        F_covariates_train, F_covariates_test, bivariate_inputs
    """
    x_train = x_train.to(device)
    x_test = x_test.to(device)
    
    if method.lower() == "dirac":
        print("dirac train:")
        univariate_responses_train, bivariate_responses_train = compute_partial_responses(x_train, model)
        print("dirac test:")
        univariate_responses_test, bivariate_responses_test = compute_partial_responses(x_test, model)
    elif method.lower() == "lebesgue":
        print("lebesgue train:")
        univariate_responses_train, bivariate_responses_train = lebesgue_partial_responses(x_train, model)
        print("lebesgue test:")
        univariate_responses_test, bivariate_responses_test = lebesgue_partial_responses(x_test, model, x_train=x_train)
    else:
        raise ValueError("Invalid method: choose 'dirac' or 'lebesgue'")

    # Combine responses
    F_covariates_train = torch.cat([univariate_responses_train, bivariate_responses_train], dim=1)
    F_covariates_test = torch.cat([univariate_responses_test, bivariate_responses_test], dim=1)

    n_features = x_train.shape[1]
    bivariate_inputs = [(i, j) for i in range(n_features) for j in range(i+1, n_features)]

    if residual:
        # Compute and add residuals
        y_pred_train = predict(x_train, model)
        y_pred_test = predict(x_test, model)
        
        logit_y_train = torch.log(y_pred_train / (1 - y_pred_train))
        logit_y_test = torch.log(y_pred_test / (1 - y_pred_test))
        
        train_approx = F_covariates_train.sum(dim=1)
        test_approx = F_covariates_test.sum(dim=1)
        
        res_logit_train = logit_y_train - train_approx
        res_logit_test = logit_y_test - test_approx
        
        # Normalize residuals
        res_temp = res_logit_train
        res_logit_train = (res_temp - res_temp.mean()) / res_temp.std()
        res_logit_test = (res_logit_test - res_temp.mean()) / res_temp.std()
        
        F_covariates_train = torch.cat([F_covariates_train, res_logit_train.unsqueeze(1)], dim=1)
        F_covariates_test = torch.cat([F_covariates_test, res_logit_test.unsqueeze(1)], dim=1)

    return F_covariates_train, F_covariates_test, bivariate_inputs