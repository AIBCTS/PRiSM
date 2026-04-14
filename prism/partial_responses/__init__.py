"""
Partial Response Calculation

This module implements the calculation of partial responses, i.e. how each feature or pair
of features contribute to a model's predictions.

The module supports two methods for calculating partial responses:

- 'dirac': Calculates the effect of each feature individually by setting other features to zero.
- 'lebesgue': Calculates the average effect of each feature over the distribution of other features
  in the training data set.

The Lebesgue method provides more realistic estimates but is more computationally intensive. It relies
on the initial training data to calculate partial responses, even for unseen data points. It works as follows:
1. For each feature (or feature pair) and each sample in the input data:
   a. Create copies of the entire training dataset, one for each sample being evaluated.
   b. In each copy, replace the value(s) of the analyzed feature(s) with the corresponding value(s)
      from the current input data sample.
2. The model predicts outcomes for all these modified datasets.
3. For each sample, the average effect is calculated across all predictions made on its modified datasets.

This process allows the Lebesgue method to consider the joint distribution of all features
when estimating the effect of changing a specific feature or feature pair.

Key Components
--------------
1. PartialResponseCalculator: Handles calculations for univariate and bivariate cases.
2. partial_responses: Calculates full partial responses for all features and feature pairs.
3. partial_responses_subset: Calculates responses for a subset of feature values: unique values
   (categorical) or linear range (continuous). For visualization.

The module is optimized for GPU usage (vectorizing calculations) and multi-threading.
"""

from typing import Any, List, Optional, Tuple

import numpy as np
import torch

from prism.device_tools import device_empty_cache

# Import and re-export public API
from .calculator import PartialResponseCalculator
from .lebesgue import get_variable_range
from .utils import _warn_if_scaled_onehot, stable_logit, to_numpy

__all__ = [
    'PartialResponseCalculator',
    'partial_responses',
    'partial_responses_subset',
    'to_numpy',
    'stable_logit',
    'get_variable_range',
    '_warn_if_scaled_onehot',
]


def partial_responses(
    x: torch.Tensor,
    model: Any,
    x_train: Optional[torch.Tensor] = None,
    method: str = 'dirac',
    device: str = 'cpu',
    batch_size: int = 1024,
    onehot_groups: Optional[List[Tuple]] = None,
    group_manager: Optional[Any] = None,
    feature_names: Optional[List[str]] = None,
    scaler: Optional[Any] = None,
    predict_batch_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Calculate partial responses for all features and feature pairs in the input data.

    This function computes both univariate (single feature) and bivariate (feature pair)
    partial responses using either the 'dirac' or 'lebesgue' method.

    Parameters
    ----------
    x : torch.Tensor
        Input data tensor for which partial responses are calculated.
    model : Any
        The machine learning model used for predictions. Must have a 'predict' method.
    x_train : Optional[torch.Tensor], default=None
        Training data tensor, required for the 'lebesgue' method.
    method : str, default='dirac'
        Method for calculating partial responses. Either 'dirac' or 'lebesgue'.
    device : str, default='cpu'
        Computation device ('cpu' or 'cuda').
    batch_size : int, default=1024
        Size of batches for processing large datasets.
    onehot_groups : Optional[List[Tuple]], optional
        **Deprecated.** Use `group_manager` instead.
        Groups of column indices representing one-hot encoded categorical features.
        Each tuple contains indices of columns that form a one-hot encoded group.
        Ignored if `group_manager` is provided.
    group_manager : Optional[OneHotGroupManager], optional
        Manager for one-hot groups. This is the preferred interface for specifying
        one-hot encoded features. Takes precedence over `onehot_groups` if both
        are provided. When specified, responses are automatically collapsed to treat
        each group as a single categorical variable (required for mathematically
        correct results).
    feature_names : Optional[List[str]], optional
        List of feature names corresponding to columns in the input data.
        Must be in the **one-hot encoded (expanded) space**, i.e., one name per
        column including all one-hot encoded columns (e.g., ['age', 'diagn_A',
        'diagn_B', 'diagn_C']).
        **Required** when using `group_manager` (to map group names to column indices).
        **Required** for collapse operations (to generate collapsed feature names).
    scaler : Optional[Any], optional
        PRiSMScaler instance used to transform training data. Required for models trained
        on scaled data (e.g., XGBoost, Random Forest via SklearnWrapper) to ensure correct
        partial response calculations for one-hot encoded features.

    Returns
    -------
    torch.Tensor
        Combined partial responses (univariate + bivariate concatenated)

    Notes
    -----
    - The 'dirac' method calculates responses by setting other features to zero.
    - The 'lebesgue' method averages over the distribution of other features and
      requires the training data (x_train) to be provided.
    - This function uses GPU acceleration if available and implements batching
      for efficient processing of large datasets.
    - When onehot_groups or group_manager is specified, one-hot groups are AUTOMATICALLY
      collapsed from N binary columns to 1 categorical column per group. This ensures
      mathematically correct reconstruction.
    """

    # Use a context manager to ensure proper GPU memory management
    with device_empty_cache(torch.device(device)):
        pr = PartialResponseCalculator(
            model,
            method,
            device,
            input_dim=x.shape[1],
            x_train=x_train,
            onehot_groups=onehot_groups,
            group_manager=group_manager,
            feature_names=feature_names,
            scaler=scaler,
            predict_batch_size=predict_batch_size,
        )

        # Calculate univariate and bivariate responses
        univariate_train, bivariate_train = pr.calculate(x, batch_size=batch_size)

        # Combine univariate and bivariate responses into a single tensor
        responses = torch.cat([univariate_train, bivariate_train], dim=1)

        responses = responses.cpu()

    return responses


def partial_responses_subset(
    x: torch.Tensor,
    model: Any,
    method: str = 'dirac',
    x_train: Optional[torch.Tensor] = None,
    device: str = 'cpu',
    n_steps: int = 15,
    categorical_threshold: int = 15,
    subtract_univariate: bool = True,
    selected_features: Optional[List[int]] = None,
    selected_feature_pairs: Optional[List[Tuple[int, int]]] = None,
    batch_size: int = 64,
    onehot_groups: Optional[List[Tuple]] = None,
    group_manager: Optional[Any] = None,
    feature_names: Optional[List[str]] = None,
    trim_quantile: Optional[float] = None,
    scaler: Optional[Any] = None,
    predict_batch_size: Optional[int] = None,
) -> Tuple[
    List[np.ndarray], List[np.ndarray], List[Tuple[int, int]], List[np.ndarray], List[np.ndarray]
]:
    """
    Calculate partial responses for a subset of feature values.

    Parameters
    ----------
    x : torch.Tensor
        Input data tensor
    model : Any
        The machine learning model used for predictions
    method : str, default='dirac'
        Method for calculating partial responses ('dirac' or 'lebesgue')
    x_train : torch.Tensor, optional
        Training data tensor, required for 'lebesgue' method
    device : str, default='cpu'
        Computation device ('cpu', 'mps', or 'cuda')
    n_steps : int, default=15
        Number of steps for continuous features
    categorical_threshold : int, default=15
        Max unique values to consider categorical
    subtract_univariate : bool, default=True
        Whether to subtract univariate contributions from bivariate responses
    selected_features : Optional[List[int]], optional
        List of feature indices to calculate univariate responses for
    selected_feature_pairs : Optional[List[Tuple[int, int]]], optional
        List of feature index pairs to calculate bivariate responses for
    batch_size : int, default=1024
        Size of batches for processing large datasets
    onehot_groups : Optional[List[Tuple]], optional
        **Deprecated.** Use `group_manager` instead.
        Groups of column indices representing one-hot encoded categorical features.
        Each tuple contains indices of columns that form a one-hot encoded group.
        Ignored if `group_manager` is provided.
    group_manager : Optional[OneHotGroupManager], optional
        Manager for one-hot groups. This is the preferred interface for specifying
        one-hot encoded features. Takes precedence over `onehot_groups` if both
        are provided. When specified, responses are automatically collapsed to treat
        each group as a single categorical variable (required for mathematically
        correct results).
    feature_names : Optional[List[str]], optional
        List of feature names corresponding to columns in the input data.
        Must be in the **one-hot encoded (expanded) space**, i.e., one name per
        column including all one-hot encoded columns (e.g., ['age', 'diagn_A',
        'diagn_B', 'diagn_C']).
        **Required** when using `group_manager` (to map group names to column indices).
        **Required** for collapse operations (to generate collapsed feature names).
    trim_quantile : Optional[float], optional
        Fraction to trim from each tail when generating grids for continuous features.
        E.g., 0.01 uses the 1st to 99th percentile range. Only affects continuous
        features; categorical features always use all unique values.
    scaler : Optional[Any], optional
        PRiSMScaler instance used to transform training data. Required for models trained
        on scaled data (e.g., XGBoost, Random Forest via SklearnWrapper) to ensure correct
        partial response calculations for one-hot encoded features.
    predict_batch_size : Optional[int], optional
        Maximum number of rows per forward pass in _batched_predict(). If None,
        auto-scaled based on available GPU VRAM.

    Returns
    -------
    Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]
        A tuple containing:
        - List of univariate responses (one array per feature)
        - List of bivariate responses (one array per feature pair)
        - List of univariate x-values (one array per feature)
        - List of bivariate x-value pairs (one array per feature pair)

    Notes
    -----
    When onehot_groups or group_manager is specified:
    - One-hot groups are AUTOMATICALLY collapsed to single categorical features
    - x-values for categorical features are integers [0, 1, 2, ...]
      representing category indices (0=reference, 1+=active categories)
    - Responses are properly averaged across one-hot group members
    - Returned data is in collapsed space (fewer features than input)
    - This is REQUIRED for mathematically correct reconstruction

    Without one-hot groups:
    - Data remains in original feature space
    - All features treated independently
    - x-values are generated based on categorical_threshold
    """

    # Use a context manager to ensure proper GPU memory management
    with device_empty_cache(torch.device(device)):
        pr = PartialResponseCalculator(
            model,
            method,
            device,
            input_dim=x.shape[1],
            x_train=x_train,
            onehot_groups=onehot_groups,
            group_manager=group_manager,
            feature_names=feature_names,
            trim_quantile=trim_quantile,
            scaler=scaler,
            predict_batch_size=predict_batch_size,
        )
        univariate_responses, bivariate_responses, x_univariate, x_bivariate = pr.calculate_subset(
            x,
            n_steps,
            categorical_threshold,
            subtract_univariate,
            selected_features,
            selected_feature_pairs,
            batch_size,
        )

    # Extract dense lists in the order of selected_features and selected_feature_pairs
    # The sparse lists use feature/pair indices, but we need dense lists aligned with selections

    # Univariate: extract in order of selected_features
    if selected_features is not None:
        univariate_responses_filtered = [univariate_responses[i] for i in selected_features]
        x_univariate_filtered = [x_univariate[i] for i in selected_features]
    else:
        # All features selected - filter out None values
        univariate_responses_filtered = [r for r in univariate_responses if r is not None]
        x_univariate_filtered = [xv for xv in x_univariate if xv is not None]

    # Bivariate: extract in order of selected_feature_pairs
    if selected_feature_pairs is not None:
        n_features = pr.n_collapsed_features if pr._is_collapsed_mode() else x.shape[1]
        bivariate_responses_filtered = []
        x_bivariate_filtered = []
        for i, j in selected_feature_pairs:
            # Calculate flat index for this pair
            pair_idx = i * n_features + j - ((i + 2) * (i + 1)) // 2
            bivariate_responses_filtered.append(bivariate_responses[pair_idx])
            x_bivariate_filtered.append(x_bivariate[pair_idx])
    else:
        # All pairs selected - filter out None values
        bivariate_responses_filtered = [r for r in bivariate_responses if r is not None]
        x_bivariate_filtered = [xv for xv in x_bivariate if xv is not None]

    # Convert PyTorch tensors to NumPy arrays
    univariate_responses_np = [to_numpy(response) for response in univariate_responses_filtered]
    bivariate_responses_np = [to_numpy(response) for response in bivariate_responses_filtered]
    x_univariate_np = [to_numpy(xv) for xv in x_univariate_filtered]
    x_bivariate_np = [to_numpy(xv) for xv in x_bivariate_filtered]

    return univariate_responses_np, bivariate_responses_np, x_univariate_np, x_bivariate_np
