"""
Lebesgue and Dirac method implementations for partial response calculations.

This module provides a mixin class containing Lebesgue-specific methods
(caching, batching) as well as shared subset calculation logic used by
both Lebesgue and Dirac methods.

Note: _calculate_dirac_subset is included in this mixin to leverage the
shared infrastructure for grid generation, one-hot group handling, and
collapsed feature space mapping that is common to both methodologies.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import torch

from prism.device_tools import _free_all_gpu_caches

from .utils import stable_logit

logger = logging.getLogger(__name__)


def get_variable_range(
    x: torch.Tensor,
    n_steps: int,
    categorical_threshold: int,
    trim_quantile: Optional[float] = None,
) -> torch.Tensor:
    """
    Get range of values for a feature, treating it as categorical or continuous.

    Parameters
    ----------
    x : torch.Tensor
        1D tensor of feature values
    n_steps : int
        Number of steps for continuous features
    categorical_threshold : int
        Max unique values to consider categorical
    trim_quantile : Optional[float], default=None
        If provided, trim this fraction from each tail of the distribution for
        continuous features. E.g., 0.01 uses the 1st to 99th percentile range.
        This limits the grid range to exclude outliers while still allowing
        histogram data to extend beyond the axis limits.
        Only applies to continuous features; categorical features always use
        all unique values regardless of this setting.

    Returns
    -------
    torch.Tensor
        Unique values (categorical) or linear range (continuous)
    """
    if x.unique().shape[0] < categorical_threshold:
        return x.unique()
    else:
        if trim_quantile is not None and trim_quantile > 0:
            # Use quantiles to limit range (exclude outliers)
            lower = torch.quantile(x.float(), trim_quantile)
            upper = torch.quantile(x.float(), 1.0 - trim_quantile)
        else:
            # Use full range (original behavior)
            lower = x.min()
            upper = x.max()
        return torch.linspace(lower, upper, steps=n_steps, device=x.device)


def _assign_column_masked(x, row_mask, col_idx, value):
    """Assign ``value`` to ``x[row_mask, col_idx]`` using MPS-safe 1-D ops.

    The direct pattern ``x[bool_mask, int_col] = scalar`` triggers an
    ``AcceleratorError`` on the Apple-Silicon Metal (MPS) backend for large
    tensors.  Decomposing into a column slice followed by a 1-D boolean
    write avoids the problematic 2-D boolean+integer indexing while
    remaining correct on CPU and CUDA.
    """
    col = x[:, col_idx]
    col[row_mask] = value
    x[:, col_idx] = col


class LebesgueMixin:
    """
    Mixin class providing Lebesgue and shared Dirac method implementations.

    This mixin is designed to be inherited by PartialResponseCalculator
    and provides all Lebesgue-specific methods including caching,
    batching, and subset calculations. It also includes the Dirac subset
    calculation logic to share the complex feature mapping and grid
    generation infrastructure.

    The mixin assumes the following attributes exist on self:
    - device: torch.device
    - x_train: torch.Tensor
    - logit_y0: float
    - input_dim: int
    - onehot_groups: Optional[List[Tuple]]
    - use_caching: bool
    - caching_threshold: Optional[int]
    - index_mapping: Dict[int, List[int]]
    - n_collapsed_features: int
    - collapse_onehot: bool
    - group_manager: Optional[Any]
    - collapsed_feature_names: List[str]
    - trim_quantile: Optional[float]
    - onehot_scaled_values: Optional[Dict]

    And the following methods:
    - predict(x) -> torch.Tensor
    - _get_onehot_value(col_idx, raw_value) -> float
    - _get_onehot_group(feature_idx) -> Optional[Tuple]
    - _should_skip_bivariate_pair(i, j) -> bool
    - _is_collapsed_mode() -> bool
    - _get_collapsed_feature_grid(collapsed_idx, x, n_steps, categorical_threshold) -> torch.Tensor
    - _create_baseline_input(n_rows, n_cols) -> torch.Tensor
    """

    def _analyze_cardinality(self, x: torch.Tensor) -> List[Dict]:
        """Analyze feature cardinality for caching optimization potential.

        Parameters
        ----------
        x : torch.Tensor
            Input data tensor

        Returns
        -------
        List[Dict]
            List of dictionaries containing feature analysis info:
            - cache_optimized: bool, whether to use caching
            - unique_values: tensor of unique values if caching
            - inverse_indices: mapping from samples to unique values if caching
        """
        logger.info("Starting feature cardinality analysis")
        logger.debug("-" * 50)

        n_samples = x.shape[0]
        threshold = self.caching_threshold or n_samples
        feature_info = []

        # Force caching when onehot_groups is provided
        force_caching = self.onehot_groups is not None
        if force_caching:
            logger.info("Forcing cached computation due to one-hot group constraints")

        for i in range(self.input_dim):
            feature_values = x[:, i]
            unique_values, inverse_indices = torch.unique(
                feature_values, return_inverse=True, sorted=True
            )

            # Use caching if forced by onehot_groups or if below threshold
            use_cache = force_caching or (
                len(unique_values) < threshold if self.use_caching else False
            )

            if use_cache:
                feature_info.append(
                    {
                        'cache_optimized': True,
                        'unique_values': unique_values,
                        'inverse_indices': inverse_indices,
                    }
                )
                if force_caching:
                    logger.debug(
                        f"Feature {i}: {len(unique_values)} unique values - using cached responses (forced by one-hot groups)"
                    )
                else:
                    logger.debug(
                        f"Feature {i}: {len(unique_values)} unique values - using cached responses"
                    )
            else:
                feature_info.append({'cache_optimized': False})
                logger.debug(
                    f"Feature {i}: {len(unique_values)} unique values - using direct computation"
                )

        return feature_info

    def _compute_collapsed_onehot_responses(
        self, collapsed_idx: int, batch_size: int
    ) -> torch.Tensor:
        """
        Compute Lebesgue responses for a collapsed one-hot group.

        Generates categorical integer grid [0, N] where:
        - 0 = reference category (all one-hot columns = 0)
        - 1 to N = active categories (corresponding one-hot column = 1)

        Parameters
        ----------
        collapsed_idx : int
            Index in collapsed feature space
        batch_size : int
            Batch size for processing

        Returns
        -------
        torch.Tensor
            Responses for each category
        """
        original_indices = self.index_mapping[collapsed_idx]
        n_categories = len(original_indices)  # Number of one-hot columns
        n_values = n_categories + 1  # Include reference (0)
        n_train_samples = self.x_train.shape[0]
        unique_responses = torch.zeros(n_values, device=self.device)

        for batch_start in range(0, n_values, batch_size):
            batch_end = min(batch_start + batch_size, n_values)
            batch_size_current = batch_end - batch_start

            # Create modified versions of training data
            x_modified = self.x_train.repeat(batch_size_current, 1)

            # For each categorical value in batch
            for batch_idx in range(batch_size_current):
                cat_value = batch_start + batch_idx
                start_row = batch_idx * n_train_samples
                end_row = start_row + n_train_samples

                if cat_value == 0:
                    # Reference category: set all one-hot columns to scaled 0
                    for orig_idx in original_indices:
                        scaled_0 = self._get_onehot_value(orig_idx, 0.0)
                        x_modified[start_row:end_row, orig_idx] = scaled_0
                else:
                    # Active category: set corresponding column to scaled 1, others to scaled 0
                    active_idx = original_indices[cat_value - 1]
                    for orig_idx in original_indices:
                        if orig_idx == active_idx:
                            scaled_1 = self._get_onehot_value(orig_idx, 1.0)
                            x_modified[start_row:end_row, orig_idx] = scaled_1
                        else:
                            scaled_0 = self._get_onehot_value(orig_idx, 0.0)
                            x_modified[start_row:end_row, orig_idx] = scaled_0

            # Get predictions and convert to logit space
            y_pred = self._batched_predict(x_modified)
            logits = stable_logit(y_pred)

            # Calculate average response for each category
            batch_responses = logits.reshape(batch_size_current, n_train_samples).mean(dim=1)

            # Store responses
            unique_responses[batch_start:batch_end] = batch_responses - self.logit_y0

            # Force computation to free memory
            _free_all_gpu_caches()

        return unique_responses

    def _compute_unique_responses(
        self, feature_idx: int, unique_values: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """Compute Lebesgue responses for unique feature values.

        For one-hot encoded features, this method handles the reference state
        by setting ALL columns in the one-hot group to 0 when the feature value is 0.
        This ensures that all features in the same one-hot group share the same
        reference state response, representing the dropped reference category.

        Parameters
        ----------
        feature_idx : int
            Index of the feature
        unique_values : torch.Tensor
            Tensor of unique values for the feature
        batch_size : int
            Batch size for processing

        Returns
        -------
        torch.Tensor
            Responses for each unique value. For one-hot encoded features:
            - value=0: Reference state (all group columns = 0)
            - value=1: Active state (target=1, siblings=0)
        """
        n_unique = len(unique_values)
        n_train_samples = self.x_train.shape[0]
        unique_responses = torch.zeros(n_unique, device=self.device)

        for batch_start in range(0, n_unique, batch_size):
            batch_end = min(batch_start + batch_size, n_unique)
            batch_size_current = batch_end - batch_start

            # Create modified versions of training data
            x_modified = self.x_train.repeat(batch_size_current, 1)

            # Replace feature values with current batch of unique values
            feature_values = unique_values[batch_start:batch_end].repeat_interleave(
                n_train_samples
            )

            # Handle one-hot group constraints for both value=0 and value=1 cases
            if self.onehot_groups:
                onehot_group = self._get_onehot_group(feature_idx)
                if onehot_group is not None:
                    # For one-hot features, use scaled values
                    scaled_feature_values = feature_values.clone()
                    for idx in range(len(feature_values)):
                        raw_val = feature_values[idx].item()
                        scaled_val = self._get_onehot_value(feature_idx, raw_val)
                        scaled_feature_values[idx] = scaled_val
                    x_modified[:, feature_idx] = scaled_feature_values
                else:
                    x_modified[:, feature_idx] = feature_values
            else:
                x_modified[:, feature_idx] = feature_values

            # Handle one-hot group constraints for both value=0 and value=1 cases
            if self.onehot_groups:
                onehot_group = self._get_onehot_group(feature_idx)
                if onehot_group is not None:
                    # For value=0: Set ALL columns in group to scaled 0 (reference state)
                    # For value=1: Set target=1, siblings=scaled 0 (active state)
                    zero_mask = feature_values == 0
                    non_zero_mask = feature_values != 0

                    # Set all columns in group to scaled 0 for reference state (value=0)
                    for col_idx in onehot_group:
                        scaled_0 = self._get_onehot_value(col_idx, 0.0)
                        _assign_column_masked(x_modified, zero_mask, col_idx, scaled_0)

                    # Set only sibling columns to scaled 0 for active state (value=1)
                    sibling_columns = [idx for idx in onehot_group if idx != feature_idx]
                    for sibling_col in sibling_columns:
                        scaled_0 = self._get_onehot_value(sibling_col, 0.0)
                        _assign_column_masked(x_modified, non_zero_mask, sibling_col, scaled_0)

            # Get predictions and convert to logit space
            y_pred = self._batched_predict(x_modified)
            logits = stable_logit(y_pred)

            # Calculate average response for each unique value
            batch_responses = logits.reshape(batch_size_current, n_train_samples).mean(dim=1)

            # Store responses
            unique_responses[batch_start:batch_end] = batch_responses - self.logit_y0

            # Force computation to free memory
            _free_all_gpu_caches()

        return unique_responses

    def _process_univariate_batch(
        self,
        x: torch.Tensor,
        i: int,
        feature_info: Dict,
        batch_start: int,
        batch_size: int,
        n_samples: int,
    ) -> Tuple[int, int, torch.Tensor]:
        """Process a batch of univariate Lebesgue partial responses with optional caching.

        Parameters
        ----------
        i : int
            Feature index
        x : torch.Tensor
            Input data tensor
        feature_info : Dict
            Feature cardinality analysis information
        batch_start : int
            Start index of batch
        batch_size : int
            Size of batch
        n_samples : int
            Total number of samples

        Returns
        -------
        Tuple[int, int, torch.Tensor]
            Feature index, batch start index, and computed responses
        """
        batch_end = min(batch_start + batch_size, n_samples)
        batch_size_current = batch_end - batch_start

        if feature_info['cache_optimized']:
            if batch_start == 0:  # Only compute unique responses once
                unique_responses = self._compute_unique_responses(
                    i, feature_info['unique_values'], batch_size
                )
                # Store for reuse by other batches
                feature_info['unique_responses'] = unique_responses

            # Map unique responses to current batch
            batch_indices = feature_info['inverse_indices'][batch_start:batch_end]
            batch_responses = feature_info['unique_responses'][batch_indices]

        else:
            # Original direct computation
            x_modified = self.x_train.repeat(batch_size_current, 1)

            # Apply scaling for one-hot columns if needed
            if hasattr(self, 'group_manager') and self.group_manager is not None:
                # Check if this feature is part of a one-hot group
                feat_info = self.group_manager.get_feature_info(i)
                if feat_info and feat_info.is_categorical_group and len(feat_info.indices) > 1:
                    # This is a one-hot column, apply scaling to the raw values
                    raw_vals = x[batch_start:batch_end, i]
                    scaled_vals = torch.zeros_like(raw_vals)
                    for v_idx, raw_val in enumerate(raw_vals):
                        if raw_val != 0:
                            scaled_vals[v_idx] = self._get_onehot_value(i, float(raw_val.item()))
                    x_modified[:, i] = scaled_vals.repeat_interleave(self.x_train.shape[0])
                else:
                    x_modified[:, i] = x[batch_start:batch_end, i].repeat_interleave(
                        self.x_train.shape[0]
                    )
            else:
                x_modified[:, i] = x[batch_start:batch_end, i].repeat_interleave(
                    self.x_train.shape[0]
                )

            y_pred = self._batched_predict(x_modified)
            logits = stable_logit(y_pred)
            batch_responses = logits.reshape(batch_size_current, -1).mean(dim=1) - self.logit_y0

        return i, batch_start, batch_responses

    def _analyze_single_pair(
        self, x: torch.Tensor, i: int, j: int, threshold: int, force_caching: bool = False
    ) -> Dict:
        """Analyze a single feature pair for caching optimization potential."""
        cardinality_start = time.time()

        # Extract and analyze feature pair values
        pairs = x[:, [i, j]]
        unique_pairs, inverse_indices = torch.unique(
            pairs, dim=0, return_inverse=True, sorted=True
        )

        # Use caching if forced by onehot_groups or if below threshold
        use_cache = force_caching or (len(unique_pairs) < threshold if self.use_caching else False)

        if use_cache:
            pair_info = {
                'cache_optimized': True,
                'indices': (i, j),
                'unique_pairs': unique_pairs,
                'inverse_indices': inverse_indices,
            }
            if force_caching:
                logger.debug(
                    f"Features {i},{j}: {len(unique_pairs)} unique pairs - will use cached responses "
                    f"(forced by one-hot groups) ({time.time() - cardinality_start:.2f}s)"
                )
            else:
                logger.debug(
                    f"Features {i},{j}: {len(unique_pairs)} unique pairs - will use cached responses "
                    f"({time.time() - cardinality_start:.2f}s)"
                )
        else:
            pair_info = {'cache_optimized': False, 'indices': (i, j)}
            logger.debug(
                f"Features {i},{j}: {len(unique_pairs)} unique pairs - will use direct computation "
                f"({time.time() - cardinality_start:.2f}s)"
            )

        return pair_info

    def _analyze_bivariate_pairs(self, x: torch.Tensor) -> List[Dict]:
        """Analyze feature pairs for caching optimization potential."""
        logger.info("Starting bivariate feature pair cardinality analysis")
        logger.debug("-" * 50)

        n_samples = x.shape[0]
        threshold = self.caching_threshold or n_samples
        pair_info = []
        skipped_pairs = []

        # Force caching when onehot_groups is provided
        force_caching = self.onehot_groups is not None
        if force_caching:
            logger.info(
                "Forcing cached computation for bivariate pairs due to one-hot group constraints"
            )

        for i in range(self.input_dim):
            for j in range(i + 1, self.input_dim):
                if self._should_skip_bivariate_pair(i, j):
                    skipped_pairs.append((i, j))
                    logger.debug(
                        f"Skipping bivariate calculation for features {i},{j} "
                        f"(same one-hot group)"
                    )
                else:
                    # Only analyze pairs that will actually be computed
                    pair_data = self._analyze_single_pair(x, i, j, threshold, force_caching)
                    pair_info.append(pair_data)

        if skipped_pairs:
            logger.info(f"Skipped {len(skipped_pairs)} same-group bivariate pairs")

        return pair_info

    def _compute_collapsed_pair_responses(
        self,
        collapsed_i: int,
        collapsed_j: int,
        x_grid_i: torch.Tensor,
        x_grid_j: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Compute bivariate responses for a collapsed feature pair.

        Handles all combinations: onehot x onehot, onehot x continuous, continuous x continuous

        Parameters
        ----------
        collapsed_i, collapsed_j : int
            Collapsed feature indices
        x_grid_i, x_grid_j : torch.Tensor
            Grid values for each feature (integers for one-hot, continuous for others)
        batch_size : int
            Batch size for processing

        Returns
        -------
        torch.Tensor
            Bivariate responses for all grid combinations
        """
        original_indices_i = self.index_mapping[collapsed_i]
        original_indices_j = self.index_mapping[collapsed_j]
        is_onehot_i = len(original_indices_i) > 1
        is_onehot_j = len(original_indices_j) > 1

        # Create cartesian product of grids
        x_ij = torch.cartesian_prod(x_grid_i, x_grid_j)
        n_unique = len(x_ij)
        n_train_samples = self.x_train.shape[0]
        unique_responses = torch.zeros(n_unique, device=self.device)

        for batch_start in range(0, n_unique, batch_size):
            batch_end = min(batch_start + batch_size, n_unique)
            batch_size_current = batch_end - batch_start

            # Create modified versions of training data
            x_modified = self.x_train.repeat(batch_size_current, 1)

            # Process each pair in batch
            for batch_idx in range(batch_size_current):
                pair_idx = batch_start + batch_idx
                value_i = x_ij[pair_idx, 0].item()
                value_j = x_ij[pair_idx, 1].item()
                start_row = batch_idx * n_train_samples
                end_row = start_row + n_train_samples

                # Expand collapsed feature i
                if is_onehot_i:
                    cat_idx_i = int(value_i)
                    if cat_idx_i == 0:
                        for orig_idx in original_indices_i:
                            scaled_0 = self._get_onehot_value(orig_idx, 0.0)
                            x_modified[start_row:end_row, orig_idx] = scaled_0
                    else:
                        active_idx = original_indices_i[cat_idx_i - 1]
                        for orig_idx in original_indices_i:
                            if orig_idx == active_idx:
                                scaled_1 = self._get_onehot_value(orig_idx, 1.0)
                                x_modified[start_row:end_row, orig_idx] = scaled_1
                            else:
                                scaled_0 = self._get_onehot_value(orig_idx, 0.0)
                                x_modified[start_row:end_row, orig_idx] = scaled_0
                else:
                    x_modified[start_row:end_row, original_indices_i[0]] = value_i

                # Expand collapsed feature j
                if is_onehot_j:
                    cat_idx_j = int(value_j)
                    if cat_idx_j == 0:
                        for orig_idx in original_indices_j:
                            scaled_0 = self._get_onehot_value(orig_idx, 0.0)
                            x_modified[start_row:end_row, orig_idx] = scaled_0
                    else:
                        active_idx = original_indices_j[cat_idx_j - 1]
                        for orig_idx in original_indices_j:
                            if orig_idx == active_idx:
                                scaled_1 = self._get_onehot_value(orig_idx, 1.0)
                                x_modified[start_row:end_row, orig_idx] = scaled_1
                            else:
                                scaled_0 = self._get_onehot_value(orig_idx, 0.0)
                                x_modified[start_row:end_row, orig_idx] = scaled_0
                else:
                    x_modified[start_row:end_row, original_indices_j[0]] = value_j

            # Get predictions and convert to logit space
            y_pred = self._batched_predict(x_modified)
            logits = stable_logit(y_pred)

            # Calculate average response for each pair
            batch_responses = logits.reshape(batch_size_current, n_train_samples).mean(dim=1)

            # Store responses
            unique_responses[batch_start:batch_end] = batch_responses - self.logit_y0

            # Force computation to free memory
            _free_all_gpu_caches()

        return unique_responses

    def _compute_unique_pair_responses(
        self,
        pair_info: Dict,
        batch_size: int,
    ) -> torch.Tensor:
        """Compute responses for unique feature pairs."""
        i, j = pair_info['indices']
        unique_pairs = pair_info['unique_pairs']
        n_unique = len(unique_pairs)
        n_train_samples = self.x_train.shape[0]
        unique_responses = torch.zeros(n_unique, device=self.device)

        for batch_start in range(0, n_unique, batch_size):
            batch_end = min(batch_start + batch_size, n_unique)
            batch_size_current = batch_end - batch_start

            # Create modified versions of training data
            x_modified = self.x_train.repeat(batch_size_current, 1)

            # Replace feature values with current batch of unique pairs
            feature_values_i = unique_pairs[batch_start:batch_end, 0].repeat_interleave(
                n_train_samples
            )
            feature_values_j = unique_pairs[batch_start:batch_end, 1].repeat_interleave(
                n_train_samples
            )
            x_modified[:, i] = feature_values_i
            x_modified[:, j] = feature_values_j

            # Handle one-hot group constraints
            if self.onehot_groups:
                # Check if i and j belong to the same group
                same_group = None
                group_i = None
                group_j = None

                for onehot_group in self.onehot_groups:
                    if i in onehot_group:
                        group_i = onehot_group
                    if j in onehot_group:
                        group_j = onehot_group

                if group_i is not None and group_j is not None and group_i == group_j:
                    # Both features are in the same group
                    same_group = group_i
                    sibling_columns = [idx for idx in same_group if idx not in [i, j]]
                    # Only keep siblings when both i and j are zero
                    both_zero_mask = (feature_values_i == 0) & (feature_values_j == 0)
                    # Zero siblings when either i or j is non-zero
                    for sibling_col in sibling_columns:
                        _assign_column_masked(x_modified, ~both_zero_mask, sibling_col, 0)
                else:
                    # Features are in different groups or not in groups - handle independently
                    if group_i is not None:
                        # For feature i: handle both value=0 (reference) and value=1 (active)
                        zero_mask_i = feature_values_i == 0
                        non_zero_mask_i = feature_values_i != 0

                        # Set all columns in group to 0 for reference state (value=0)
                        for col_idx in group_i:
                            _assign_column_masked(x_modified, zero_mask_i, col_idx, 0)

                        # Set only sibling columns to 0 for active state (value=1)
                        sibling_columns_i = [idx for idx in group_i if idx != i]
                        for sibling_col in sibling_columns_i:
                            _assign_column_masked(x_modified, non_zero_mask_i, sibling_col, 0)

                    if group_j is not None:
                        # For feature j: handle both value=0 (reference) and value=1 (active)
                        zero_mask_j = feature_values_j == 0
                        non_zero_mask_j = feature_values_j != 0

                        # Set all columns in group to 0 for reference state (value=0)
                        for col_idx in group_j:
                            _assign_column_masked(x_modified, zero_mask_j, col_idx, 0)

                        # Set only sibling columns to 0 for active state (value=1)
                        sibling_columns_j = [idx for idx in group_j if idx != j]
                        for sibling_col in sibling_columns_j:
                            _assign_column_masked(x_modified, non_zero_mask_j, sibling_col, 0)

            # Get predictions and convert to logit space
            y_pred = self._batched_predict(x_modified)
            logits = stable_logit(y_pred)

            # Calculate average response for each unique pair
            batch_responses = logits.reshape(batch_size_current, n_train_samples).mean(dim=1)

            # Store responses
            unique_responses[batch_start:batch_end] = batch_responses - self.logit_y0

            # Force computation to free memory
            _free_all_gpu_caches()

        return unique_responses

    def _process_bivariate_batch(
        self,
        x: torch.Tensor,
        pair_info: Dict,
        univariate_responses: torch.Tensor,
        batch_start: int,
        batch_size: int,
        n_samples: int,
    ) -> Tuple[Tuple[int, int], int, torch.Tensor]:
        """Process a batch of bivariate responses with optional caching."""
        i, j = pair_info['indices']
        batch_end = min(batch_start + batch_size, n_samples)
        batch_size_current = batch_end - batch_start

        if pair_info['cache_optimized']:
            if batch_start == 0:  # Only compute unique responses once
                unique_responses = self._compute_unique_pair_responses(pair_info, batch_size)
                # Store for reuse by other batches
                pair_info['unique_responses'] = unique_responses

            # Map unique responses to current batch
            batch_indices = pair_info['inverse_indices'][batch_start:batch_end]
            batch_responses = pair_info['unique_responses'][batch_indices]

        else:
            # Original direct computation
            x_modified = self.x_train.repeat(batch_size_current, 1)

            # Apply scaling for one-hot columns if needed
            if hasattr(self, 'group_manager') and self.group_manager is not None:
                # Check if feature i is part of a one-hot group
                feature_info_i = self.group_manager.get_feature_info(i)
                if (
                    feature_info_i
                    and feature_info_i.is_categorical_group
                    and len(feature_info_i.indices) > 1
                ):
                    # This is a one-hot column, apply scaling to the raw values
                    raw_vals_i = x[batch_start:batch_end, i]
                    scaled_vals_i = torch.zeros_like(raw_vals_i)
                    for v_idx, raw_val in enumerate(raw_vals_i):
                        if raw_val != 0:
                            scaled_vals_i[v_idx] = self._get_onehot_value(i, float(raw_val.item()))
                    x_modified[:, i] = scaled_vals_i.repeat_interleave(self.x_train.shape[0])
                else:
                    x_modified[:, i] = x[batch_start:batch_end, i].repeat_interleave(
                        self.x_train.shape[0]
                    )
            else:
                x_modified[:, i] = x[batch_start:batch_end, i].repeat_interleave(
                    self.x_train.shape[0]
                )

            # Apply scaling for one-hot columns if needed
            if hasattr(self, 'group_manager') and self.group_manager is not None:
                # Check if feature j is part of a one-hot group
                feature_info_j = self.group_manager.get_feature_info(j)
                if (
                    feature_info_j
                    and feature_info_j.is_categorical_group
                    and len(feature_info_j.indices) > 1
                ):
                    # This is a one-hot column, apply scaling to the raw values
                    raw_vals_j = x[batch_start:batch_end, j]
                    scaled_vals_j = torch.zeros_like(raw_vals_j)
                    for v_idx, raw_val in enumerate(raw_vals_j):
                        if raw_val != 0:
                            scaled_vals_j[v_idx] = self._get_onehot_value(j, float(raw_val.item()))
                    x_modified[:, j] = scaled_vals_j.repeat_interleave(self.x_train.shape[0])
                else:
                    x_modified[:, j] = x[batch_start:batch_end, j].repeat_interleave(
                        self.x_train.shape[0]
                    )
            else:
                x_modified[:, j] = x[batch_start:batch_end, j].repeat_interleave(
                    self.x_train.shape[0]
                )

            y_pred = self._batched_predict(x_modified)
            logits = stable_logit(y_pred)
            batch_responses = logits.reshape(batch_size_current, -1).mean(dim=1) - self.logit_y0

        # Subtract univariate contributions
        batch_responses -= (
            univariate_responses[batch_start:batch_end, i]
            + univariate_responses[batch_start:batch_end, j]
        )

        return (i, j), batch_start, batch_responses

    def _calculate_lebesgue(
        self,
        x: torch.Tensor,
        batch_size: int = 1024,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
        """
        Calculate partial responses using the Lebesgue method.

        The Lebesgue method computes the effect of each feature value (univariate) and feature
        value pair (bivariate) by averaging the model's output over the distribution of other
        features in the training data.

        Parameters
        ----------
        x : torch.Tensor
            Input data tensor of shape (n_samples, n_features).
        batch_size : int, optional
            Size of batches for processing large datasets.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]]]
            - univariate_responses: Tensor of shape (n_samples, n_features) containing
            the partial responses for each feature value in each sample.
            - bivariate_responses: Tensor of shape (n_samples, n_bivariate) containing
            the partial responses for each feature value pair in each sample.
            - List of feature index pairs corresponding to the bivariate responses.

        Notes
        -----
        The Lebesgue method calculates partial responses as follows:
        1. Univariate responses
        - For each feature and ach sample:
            - Create a copy of the training data, replacing the feature's values with
            the current sample's value.
            - Calculate the model's output for all these modified trainin data samples.
            - Average the all outputs and compute the difference from the baseline (logit_y0).

        2. Bivariate responses
        - Similar to univariat, but replacing two features' values simultaneously.
        - Subtract univariate responses to isolate the interaction effect.

        Vectorization and Performance Optimization
        - The method uses vectorized operations to process multiple samples
          simultaneously, significantly improving computation speed, especially on GPUs.
        - Batching is employed to handle large datasets efficiently.
        """
        n_features = x.shape[1]
        n_samples = x.shape[0]
        x = x.to(self.device)
        total_start_time = time.time()

        # Preallocate tensors for responses
        univariate_responses = torch.zeros((n_samples, n_features), device=self.device)
        n_bivariate = n_features * (n_features - 1) // 2
        bivariate_responses = torch.zeros((n_samples, n_bivariate), device=self.device)

        # Analyze features for caching potential
        feature_info = self._analyze_cardinality(x)

        try:
            # Log resource status
            logger.info(f"Compute device: {self.device}")
            logger.info(f"Batch size: {batch_size}")

            # Process univariate responses sequentially
            logger.info(f"Processing {n_features} univariate responses")
            logger.debug("-" * 50)

            # Track timing per feature
            feature_times = {i: {'start': None, 'end': None} for i in range(n_features)}

            for i in range(n_features):
                feature_times[i]['start'] = time.time()

                for batch_start in range(0, n_samples, batch_size):
                    i, batch_start, response = self._process_univariate_batch(
                        x, i, feature_info[i], batch_start, batch_size, n_samples
                    )
                    batch_end = min(batch_start + batch_size, n_samples)
                    univariate_responses[batch_start:batch_end, i] = response

                # Record completion time for feature
                feature_times[i]['end'] = time.time()
                processing_time = feature_times[i]['end'] - feature_times[i]['start']
                if feature_info[i]['cache_optimized']:
                    logger.debug(f"Feature {i}: {processing_time:.2f}s using cached responses")
                else:
                    logger.debug(f"Feature {i}: {processing_time:.2f}s using direct computation")

                # Force computation to free memory
                _free_all_gpu_caches()

            univariate_end_time = time.time()
            logger.info(
                f"Total univariate processing time: {univariate_end_time - total_start_time:.2f} seconds"
            )

            # Analyze and process bivariate pairs (only non-skipped pairs)
            pair_info = self._analyze_bivariate_pairs(x)

            logger.info(f"Processing {len(pair_info)} bivariate responses")
            logger.debug("-" * 50)

            # Track timing per pair
            pair_times = {idx: {'start': None, 'end': None} for idx in range(len(pair_info))}

            # Process bivariate responses sequentially (only for non-skipped pairs)
            for pair_idx, pair_info_dict in enumerate(pair_info):
                pair_times[pair_idx]['start'] = time.time()
                i, j = pair_info_dict['indices']

                for batch_start in range(0, n_samples, batch_size):
                    (_, _), _, response = self._process_bivariate_batch(
                        x,
                        pair_info_dict,
                        univariate_responses,
                        batch_start,
                        batch_size,
                        n_samples,
                    )
                    batch_end = min(batch_start + batch_size, n_samples)

                    # Map to lower triangular matrix position
                    idx = i * n_features + j - ((i + 2) * (i + 1)) // 2
                    bivariate_responses[batch_start:batch_end, idx] = response

                # Record completion time for this feature pair
                pair_times[pair_idx]['end'] = time.time()
                processing_time = pair_times[pair_idx]['end'] - pair_times[pair_idx]['start']
                if pair_info[pair_idx]['cache_optimized']:
                    logger.debug(
                        f"Features {i},{j}: {processing_time:.2f}s using cached responses"
                    )
                else:
                    logger.debug(
                        f"Features {i},{j}: {processing_time:.2f}s using direct computation"
                    )

            total_end_time = time.time()
            logger.info(
                f"Total bivariate processing time: {total_end_time - total_start_time:.2f} seconds"
            )

        finally:
            # Clean up GPU memory if needed
            _free_all_gpu_caches()

        # Note: Collapsing is now applied in calculate() to work for both methods
        return univariate_responses, bivariate_responses

    def _calculate_dirac_subset(
        self,
        x: torch.Tensor,
        n_steps: int,
        categorical_threshold: int,
        subtract_univariate: bool,
        selected_features: Optional[List[int]] = None,
        selected_feature_pairs: Optional[List[Tuple[int, int]]] = None,
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
    ]:
        """
        Calculate partial responses for value subsets using the Dirac method.

        The Dirac method evaluates the model at specific grid points while holding
        other features at their baseline values. This method is hosted in LebesgueMixin
        to share the infrastructure for grid generation, one-hot group expansion,
        and collapsed mode handling that is common to both methodologies.

        Parameters
        ----------
        x : torch.Tensor
            Input data tensor.
        n_steps : int
            Number of steps for continuous feature grids.
        categorical_threshold : int
            Threshold for treating features as categorical.
        subtract_univariate : bool
            Whether to subtract univariate effects from bivariate responses.
        selected_features : Optional[List[int]]
            Indices of features to process. If None, processes all features.
        selected_feature_pairs : Optional[List[Tuple[int, int]]]
            Pairs of feature indices to process. If None, processes all pairs.

        Returns
        -------
        Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]
            - univariate_responses: List of tensors containing univariate responses.
            - bivariate_responses: List of tensors containing bivariate responses.
            - x_univariate: List of tensors containing univariate grid values.
            - x_bivariate: List of tensors containing bivariate grid values.
        """
        n_features_original = x.shape[1]
        x = x.to(self.device)

        # Determine if we're in collapsed mode
        in_collapsed_mode = self._is_collapsed_mode()

        if in_collapsed_mode:
            n_features = self.n_collapsed_features
        else:
            n_features = n_features_original

        # --- Feature Selection Logic (Same as Lebesgue) ---
        features_to_process = set(selected_features if selected_features is not None else [])
        if selected_feature_pairs is not None:
            for i, j in selected_feature_pairs:
                features_to_process.add(i)
                features_to_process.add(j)

        selected_features = sorted(list(features_to_process))
        if not selected_features:
            selected_features = list(range(n_features))

        if selected_feature_pairs is None:
            feature_pairs = [(i, j) for i in selected_features for j in selected_features if i < j]
        else:
            feature_pairs = selected_feature_pairs

        # Filter pairs
        valid_pairs = []
        skipped_pairs = []
        for i, j in feature_pairs:
            should_skip = False if in_collapsed_mode else self._should_skip_bivariate_pair(i, j)
            if should_skip:
                skipped_pairs.append((i, j))
            else:
                valid_pairs.append((i, j))
        feature_pairs = valid_pairs
        # --------------------------------------------------

        univariate_responses = [None] * n_features
        x_univariate = [None] * n_features

        # Calculate Univariate Responses
        for i in selected_features:
            if in_collapsed_mode:
                x_subset = self._get_collapsed_feature_grid(i, x, n_steps, categorical_threshold)
            else:
                x_subset = get_variable_range(
                    x[:, i], n_steps, categorical_threshold, self.trim_quantile
                )

            x_univariate[i] = x_subset

            # Construct input for model
            # Start with baseline values (Dirac baseline)
            x_input = self._create_baseline_input(len(x_subset), n_features_original)

            if in_collapsed_mode:
                # Expand collapsed values to original features
                original_indices = self.index_mapping[i]

                is_group = len(original_indices) > 1
                if not is_group and self.group_manager is not None:
                    feature_name = self.collapsed_feature_names[i]
                    is_group = self.group_manager.is_categorical_group(feature_name)

                if is_group:  # One-hot group
                    # x_subset contains integers 0..K
                    for row_idx, val in enumerate(x_subset):
                        cat_idx = int(val.item())
                        if cat_idx > 0:
                            # Active category: set corresponding original col to scaled value of 1
                            # cat_idx 1 -> original_indices[0]
                            active_col = original_indices[cat_idx - 1]
                            x_input[row_idx, active_col] = self._get_onehot_value(active_col, 1.0)
                else:  # Continuous
                    orig_idx = original_indices[0]
                    x_input[:, orig_idx] = x_subset
            else:
                x_input[:, i] = x_subset

            y_i = self.predict(x_input)
            response = stable_logit(y_i) - self.logit_y0
            univariate_responses[i] = response

        # Calculate Bivariate Responses
        max_bivariate_idx = n_features * (n_features - 1) // 2
        bivariate_responses = [None] * max_bivariate_idx
        x_bivariate = [None] * max_bivariate_idx

        # Handle skipped pairs
        for i, j in skipped_pairs:
            pair_idx = i * n_features + j - ((i + 2) * (i + 1)) // 2
            bivariate_responses[pair_idx] = torch.zeros(1, device=self.device)
            x_bivariate[pair_idx] = torch.zeros((1, 2), device=self.device)

        for i, j in feature_pairs:
            pair_idx = i * n_features + j - ((i + 2) * (i + 1)) // 2

            if in_collapsed_mode:
                x_subset_i = self._get_collapsed_feature_grid(i, x, n_steps, categorical_threshold)
                x_subset_j = self._get_collapsed_feature_grid(j, x, n_steps, categorical_threshold)
            else:
                x_subset_i = get_variable_range(
                    x[:, i], n_steps, categorical_threshold, self.trim_quantile
                )
                x_subset_j = get_variable_range(
                    x[:, j], n_steps, categorical_threshold, self.trim_quantile
                )

            x_ij = torch.cartesian_prod(x_subset_i, x_subset_j)
            x_bivariate[pair_idx] = x_ij

            # Construct input with baseline values
            x_input = self._create_baseline_input(len(x_ij), n_features_original)

            # Apply feature i
            if in_collapsed_mode:
                orig_indices_i = self.index_mapping[i]

                is_group_i = len(orig_indices_i) > 1
                if not is_group_i and self.group_manager is not None:
                    feature_name = self.collapsed_feature_names[i]
                    is_group_i = self.group_manager.is_categorical_group(feature_name)

                if is_group_i:  # One-hot
                    for row_idx, val in enumerate(x_ij[:, 0]):
                        cat_idx = int(val.item())
                        if cat_idx > 0:
                            active_col_i = orig_indices_i[cat_idx - 1]
                            scaled_val = self._get_onehot_value(active_col_i, 1.0)
                            x_input[row_idx, active_col_i] = scaled_val
                            # Debug: Track when we use scaled values
                            if hasattr(self, '_debug_count'):
                                self._debug_count += 1
                            else:
                                self._debug_count = 1
                            if self._debug_count <= 10:  # Only log first 10
                                logger.info(f"DEBUG: Set {active_col_i} to {scaled_val} (raw 1.0)")
                else:
                    x_input[:, orig_indices_i[0]] = x_ij[:, 0]
            else:
                x_input[:, i] = x_ij[:, 0]

            # Apply feature j
            if in_collapsed_mode:
                orig_indices_j = self.index_mapping[j]

                is_group_j = len(orig_indices_j) > 1
                if not is_group_j and self.group_manager is not None:
                    feature_name = self.collapsed_feature_names[j]
                    is_group_j = self.group_manager.is_categorical_group(feature_name)

                if is_group_j:  # One-hot
                    for row_idx, val in enumerate(x_ij[:, 1]):
                        cat_idx = int(val.item())
                        if cat_idx > 0:
                            active_col_j = orig_indices_j[cat_idx - 1]
                            x_input[row_idx, active_col_j] = self._get_onehot_value(
                                active_col_j, 1.0
                            )
                else:
                    x_input[:, orig_indices_j[0]] = x_ij[:, 1]
            else:
                x_input[:, j] = x_ij[:, 1]

            y_ij = self.predict(x_input)
            bivariate_response = stable_logit(y_ij) - self.logit_y0

            if subtract_univariate:
                # Need to map values to indices in univariate response
                # For continuous, use searchsorted. For categorical (integers), use values as indices?
                # Actually searchsorted works for sorted integers too.

                # Note: x_subset_i/j are sorted.

                idx_i = torch.searchsorted(x_subset_i, x_ij[:, 0])
                idx_j = torch.searchsorted(x_subset_j, x_ij[:, 1])

                bivariate_response -= (
                    univariate_responses[i][idx_i] + univariate_responses[j][idx_j]
                )

            bivariate_responses[pair_idx] = bivariate_response

        return univariate_responses, bivariate_responses, x_univariate, x_bivariate

    def _calculate_lebesgue_subset(
        self,
        x: torch.Tensor,
        n_steps: int,
        categorical_threshold: int,
        subtract_univariate: bool,
        batch_size: int,
        selected_features: Optional[List[int]] = None,
        selected_feature_pairs: Optional[List[Tuple[int, int]]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Calculate partial responses for value subsets using Lebesgue method.
        Since n_steps constrains cardinality, all features use cached computation.

        If a feature appears in selected_feature_pairs but not in selected_features,
        it will be automatically added to ensure all necessary univariate responses
        are calculated.

        Parameters
        ----------
        batch_size : int
            Size of batches for processing large datasets
        selected_features : Optional[List[int]]
            List of feature indices to calculate univariate responses for.
            If None, calculates for all features.
        selected_feature_pairs : Optional[List[Tuple[int, int]]]
            List of feature index pairs to calculate bivariate responses for.
            If None, calculates for all feature pairs using selected_features.
        """
        n_features_original = x.shape[1]  # Original (uncollapsed) feature count
        x = x.to(self.device)

        # Determine if we're in collapsed mode
        in_collapsed_mode = self._is_collapsed_mode()

        if in_collapsed_mode:
            # In collapsed mode: selected_features are collapsed indices
            n_features = self.n_collapsed_features
            logger.info(
                f"Operating in collapsed mode: {n_features_original} -> {n_features} features"
            )
        else:
            # Normal mode: selected_features are original indices
            n_features = n_features_original

        # Start with initial selected features or empty list
        features_to_process = set(selected_features if selected_features is not None else [])

        # Add features from pairs if provided
        if selected_feature_pairs is not None:
            # Validate provided pairs
            for i, j in selected_feature_pairs:
                if not (0 <= i < n_features and 0 <= j < n_features):
                    raise ValueError(f"Feature pair ({i},{j}) out of range [0, {n_features-1}]")
                # Add both features from each pair
                features_to_process.add(i)
                features_to_process.add(j)

        # Convert to sorted list
        selected_features = sorted(list(features_to_process))

        # If no features selected, use all features
        if not selected_features:
            selected_features = list(range(n_features))

        # Determine feature pairs to process
        if selected_feature_pairs is None:
            # Generate all pairs from selected features
            feature_pairs = [(i, j) for i in selected_features for j in selected_features if i < j]
        else:
            feature_pairs = selected_feature_pairs

        # Filter out same-group pairs
        valid_pairs = []
        skipped_pairs = []
        for i, j in feature_pairs:
            # In collapsed mode, i and j are collapsed indices, so they are never in the same group
            # (unless i == j, but we ensure i < j)
            should_skip = False if in_collapsed_mode else self._should_skip_bivariate_pair(i, j)

            if should_skip:
                skipped_pairs.append((i, j))
                logger.info(f"Skipping bivariate subset for features {i},{j} (same one-hot group)")
            else:
                valid_pairs.append((i, j))

        feature_pairs = valid_pairs

        # Initialize response lists with None placeholders for all features/pairs
        univariate_responses = [None] * n_features
        x_univariate = [None] * n_features

        logger.info("Processing univariate response subset")
        logger.debug("-" * 50)

        # Process selected univariate responses
        for i in selected_features:
            if in_collapsed_mode:
                # Collapsed mode: i is a collapsed index
                original_indices = self.index_mapping[i]

                if len(original_indices) > 1:
                    # One-hot group: generate integer grid and compute collapsed responses
                    x_subset = self._get_collapsed_feature_grid(
                        i, x, n_steps, categorical_threshold
                    )
                    x_univariate[i] = x_subset
                    response = self._compute_collapsed_onehot_responses(i, batch_size)
                    univariate_responses[i] = response
                    logger.debug(
                        f"Collapsed feature {i} (one-hot): computed responses for {len(x_subset)} categories"
                    )
                else:
                    # Single continuous feature: use standard approach
                    orig_idx = original_indices[0]
                    x_subset = get_variable_range(
                        x[:, orig_idx], n_steps, categorical_threshold, self.trim_quantile
                    )
                    x_univariate[i] = x_subset
                    response = self._compute_unique_responses(orig_idx, x_subset, batch_size)
                    univariate_responses[i] = response
                    logger.debug(
                        f"Collapsed feature {i} (continuous): computed responses for {len(x_subset)} values"
                    )
            else:
                # Normal mode: i is an original feature index
                x_subset = get_variable_range(
                    x[:, i], n_steps, categorical_threshold, self.trim_quantile
                )
                x_univariate[i] = x_subset
                response = self._compute_unique_responses(i, x_subset, batch_size)
                univariate_responses[i] = response
                logger.debug(f"Feature {i}: computed responses for {len(x_subset)} values")

        # Initialize bivariate response lists
        max_bivariate_idx = n_features * (n_features - 1) // 2
        bivariate_responses = [None] * max_bivariate_idx
        x_bivariate = [None] * max_bivariate_idx

        # Set skipped pairs to zero tensors
        for i, j in skipped_pairs:
            pair_idx = i * n_features + j - ((i + 2) * (i + 1)) // 2
            bivariate_responses[pair_idx] = torch.zeros(1, device=self.device)  # Single zero
            x_bivariate[pair_idx] = torch.zeros((1, 2), device=self.device)  # Dummy x values

        if feature_pairs:  # Only process if we have pairs to compute
            logger.info(f"Processing {len(feature_pairs)} bivariate response subset")
            logger.debug("-" * 50)

            # Process each selected feature pair
            for i, j in feature_pairs:
                logger.debug(f"Processing feature pair ({i}, {j})")

                # Calculate flat index for (i,j) pair
                pair_idx = i * n_features + j - ((i + 2) * (i + 1)) // 2

                if in_collapsed_mode:
                    # Collapsed mode: generate grids for collapsed features
                    x_subset_i = self._get_collapsed_feature_grid(
                        i, x, n_steps, categorical_threshold
                    )
                    x_subset_j = self._get_collapsed_feature_grid(
                        j, x, n_steps, categorical_threshold
                    )

                    # Store grid values
                    x_ij = torch.cartesian_prod(x_subset_i, x_subset_j)
                    x_bivariate[pair_idx] = x_ij

                    # Compute collapsed bivariate responses
                    bivariate_response = self._compute_collapsed_pair_responses(
                        i, j, x_subset_i, x_subset_j, batch_size
                    )

                    if subtract_univariate:
                        # Get indices into univariate responses
                        idx_i = torch.arange(len(x_subset_i)).repeat_interleave(len(x_subset_j))
                        idx_j = torch.arange(len(x_subset_j)).repeat(len(x_subset_i))

                        # Subtract univariate contributions
                        bivariate_response -= (
                            univariate_responses[i][idx_i] + univariate_responses[j][idx_j]
                        )

                    bivariate_responses[pair_idx] = bivariate_response
                    logger.debug(
                        f"Collapsed features {i},{j}: computed responses for {len(x_ij)} value pairs"
                    )
                else:
                    # Normal mode: get grids from original features
                    x_subset_i = get_variable_range(
                        x[:, i], n_steps, categorical_threshold, self.trim_quantile
                    )
                    x_subset_j = get_variable_range(
                        x[:, j], n_steps, categorical_threshold, self.trim_quantile
                    )

                    # Create all combinations
                    x_ij = torch.cartesian_prod(x_subset_i, x_subset_j)
                    x_bivariate[pair_idx] = x_ij

                    # Create pair info structure
                    pair_info = {'indices': (i, j), 'unique_pairs': x_ij}

                    # Compute responses for all pairs
                    bivariate_response = self._compute_unique_pair_responses(pair_info, batch_size)

                    if subtract_univariate:
                        # We should always have univariate responses since we added features from pairs
                        # Get indices into univariate responses
                        idx_i = torch.arange(len(x_subset_i)).repeat_interleave(len(x_subset_j))
                        idx_j = torch.arange(len(x_subset_j)).repeat(len(x_subset_i))

                        # Subtract univariate contributions
                        bivariate_response -= (
                            univariate_responses[i][idx_i] + univariate_responses[j][idx_j]
                        )

                    bivariate_responses[pair_idx] = bivariate_response
                    logger.debug(
                        f"Features {i},{j}: computed responses for {len(x_ij)} value pairs"
                    )

        return univariate_responses, bivariate_responses, x_univariate, x_bivariate
