"""
Utility functions for partial response calculations.

This module contains pure utility functions that don't depend on the
PartialResponseCalculator class.
"""

import logging
import warnings

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _warn_if_scaled_onehot(x, onehot_groups, tolerance=0.1):
    """
    Check if one-hot encoded columns appear to have been scaled and issue a warning.
    Also checks for invalid "two active" states within one-hot groups.

    One-hot columns should contain only 0s and 1s. If they contain other values,
    they may have been incorrectly scaled, which can cause issues with partial
    response calculations.

    Parameters
    ----------
    x : torch.Tensor or array-like
        Data matrix (potentially scaled)
    onehot_groups : list of tuple
        Groups of column indices representing one-hot encoded features
    tolerance : float
        Tolerance for detecting non-binary values
    """
    if onehot_groups is None:
        return

    # Convert to numpy for analysis
    if isinstance(x, torch.Tensor):
        x_np = x.cpu().numpy()
    else:
        x_np = np.asarray(x)

    # Collect all one-hot column indices
    onehot_indices = []
    for group in onehot_groups:
        onehot_indices.extend(group)

    # Check each column for scaled values
    scaled_cols = []
    for col_idx in onehot_indices:
        if col_idx >= x_np.shape[1]:
            continue

        col = x_np[:, col_idx]
        unique_vals = np.unique(col[~np.isnan(col)])

        # Check if any value is not close to 0 or 1
        is_zero = np.abs(unique_vals) < tolerance
        is_one = np.abs(unique_vals - 1.0) < tolerance

        if not np.all(is_zero | is_one):
            scaled_cols.append(col_idx)

    if scaled_cols:
        warnings.warn(
            f"One-hot encoded columns appear to have been scaled (columns {scaled_cols[:5]}... "
            f"contain values other than 0/1). This can cause incorrect partial response "
            f"calculations. Consider using PRiSMScaler with auto_detect_binary=True to "
            f"exclude binary columns from scaling, or pass unscaled data for one-hot columns.",
            UserWarning,
        )
        logger.warning(
            f"Detected {len(scaled_cols)} potentially scaled one-hot columns. "
            f"Example values in column {scaled_cols[0]}: {np.unique(x_np[:, scaled_cols[0]])[:5]}"
        )

    # Check for "two active" constraint violations (multiple categories active in same group)
    # This is a fast O(n_samples * n_groups) check - negligible compared to PR calculation
    two_active_groups = []
    for group_idx, group in enumerate(onehot_groups):
        valid_indices = [idx for idx in group if idx < x_np.shape[1]]
        if not valid_indices:
            continue

        # Sum across group columns for each sample
        group_sums = x_np[:, valid_indices].sum(axis=1)

        # Check if any sample has sum > 1 (two or more active)
        # Use tolerance to handle scaled values near 1
        if np.any(group_sums > 1.0 + tolerance):
            n_violations = np.sum(group_sums > 1.0 + tolerance)
            two_active_groups.append((group_idx, n_violations))

    if two_active_groups:
        warnings.warn(
            f"One-hot constraint violated: {len(two_active_groups)} group(s) have samples with "
            f"multiple active categories. This indicates a data preprocessing error. "
            f"Groups with violations: {[g[0] for g in two_active_groups[:5]]}",
            UserWarning,
        )
        logger.warning(
            f"Two-active constraint violations detected in {len(two_active_groups)} groups. "
            f"First violation: group {two_active_groups[0][0]} has {two_active_groups[0][1]} "
            f"samples with multiple active categories."
        )


def to_numpy(tensor_or_array):
    """
    Convert a tensor or array-like to a numpy array.

    Parameters
    ----------
    tensor_or_array : torch.Tensor or array-like
        Input data to convert

    Returns
    -------
    np.ndarray
        Numpy array representation
    """
    if isinstance(tensor_or_array, torch.Tensor):
        return tensor_or_array.cpu().numpy()
    return np.asarray(tensor_or_array)


@torch.no_grad()
def stable_logit(y: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Compute logit transformation with numerical stability.

    Parameters
    ----------
    y : torch.Tensor
        Input probabilities in range [0, 1]
    eps : float, optional
        Small constant for numerical stability, by default 1e-7

    Returns
    -------
    torch.Tensor
        Logit transformed values log(y/(1-y))
    """
    # Clamp values to prevent numerical instability
    y_stable = torch.clamp(y, eps, 1 - eps)

    # Compute logit
    return torch.log(y_stable / (1 - y_stable))
