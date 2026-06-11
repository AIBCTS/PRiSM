"""
CSV export utilities for nomogram data.

This module provides functions for saving nomogram data to CSV files,
using the PlottingDataBundle from the new plotting architecture.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from prism.config import MODELS_DIR

if TYPE_CHECKING:
    from prism.lasso import LassoResultsManager
    from prism.plotting_data import PlottingDataBundle

logger = logging.getLogger(__name__)


def save_nomogram_csv(
    bundle: 'PlottingDataBundle',
    lasso_results: 'LassoResultsManager',
    model_info: Optional[Dict[str, Any]] = None,
    file_path: Optional[Path] = None,
    use_odds_ratio: bool = False,
) -> Tuple[Path, Path]:
    """Save nomogram data to CSV files using the new PlottingDataBundle.

    Saves denormalized x-values and partial responses for both univariate
    and bivariate features to separate CSV files with metadata headers.

    Parameters
    ----------
    bundle : PlottingDataBundle
        Bundle containing partial responses and feature data (x-values already denormalized)
    lasso_results : LassoResultsManager
        LASSO results for extracting intercept and base model name
    model_info : Optional[Dict[str, Any]]
        Dictionary containing model metadata (e.g., 'comment', 'method')
    file_path : Optional[Path]
        Base path for saving CSV files. If None, auto-generates based on timestamp.
    use_odds_ratio : bool
        Whether responses are in odds ratio scale (affects metadata only)

    Returns
    -------
    Tuple[Path, Path]
        Paths to the saved univariate and bivariate CSV files

    Examples
    --------
    >>> univariate_path, bivariate_path = save_nomogram_csv(
    ...     bundle=plotting_bundle,
    ...     lasso_results=lasso_results,
    ...     model_info={'comment': 'PRN Nomogram', 'method': 'lebesgue'},
    ... )
    """
    model_info = model_info or {}

    # Generate file paths
    if file_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        nomogram_dir = Path(MODELS_DIR) / "nomogram"

        # Include base model name in directory and filename if available
        base_model = lasso_results.base_model_name
        if base_model:
            nomogram_dir = nomogram_dir / base_model
            base_name = f"nomogram_{base_model}_{timestamp}"
        else:
            base_name = f"nomogram_{timestamp}"

        nomogram_dir.mkdir(parents=True, exist_ok=True)
        file_path = nomogram_dir / f"{base_name}.csv"
    else:
        file_path = Path(file_path)

    univariate_file_path = file_path.with_name(file_path.stem + "_univariate.csv")
    bivariate_file_path = file_path.with_name(file_path.stem + "_bivariate.csv")

    def write_metadata(fpath: Path, data_type: str):
        """Write metadata header to CSV file."""
        # Get the intercept from the selected model
        model = lasso_results.get_selected_model()
        intercept = model.intercept_[0]

        # Format intercept display
        if use_odds_ratio:
            intercept_display = f"Intercept (beta_0) odds ratio: {np.exp(intercept):.6f}"
        else:
            intercept_display = f"Intercept (beta_0) log odds: {intercept:.6f}"

        metadata = [
            f"# Nomogram Data - {data_type}",
            f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"# Comment: {model_info.get('comment', 'Unknown')}",
            f"# Method: {model_info.get('method', 'Unknown')}",
            f"# Base model: {lasso_results.base_model_name or 'Unknown'}",
            f"# Number of discretization steps: {bundle.n_steps}",
            f"# Categorical threshold: {bundle.categorical_threshold}",
            f"# Response scale: {'Odds ratio' if use_odds_ratio else 'Log odds ratio'}",
            f"# {intercept_display}",
            "",
        ]
        with open(fpath, 'w') as f:
            f.write('\n'.join(metadata))

    # Process univariate data
    univariate_data = {}
    max_length = 0

    for info in bundle.univariate_features():
        feature_name = info.name
        # x_values are already denormalized in the bundle
        x_values = info.x_values if info.x_values is not None else np.array([])
        responses = info.response if info.response is not None else np.array([])

        # Flatten arrays if needed
        x_values = np.asarray(x_values).flatten()
        responses = np.asarray(responses).flatten()

        max_length = max(max_length, len(x_values), len(responses))
        univariate_data[f"{feature_name}_x"] = x_values
        univariate_data[f"{feature_name}_response"] = responses

    # Pad univariate data to equal length
    for key in univariate_data:
        arr = univariate_data[key]
        if len(arr) < max_length:
            univariate_data[key] = np.pad(
                arr,
                (0, max_length - len(arr)),
                mode='constant',
                constant_values=np.nan,
            )

    # Save univariate data
    write_metadata(univariate_file_path, "Univariate")
    pd.DataFrame(univariate_data).to_csv(univariate_file_path, mode='a', index=False)

    # Process bivariate data
    bivariate_data = {}
    max_length = 0

    for info in bundle.bivariate_pairs():
        if info.skipped:
            continue

        feature1, feature2 = info.names
        pair_name = f"{feature1}_{feature2}"

        # x_values are already denormalized in the bundle
        x_values = info.x_values if info.x_values is not None else np.zeros((0, 2))
        responses = info.response if info.response is not None else np.array([])

        # Ensure proper shape
        x_values = np.asarray(x_values)
        responses = np.asarray(responses).flatten()

        if x_values.ndim == 1:
            # Handle edge case where x_values might be flat
            x_values = x_values.reshape(-1, 2) if len(x_values) > 0 else np.zeros((0, 2))

        max_length = max(max_length, len(x_values), len(responses))
        bivariate_data[f"{pair_name}_x1"] = x_values[:, 0] if len(x_values) > 0 else np.array([])
        bivariate_data[f"{pair_name}_x2"] = x_values[:, 1] if len(x_values) > 0 else np.array([])
        bivariate_data[f"{pair_name}_response"] = responses

    # Pad bivariate data to equal length
    for key in bivariate_data:
        arr = bivariate_data[key]
        if len(arr) < max_length:
            bivariate_data[key] = np.pad(
                arr,
                (0, max_length - len(arr)),
                mode='constant',
                constant_values=np.nan,
            )

    # Save bivariate data
    write_metadata(bivariate_file_path, "Bivariate")
    pd.DataFrame(bivariate_data).to_csv(bivariate_file_path, mode='a', index=False)

    print(f"Univariate nomogram data saved to {univariate_file_path}")
    print(f"Bivariate nomogram data saved to {bivariate_file_path}")

    return univariate_file_path, bivariate_file_path
