"""
JSON export utilities for nomogram data.

This module provides functions for saving and loading nomogram data to/from JSON files,
using the PlottingDataBundle from the plotting architecture. The JSON format
is human-readable and contains all data necessary to reconstruct nomogram plots.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import numpy as np

from prism import __version__
from prism.config import MODELS_DIR

if TYPE_CHECKING:
    from prism.lasso import LassoResultsManager
    from prism.plotting_data import PlottingDataBundle

logger = logging.getLogger(__name__)

# JSON schema version for future compatibility
JSON_SCHEMA_VERSION = "1.0"


@dataclass
class NomogramData:
    """Loaded nomogram data from JSON file.

    Contains all data needed to reconstruct a nomogram plot without
    requiring the original model, scaler, or LASSO results.

    .. note::
        The ``response`` arrays are already **beta-scaled** (i.e. multiplied
        by the LASSO coefficient). The ``beta`` field is stored for reference
        only and must NOT be applied again when reconstructing the nomogram.
        To obtain the total logit prediction, sum all response contributions
        and add the model intercept.

    Attributes
    ----------
    version : str
        JSON schema version (e.g., "1.0")
    metadata : Dict[str, Any]
        Metadata including base_model, method, n_steps, categorical_threshold,
        beta_scaled, comment
    model : Dict[str, Any]
        Model information including intercept, selected_lambda, selected_lambda_index
    univariate : Dict[str, Dict]
        Univariate feature data keyed by feature name. Each entry contains:
        index, name, label, is_categorical, x_values, response (beta-scaled),
        beta, histogram, category_labels
    bivariate : Dict[str, Dict]
        Bivariate pair data keyed by "feat1__feat2". Each entry contains:
        indices, names, labels, is_categorical, x_values_1, x_values_2,
        response (beta-scaled), beta, skipped
    file_path : Path
        Path to the source JSON file
    """

    version: str
    metadata: Dict[str, Any]
    model: Dict[str, Any]
    univariate: Dict[str, Dict]
    bivariate: Dict[str, Dict]
    file_path: Path

    @property
    def intercept(self) -> float:
        """Get model intercept (beta_0)."""
        return self.model.get('intercept', 0.0)

    @property
    def n_steps(self) -> int:
        """Get number of discretization steps used."""
        return self.metadata.get('n_steps', 50)

    @property
    def categorical_threshold(self) -> int:
        """Get categorical threshold used."""
        return self.metadata.get('categorical_threshold', 15)

    @property
    def base_model(self) -> str:
        """Get base model name."""
        return self.metadata.get('base_model', 'unknown')

    def get_category_labels(self) -> Dict[str, Dict[str, str]]:
        """Extract all category labels from univariate features.

        Returns
        -------
        Dict[str, Dict[str, str]]
            Mapping of feature_name -> {category_value: label}
        """
        labels = {}
        for feat_name, feat_data in self.univariate.items():
            if 'category_labels' in feat_data and feat_data['category_labels']:
                labels[feat_name] = feat_data['category_labels']
        return labels


def load_nomogram_json(file_path: Union[str, Path]) -> NomogramData:
    """Load nomogram data from a JSON file.

    Loads previously saved nomogram data for inspection or plotting.
    The returned NomogramData contains all information needed to
    reconstruct the nomogram without the original model.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the JSON file to load

    Returns
    -------
    NomogramData
        Loaded nomogram data with all features and metadata

    Raises
    ------
    FileNotFoundError
        If the JSON file does not exist
    ValueError
        If the JSON file has an unsupported schema version

    Examples
    --------
    >>> data = load_nomogram_json("models/nomogram/htx_example_mlp/nomogram.json")
    >>> print(f"Model intercept: {data.intercept}")
    >>> print(f"Number of univariate features: {len(data.univariate)}")
    >>> for name, feat in data.univariate.items():
    ...     print(f"  {name}: {len(feat['x_values'])} points")
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Nomogram JSON file not found: {file_path}")

    with open(file_path, 'r') as f:
        raw_data = json.load(f)

    # Validate version
    version = raw_data.get('version', '1.0')
    if version != JSON_SCHEMA_VERSION:
        logger.warning(
            f"JSON schema version mismatch: file has {version}, "
            f"expected {JSON_SCHEMA_VERSION}. Attempting to load anyway."
        )

    return NomogramData(
        version=version,
        metadata=raw_data.get('metadata', {}),
        model=raw_data.get('model', {}),
        univariate=raw_data.get('univariate', {}),
        bivariate=raw_data.get('bivariate', {}),
        file_path=file_path,
    )


def _convert_to_json_serializable(obj: Any) -> Any:
    """Convert numpy types and other non-serializable types to JSON-serializable format.

    Recursively handles numpy arrays, numpy scalar types, and nested structures
    (dicts, lists, tuples).

    Parameters
    ----------
    obj : Any
        Object to convert (numpy array, scalar, dict, list, or primitive)

    Returns
    -------
    Any
        JSON-serializable version of the input
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: _convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_json_serializable(item) for item in obj]
    return obj


def _compute_histogram_stats(
    data: np.ndarray,
    is_categorical: bool,
    n_bins: int = 20,
) -> Dict[str, Any]:
    """Compute histogram statistics for a feature.

    For continuous features, computes bin edges and counts using numpy histogram.
    For categorical features, computes unique category values and their counts.

    Parameters
    ----------
    data : np.ndarray
        1D array of feature values from x_data
    is_categorical : bool
        Whether the feature is categorical
    n_bins : int
        Number of bins for continuous features (default: 20)

    Returns
    -------
    Dict[str, Any]
        For continuous: {"bin_edges": [...], "counts": [...]}
        For categorical: {"categories": [...], "counts": [...]}
    """
    # Filter out NaN values
    valid_data = data[~np.isnan(data)]

    if len(valid_data) == 0:
        if is_categorical:
            return {"categories": [], "counts": []}
        else:
            return {"bin_edges": [], "counts": []}

    if is_categorical:
        unique_vals, counts = np.unique(valid_data, return_counts=True)
        return {
            "categories": unique_vals.tolist(),
            "counts": counts.tolist(),
        }
    else:
        counts, bin_edges = np.histogram(valid_data, bins=n_bins)
        return {
            "bin_edges": bin_edges.tolist(),
            "counts": counts.tolist(),
        }


def save_nomogram_json(
    bundle: 'PlottingDataBundle',
    lasso_results: 'LassoResultsManager',
    file_path: Optional[Union[str, Path]] = None,
    n_histogram_bins: int = 20,
    comment: Optional[str] = None,
    method: str = "unknown",
    category_labels: Optional[Dict[str, Dict[int, str]]] = None,
) -> Path:
    """Save nomogram data to a JSON file for later reconstruction.

    Creates a self-contained JSON file with all data needed to reconstruct
    the nomogram plot, including partial responses, histogram summaries,
    and model metadata.

    .. important::
        The bundle's ``response`` arrays are expected to already be
        **beta-scaled** (via ``PlottingPipeline.apply_beta_scaling``).
        The raw LASSO ``beta`` coefficient is stored alongside each feature
        for reference/auditability but must **not** be applied again when
        reading the JSON back.

    Parameters
    ----------
    bundle : PlottingDataBundle
        Bundle containing partial responses and feature data.
        Responses must already be beta-scaled; x-values should be
        denormalized.
    lasso_results : LassoResultsManager
        LASSO results for extracting intercept, beta values, and lambda info
    file_path : Optional[Union[str, Path]]
        Path for saving JSON file. If None, auto-generates based on timestamp
        in models/nomogram/{base_model}/ directory.
    n_histogram_bins : int
        Number of bins for continuous feature histograms (default: 20)
    comment : Optional[str]
        Optional user comment to include in metadata
    method : str
        Partial response method used ("dirac" or "lebesgue")
    category_labels : Optional[Dict[str, Dict[int, str]]]
        Optional mapping of feature names to category label dicts.
        E.g., {"diagnosis": {0: "CAD", 1: "Cardiomyopathy"}}

    Returns
    -------
    Path
        Path to the saved JSON file

    Examples
    --------
    >>> from prism.plotting import PlottingPipeline, save_nomogram_json
    >>> pipeline = PlottingPipeline(lasso_results, group_manager, label_manager)
    >>> bundle = pipeline.prepare_plotting_bundle(
    ...     x=x, model=model, scaler=scaler, n_steps=100, trim_quantile=None
    ... )
    >>> bundle = pipeline.apply_beta_scaling(bundle)
    >>> json_path = save_nomogram_json(
    ...     bundle=bundle,
    ...     lasso_results=lasso_results,
    ...     comment='High-resolution nomogram data',
    ...     method='lebesgue',
    ... )
    """
    category_labels = category_labels or {}

    # Generate file path if not provided
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
        file_path = nomogram_dir / f"{base_name}.json"
    else:
        file_path = Path(file_path)
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract model information
    selected_model = lasso_results.get_selected_model()
    intercept = selected_model.intercept_[0]
    selected_lambda_index = lasso_results.selected_lambda_index
    selected_lambda = (
        lasso_results.lambdas[selected_lambda_index] if selected_lambda_index is not None else None
    )

    # Get beta coefficients for all features
    beta_vector = lasso_results.get_selected_beta()

    # Build the JSON structure
    data = {
        "version": JSON_SCHEMA_VERSION,
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "prism_version": __version__,
            "base_model": lasso_results.base_model_name or "unknown",
            "method": method,
            "n_steps": bundle.n_steps,
            "categorical_threshold": bundle.categorical_threshold,
            "beta_scaled": True,
            "comment": comment,
        },
        "model": {
            "intercept": float(intercept),
            "selected_lambda": float(selected_lambda) if selected_lambda is not None else None,
            "selected_lambda_index": (
                int(selected_lambda_index) if selected_lambda_index is not None else None
            ),
        },
        "univariate": {},
        "bivariate": {},
    }

    # Build univariate section
    for info in bundle.univariate_features():
        feature_name = info.name
        feature_idx = info.index

        # Get beta coefficient for this feature
        beta = float(beta_vector[feature_idx]) if feature_idx < len(beta_vector) else None

        # Compute histogram stats if x_data is available
        histogram = {}
        if bundle.x_data is not None and feature_idx < bundle.x_data.shape[1]:
            histogram = _compute_histogram_stats(
                bundle.x_data[:, feature_idx],
                info.is_categorical,
                n_histogram_bins,
            )

        # Build feature entry
        feature_entry = {
            "index": feature_idx,
            "name": feature_name,
            "label": info.label,
            "is_categorical": info.is_categorical,
            "x_values": info.x_values.tolist() if info.x_values is not None else [],
            "response": info.response.tolist() if info.response is not None else [],
            "beta": beta,
            "histogram": histogram,
        }

        # Add category labels if available for this feature
        if feature_name in category_labels:
            # Convert keys to strings for JSON compatibility
            feature_entry["category_labels"] = {
                str(k): v for k, v in category_labels[feature_name].items()
            }

        data["univariate"][feature_name] = feature_entry

    # Build bivariate section
    for pair_info in bundle.bivariate_pairs():
        if pair_info.skipped:
            continue

        feature1_name, feature2_name = pair_info.names
        pair_key = f"{feature1_name}__{feature2_name}"
        idx1, idx2 = pair_info.indices

        # Get beta coefficient for this interaction.
        # The beta vector is structured as:
        #   [0 .. n_univ-1]  = univariate betas (ALL features, including zeros)
        #   [n_univ .. ]      = bivariate betas in canonical pair order
        #                       i.e. combinations(range(n_univ), 2)
        # We compute the combinatorial offset for pair (i, j) the same way
        # as PlottingPipeline.apply_beta_scaling.
        n_univ_total = lasso_results.n_univ
        pair_offset = idx1 * n_univ_total - idx1 * (idx1 + 1) // 2 + (idx2 - idx1 - 1)
        beta_idx = n_univ_total + pair_offset
        beta = None
        if beta_idx < len(beta_vector):
            beta = float(beta_vector[beta_idx])

        # Extract x_values for each feature
        x_values = pair_info.x_values if pair_info.x_values is not None else np.zeros((0, 2))

        # Separate x_values into two arrays
        if len(x_values) > 0:
            x_values_1 = x_values[:, 0].tolist()
            x_values_2 = x_values[:, 1].tolist()
        else:
            x_values_1 = []
            x_values_2 = []

        # Determine response shape for proper 2D storage
        response = pair_info.response if pair_info.response is not None else np.array([])

        # Build pair entry
        pair_entry = {
            "indices": list(pair_info.indices),
            "names": list(pair_info.names),
            "labels": list(pair_info.labels),
            "is_categorical": list(pair_info.is_categorical),
            "x_values_1": x_values_1,
            "x_values_2": x_values_2,
            "response": response.tolist() if isinstance(response, np.ndarray) else list(response),
            "beta": beta,
            "skipped": pair_info.skipped,
        }

        data["bivariate"][pair_key] = pair_entry

    # Convert all numpy types to JSON-serializable format
    data = _convert_to_json_serializable(data)

    # Write to file with pretty formatting
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

    logger.info(f"Nomogram JSON data saved to {file_path}")
    print(f"Nomogram JSON data saved to {file_path}")

    return file_path
