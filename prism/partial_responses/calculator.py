"""
Core PartialResponseCalculator class for partial response calculations.

This module implements the main calculator class that handles:
- Model interpretability through partial response analysis
- Both 'dirac' and 'lebesgue' methods
- One-hot encoding handling and collapsing
- GPU optimization and device management
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from prism.device_tools import _free_all_gpu_caches, device_empty_cache

from .lebesgue import LebesgueMixin, get_variable_range
from .utils import _warn_if_scaled_onehot, stable_logit

logger = logging.getLogger(__name__)


class PartialResponseCalculator(LebesgueMixin):
    """
    A class for calculating partial responses in machine learning models which
    help interpret the impact of individual features and feature pairs on model
    predictions.

    It supports both 'dirac' and 'lebesgue' methods for computing partial
    responses.

    One-Hot Encoded Feature Handling
    --------------------------------

    When `onehot_groups` is specified, the calculator handles one-hot encoding constraints:

    **Key Concept**: For features within the same one-hot group, bivariate effects are zero by design.
    Since one-hot encoding means only one feature can be active, setting one feature fully determines
    all others in the group (e.g., diagnosis_A=1 implies diagnosis_B=0). The bivariate effect, which
    measures additional impact beyond individual effects, is zero because there's no additional
    information in specifying both values.

    **Univariate Partial Responses**:
    - **value=0 (Reference state)**: ALL columns in the one-hot group are set to 0, representing
      the dropped reference category. All features in the same group share the same value=0 response.
    - **value=1 (Active state)**: Target feature is 1, siblings are 0, representing that specific category.

    **Bivariate Partial Responses**:
    - **Same-group pairs**: Skipped entirely (bivariate effect is zero by design)
    - **Different-group pairs**: Apply one-hot constraints independently to each group

    **Assumptions**: Groups are mutually exclusive; each represents one categorical variable

    Attributes
    ----------
    original_model : Any
        The original machine learning model.
    method : str
        The method used for calculating partial responses ('dirac' or 'lebesgue').
    device : torch.device
        The computation device (CPU or GPU).
    input_dim : int
        The input dimension of the model.
    logit_y0 : float
        The baseline logit value for partial response calculations.
    x_train : torch.Tensor
        The training data, required for the 'lebesgue' method.
    onehot_groups : Optional[List[Tuple]]
        Groups of column indices representing one-hot encoded categorical features.

    Methods
    -------
    calculate(x: torch.Tensor, batch_size: int = 1024) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]]]
        Calculate partial responses for the given input data.
    calculate_subset(x: torch.Tensor, n_steps: int = 15, categorical_threshold: int = 15, subtract_univariate: bool = False) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Tuple[int, int]], List[torch.Tensor], List[torch.Tensor]]
        Calculate partial responses for a subset of feature values.
    """

    def __init__(
        self,
        model: Any,
        method: str = 'dirac',
        device: Optional[str] = None,
        input_dim: int = 1,
        x_train: Optional[torch.Tensor] = None,
        use_caching: bool = True,
        caching_threshold: Optional[int] = None,
        onehot_groups: Optional[List[Tuple]] = None,
        group_manager: Optional[Any] = None,
        feature_names: Optional[List[str]] = None,
        trim_quantile: Optional[float] = None,
        scaler: Optional[Any] = None,
        predict_batch_size: Optional[int] = None,
    ):
        """Initialize the calculator with model and training data.

        Parameters
        ----------
        model : Any
            The machine learning model to analyze
        method : str, optional
            Method for calculating partial responses ('dirac' or 'lebesgue')
        device : Optional[str], optional
            Computation device ('cpu', 'mps', or 'cuda')
        input_dim : int, optional
            Input dimension of the model
        x_train : Optional[torch.Tensor], optional
            Training data, required for 'lebesgue' method
        use_caching : bool, optional
            Whether to enable response caching for features with few unique values
        caching_threshold : int, optional
            Max unique values for a feature to be cached. If None, defaults to input size
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
            PRiSMScaler instance used to transform training data. When provided, ensures
            that one-hot encoded features are set to properly scaled values during partial
            response calculations. This is critical for models trained on scaled data (e.g.,
            XGBoost, Random Forest via SklearnWrapper) to prevent identical partial responses
            for all categories within a one-hot group.
        predict_batch_size : Optional[int], optional
            Maximum number of rows per forward pass in _batched_predict(). If None,
            auto-scaled based on available GPU VRAM. Set explicitly to control memory
            usage for large models (e.g., PRN with many hidden units).

        Notes
        -----
        One-hot groups are ALWAYS collapsed when specified. This ensures mathematically
        correct reconstruction by treating each one-hot encoded group as a single
        categorical variable rather than multiple binary features.
        """

        self.model = model
        self.method = method
        self.device = (
            torch.device(device)
            if device is not None
            else (
                next(model.parameters()).device
                if hasattr(model, 'parameters')
                else torch.device('cpu')
            )
        )
        self.input_dim = input_dim
        self.logit_y0 = None
        self.use_caching = use_caching
        self.caching_threshold = caching_threshold
        self.feature_names = feature_names
        self.group_manager = group_manager
        self.trim_quantile = trim_quantile
        self.predict_batch_size = predict_batch_size or self._auto_predict_batch_size(
            self.device, self.model
        )

        # Validate input dimensions early to catch mismatches
        self._validate_input_dimensions(x_train, feature_names)

        # Compute indices from manager if provided (group_manager takes precedence)
        if self.group_manager is not None:
            if feature_names is None:
                raise ValueError("feature_names must be provided when using group_manager")
            if onehot_groups is not None:
                import warnings

                warnings.warn(
                    "Both 'group_manager' and 'onehot_groups' were provided. "
                    "'group_manager' takes precedence; 'onehot_groups' is ignored. "
                    "Note: 'onehot_groups' is deprecated; use 'group_manager' instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            self.onehot_groups = self.group_manager.to_indices(feature_names)
        elif onehot_groups is not None:
            import warnings

            warnings.warn(
                "'onehot_groups' is deprecated and will be removed in a future version. "
                "Use 'group_manager' (OneHotGroupManager) instead for better feature naming "
                "and consistency with the preprocessing pipeline.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.onehot_groups = onehot_groups
        else:
            self.onehot_groups = None

        # Automatically enable collapsing when one-hot groups are present
        # This is REQUIRED for mathematically correct reconstruction
        self.collapse_onehot = self.onehot_groups is not None or self.group_manager is not None

        if self.collapse_onehot:
            logger.info(
                "One-hot group collapsing ENABLED: Treating one-hot encoded groups as "
                "single categorical variables for mathematically correct reconstruction."
            )

        # Setup collapse mappings
        self.collapsed_feature_names = None
        self.index_mapping = None  # collapsed_idx -> [original_indices]
        self.n_collapsed_features = None

        self._check_model_compatibility(x_train)
        self._validate_onehot_groups()

        # Create collapsed mapping if collapsing is enabled
        if self.collapse_onehot:
            self._create_collapsed_mapping()

        # Store scaler and compute scaled values for one-hot columns
        self.scaler = scaler
        self.onehot_scaled_values = (
            None  # Will store {col_idx: {'scaled_0': val, 'scaled_1': val}}
        )
        if self.onehot_groups is not None and scaler is None:
            raise ValueError(
                "One-hot groups provided without scaler. When one-hot encoded features are present, "
                "the scaler is required to correctly detect active categories in scaled data. "
                "Pass scaler=scaler to ensure correct one-hot category detection. "
                "If no scaling was applied to the data, pass scaler=NoScaler()."
            )
        if scaler is not None and self.onehot_groups is not None:
            self._compute_onehot_scaled_values()
            # Precompute tolerance tensors for vectorized collapse operations
            self._precompute_group_tolerances()

        if method == 'lebesgue':
            if x_train is None:
                error_msg = "The x_train argument must be provided for the Lebesgue method."
                logger.error(error_msg)
                raise ValueError(error_msg)
            self.x_train = x_train.to(self.device)
            self.calculate_baseline(x_train)

    @torch.no_grad()
    def _check_model_compatibility(self, x_train: Optional[torch.Tensor]):
        """
        Check if the provided model is compatible with the predict method.

        This method attempts to make a prediction using a dummy input to ensure
        that the model's predict method works as expected.

        Parameters
        ----------
        x_train : Optional[torch.Tensor]
            The training data, used for creating a dummy input in the 'lebesgue' method.

        Raises
        ------
        ValueError
            If the model is not compatible with the predict method.
        """
        try:
            if self.method == 'dirac':
                # For 'dirac' method, use a zero tensor as dummy input
                dummy_input = torch.zeros((1, self.input_dim), device=self.device)
            else:  # lebesgue
                # For 'lebesgue' method, use the first row of training data
                dummy_input = x_train[:1].to(self.device)

            # Attempt to make a prediction
            _ = self.predict_proba(dummy_input)
        except Exception as e:
            error_msg = (
                f"The provided model is not compatible with the predict method. Error: {str(e)}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _validate_onehot_groups(self):
        """
        Validate that onehot_groups meet the required assumptions.

        Raises
        ------
        ValueError
            If onehot_groups violate assumptions (overlapping groups,
            out-of-range indices, empty groups, or invalid one-hot encoding)
        """
        if self.onehot_groups is None:
            return

        if not isinstance(self.onehot_groups, (list, tuple)):
            raise ValueError("onehot_groups must be a list or tuple of tuples")

        seen_indices = set()

        for i, group in enumerate(self.onehot_groups):
            if not isinstance(group, (list, tuple)):
                raise ValueError(f"onehot_groups[{i}] must be a list or tuple, got {type(group)}")

            if len(group) < 2:
                raise ValueError(
                    f"onehot_groups[{i}] must contain at least 2 feature indices, got {len(group)}"
                )

            for feature_idx in group:
                if not isinstance(feature_idx, int):
                    raise ValueError(
                        f"Feature indices must be integers, got {type(feature_idx)} in group {i}"
                    )

                if feature_idx < 0 or feature_idx >= self.input_dim:
                    raise ValueError(
                        f"Feature index {feature_idx} in group {i} is out of range [0, {self.input_dim-1}]"
                    )

                if feature_idx in seen_indices:
                    raise ValueError(
                        f"Feature index {feature_idx} appears in multiple groups (groups must be mutually exclusive)"
                    )

                seen_indices.add(feature_idx)

        # Validate one-hot encoding if training data is available
        if hasattr(self, 'x_train') and self.x_train is not None:
            self._validate_onehot_encoding()

    def _validate_onehot_encoding(self):
        """
        Validate that the specified groups actually represent one-hot encoded data.

        Raises
        ------
        ValueError
            If the groups don't represent valid one-hot encoding
        """
        for i, group in enumerate(self.onehot_groups):
            group_data = self.x_train[:, group]

            # Check that all values are 0 or 1
            if not torch.all((group_data == 0) | (group_data == 1)):
                raise ValueError(
                    f"onehot_groups[{i}] contains non-binary values. One-hot encoding requires only 0s and 1s."
                )

            # Check that each row sums to at most 1 (one-hot constraint)
            row_sums = group_data.sum(dim=1)
            if torch.any(row_sums > 1):
                raise ValueError(
                    f"onehot_groups[{i}] violates one-hot constraint: some rows have multiple 1s."
                )

            # Warn if any rows are all zeros (though this might be valid)
            if torch.any(row_sums == 0):
                logger.warning(
                    f"onehot_groups[{i}] contains rows with all zeros. This may indicate missing categories or incomplete one-hot encoding."
                )

    def _validate_input_dimensions(
        self, x_train: Optional[torch.Tensor], feature_names: Optional[List[str]]
    ):
        """
        Validate that input dimensions are consistent.

        This catches common errors where x_train or feature_names dimensions
        don't match input_dim, which would cause subtle bugs downstream.

        Parameters
        ----------
        x_train : Optional[torch.Tensor]
            Training data to validate
        feature_names : Optional[List[str]]
            Feature names to validate

        Raises
        ------
        ValueError
            If dimensions don't match input_dim
        """
        if x_train is not None:
            x_train_dim = x_train.shape[1] if x_train.ndim > 1 else 1
            if x_train_dim != self.input_dim:
                raise ValueError(
                    f"x_train has {x_train_dim} features, but input_dim={self.input_dim}. "
                    f"Ensure x_train is in the original (OHE) feature space matching input_dim."
                )

        if feature_names is not None:
            if len(feature_names) != self.input_dim:
                raise ValueError(
                    f"feature_names has {len(feature_names)} names, but input_dim={self.input_dim}. "
                    f"Ensure feature_names includes all columns (OHE columns if using one-hot encoding)."
                )

    def _create_collapsed_mapping(self):
        """
        Create mappings between original one-hot features and collapsed categorical features.

        Sets:
            self.collapsed_feature_names: List of collapsed feature names
            self.index_mapping: Dict[int, List[int]] - collapsed_idx -> original_indices
            self.n_collapsed_features: Number of features after collapsing

        Note
        ----
        The index ordering within each group MUST match the ordering used by
        collapse_onehot_features() in preprocessing.py. Both use the explicit
        order from groups_dict[group_name] to ensure consistency:
        - Category 1 maps to groups_dict[group_name][0]
        - Category 2 maps to groups_dict[group_name][1]
        - etc.

        Raises
        ------
        ValueError
            If feature_names is not provided. Feature names are required for collapsing
            to maintain a single source of truth for feature naming. Use group_manager
            (preferred) or provide feature_names explicitly with onehot_groups.
        """
        if self.feature_names is None:
            raise ValueError(
                "feature_names must be provided when collapsing one-hot groups. "
                "This ensures consistent feature naming across the pipeline. "
                "Preferred: use group_manager with feature_names. "
                "Alternative: provide feature_names explicitly with onehot_groups."
            )

        self.collapsed_feature_names = []
        self.index_mapping = {}

        collapsed_idx = 0
        grouped_indices = set()

        # Process one-hot groups
        if self.group_manager:
            # Use manager for group names and ordering
            # CRITICAL: Use groups_dict order explicitly to match collapse_onehot_features()
            for group_name, group_features in self.group_manager.groups_dict.items():
                # Get indices in groups_dict order (not feature_names order!)
                # This matches the encoding order in collapse_onehot_features():
                #   indices = [feature_names.index(f) for f in group_features]
                indices = [
                    self.feature_names.index(f) for f in group_features if f in self.feature_names
                ]

                self.collapsed_feature_names.append(group_name)
                self.index_mapping[collapsed_idx] = indices
                grouped_indices.update(indices)
                collapsed_idx += 1
        else:
            # Fallback: extract group name from feature name
            for group_tuple in self.onehot_groups:
                indices = list(group_tuple)
                # Extract group name from first member (e.g., 'diagn_CAD' -> 'diagn')
                first_feature = self.feature_names[indices[0]]
                group_name = first_feature.rsplit('_', 1)[0]

                self.collapsed_feature_names.append(group_name)
                self.index_mapping[collapsed_idx] = indices
                grouped_indices.update(indices)
                collapsed_idx += 1

        # Add non-grouped features
        for orig_idx, fname in enumerate(self.feature_names):
            if orig_idx not in grouped_indices:
                self.collapsed_feature_names.append(fname)
                self.index_mapping[collapsed_idx] = [orig_idx]
                collapsed_idx += 1

        self.n_collapsed_features = collapsed_idx
        logger.info(
            f"Mapped {len(self.feature_names)} features collapsed to {self.n_collapsed_features} features"
        )

    def _compute_onehot_scaled_values(self):
        """
        Compute scaled values for 0 and 1 for each one-hot encoded column.

        This ensures that when setting one-hot columns during partial response calculations,
        we use the correctly scaled values that match what the model was trained on.

        Uses the scaler's transform() method for flexibility with any scaler type
        (MedianStdScaler, StandardScaler, MinMaxScaler, etc.).

        Sets:
            self.onehot_scaled_values: Dict[int, Dict[str, float]]
                {column_index: {'scaled_0': value, 'scaled_1': value}}

        Raises:
            TypeError: If scaler does not have a transform() method.
        """
        self.onehot_scaled_values = {}

        # Collect all one-hot column indices
        onehot_indices = sorted(
            set(idx for group_tuple in self.onehot_groups for idx in group_tuple)
        )

        if not onehot_indices:
            return

        # Require scaler to have transform method
        if not hasattr(self.scaler, 'transform'):
            raise TypeError(
                f"Scaler type {type(self.scaler).__name__} does not have a transform() method. "
                f"When using onehot_groups, the scaler must support transform() to compute "
                f"scaled values for one-hot columns."
            )

        # Use scaler.transform() to compute what 0 and 1 become after scaling
        # This works with any sklearn-compatible scaler
        n_features = self.input_dim

        # Create test rows: one with all zeros, one with 1s in one-hot columns
        x_zeros = np.zeros((1, n_features))
        x_ones = np.zeros((1, n_features))
        for col_idx in onehot_indices:
            x_ones[0, col_idx] = 1.0

        # Transform both
        x_zeros_scaled = self.scaler.transform(x_zeros)
        x_ones_scaled = self.scaler.transform(x_ones)

        # Extract scaled values for each one-hot column
        for col_idx in onehot_indices:
            self.onehot_scaled_values[col_idx] = {
                'scaled_0': float(x_zeros_scaled[0, col_idx]),
                'scaled_1': float(x_ones_scaled[0, col_idx]),
            }

        logger.info(
            f"Computed scaled values for {len(self.onehot_scaled_values)} one-hot encoded columns"
        )

    def _precompute_group_tolerances(self):
        """
        Precompute and cache tolerance tensors for all one-hot groups.

        This method creates tensors for group indices, scaled_1 values, and tolerances
        that are used in vectorized collapse operations. These tensors enable batch
        processing of active index detection without Python loops.

        Sets:
            self._group_indices_tensor: Dict[int, torch.Tensor]
                {collapsed_idx: tensor of original indices for the group}
            self._group_scaled_1: Dict[int, torch.Tensor]
                {collapsed_idx: tensor of scaled_1 values for each column in group}
            self._group_tolerance: Dict[int, torch.Tensor]
                {collapsed_idx: tensor of tolerance values for each column in group}
        """
        if self.onehot_groups is None:
            return

        self._group_indices_tensor = {}
        self._group_scaled_1 = {}
        self._group_tolerance = {}

        for collapsed_idx, original_indices in self.index_mapping.items():
            if len(original_indices) <= 1:
                # Not a one-hot group, skip
                continue

            # Cache group indices as tensor
            self._group_indices_tensor[collapsed_idx] = torch.tensor(
                original_indices, dtype=torch.long, device=self.device
            )

            # Build scaled_1 and tolerance tensors for this group
            use_scaled = (
                self.onehot_scaled_values is not None
                and original_indices[0] in self.onehot_scaled_values
            )

            if use_scaled:
                scaled_1_values = []
                tolerances = []
                for orig_idx in original_indices:
                    scaled_info = self.onehot_scaled_values.get(orig_idx, {})
                    s1 = scaled_info.get('scaled_1', 1.0)
                    s0 = scaled_info.get('scaled_0', 0.0)
                    scaled_1_values.append(s1)
                    tolerances.append(abs(s1 - s0) * 0.5)

                self._group_scaled_1[collapsed_idx] = torch.tensor(
                    scaled_1_values, dtype=torch.float32, device=self.device
                )
                self._group_tolerance[collapsed_idx] = torch.tensor(
                    tolerances, dtype=torch.float32, device=self.device
                )
            else:
                # Unscaled data
                self._group_scaled_1[collapsed_idx] = torch.ones(
                    len(original_indices), dtype=torch.float32, device=self.device
                )
                self._group_tolerance[collapsed_idx] = torch.full(
                    (len(original_indices),), 0.5, dtype=torch.float32, device=self.device
                )

        logger.info(
            f"Precomputed tolerance tensors for {len(self._group_indices_tensor)} one-hot groups"
        )

    def _precompute_active_indices(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Precompute active original column indices for all one-hot groups across all samples.

        This method vectorizes the detection of which original feature is active in each
        one-hot group for each sample. It handles both scaled and unscaled data by using
        precomputed tolerance tensors.

        Parameters
        ----------
        x : torch.Tensor
            Input data with one-hot encoding, shape (n_samples, n_features)

        Returns
        -------
        Dict[int, torch.Tensor]
            Mapping from collapsed_idx to tensor of active original indices.
            Shape of each tensor: (n_samples,)
            For reference state (all zeros), returns index of first group member.

        Notes
        -----
        Uses argmax on boolean masks to find active indices. When all values are False
        (reference state), argmax returns 0, which correctly maps to group_indices[0].
        """
        if not hasattr(self, '_group_indices_tensor'):
            # Fallback if precompute_group_tolerances wasn't called
            return {}

        active_indices = {}
        x = x.to(self.device)

        for collapsed_idx in self._group_indices_tensor.keys():
            group_indices = self.index_mapping[collapsed_idx]
            group_data = x[:, group_indices]  # (n_samples, n_group_members)

            # Get precomputed tensors
            scaled_1_vals = self._group_scaled_1[collapsed_idx]  # (n_group_members,)
            tolerance_vals = self._group_tolerance[collapsed_idx]  # (n_group_members,)

            # Vectorized comparison: which columns are active for each sample
            # Shape: (n_samples, n_group_members)
            active_mask = torch.abs(group_data - scaled_1_vals) < tolerance_vals

            # Find first active index for each sample (or 0 if none active)
            # argmax returns 0 for all-False rows, which correctly maps to reference state
            local_active_idx = active_mask.int().argmax(dim=1)  # (n_samples,)

            # Map local indices to original feature indices
            group_indices_tensor = self._group_indices_tensor[collapsed_idx]
            active_orig_idx = group_indices_tensor[local_active_idx]  # (n_samples,)

            active_indices[collapsed_idx] = active_orig_idx

        return active_indices

    def _get_onehot_value(self, col_idx: int, raw_value: float) -> float:
        """
        Get the scaled value for a one-hot column given a raw value.

        Parameters
        ----------
        col_idx : int
            Column index
        raw_value : float
            Raw value (0.0 or 1.0 for one-hot columns)

        Returns
        -------
        float
            Scaled value if scaler is available and column is one-hot, otherwise raw value
        """
        if self.onehot_scaled_values is not None and col_idx in self.onehot_scaled_values:
            if abs(raw_value - 0.0) < 1e-9:
                return self.onehot_scaled_values[col_idx]['scaled_0']
            elif abs(raw_value - 1.0) < 1e-9:
                return self.onehot_scaled_values[col_idx]['scaled_1']
        return raw_value

    def _create_baseline_input(self, n_rows: int, n_cols: int) -> torch.Tensor:
        """
        Create an input tensor with all features at their reference/baseline values.

        For the Dirac method, the baseline represents all features at their "zero" state
        after any scaling transformations have been applied. This method uses the scaler's
        transform() to ensure consistency with how the model was trained.

        - Continuous features: value after transforming 0 (typically the median after median scaling)
        - One-hot columns: value after transforming 0 (reference category)

        Parameters
        ----------
        n_rows : int
            Number of rows in the tensor
        n_cols : int
            Number of columns (features)

        Returns
        -------
        torch.Tensor
            Input tensor with all features at baseline values, ready for model prediction
        """
        baseline = torch.zeros((n_rows, n_cols), device=self.device)

        # Apply scaled values for one-hot columns if computed
        if self.onehot_scaled_values is not None:
            for col_idx, scaled_vals in self.onehot_scaled_values.items():
                if col_idx < n_cols:  # Safety check
                    baseline[:, col_idx] = scaled_vals['scaled_0']

        return baseline

    def _collapse_univariate_responses(
        self, x: torch.Tensor, univariate_responses: torch.Tensor
    ) -> torch.Tensor:
        """
        Collapse univariate responses from one-hot to categorical structure.

        Uses vectorized operations via precomputed active indices for significant
        performance improvement over the legacy sample-by-sample approach.

        Parameters
        ----------
        x : torch.Tensor
            Original input with one-hot encoding (n_samples, n_features)
        univariate_responses : torch.Tensor
            Uncollapsed responses (n_samples, n_features)

        Returns
        -------
        torch.Tensor
            Collapsed responses (n_samples, n_collapsed_features)
        """
        n_samples = x.shape[0]
        collapsed = torch.zeros((n_samples, self.n_collapsed_features), device=self.device)

        # Precompute active indices for all one-hot groups once
        active_indices = self._precompute_active_indices(x)

        for collapsed_idx, original_indices in self.index_mapping.items():
            if len(original_indices) == 1:
                # Non-grouped feature: direct copy
                collapsed[:, collapsed_idx] = univariate_responses[:, original_indices[0]]
            else:
                # One-hot group: collapse to single column using vectorized operation
                collapsed[:, collapsed_idx] = self._collapse_group(
                    x,
                    original_indices,
                    univariate_responses,
                    precomputed_active=active_indices.get(collapsed_idx),
                )

        return collapsed

    def _collapse_group(
        self,
        x: torch.Tensor,
        group_indices: List[int],
        responses: torch.Tensor,
        precomputed_active: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Collapse one-hot group responses to single column using vectorized operations.

        For each sample, select the response corresponding to the active category:
        - If all zeros (reference): use response from any member (all identical)
        - If one active: use that member's response

        Parameters
        ----------
        x : torch.Tensor
            Original input with one-hot encoding (may be scaled)
        group_indices : List[int]
            Indices of features in the one-hot group
        responses : torch.Tensor
            Responses for all features
        precomputed_active : Optional[torch.Tensor]
            Precomputed tensor of active original indices, shape (n_samples,)
            If provided, skips active index detection and uses these directly

        Returns
        -------
        torch.Tensor
            Collapsed responses for this group, shape (n_samples,)
        """
        if precomputed_active is not None:
            # Vectorized path: use torch.gather to select responses based on precomputed indices
            # Convert original indices to column positions in responses tensor
            # precomputed_active contains original feature indices, we need to gather from responses
            collapsed = responses.gather(dim=1, index=precomputed_active.unsqueeze(1)).squeeze(1)
            return collapsed

        # Legacy fallback path (not used when precomputed_active is provided)
        n_samples = x.shape[0]
        collapsed = torch.zeros(n_samples, device=self.device)

        # Extract one-hot group data and ensure it's on the correct device
        group_data = x[:, group_indices].to(self.device)  # (n_samples, len(group_indices))

        # Build per-column comparison values and tolerances
        # Each column in a one-hot group may have different scaled_1 values
        # due to different statistics (median, std) from the scaler
        use_scaled = (
            self.onehot_scaled_values is not None and group_indices[0] in self.onehot_scaled_values
        )

        if use_scaled:
            # Build arrays of scaled_1 and tolerance for each column in the group
            scaled_1_values = []
            tolerances = []
            for orig_idx in group_indices:
                scaled_info = self.onehot_scaled_values.get(orig_idx, {})
                s1 = scaled_info.get('scaled_1', 1.0)
                s0 = scaled_info.get('scaled_0', 0.0)
                scaled_1_values.append(s1)
                tolerances.append(abs(s1 - s0) * 0.5)
            scaled_1_tensor = torch.tensor(scaled_1_values, device=self.device)
            tolerance_tensor = torch.tensor(tolerances, device=self.device)
        else:
            # Use raw value for comparison (unscaled data)
            scaled_1_tensor = torch.ones(len(group_indices), device=self.device)
            tolerance_tensor = torch.full((len(group_indices),), 0.5, device=self.device)

        for sample_idx in range(n_samples):
            # Compare each column against its own scaled_1 value with per-column tolerance
            active_mask = torch.abs(group_data[sample_idx] - scaled_1_tensor) < tolerance_tensor

            if active_mask.any():
                # Find which member is active
                active_local_idx = torch.where(active_mask)[0][0].item()
                active_orig_idx = group_indices[active_local_idx]
                collapsed[sample_idx] = responses[sample_idx, active_orig_idx]
            else:
                # Reference state: all members have identical response
                collapsed[sample_idx] = responses[sample_idx, group_indices[0]]

        return collapsed

    def _collapse_bivariate_responses(
        self, x: torch.Tensor, bivariate_responses: torch.Tensor
    ) -> torch.Tensor:
        """
        Collapse bivariate responses from one-hot to categorical structure using vectorized operations.

        This method replaces triple nested loops with batch tensor operations for significant
        performance improvement (expected 10-100x speedup on GPU).

        Parameters
        ----------
        x : torch.Tensor
            Original input with one-hot encoding (n_samples, n_features)
        bivariate_responses : torch.Tensor
            Uncollapsed bivariate responses (n_samples, n_original_pairs)
            where n_original_pairs = n_features * (n_features - 1) / 2

        Returns
        -------
        torch.Tensor
            Collapsed bivariate responses (n_samples, n_collapsed_pairs)
            where n_collapsed_pairs = n_collapsed_features * (n_collapsed_features - 1) / 2
        """
        n_samples = x.shape[0]
        n_original_features = x.shape[1]
        _ = self.n_collapsed_features * (self.n_collapsed_features - 1) // 2

        # Step 1: Precompute active indices for all collapsed features
        active_indices = self._precompute_active_indices(x)

        # Step 2: Build active indices tensor for all collapsed features
        # For non-grouped features, use their single original index
        active_indices_stacked = torch.zeros(
            (self.n_collapsed_features, n_samples), dtype=torch.long, device=self.device
        )
        for collapsed_idx in range(self.n_collapsed_features):
            original_indices = self.index_mapping[collapsed_idx]
            if len(original_indices) == 1:
                # Non-grouped feature: use the single index for all samples
                active_indices_stacked[collapsed_idx, :] = original_indices[0]
            else:
                # One-hot group: use precomputed active indices
                active_indices_stacked[collapsed_idx, :] = active_indices[collapsed_idx]

        # Step 3: Generate all collapsed pairs and compute their original pair indices
        collapsed_pairs = []
        for i_collapsed in range(self.n_collapsed_features):
            for j_collapsed in range(i_collapsed + 1, self.n_collapsed_features):
                collapsed_pairs.append((i_collapsed, j_collapsed))

        # Convert to tensors for vectorized operations
        collapsed_pairs_tensor = torch.tensor(
            collapsed_pairs, dtype=torch.long, device=self.device
        )
        i_collapsed = collapsed_pairs_tensor[:, 0]  # (n_collapsed_pairs,)
        j_collapsed = collapsed_pairs_tensor[:, 1]  # (n_collapsed_pairs,)

        # Step 4: For each pair, get the active original indices for all samples
        # Shape: (n_collapsed_pairs, n_samples)
        i_orig = active_indices_stacked[i_collapsed]  # (n_collapsed_pairs, n_samples)
        j_orig = active_indices_stacked[j_collapsed]  # (n_collapsed_pairs, n_samples)

        # Step 5: Ensure i < j for pair indexing (swap if needed)
        i_sorted = torch.min(i_orig, j_orig)
        j_sorted = torch.max(i_orig, j_orig)

        # Step 6: Compute original pair indices using lower triangular formula
        # orig_pair_idx = i * n + j - ((i + 2) * (i + 1)) // 2
        orig_pair_idx = (
            i_sorted * n_original_features + j_sorted - ((i_sorted + 2) * (i_sorted + 1)) // 2
        )  # (n_collapsed_pairs, n_samples)

        # Step 7: Gather all responses at once
        # Transpose to (n_samples, n_collapsed_pairs) for gather operation
        orig_pair_idx_t = orig_pair_idx.t()  # (n_samples, n_collapsed_pairs)
        collapsed = bivariate_responses.gather(dim=1, index=orig_pair_idx_t)

        return collapsed

    def _should_skip_bivariate_pair(self, i: int, j: int) -> bool:
        """Check if a bivariate pair should be skipped (same one-hot group)."""
        if self.onehot_groups:
            for group in self.onehot_groups:
                if i in group and j in group:
                    return True
        return False

    def _get_onehot_group(self, feature_idx: int) -> Optional[Tuple]:
        """Get the one-hot group containing the feature, or None if not in any group.

        Parameters
        ----------
        feature_idx : int
            Index of the feature

        Returns
        -------
        Optional[Tuple]
            The one-hot group tuple containing feature_idx, or None if feature is not in any group
        """
        if self.onehot_groups:
            for group in self.onehot_groups:
                if feature_idx in group:
                    return group
        return None

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Probability of the positive class from the wrapped model.

        Parameters
        ----------
        x : torch.Tensor
            Input data for prediction.

        Returns
        -------
        torch.Tensor
            P(y=1) per sample, squeezed to shape (n_samples,).
        """
        return self.model.predict_proba(x, device=self.device).squeeze()

    @staticmethod
    def _estimate_bytes_per_row(model) -> int:
        """Estimate GPU memory per row during prediction from model architecture.

        Inspects the model to compute a per-row byte estimate:
        - nn.Module (MLP, PRN): sums Linear layer output sizes * 4 bytes * 2 safety
        - XGBoost/XGBRFClassifier (via SklearnWrapper with GPU): estimates from
          n_estimators and max_depth, since tree traversal allocates per-row buffers
          proportional to ensemble size
        - Other models: returns 0 (caller falls back to a generic estimate)

        Returns bytes per row, or 0 if the model cannot be inspected.
        """
        import torch.nn as nn

        # Unwrap SklearnWrapper if present
        actual_model = model.model if hasattr(model, 'model') else model

        # Path 1: PyTorch nn.Module (MLP, PRN, MaskedMLP)
        if isinstance(actual_model, nn.Module):
            total_units = 0
            for module in actual_model.modules():
                if isinstance(module, nn.Linear):
                    total_units += module.out_features
            # float32 = 4 bytes per activation, x2 safety margin for
            # intermediates (gradients disabled, but pytorch may keep
            # input tensors alive until backward is confirmed skipped)
            return total_units * 4 * 2

        # Path 2: XGBoost / XGBRFClassifier (GPU inference via cupy)
        # Check if the wrapper indicates GPU-enabled XGBoost
        is_xgb_gpu = getattr(model, '_xgb_gpu_enabled', False)
        if is_xgb_gpu:
            try:
                params = actual_model.get_params()
                n_estimators = params.get('n_estimators', 100)
                max_depth = params.get('max_depth', 6)
                # XGBoost GPU prediction allocates per-row buffers proportional
                # to the number of tree nodes traversed. Each tree has up to
                # 2^(depth+1)-1 nodes; predict touches one path of depth nodes.
                # Empirically ~32 bytes per tree per row covers the prediction
                # buffer, node indices, and intermediate margin storage.
                return n_estimators * max_depth * 32
            except Exception:
                pass

        return 0

    @staticmethod
    def _auto_predict_batch_size(device: torch.device, model=None) -> int:
        """Choose predict sub-batch size based on model and available GPU VRAM.

        Estimates per-row activation memory from the model's Linear layers,
        then sizes the batch to use ~25% of free GPU VRAM. Falls back to
        65536 on CPU or if VRAM query fails.

        Returns a row count, always a power-of-two for alignment.
        """
        VRAM_FRACTION = 0.25
        FALLBACK = 65_536
        MIN_BATCH = 4_096
        MAX_BATCH = 524_288  # 512K rows cap
        FALLBACK_BYTES_PER_ROW = 20_480  # ~20 KB if model can't be inspected

        bytes_per_row = 0
        if model is not None:
            bytes_per_row = PartialResponseCalculator._estimate_bytes_per_row(model)
        if bytes_per_row == 0:
            bytes_per_row = FALLBACK_BYTES_PER_ROW

        try:
            if device.type == 'cuda':
                free, _total = torch.cuda.mem_get_info(device)
                budget = int(free * VRAM_FRACTION)
                rows = max(MIN_BATCH, budget // bytes_per_row)
                # Round down to nearest power of 2
                rows = 1 << (rows.bit_length() - 1)
                return min(rows, MAX_BATCH)
        except Exception:
            pass
        return FALLBACK

    def _batched_predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict in chunks to avoid GPU OOM with large models.

        Splits x into sub-batches of self.predict_batch_size rows,
        runs predict_proba on each, and concatenates results. Calls
        _free_all_gpu_caches() between sub-batches to release
        intermediate activation memory.

        Mathematically identical to self.predict_proba(x) -- predictions
        are independent across rows.
        """
        n_rows = x.shape[0]
        if n_rows <= self.predict_batch_size:
            return self.predict_proba(x)

        chunks = []
        for start in range(0, n_rows, self.predict_batch_size):
            end = min(start + self.predict_batch_size, n_rows)
            chunks.append(self.predict_proba(x[start:end]))
            _free_all_gpu_caches()
        return torch.cat(chunks, dim=0)

    @torch.no_grad()
    def calculate_baseline(self, x: torch.Tensor) -> None:
        """
        Calculate the baseline logit value for partial responses.

        IMPORTANT: Dirac and Lebesgue use DIFFERENT baseline definitions by design.
        This is intentional and mathematically correct for each method:

        For 'dirac' method:
        - Baseline = model prediction at a single "reference point" (all features at zero/reference)
        - Uses _create_baseline_input() to ensure consistency with _calculate_dirac
        - For one-hot groups: baseline is the reference category (all zeros)
        - For continuous features: baseline is scaled zero (corresponding to median)
        - Interpretation: PR measures effect relative to a fixed reference patient

        For 'lebesgue' method:
        - Baseline = mean model prediction over the training data distribution
        - Interpretation: PR measures effect relative to the population average
        - This is consistent with the Lebesgue integration approach which marginalizes
          over the empirical distribution of other features

        The different baselines mean PR values are NOT directly comparable between methods,
        but reconstruction (logit_y0 + sum(phi_i) + sum(phi_ij)) works correctly for each.

        Parameters
        ----------
        x : torch.Tensor
            Input data (used only for 'lebesgue' method to compute mean prediction).
        """
        if self.method == 'dirac':
            # Use _create_baseline_input to ensure consistency with _calculate_dirac
            # This ensures the baseline uses the same reference point:
            # - One-hot columns: scaled_0 (which equals 0 when not scaled, or the scaled
            #   reference category value)
            # - Continuous columns: scaled_0 (corresponding to median value)
            x0 = self._create_baseline_input(1, self.input_dim)
            y0 = self.predict_proba(x0)
            self.logit_y0 = stable_logit(y0).item()
            logger.debug(f"Baseline logit (Dirac method): {self.logit_y0:.6f}")
        else:  # lebesgue
            y0 = self.predict_proba(x)
            self.logit_y0 = stable_logit(y0).mean().item()
            logger.debug(f"Baseline logit (Lebesgue method): {self.logit_y0:.6f}")

    def _is_collapsed_mode(self) -> bool:
        """Check if calculator is in collapsed mode."""
        return self.collapse_onehot and self.group_manager is not None

    def _get_collapsed_feature_grid(
        self, collapsed_idx: int, x: torch.Tensor, n_steps: int, categorical_threshold: int
    ) -> torch.Tensor:
        """
        Generate grid for a collapsed feature.

        Parameters
        ----------
        collapsed_idx : int
            Index in collapsed feature space
        x : torch.Tensor
            Original (uncollapsed) input data
        n_steps : int
            Number of steps for continuous features
        categorical_threshold : int
            Threshold for categorical detection

        Returns
        -------
        torch.Tensor
            Grid values - integers [0, N-1] for one-hot groups, linspace for continuous
        """
        original_indices = self.index_mapping[collapsed_idx]

        # Check if it's a group (either >1 members OR explicitly defined as group)
        is_group = len(original_indices) > 1
        if not is_group and self.group_manager is not None:
            feature_name = self.collapsed_feature_names[collapsed_idx]
            is_group = self.group_manager.is_categorical_group(feature_name)

        if is_group:
            # One-hot group: return categorical integers
            n_categories = len(original_indices)
            # Account for reference category (all zeros state)
            return torch.arange(n_categories + 1, device=self.device, dtype=torch.float32)
        else:
            # Single continuous feature: use standard range
            orig_idx = original_indices[0]
            return get_variable_range(
                x[:, orig_idx], n_steps, categorical_threshold, self.trim_quantile
            )

    def _expand_collapsed_value_to_onehot(
        self, collapsed_idx: int, collapsed_value: float
    ) -> Dict[int, float]:
        """
        Expand a collapsed categorical value to one-hot representation.

        Parameters
        ----------
        collapsed_idx : int
            Index in collapsed feature space
        collapsed_value : float
            Categorical integer (0 = reference, 1 = first category, etc.)

        Returns
        -------
        Dict[int, float]
            Mapping from original feature indices to values
        """
        original_indices = self.index_mapping[collapsed_idx]

        if len(original_indices) == 1:
            # Not a one-hot group, return as-is
            return {original_indices[0]: collapsed_value}

        # One-hot group: expand integer to one-hot
        cat_idx = int(collapsed_value)
        result = {}

        if cat_idx == 0:
            # Reference category: all zeros
            for idx in original_indices:
                result[idx] = 0.0
        else:
            # Active category: set corresponding index to 1, others to 0
            for i, idx in enumerate(original_indices):
                result[idx] = 1.0 if (i == cat_idx - 1) else 0.0

        return result

    @torch.no_grad()
    def calculate(
        self, x: torch.Tensor, batch_size: int = 1024
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
        """
        Calculate partial responses for all features and feature pairs.

        Parameters
        ----------
        x : torch.Tensor
            Input data.
        batch_size : int, optional
            Size of batches for processing.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]]]
            Univariate responses, bivariate responses, and feature pair indices.
        """
        with device_empty_cache(self.device):
            x = x.to(self.device)  # Ensure input tensor is on correct device

            # Warn if one-hot columns appear to be scaled
            _warn_if_scaled_onehot(x, self.onehot_groups)

            if self.method == 'dirac':
                if self.logit_y0 is None:
                    self.calculate_baseline(x)
                univariate_responses, bivariate_responses = self._calculate_dirac(x)
            elif self.method == 'lebesgue':
                univariate_responses, bivariate_responses = self._calculate_lebesgue(
                    x, batch_size=batch_size
                )
            else:
                raise ValueError(
                    f"Method {self.method} not implemented. Choose 'dirac' or 'lebesgue'."
                )

            # Apply collapsing if enabled (moved outside method-specific code)
            if self.collapse_onehot:
                n_features = x.shape[1]
                logger.info("Applying one-hot collapse to univariate responses")
                univariate_responses = self._collapse_univariate_responses(x, univariate_responses)
                logger.info(
                    f"Collapsed from {n_features} to {self.n_collapsed_features} univariate features"
                )

                logger.info("Applying one-hot collapse to bivariate responses")
                n_original_pairs = n_features * (n_features - 1) // 2
                n_collapsed_pairs = (
                    self.n_collapsed_features * (self.n_collapsed_features - 1) // 2
                )
                bivariate_responses = self._collapse_bivariate_responses(x, bivariate_responses)
                logger.info(
                    f"Collapsed from {n_original_pairs} to {n_collapsed_pairs} bivariate pairs"
                )

            return univariate_responses, bivariate_responses

    def _calculate_dirac(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
        """
        Calculate partial responses using the Dirac method.

        The Dirac method computes the effect of each feature value (univariate) and feature value
        pair (bivariate) by setting all other features to zero and measuring the change in the
        model's output. This is done for every sample in the input data.

        Parameters
        ----------
        x : torch.Tensor
            Input data tensor of shape (n_samples, n_features).

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
        The Dirac method calculates partial responses as follows:
        1. Univariate responses
        - For each feature and ach sample:
            - Create a tensor with only that feature's value, setting all other features to zero.
            - Calculate the model's output for this modified input.
            - Compute the difference between this output and the baseline (logit_y0).

        2. Bivariate responses
        - For each pair of feaures and each sample:
            - Create a tensor with only those two features' values, setting all other features to zero.
            - Calculate the model's output for this modified input.
            - Subtract the baseline and the corresponding univariate responses to isolate
            the interaction effect.

        The method uses the logit scale for calculations to ensure additivity of the effects.
        This process is repeated for every sample in the input data, resulting in a response
        for each feature value and feature value pair in each sample.
        """
        n_features = x.shape[1]
        n_samples = x.shape[0]
        x = x.to(self.device)

        # Calculate univariate responses
        univariate_responses = torch.zeros((n_samples, n_features), device=self.device)
        for i in range(n_features):
            # Create a tensor with only the i-th feature, others set to baseline values
            x_i = self._create_baseline_input(n_samples, n_features)
            x_i[:, i] = x[:, i]

            # Calculate the model's output and convert to log-odds
            y_i = self.predict_proba(x_i)
            univariate_responses[:, i] = stable_logit(y_i) - self.logit_y0

        # Calculate bivariate responses
        bivariate_responses = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                # Create a tensor with only the i-th and j-th features, others set to baseline values
                x_input = self._create_baseline_input(n_samples, n_features)
                x_input[:, i] = x[:, i]
                x_input[:, j] = x[:, j]

                # Calculate the model's output and convert to log-odds
                y_ij = self.predict_proba(x_input)

                # Calculate isolated bivariate response by subtracting baseline and univariate responses
                bivariate_response = (
                    stable_logit(y_ij)
                    - self.logit_y0
                    - univariate_responses[:, i]
                    - univariate_responses[:, j]
                )
                bivariate_responses.append(bivariate_response)

        # Stack bivariate responses into a single tensor
        bivariate_responses = torch.stack(bivariate_responses, dim=1)

        return univariate_responses, bivariate_responses

    @torch.no_grad()
    def calculate_subset(
        self,
        x: torch.Tensor,
        n_steps: int = 15,
        categorical_threshold: int = 15,
        subtract_univariate: bool = False,
        selected_features: Optional[List[int]] = None,
        selected_feature_pairs: Optional[List[Tuple[int, int]]] = None,
        batch_size: int = 1024,
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[Tuple[int, int]],
        List[torch.Tensor],
        List[torch.Tensor],
    ]:
        """
        Calculate partial responses for a subset of features on synthetic grids.

        When OneHotGroupManager is provided and collapse_onehot=True, this method
        operates in collapsed space:
        - selected_features are interpreted as collapsed feature indices
        - For one-hot groups, generates categorical integer grids
        - Expands to original one-hot encoding before model query
        - Returns responses indexed by collapsed features

        When OneHotGroupManager is None, operates on original feature space.

        Parameters
        ----------
        x : torch.Tensor
            Input data in original (uncollapsed) feature space
        n_steps : int
            Number of grid points for continuous features
        categorical_threshold : int
            Max unique values to treat as categorical
        subtract_univariate : bool
            Whether to subtract univariate from bivariate responses
        selected_features : Optional[List[int]]
            Feature indices to calculate (collapsed indices if in collapsed mode)
        selected_feature_pairs : Optional[List[Tuple[int, int]]]
            Feature pairs to calculate (collapsed indices if in collapsed mode)
        batch_size : int
            Batch size for processing

        Returns
        -------
        Tuple containing:
            - univariate_responses: List[Tensor] indexed by (collapsed) feature
            - bivariate_responses: List[Tensor] indexed by (collapsed) pair
            - x_univariate: List[Tensor] - grid values
            - x_bivariate: List[Tensor] - grid value pairs
        """
        if self.method == 'dirac':
            if self.logit_y0 is None:
                self.calculate_baseline(x)
            return self._calculate_dirac_subset(
                x,
                n_steps,
                categorical_threshold,
                subtract_univariate,
                selected_features,
                selected_feature_pairs,
            )
        elif self.method == 'lebesgue':
            return self._calculate_lebesgue_subset(
                x,
                n_steps,
                categorical_threshold,
                subtract_univariate,
                batch_size,
                selected_features=selected_features,
                selected_feature_pairs=selected_feature_pairs,
            )
        else:
            raise ValueError(
                f"Method {self.method} not implemented. Choose 'dirac' or 'lebesgue'."
            )
