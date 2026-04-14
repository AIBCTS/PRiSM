"""
Feature metadata registry for centralized feature property management.

This module provides the FeatureMetadataRegistry class for storing and looking up
feature properties (categorical vs continuous, labels, statistics, etc.) by various
keys (dense index, collapsed index, or feature name).
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from prism.feature_labels import FeatureLabelManager
    from prism.plotting.index_mapper import IndexMapper
    from prism.preprocessing import OneHotGroupManager

logger = logging.getLogger(__name__)


@dataclass
class FeatureMetadata:
    """Complete metadata for a single feature.

    Attributes
    ----------
    dense_idx : int
        Position in the selected features list (0, 1, 2, ...)
    collapsed_idx : int
        Index in collapsed space (after one-hot group collapse)
    original_indices : List[int]
        Indices in original OHE space (multiple if collapsed group, single otherwise)
    column_name : str
        Feature name for lookups and code (e.g., 'diagn', 'age')
    user_label : str
        Feature label for display (e.g., 'Diagnosis\\n(categorical)', 'Age (years)')
    is_categorical : bool
        True if feature should be treated as categorical
    is_collapsed_group : bool
        True if this feature is a collapsed one-hot group
    unique_values : Optional[np.ndarray]
        Unique values for categorical features (None for continuous)
    min_value : Optional[float]
        Minimum value for continuous features (None for categorical)
    max_value : Optional[float]
        Maximum value for continuous features (None for categorical)
    """

    dense_idx: int
    collapsed_idx: int
    original_indices: List[int]
    column_name: str
    user_label: str
    is_categorical: bool
    is_collapsed_group: bool
    unique_values: Optional[np.ndarray] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None


class FeatureMetadataRegistry:
    """Registry for looking up feature metadata by multiple keys.

    This class centralizes all feature metadata and provides clean lookup methods
    by dense index, collapsed index, or feature name. It resolves the ambiguity
    between column names (for code) and user labels (for display).

    Parameters
    ----------
    index_mapper : IndexMapper
        Mapper for converting between index spaces
    all_feature_names : List[str]
        All feature names in collapsed space (for lookup)
    x_data : np.ndarray
        Feature data in collapsed space (for computing statistics)
    categorical_threshold : int
        Maximum number of unique values to consider a feature categorical
        (only used as fallback if not semantically defined)
    group_manager : Optional[OneHotGroupManager]
        Manager for one-hot group definitions (for semantic categorical detection)
    label_manager : Optional[FeatureLabelManager]
        Manager for user-friendly feature labels (for display)

    Examples
    --------
    >>> registry = FeatureMetadataRegistry(
    ...     index_mapper=mapper,
    ...     all_feature_names=['age', 'bmi', 'diagn'],
    ...     x_data=X_collapsed,
    ...     categorical_threshold=10,
    ...     group_manager=group_mgr,
    ...     label_manager=label_mgr,
    ... )
    >>> # Lookup by dense index
    >>> metadata = registry.get_by_dense(0)
    >>> print(metadata.column_name, metadata.user_label)
    'age' 'Age (years)'
    >>> # Lookup by name
    >>> metadata = registry.get_by_name('diagn')
    >>> print(metadata.is_categorical, metadata.is_collapsed_group)
    True True
    """

    def __init__(
        self,
        index_mapper: 'IndexMapper',
        all_feature_names: List[str],
        x_data: np.ndarray,
        categorical_threshold: int,
        group_manager: Optional['OneHotGroupManager'] = None,
        label_manager: Optional['FeatureLabelManager'] = None,
    ):
        self._metadata: List[FeatureMetadata] = []
        self._name_to_metadata: Dict[str, FeatureMetadata] = {}
        self._index_mapper = index_mapper

        # Build metadata for each selected univariate feature
        for dense_idx in range(index_mapper.n_dense):
            collapsed_idx = index_mapper.dense_to_collapsed(dense_idx)
            column_name = all_feature_names[collapsed_idx]

            # Get user label (fallback to column name if no label manager)
            user_label = label_manager.get_label(column_name) if label_manager else column_name

            # Get original indices (for OHE data access)
            original_indices = index_mapper.collapsed_to_original(collapsed_idx)

            # Get feature data for this feature (in collapsed space)
            feature_data = x_data[:, collapsed_idx]

            # Determine if categorical
            is_categorical = self._determine_categorical(
                column_name, feature_data, categorical_threshold, group_manager
            )

            # Check if this is a collapsed group
            is_collapsed_group = (
                group_manager.is_categorical_group(column_name) if group_manager else False
            )

            # Compute statistics based on type
            if is_categorical:
                unique_values = np.unique(feature_data)
                min_val, max_val = None, None
            else:
                unique_values = None
                min_val, max_val = float(np.min(feature_data)), float(np.max(feature_data))

            # Create metadata entry
            metadata = FeatureMetadata(
                dense_idx=dense_idx,
                collapsed_idx=collapsed_idx,
                original_indices=original_indices,
                column_name=column_name,
                user_label=user_label,
                is_categorical=is_categorical,
                is_collapsed_group=is_collapsed_group,
                unique_values=unique_values,
                min_value=min_val,
                max_value=max_val,
            )

            self._metadata.append(metadata)
            self._name_to_metadata[column_name] = metadata

        logger.debug(f"FeatureMetadataRegistry initialized with {len(self._metadata)} features")

    def _determine_categorical(
        self,
        column_name: str,
        data: np.ndarray,
        threshold: int,
        group_manager: Optional['OneHotGroupManager'],
    ) -> bool:
        """Determine if feature is categorical.

        Priority: 1) Semantic (from group_manager), 2) Cardinality threshold

        Parameters
        ----------
        column_name : str
            Feature name
        data : np.ndarray
            Feature values
        threshold : int
            Maximum unique values to consider categorical
        group_manager : Optional[OneHotGroupManager]
            Group manager for semantic detection

        Returns
        -------
        bool
            True if categorical
        """
        # Priority 1: Semantic detection (collapsed groups are categorical)
        if group_manager and group_manager.is_categorical_group(column_name):
            return True

        # Priority 2: Cardinality-based detection
        n_unique = len(np.unique(data))
        return n_unique < threshold

    def get_by_dense(self, dense_idx: int) -> FeatureMetadata:
        """Get metadata by dense index.

        Parameters
        ----------
        dense_idx : int
            Position in selected features list (0, 1, 2, ...)

        Returns
        -------
        FeatureMetadata
            Feature metadata

        Raises
        ------
        IndexError
            If dense_idx is out of range
        """
        if dense_idx < 0 or dense_idx >= len(self._metadata):
            raise IndexError(
                f"Dense index {dense_idx} out of range. "
                f"Valid range: [0, {len(self._metadata)-1}]"
            )
        return self._metadata[dense_idx]

    def get_by_collapsed(self, collapsed_idx: int) -> Optional[FeatureMetadata]:
        """Get metadata by collapsed index.

        Parameters
        ----------
        collapsed_idx : int
            Index in collapsed space

        Returns
        -------
        FeatureMetadata or None
            Feature metadata, or None if this collapsed feature was not selected
        """
        for metadata in self._metadata:
            if metadata.collapsed_idx == collapsed_idx:
                return metadata
        return None

    def get_by_name(self, column_name: str) -> Optional[FeatureMetadata]:
        """Get metadata by feature name.

        Parameters
        ----------
        column_name : str
            Feature name (e.g., 'age', 'diagn')

        Returns
        -------
        FeatureMetadata or None
            Feature metadata, or None if feature not found
        """
        return self._name_to_metadata.get(column_name)

    def __len__(self) -> int:
        """Get number of features in registry."""
        return len(self._metadata)

    def __iter__(self):
        """Iterate over all metadata entries (in dense order)."""
        return iter(self._metadata)

    def __repr__(self) -> str:
        return f"FeatureMetadataRegistry(n_features={len(self._metadata)})"
