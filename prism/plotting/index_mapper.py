"""
Index space mapping utilities.

This module provides the IndexMapper class for handling conversions between
three different index spaces used in the plotting pipeline:

1. Original OHE (One-Hot Encoded): Model input space with all one-hot features
2. Collapsed: LASSO space where one-hot groups are collapsed to integers
3. Dense: Plotting space with only selected features (0, 1, 2, ...)

The IndexMapper eliminates index confusion by providing explicit conversion methods.
"""

import logging
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from prism.preprocessing import OneHotGroupManager

logger = logging.getLogger(__name__)


class IndexMapper:
    """Maps between three index spaces: Original OHE, Collapsed, and Dense.

    This class provides explicit conversions between the three index spaces used
    in the plotting pipeline, eliminating the confusion between dense (sequential)
    and sparse (feature space) indices.

    Index Spaces:
    - **Original OHE**: One-hot encoded features (model input space)
      Example: [age, bmi, diagn_CAD, diagn_Valve, diagn_Other, ...]  (50 features)

    - **Collapsed**: Groups collapsed to integers (LASSO space)
      Example: [age, bmi, diagn, ...]  (30 features)
      Where diagn = {0:CAD, 1:Valve, 2:Other}

    - **Dense**: Position in selected features list (plotting space)
      Example: If LASSO selected indices [1, 5, 17] from collapsed space,
      dense indices are [0, 1, 2] mapping to collapsed [1, 5, 17]

    Parameters
    ----------
    original_names : List[str]
        Feature names in original one-hot encoded space
    collapsed_names : List[str]
        Feature names in collapsed space (groups as single features)
    selected_indices : List[int]
        Selected feature indices in collapsed space (from LASSO)
    group_manager : Optional[OneHotGroupManager]
        Manager for one-hot group definitions (for OHE <-> Collapsed conversions)

    Examples
    --------
    >>> # Without collapse (1:1:1 mapping)
    >>> mapper = IndexMapper(
    ...     original_names=['age', 'bmi', 'glucose'],
    ...     collapsed_names=['age', 'bmi', 'glucose'],
    ...     selected_indices=[0, 2],  # LASSO selected age and glucose
    ... )
    >>> mapper.dense_to_collapsed(0)  # First selected feature
    0  # -> age (index 0 in collapsed space)
    >>> mapper.dense_to_collapsed(1)  # Second selected feature
    2  # -> glucose (index 2 in collapsed space)

    >>> # With collapse (group mapping)
    >>> mapper = IndexMapper(
    ...     original_names=['age', 'diagn_CAD', 'diagn_Valve'],
    ...     collapsed_names=['age', 'diagn'],
    ...     selected_indices=[1],  # LASSO selected diagn
    ...     group_manager=group_manager,  # knows diagn = [diagn_CAD, diagn_Valve]
    ... )
    >>> mapper.collapsed_to_original(1)  # diagn in collapsed space
    [1, 2]  # -> [diagn_CAD, diagn_Valve] in original space
    """

    def __init__(
        self,
        original_names: List[str],
        collapsed_names: List[str],
        selected_indices: List[int],
        group_manager: Optional['OneHotGroupManager'] = None,
    ):
        self._original_names = original_names
        self._collapsed_names = collapsed_names
        self._selected_indices = selected_indices
        self._group_manager = group_manager

        # Build reverse lookup: collapsed_idx -> dense_idx
        self._collapsed_to_dense_map = {
            collapsed_idx: dense_idx for dense_idx, collapsed_idx in enumerate(selected_indices)
        }

        logger.debug(
            f"IndexMapper initialized: {len(original_names)} original, "
            f"{len(collapsed_names)} collapsed, {len(selected_indices)} selected"
        )

    def dense_to_collapsed(self, dense_idx: int) -> int:
        """Convert dense position to collapsed feature index.

        Parameters
        ----------
        dense_idx : int
            Position in the selected features list (0, 1, 2, ...)

        Returns
        -------
        int
            Feature index in collapsed space

        Raises
        ------
        IndexError
            If dense_idx is out of range

        Examples
        --------
        >>> # If selected_indices = [1, 5, 17]
        >>> mapper.dense_to_collapsed(0)
        1  # First selected feature is at collapsed index 1
        >>> mapper.dense_to_collapsed(2)
        17  # Third selected feature is at collapsed index 17
        """
        if dense_idx < 0 or dense_idx >= len(self._selected_indices):
            raise IndexError(
                f"Dense index {dense_idx} out of range. "
                f"Valid range: [0, {len(self._selected_indices)-1}]"
            )
        return self._selected_indices[dense_idx]

    def collapsed_to_dense(self, collapsed_idx: int) -> Optional[int]:
        """Convert collapsed feature index to dense position.

        Parameters
        ----------
        collapsed_idx : int
            Feature index in collapsed space

        Returns
        -------
        int or None
            Position in selected features list, or None if not selected

        Examples
        --------
        >>> # If selected_indices = [1, 5, 17]
        >>> mapper.collapsed_to_dense(5)
        1  # Collapsed index 5 is the 2nd selected feature (dense index 1)
        >>> mapper.collapsed_to_dense(10)
        None  # Collapsed index 10 was not selected
        """
        return self._collapsed_to_dense_map.get(collapsed_idx)

    def collapsed_to_original(self, collapsed_idx: int) -> List[int]:
        """Convert collapsed index to original OHE indices.

        If the feature is a collapsed group, returns all member indices.
        Otherwise, returns a single-element list with the original index.

        Parameters
        ----------
        collapsed_idx : int
            Feature index in collapsed space

        Returns
        -------
        List[int]
            Indices in original OHE space (multiple if group, single otherwise)

        Raises
        ------
        IndexError
            If collapsed_idx is out of range
        ValueError
            If group mapping is inconsistent

        Examples
        --------
        >>> # Without group: 1:1 mapping
        >>> mapper.collapsed_to_original(2)
        [2]

        >>> # With group: diagn at collapsed index 1 -> [diagn_CAD, diagn_Valve]
        >>> mapper.collapsed_to_original(1)
        [2, 3]  # Original indices of the one-hot group members
        """
        if collapsed_idx < 0 or collapsed_idx >= len(self._collapsed_names):
            raise IndexError(
                f"Collapsed index {collapsed_idx} out of range. "
                f"Valid range: [0, {len(self._collapsed_names)-1}]"
            )

        if self._group_manager is None:
            # No collapse: 1:1 mapping
            return [collapsed_idx]

        name = self._collapsed_names[collapsed_idx]

        # Check if this is a collapsed group
        if name in self._group_manager.groups_dict:
            # This is a collapsed group - return all member indices
            group_members = self._group_manager.groups_dict[name]
            try:
                return [self._original_names.index(member) for member in group_members]
            except ValueError as e:
                raise ValueError(
                    f"Group member not found in original names. "
                    f"Group: {name}, Members: {group_members}, "
                    f"Original names: {self._original_names[:10]}..."
                ) from e
        else:
            # Regular feature - find in original names
            try:
                return [self._original_names.index(name)]
            except ValueError:
                # If not found, it might be that collapsed and original are identical
                # Try using the same index
                if collapsed_idx < len(self._original_names):
                    logger.warning(
                        f"Feature '{name}' not found in original names, "
                        f"using index {collapsed_idx}"
                    )
                    return [collapsed_idx]
                raise ValueError(
                    f"Feature '{name}' not found in original names and "
                    f"collapsed index {collapsed_idx} exceeds original length"
                )

    @property
    def n_dense(self) -> int:
        """Number of selected features (dense space dimension)."""
        return len(self._selected_indices)

    @property
    def n_collapsed(self) -> int:
        """Number of collapsed features (collapsed space dimension)."""
        return len(self._collapsed_names)

    @property
    def n_original(self) -> int:
        """Number of original OHE features (original space dimension)."""
        return len(self._original_names)

    @property
    def is_collapse_mode(self) -> bool:
        """Check if collapse mode is active (group_manager present)."""
        return self._group_manager is not None

    def __repr__(self) -> str:
        return (
            f"IndexMapper(original={self.n_original}, "
            f"collapsed={self.n_collapsed}, "
            f"dense={self.n_dense}, "
            f"collapse_mode={self.is_collapse_mode})"
        )
