"""
Plotting Data Management for PRiSM

Provides a unified data structure for managing partial response data for plotting.
Solves the index mapping problem between dense lists and sparse feature indices.

The key insight: partial_responses_subset() returns DENSE lists (indexed 0, 1, 2, ...)
but the rest of the code expects to access by FEATURE INDEX (e.g., 5, 10, 17).
This module bridges that gap with clear semantics.
"""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from prism.feature_labels import FeatureLabelManager
    from prism.plotting.index_mapper import IndexMapper
    from prism.plotting.metadata import FeatureMetadataRegistry
    from prism.preprocessing import OneHotGroupManager, PRiSMScaler

logger = logging.getLogger(__name__)


@dataclass
class FeatureInfo:
    """Information about a single feature."""

    index: int  # Index in collapsed feature space
    name: str  # Column name in data
    label: str  # User-friendly label for plots (may contain \n)
    is_categorical: bool  # Whether feature is categorical
    response: Optional[np.ndarray] = None  # Partial response values
    x_values: Optional[np.ndarray] = None  # Grid values for response


@dataclass
class FeaturePairInfo:
    """Information about a feature pair (bivariate interaction)."""

    indices: Tuple[int, int]  # (i, j) in collapsed feature space, i < j
    names: Tuple[str, str]  # Column names
    labels: Tuple[str, str]  # User-friendly labels
    is_categorical: Tuple[bool, bool]  # Categorical flags for each
    response: Optional[np.ndarray] = None  # Bivariate response values
    x_values: Optional[np.ndarray] = None  # Grid values (n_points, 2)
    skipped: bool = False  # True if same-group pair (response is zero)


@dataclass
class BinaryFeatureGroup:
    """Information about a group of binary categorical features for combined rendering."""

    # List of features in this group (1-3 features)
    features: List[FeatureInfo]
    # Y-positions for each feature line (dynamically calculated)
    y_positions: List[float] = field(default_factory=list)

    def __post_init__(self):
        """Calculate y-positions based on number of features."""
        if not self.y_positions:  # Only calculate if not explicitly set
            n = len(self.features)
            if n == 1:
                self.y_positions = [0.5]
            elif n == 2:
                self.y_positions = [0.33, 0.66]
            elif n == 3:
                self.y_positions = [0.17, 0.5, 0.83]
            else:
                # Fallback for any other count (shouldn't happen)
                step = 1.0 / (n + 1)
                self.y_positions = [step * (i + 1) for i in range(n)]

    @property
    def n_features(self) -> int:
        """Number of features in this group."""
        return len(self.features)


@dataclass
class PlottingDataBundle:
    """
    Unified data structure for partial response plotting.

    Solves the indexing problem by providing:
    - Dense iteration over selected features/pairs
    - Sparse access by feature index when needed
    - Clear separation of names, labels, and data

    Example Usage
    -------------
    >>> bundle = PlottingDataBundle.from_partial_responses(...)
    >>>
    >>> # Iterate over selected univariate features (dense)
    >>> for info in bundle.univariate_features():
    ...     plot(info.x_values, info.response, label=info.label)
    >>>
    >>> # Access by feature index (sparse)
    >>> info = bundle.get_univariate(feature_idx=5)
    >>>
    >>> # Get all feature names/labels
    >>> names = bundle.selected_feature_names
    >>> labels = bundle.selected_feature_labels
    """

    # All collapsed feature names (full list)
    all_feature_names: List[str]

    # Selected features (dense lists, aligned)
    _univariate_info: List[FeatureInfo] = field(default_factory=list)
    _bivariate_info: List[FeaturePairInfo] = field(default_factory=list)

    # Optional scaler for denormalization
    scaler: Optional[Any] = None

    # Original data tensor (for histogram overlays)
    x_data: Optional[np.ndarray] = None

    # Metadata
    n_steps: int = 50
    categorical_threshold: int = 15

    # NEW: Service objects for refactored architecture
    index_mapper: Optional['IndexMapper'] = None
    metadata_registry: Optional['FeatureMetadataRegistry'] = None

    def univariate_features(self) -> List[FeatureInfo]:
        """Iterate over selected univariate features in order."""
        return self._univariate_info

    def bivariate_pairs(self) -> List[FeaturePairInfo]:
        """Iterate over selected bivariate pairs in order."""
        return self._bivariate_info

    def get_univariate(self, feature_idx: int) -> Optional[FeatureInfo]:
        """
        Get univariate info by feature index (sparse access).

        **Note:** This method is for testing only. Production code should use
        dense iteration via `univariate_features()` for better performance.
        Uses linear search - acceptable for testing.
        """
        for info in self._univariate_info:
            if info.index == feature_idx:
                return info
        return None

    def get_bivariate(self, pair: Tuple[int, int]) -> Optional[FeaturePairInfo]:
        """
        Get bivariate info by feature pair (sparse access).

        **Note:** This method is for testing only. Production code should use
        dense iteration via `bivariate_pairs()` for better performance.
        Uses linear search - acceptable for testing.
        """
        # Ensure pair is in canonical order (i < j)
        if pair[0] > pair[1]:
            pair = (pair[1], pair[0])
        for info in self._bivariate_info:
            if info.indices == pair:
                return info
        return None

    @property
    def selected_univariate_indices(self) -> List[int]:
        """Feature indices of selected univariate features."""
        return [info.index for info in self._univariate_info]

    @property
    def selected_bivariate_pairs(self) -> List[Tuple[int, int]]:
        """Feature index pairs of selected bivariate interactions."""
        return [info.indices for info in self._bivariate_info]

    @property
    def n_univariate(self) -> int:
        """Number of selected univariate features."""
        return len(self._univariate_info)

    @property
    def n_bivariate(self) -> int:
        """Number of selected bivariate pairs."""
        return len(self._bivariate_info)

    @property
    def n_plots(self) -> int:
        """Total number of plots needed."""
        return self.n_univariate + self.n_bivariate

    def is_categorical(self, feature_idx: int) -> bool:
        """Check if a feature is categorical."""
        info = self.get_univariate(feature_idx)
        if info:
            return info.is_categorical
        # Fallback: check x values
        return False

    def get_label(self, feature_idx: int, clean: bool = False) -> str:
        """Get label for a feature, optionally removing newlines."""
        info = self.get_univariate(feature_idx)
        label = info.label if info else self.all_feature_names[feature_idx]
        if clean:
            label = label.replace('\n', ' ')
        return label

    def denormalize(self, values: np.ndarray, feature_idx: int) -> np.ndarray:
        """Denormalize values for a feature using the scaler."""
        if self.scaler is None:
            return values

        # Create a full-width array to inverse transform
        n_features = len(self.all_feature_names)
        full_array = np.zeros((len(values), n_features))
        full_array[:, feature_idx] = values.flatten()

        # Inverse transform and extract the column
        denorm = self.scaler.inverse_transform(full_array)
        return denorm[:, feature_idx]

    @property
    def has_services(self) -> bool:
        """Check if bundle has new service objects (refactored architecture)."""
        return self.index_mapper is not None and self.metadata_registry is not None

    @classmethod
    def from_partial_responses(
        cls,
        univariate_responses: List[np.ndarray],
        bivariate_responses: List[np.ndarray],
        x_univariate: List[np.ndarray],
        x_bivariate: List[np.ndarray],
        selected_univariate_indices: List[int],
        selected_bivariate_pairs: List[Tuple[int, int]],
        all_feature_names: List[str],
        label_manager: Optional['FeatureLabelManager'] = None,
        is_categorical: Optional[List[bool]] = None,
        scaler: Optional['PRiSMScaler'] = None,
        x_data: Optional[np.ndarray] = None,
        n_steps: int = 50,
        categorical_threshold: int = 15,
    ) -> 'PlottingDataBundle':
        """
        Create PlottingDataBundle from partial_responses_subset() output.

        Parameters
        ----------
        univariate_responses : List[np.ndarray]
            Dense list of univariate responses, aligned with selected_univariate_indices
        bivariate_responses : List[np.ndarray]
            Dense list of bivariate responses, aligned with selected_bivariate_pairs
        x_univariate : List[np.ndarray]
            Dense list of x grid values for univariate responses
        x_bivariate : List[np.ndarray]
            Dense list of x grid values for bivariate responses
        selected_univariate_indices : List[int]
            Feature indices that were selected (in collapsed space)
        selected_bivariate_pairs : List[Tuple[int, int]]
            Feature pairs that were selected
        all_feature_names : List[str]
            Complete list of feature names (collapsed space)
        label_manager : Optional['FeatureLabelManager']
            Manager for getting user-friendly labels
        is_categorical : Optional[List[bool]]
            Pre-computed categorical flags, aligned with selected_univariate_indices
        scaler : Optional['PRiSMScaler']
            Scaler for denormalization
        x_data : Optional[np.ndarray]
            Original data for histograms
        n_steps : int
            Number of grid steps
        categorical_threshold : int
            Threshold for categorical detection

        Returns
        -------
        PlottingDataBundle
            Fully populated bundle ready for plotting
        """
        bundle = cls(
            all_feature_names=all_feature_names,
            scaler=scaler,
            x_data=x_data,
            n_steps=n_steps,
            categorical_threshold=categorical_threshold,
        )

        # Build univariate info
        for pos, (feat_idx, response, x_vals) in enumerate(
            zip(selected_univariate_indices, univariate_responses, x_univariate)
        ):
            name = all_feature_names[feat_idx]

            # Get label from manager or fall back to name
            if label_manager is not None:
                label = label_manager.get_label(name)
            else:
                label = name

            # Determine categorical flag
            if is_categorical is not None and pos < len(is_categorical):
                is_cat = is_categorical[pos]
            else:
                # Infer from x values
                is_cat = (
                    len(np.unique(x_vals)) < categorical_threshold if x_vals is not None else False
                )

            info = FeatureInfo(
                index=feat_idx,
                name=name,
                label=label,
                is_categorical=is_cat,
                response=response,
                x_values=x_vals,
            )
            bundle._univariate_info.append(info)

        # Build bivariate info
        for pos, (pair, response, x_vals) in enumerate(
            zip(selected_bivariate_pairs, bivariate_responses, x_bivariate)
        ):
            i, j = pair
            name1 = all_feature_names[i]
            name2 = all_feature_names[j]

            # Get labels
            if label_manager is not None:
                label1 = label_manager.get_label(name1)
                label2 = label_manager.get_label(name2)
            else:
                label1, label2 = name1, name2

            # Determine categorical flags
            is_cat1 = (
                bundle.is_categorical(i)
                if bundle.get_univariate(i) is not None
                else (
                    len(np.unique(x_vals[:, 0])) < categorical_threshold
                    if x_vals is not None and x_vals.ndim == 2
                    else False
                )
            )
            is_cat2 = (
                bundle.is_categorical(j)
                if bundle.get_univariate(j) is not None
                else (
                    len(np.unique(x_vals[:, 1])) < categorical_threshold
                    if x_vals is not None and x_vals.ndim == 2
                    else False
                )
            )

            # Check if this was a skipped pair (same-group, zero response)
            skipped = (
                response is not None and response.size == 1 and np.isclose(response.item(), 0)
            )

            info = FeaturePairInfo(
                indices=pair,
                names=(name1, name2),
                labels=(label1, label2),
                is_categorical=(is_cat1, is_cat2),
                response=response,
                x_values=x_vals,
                skipped=skipped,
            )
            bundle._bivariate_info.append(info)

        logger.debug(
            f"Created PlottingDataBundle with {bundle.n_univariate} univariate, "
            f"{bundle.n_bivariate} bivariate features"
        )

        return bundle

    @classmethod
    def from_partial_responses_with_services(
        cls,
        univariate_responses: List[np.ndarray],
        bivariate_responses: List[np.ndarray],
        x_univariate: List[np.ndarray],
        x_bivariate: List[np.ndarray],
        selected_univariate_indices: List[int],
        selected_bivariate_pairs: List[Tuple[int, int]],
        all_feature_names: List[str],
        original_feature_names: List[str],
        collapsed_feature_names: List[str],
        scaler: Optional['PRiSMScaler'],
        x_data: np.ndarray,
        n_steps: int,
        categorical_threshold: int,
        group_manager: Optional['OneHotGroupManager'] = None,
        label_manager: Optional['FeatureLabelManager'] = None,
    ) -> 'PlottingDataBundle':
        """
        Create PlottingDataBundle with new service objects (refactored architecture).

        This factory method creates the bundle with the new architecture components:
        IndexMapper, FeatureMetadataRegistry, and DenormalizationService.

        Parameters
        ----------
        univariate_responses : List[np.ndarray]
            Dense list of univariate responses
        bivariate_responses : List[np.ndarray]
            Dense list of bivariate responses
        x_univariate : List[np.ndarray]
            Dense list of x grid values for univariate responses
        x_bivariate : List[np.ndarray]
            Dense list of x grid values for bivariate responses
        selected_univariate_indices : List[int]
            Feature indices that were selected (in collapsed space)
        selected_bivariate_pairs : List[Tuple[int, int]]
            Feature pairs that were selected
        all_feature_names : List[str]
            Complete list of feature names (for backward compatibility)
        original_feature_names : List[str]
            Feature names in original OHE space
        collapsed_feature_names : List[str]
            Feature names in collapsed space
        scaler : Optional[PRiSMScaler]
            Scaler for denormalization
        x_data : np.ndarray
            Original data (in collapsed space) for histograms
        n_steps : int
            Number of grid steps
        categorical_threshold : int
            Threshold for categorical detection
        group_manager : Optional[OneHotGroupManager]
            Manager for one-hot groups
        label_manager : Optional[FeatureLabelManager]
            Manager for user-friendly labels

        Returns
        -------
        PlottingDataBundle
            Fully populated bundle with service objects
        """
        from prism.plotting.index_mapper import IndexMapper
        from prism.plotting.metadata import FeatureMetadataRegistry

        # 1. Create IndexMapper
        index_mapper = IndexMapper(
            original_names=original_feature_names,
            collapsed_names=collapsed_feature_names,
            selected_indices=selected_univariate_indices,
            group_manager=group_manager,
        )

        # 2. Create FeatureMetadataRegistry
        metadata_registry = FeatureMetadataRegistry(
            index_mapper=index_mapper,
            all_feature_names=collapsed_feature_names,
            x_data=x_data,
            categorical_threshold=categorical_threshold,
            group_manager=group_manager,
            label_manager=label_manager,
        )

        # 3. Build FeatureInfo using metadata
        univariate_info = []
        for dense_idx, (response, x_vals) in enumerate(zip(univariate_responses, x_univariate)):
            metadata = metadata_registry.get_by_dense(dense_idx)
            info = FeatureInfo(
                index=metadata.collapsed_idx,
                name=metadata.column_name,
                label=metadata.user_label,
                is_categorical=metadata.is_categorical,
                response=response,
                x_values=x_vals,
            )
            univariate_info.append(info)

        # 5. Build FeaturePairInfo using metadata
        bivariate_info = []
        for (i, j), response, x_vals in zip(
            selected_bivariate_pairs, bivariate_responses, x_bivariate
        ):
            # Get metadata for both features
            metadata_i = metadata_registry.get_by_collapsed(i)
            metadata_j = metadata_registry.get_by_collapsed(j)

            # Check if same-group pair (skip if so)
            skipped = False
            if metadata_i is not None and metadata_j is not None:
                skipped = (
                    metadata_i.is_collapsed_group
                    and metadata_j.is_collapsed_group
                    and metadata_i.column_name == metadata_j.column_name
                )

            if skipped:
                # Create dummy entry for skipped pairs
                response = np.array([0.0])
                x_vals = np.array([[0.0, 0.0]])

            # Get names, labels, and categorical flags with fallbacks for missing metadata
            name_i = (
                metadata_i.column_name if metadata_i is not None else collapsed_feature_names[i]
            )
            name_j = (
                metadata_j.column_name if metadata_j is not None else collapsed_feature_names[j]
            )
            label_i = (
                metadata_i.user_label if metadata_i is not None else collapsed_feature_names[i]
            )
            label_j = (
                metadata_j.user_label if metadata_j is not None else collapsed_feature_names[j]
            )

            # Determine categorical flags with fallback to cardinality-based detection
            # This handles features that are only in bivariate pairs (no univariate selection)
            if metadata_i is not None:
                is_cat_i = metadata_i.is_categorical
            else:
                # Fallback: use cardinality-based detection from x_data
                # Also check if it's a collapsed one-hot group
                if group_manager and group_manager.is_categorical_group(
                    collapsed_feature_names[i]
                ):
                    is_cat_i = True
                elif x_data is not None:
                    col_data = x_data[:, i]
                    n_unique = len(np.unique(col_data))
                    is_cat_i = n_unique < categorical_threshold
                else:
                    is_cat_i = False

            if metadata_j is not None:
                is_cat_j = metadata_j.is_categorical
            else:
                # Fallback: use cardinality-based detection from x_data
                if group_manager and group_manager.is_categorical_group(
                    collapsed_feature_names[j]
                ):
                    is_cat_j = True
                elif x_data is not None:
                    col_data = x_data[:, j]
                    n_unique = len(np.unique(col_data))
                    is_cat_j = n_unique < categorical_threshold
                else:
                    is_cat_j = False

            # For features without metadata (not selected univariate),
            # try to get labels from the label_manager directly
            if metadata_i is None and label_manager:
                label_i = label_manager.get_label(collapsed_feature_names[i])
            if metadata_j is None and label_manager:
                label_j = label_manager.get_label(collapsed_feature_names[j])

            info = FeaturePairInfo(
                indices=(i, j),
                names=(name_i, name_j),
                labels=(label_i, label_j),
                is_categorical=(is_cat_i, is_cat_j),
                response=response,
                x_values=x_vals,
                skipped=skipped,
            )
            bivariate_info.append(info)

        # 5. Create bundle with services
        bundle = cls(
            all_feature_names=all_feature_names,
            _univariate_info=univariate_info,
            _bivariate_info=bivariate_info,
            scaler=scaler,
            x_data=x_data,
            n_steps=n_steps,
            categorical_threshold=categorical_threshold,
            index_mapper=index_mapper,
            metadata_registry=metadata_registry,
        )

        logger.debug(
            f"Created PlottingDataBundle with services: {bundle.n_univariate} univariate, "
            f"{bundle.n_bivariate} bivariate features (collapse_mode={index_mapper.is_collapse_mode})"
        )

        return bundle
