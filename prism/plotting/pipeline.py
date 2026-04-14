"""
Plotting data preparation pipeline.

This module provides the PlottingPipeline class that orchestrates the preparation
of plotting data from partial responses, coordinating the architecture
components (IndexMapper, FeatureMetadataRegistry).

The pipeline isolates LASSO coupling to the initial data preparation phase,
enabling clean separation of concerns.

Data Flow Architecture:
- Model predictions are computed in SCALED space (required for model accuracy)
- All x-values passed to PlottingDataBundle are UNSCALED (ready for display)
- This includes: x_univariate grids, x_bivariate grids, and x_data (histogram data)
- Denormalization happens ONCE in this pipeline, not in the renderer

This design:
1. Ensures collapse_onehot_features() always receives unscaled data (required)
2. Simplifies the renderer (no denormalization logic needed)
3. Provides a clear data contract for the bundle (all x-values are display-ready)
"""

import logging
from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np
import torch

from prism.partial_responses import partial_responses_subset, to_numpy
from prism.plotting_data import PlottingDataBundle
from prism.preprocessing import collapse_onehot_features

if TYPE_CHECKING:
    from prism.feature_labels import FeatureLabelManager
    from prism.lasso import LassoResultsManager
    from prism.preprocessing import OneHotGroupManager, PRiSMScaler

logger = logging.getLogger(__name__)


class PlottingPipeline:
    """Orchestrates plotting data preparation and transformation.

    This class coordinates the entire plotting data preparation workflow:
    1. Extract feature selections from LASSO results
    2. Calculate partial responses with automatic collapse
    3. Prepare scaler (collapse if needed)
    4. Create PlottingDataBundle with service objects
    5. Optionally apply beta scaling

    After bundle creation, LASSO access is isolated to beta scaling only.

    Parameters
    ----------
    lasso_results : LassoResultsManager
        LASSO results containing feature selections and beta values
    group_manager : Optional[OneHotGroupManager]
        Manager for one-hot group collapse (None for no collapse)
    label_manager : Optional[FeatureLabelManager]
        Manager for user-friendly feature labels (None for default names)

    Examples
    --------
    >>> pipeline = PlottingPipeline(
    ...     lasso_results=lasso_mgr,
    ...     group_manager=group_mgr,
    ...     label_manager=label_mgr,
    ... )
    >>> # Prepare data bundle
    >>> bundle = pipeline.prepare_plotting_bundle(
    ...     x=x_test,
    ...     model=trained_model,
    ...     scaler=scaler,
    ...     feature_names=original_feature_names,
    ... )
    >>> # Apply beta scaling
    >>> bundle = pipeline.apply_beta_scaling(bundle)
    """

    def __init__(
        self,
        lasso_results: 'LassoResultsManager',
        group_manager: Optional['OneHotGroupManager'] = None,
        label_manager: Optional['FeatureLabelManager'] = None,
    ):
        self.lasso_results = lasso_results
        self.group_manager = group_manager
        self.label_manager = label_manager

    def prepare_plotting_bundle(
        self,
        x: torch.Tensor,
        model: Any,
        scaler: Optional['PRiSMScaler'] = None,
        n_steps: int = 50,
        method: str = "dirac",
        x_train: Optional[torch.Tensor] = None,
        device: str = "cpu",
        categorical_threshold: int = 15,
        subtract_univariate: bool = True,
        feature_names: Optional[List[str]] = None,
        trim_quantile: Optional[float] = None,
    ) -> PlottingDataBundle:
        """Prepare complete plotting bundle with all services.

        This method orchestrates the entire data preparation workflow:
        - Determines collapse mode from group_manager
        - Gets feature names (provided, inferred, or reconstructed)
        - Derives collapsed names if needed
        - Calculates partial responses
        - Prepares scaler (collapses if needed)
        - Creates bundle with service objects

        Parameters
        ----------
        x : torch.Tensor
            Input data (test set)
        model : Any
            Trained model for partial response calculation
        scaler : Optional[PRiSMScaler]
            Scaler for denormalization (may be in OHE space, will collapse if needed)
        n_steps : int
            Number of grid steps for partial responses
        method : str
            Partial response method ("dirac" or "lebesgue")
        x_train : Optional[torch.Tensor]
            Training data (required for lebesgue method)
        device : str
            Device for computation ("cpu" or "cuda")
        categorical_threshold : int
            Threshold for categorical detection
        subtract_univariate : bool
            Whether to subtract univariate from bivariate responses
        feature_names : Optional[List[str]]
            Original feature names (OHE space). If None, will try to infer or reconstruct.
        trim_quantile : Optional[float]
            Fraction to trim from each tail when generating grids for continuous features.
            E.g., 0.01 uses the 1st to 99th percentile range.

        Returns
        -------
        PlottingDataBundle
            Bundle with partial responses and service objects (not yet beta-scaled)

        Raises
        ------
        ValueError
            If input dimensions are inconsistent across spaces
        """
        # Determine collapse mode
        collapse_mode = self.group_manager is not None

        # Validate input data dimensions
        x_dim = x.shape[1] if hasattr(x, 'shape') else len(x[0])

        # Get feature names
        if feature_names is not None:
            original_feature_names = feature_names
            # Validate feature names match input dimension
            if len(original_feature_names) != x_dim:
                raise ValueError(
                    f"feature_names has {len(original_feature_names)} names, but x has "
                    f"{x_dim} features. Ensure feature_names matches x dimensions."
                )
        elif hasattr(x, 'columns'):
            original_feature_names = list(x.columns)
        else:
            # Fallback: reconstruct from LASSO names
            original_feature_names = self._reconstruct_feature_names()
            logger.warning(
                "No feature names provided and cannot infer from data. "
                "Reconstructed from LASSO results (may be incomplete for collapsed mode)."
            )

        # =======================================================================
        # EARLY DENORMALIZATION: Get unscaled data for collapse operations
        # =======================================================================
        # collapse_onehot_features REQUIRES unscaled data because:
        # - When data is scaled (especially with MedianStdScaler), majority categories
        #   (>50% prevalence) can have their "active" value scaled to 0, making it
        #   impossible to distinguish from reference categories.
        # - See: notes/issue_creditg_missing_categories.md
        if scaler is not None:
            x_unscaled = scaler.inverse_transform(to_numpy(x))
        else:
            x_unscaled = to_numpy(x)

        # Derive collapsed names (uses unscaled data)
        if collapse_mode and original_feature_names is not None:
            _, collapsed_column_names = collapse_onehot_features(
                x_unscaled, self.group_manager, original_feature_names
            )
        else:
            collapsed_column_names = self.lasso_results.univariate_feature_names

        # Get selected features (in collapsed space)
        selected_univ_indices = self.lasso_results.get_selected_univariate_indices()
        selected_biv_pairs = self.lasso_results.get_selected_bivariate_index_pairs()

        # Validate selected indices are within bounds of collapsed space
        n_collapsed = len(collapsed_column_names)
        for idx in selected_univ_indices:
            if idx < 0 or idx >= n_collapsed:
                raise ValueError(
                    f"Selected univariate index {idx} is out of bounds for collapsed space "
                    f"with {n_collapsed} features. This indicates an index space mismatch."
                )

        logger.debug(
            f"Preparing plotting bundle: {len(selected_univ_indices)} univariate, "
            f"{len(selected_biv_pairs)} bivariate features (collapse_mode={collapse_mode})"
        )

        # Calculate partial responses (with automatic collapse)
        # NOTE: Model predictions require SCALED data, so we pass x as-is here.
        # The returned x_univariate and x_bivariate grids are in SCALED space.
        # We will denormalize them below before passing to the bundle.
        univariate_responses, bivariate_responses, x_univariate, x_bivariate = (
            partial_responses_subset(
                x,
                model,
                method=method,
                x_train=x_train,
                device=device,
                n_steps=n_steps,
                categorical_threshold=categorical_threshold,
                subtract_univariate=subtract_univariate,
                selected_features=selected_univ_indices,
                selected_feature_pairs=selected_biv_pairs,
                group_manager=self.group_manager,
                feature_names=original_feature_names,
                trim_quantile=trim_quantile,
                scaler=scaler,
            )
        )

        # Select/collapse scaler if needed (for denormalization below)
        scaler_to_use = self._prepare_scaler(
            scaler, original_feature_names, collapsed_column_names, collapse_mode
        )

        # =======================================================================
        # DENORMALIZATION PHASE: Convert all x-values from scaled to unscaled
        # =======================================================================
        # This ensures the PlottingDataBundle receives all x-values ready for display.
        # The renderer will NOT need to denormalize anything.
        #
        # Note on one-hot groups: For collapsed one-hot groups, the grids contain
        # integer category labels (0, 1, 2, ...) which do NOT need denormalization.
        # Only continuous features need inverse transform.

        # Denormalize x_univariate grids (aligned with selected_univ_indices)
        x_univariate_denorm = self._denormalize_grids(
            x_univariate, scaler_to_use, collapsed_column_names, selected_univ_indices
        )

        # Denormalize x_bivariate grids (aligned with selected_biv_pairs)
        x_bivariate_denorm = self._denormalize_bivariate_grids(
            x_bivariate, scaler_to_use, collapsed_column_names, selected_biv_pairs
        )

        # Prepare x_data (collapsed for histograms) from UNSCALED input
        # NOTE: x_unscaled was computed early (before collapse_onehot_features calls)
        # to ensure all collapse operations use unscaled data.
        if collapse_mode:
            # Collapse using unscaled data (correct integer labels for one-hot groups,
            # and correct denormalized values for continuous features)
            x_numpy, _ = collapse_onehot_features(
                x_unscaled, self.group_manager, original_feature_names
            )
        else:
            # No collapse - x_unscaled is already in the right form
            x_numpy = x_unscaled

        # Create bundle with services
        # NOTE: We pass the DENORMALIZED grids (x_univariate_denorm, x_bivariate_denorm)
        # so the bundle contains display-ready x-values. The renderer should NOT
        # call denormalize_feature() on these values.
        bundle = PlottingDataBundle.from_partial_responses_with_services(
            univariate_responses=univariate_responses,
            bivariate_responses=bivariate_responses,
            x_univariate=x_univariate_denorm,
            x_bivariate=x_bivariate_denorm,
            selected_univariate_indices=selected_univ_indices,
            selected_bivariate_pairs=selected_biv_pairs,
            all_feature_names=self.lasso_results.all_feature_names,
            original_feature_names=original_feature_names,
            collapsed_feature_names=collapsed_column_names,
            scaler=scaler_to_use,
            x_data=x_numpy,
            n_steps=n_steps,
            categorical_threshold=categorical_threshold,
            group_manager=self.group_manager,
            label_manager=self.label_manager,
        )

        logger.info(
            f"Plotting bundle prepared: {bundle.n_univariate} univariate, "
            f"{bundle.n_bivariate} bivariate (has_services={bundle.has_services})"
        )

        return bundle

    def apply_beta_scaling(self, bundle: PlottingDataBundle) -> PlottingDataBundle:
        """Apply LASSO beta scaling to responses.

        This is the ONLY post-bundle LASSO access point. After this step,
        the bundle is fully independent of LASSO results.

        The beta array from get_selected_beta() contains ALL features:
        - beta[0:n_univariate_total]: all univariate betas (18 features in collapsed space)
        - beta[n_univariate_total:]: all bivariate betas in canonical pair order
        - Zero values for unselected features

        We index using COLLAPSED indices (info.index), not dense position.

        Parameters
        ----------
        bundle : PlottingDataBundle
            Bundle with un-scaled partial responses

        Returns
        -------
        PlottingDataBundle
            Same bundle with responses scaled by beta values (mutated in-place)
        """
        beta = self.lasso_results.get_selected_beta()
        n_univariate_total = len(self.lasso_results.univariate_feature_names)
        n_all_features = len(self.lasso_results.all_feature_names)

        # Validate beta array format (must be FULL format)
        if len(beta) != n_all_features:
            raise ValueError(
                f"Beta array has unexpected size: {len(beta)}. "
                f"Expected {n_all_features} (all features). "
                f"get_selected_beta() must return the full beta array with zeros for unselected features."
            )

        # Scale univariate responses
        # Use info.index (collapsed index) to look up beta from full array
        for info in bundle.univariate_features():
            if info.response is not None:
                beta_idx = info.index  # Collapsed index (0-17 range for 18 total features)
                info.response = info.response * beta[beta_idx]

        # Scale bivariate responses
        # Convert pair indices (i, j) to position in all_feature_names
        for info in bundle.bivariate_pairs():
            if info.skipped or info.response is None:
                continue

            # Get collapsed indices for this pair
            i, j = info.indices

            # Calculate position in all_feature_names bivariate section
            # Pairs are in canonical order: (0,1), (0,2), ..., (0,n-1), (1,2), ..., (n-2, n-1)
            # Position = n_univariate + sum of pairs before this one
            # For pair (i, j): position = n_univariate + (i * n_collapsed - i*(i+1)/2) + (j - i - 1)
            n_collapsed = n_univariate_total
            pair_offset = i * n_collapsed - i * (i + 1) // 2 + (j - i - 1)
            beta_idx = n_univariate_total + pair_offset

            info.response = info.response * beta[beta_idx]

        logger.debug("Applied beta scaling to partial responses")
        return bundle

    def _prepare_scaler(
        self,
        scaler: Optional['PRiSMScaler'],
        original_names: List[str],
        collapsed_names: List[str],
        collapse_mode: bool,
    ) -> Optional['PRiSMScaler']:
        """Select appropriate scaler (may need to collapse).

        Parameters
        ----------
        scaler : Optional[PRiSMScaler]
            Input scaler (may be in OHE or collapsed space)
        original_names : List[str]
            Original feature names (OHE space)
        collapsed_names : List[str]
            Collapsed feature names
        collapse_mode : bool
            Whether collapse mode is active

        Returns
        -------
        Optional[PRiSMScaler]
            Scaler in collapsed space (or None if no scaler)
        """
        if not collapse_mode or scaler is None:
            return scaler

        n_original = len(original_names)
        n_collapsed = len(collapsed_names)

        # Determine scaler dimension
        scaler_dim = None
        if hasattr(scaler, 'scaler') and scaler.scaler is not None:
            if hasattr(scaler.scaler, 'median_'):
                scaler_dim = len(scaler.scaler.median_)
            elif hasattr(scaler.scaler, 'mean_'):
                scaler_dim = len(scaler.scaler.mean_)

        if scaler_dim == n_original:
            # Scaler is in OHE space - collapse it
            logger.debug(
                f"Collapsing scaler from {n_original} (OHE) to {n_collapsed} (collapsed) features"
            )
            return self.group_manager.create_collapsed_scaler(scaler, original_names)
        else:
            # Already collapsed or cannot determine - use as-is
            if scaler_dim is not None and scaler_dim != n_collapsed:
                logger.warning(
                    f"Scaler dimension ({scaler_dim}) does not match expected "
                    f"collapsed dimension ({n_collapsed}). Using as-is."
                )
            return scaler

    def _reconstruct_feature_names(self) -> List[str]:
        """Reconstruct original feature names from collapsed names.

        This is a fallback for when feature names are not provided.
        For collapsed mode, this expands groups back to original names.

        Returns
        -------
        List[str]
            Original feature names (best effort)
        """
        collapsed_names = self.lasso_results.univariate_feature_names

        if not self.group_manager:
            return collapsed_names

        original_names = []
        for cname in collapsed_names:
            if cname in self.group_manager.groups_dict:
                # Expand group
                original_names.extend(self.group_manager.groups_dict[cname])
            else:
                original_names.append(cname)

        logger.debug(
            f"Reconstructed {len(original_names)} original names from "
            f"{len(collapsed_names)} collapsed names"
        )

        return original_names

    def _denormalize_grids(
        self,
        x_grids: List[Optional[np.ndarray]],
        scaler: Optional['PRiSMScaler'],
        collapsed_names: List[str],
        selected_indices: List[int],
    ) -> List[Optional[np.ndarray]]:
        """Denormalize univariate x-value grids from scaled to unscaled space.

        For collapsed one-hot groups, the grids are already integer category labels
        (0, 1, 2, ...) and are returned as-is.

        For continuous features, the scaler's inverse_transform is applied.

        Parameters
        ----------
        x_grids : List[Optional[np.ndarray]]
            Scaled x-value grids (dense list aligned with selected_indices)
        scaler : Optional[PRiSMScaler]
            Scaler for inverse transform (None means no scaling)
        collapsed_names : List[str]
            All collapsed feature names (used to identify one-hot groups)
        selected_indices : List[int]
            Selected feature indices in collapsed space (aligned with x_grids)

        Returns
        -------
        List[Optional[np.ndarray]]
            Denormalized x-value grids, same structure as input
        """
        from prism.preprocessing import NoScaler

        result = []
        n_features = len(collapsed_names)

        for dense_idx, grid in enumerate(x_grids):
            if grid is None:
                result.append(None)
                continue

            # Get the actual collapsed feature index
            collapsed_idx = selected_indices[dense_idx]

            # Convert to numpy if needed
            grid_np = to_numpy(grid) if hasattr(grid, 'numpy') else np.asarray(grid)

            # Check if this is a collapsed one-hot group
            if self.group_manager is not None and collapsed_idx < len(collapsed_names):
                feature_name = collapsed_names[collapsed_idx]
                if feature_name in self.group_manager.groups_dict:
                    # One-hot group: integer labels (0, 1, 2, ...), no denormalization needed
                    result.append(grid_np)
                    continue

            # Continuous feature: apply inverse transform
            if scaler is None or isinstance(scaler, NoScaler):
                result.append(grid_np)
            else:
                # Create dummy array for inverse transform
                dummy = np.zeros((len(grid_np), n_features))
                dummy[:, collapsed_idx] = grid_np
                denorm = scaler.inverse_transform(dummy)
                result.append(denorm[:, collapsed_idx])

        return result

    def _denormalize_bivariate_grids(
        self,
        x_grids: List[Optional[np.ndarray]],
        scaler: Optional['PRiSMScaler'],
        collapsed_names: List[str],
        selected_pairs: List[tuple],
    ) -> List[Optional[np.ndarray]]:
        """Denormalize bivariate x-value grids from scaled to unscaled space.

        Each grid has shape (n_points, 2) containing values for feature pair (i, j).
        The grids are aligned with selected_pairs (dense list).

        For one-hot groups, the values are integer category labels (0, 1, 2, ...)
        which do NOT need denormalization. Only continuous features need inverse transform.

        Parameters
        ----------
        x_grids : List[Optional[np.ndarray]]
            Scaled x-value grids (dense list aligned with selected_pairs)
        scaler : Optional[PRiSMScaler]
            Scaler for inverse transform (None means no scaling)
        collapsed_names : List[str]
            All collapsed feature names (used to identify one-hot groups)
        selected_pairs : List[tuple]
            Selected feature pairs as (i, j) tuples (aligned with x_grids)

        Returns
        -------
        List[Optional[np.ndarray]]
            Denormalized x-value grids, same structure as input
        """
        from prism.preprocessing import NoScaler

        result = []
        n_features = len(collapsed_names)

        for dense_idx, grid in enumerate(x_grids):
            if grid is None:
                result.append(None)
                continue

            # Convert to numpy if needed
            grid_np = to_numpy(grid) if hasattr(grid, 'numpy') else np.asarray(grid)

            if scaler is None or isinstance(scaler, NoScaler):
                result.append(grid_np)
                continue

            # Get the actual feature indices from selected_pairs
            i, j = selected_pairs[dense_idx]

            # Check if each feature is a one-hot group (no denorm needed for those)
            def is_onehot_group(idx: int) -> bool:
                if self.group_manager is None or idx >= len(collapsed_names):
                    return False
                feature_name = collapsed_names[idx]
                return feature_name in self.group_manager.groups_dict

            i_is_group = is_onehot_group(i)
            j_is_group = is_onehot_group(j)

            # Denormalize each column separately (only if not a one-hot group)
            denorm_grid = np.zeros_like(grid_np)

            # Column 0 -> feature i
            if i_is_group:
                # One-hot group: integer labels, keep as-is
                denorm_grid[:, 0] = grid_np[:, 0]
            else:
                dummy_i = np.zeros((len(grid_np), n_features))
                dummy_i[:, i] = grid_np[:, 0]
                denorm_i = scaler.inverse_transform(dummy_i)
                denorm_grid[:, 0] = denorm_i[:, i]

            # Column 1 -> feature j
            if j_is_group:
                # One-hot group: integer labels, keep as-is
                denorm_grid[:, 1] = grid_np[:, 1]
            else:
                dummy_j = np.zeros((len(grid_np), n_features))
                dummy_j[:, j] = grid_np[:, 1]
                denorm_j = scaler.inverse_transform(dummy_j)
                denorm_grid[:, 1] = denorm_j[:, j]

            result.append(denorm_grid)

        return result
