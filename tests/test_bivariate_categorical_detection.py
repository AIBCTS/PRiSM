"""
Investigation test for Issue #6-8: Binary features incorrectly classified as continuous.

This test module investigates why bivariate pairs involving binary features
(rececmo, recinfections2weeks) are incorrectly routed to contour plots instead
of mixed/heatmap plots.

Root Cause Analysis:
- Features in selected bivariate pairs but NOT selected as univariate have no
  metadata entry in FeatureMetadataRegistry
- When metadata is None, is_categorical defaults to False
- This causes binary features to be treated as continuous

Solutions to investigate:
1. Extend metadata registry to include bivariate pair components
2. Add fallback categorical detection from raw data in FeaturePairInfo creation
3. Pass explicit binary/categorical info from preprocessing metadata
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from prism.feature_labels import FeatureLabelManager
from prism.lasso import LassoResultsManager
from prism.maskedmlp import MaskedMLP
from prism.plotting import PlottingPipeline
from prism.preprocessing import OneHotGroupManager, collapse_onehot_features

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# =============================================================================
# Fixtures (same as test_visual_validation_production.py)
# =============================================================================

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "htx_example_replica"
TRAIN_DATA_PATH = FIXTURES_DIR / "htx_example_mlp_train.csv"
METADATA_PATH = FIXTURES_DIR / "preprocessing_metadata.json"
MODEL_PATH = FIXTURES_DIR / "htx_example_mlp_model.pt"
LASSO_PATH = FIXTURES_DIR / "blackbox_lasso_results.pt"
LABELS_PATH = FIXTURES_DIR / "variable_labels.csv"

TARGET_CANDIDATES = ['var1', 'event_oneyear', 'target', 'outcome', 'y', 'label']
ID_CANDIDATES = ['trr_id_code', 'id', 'patient_id', 'subject_id', 'var_id']


def skip_if_missing_fixtures():
    """Skip test if fixture files are missing."""
    for path in [TRAIN_DATA_PATH, METADATA_PATH, MODEL_PATH, LASSO_PATH]:
        if not path.exists():
            pytest.skip(f"Missing fixture: {path}")


@pytest.fixture(scope="module")
def production_data():
    """Load all production data for investigation."""
    skip_if_missing_fixtures()

    # Load training data
    train_df = pd.read_csv(TRAIN_DATA_PATH, comment='#')

    # Find target column
    target_column = None
    id_column = None
    for c in TARGET_CANDIDATES:
        if c in train_df.columns:
            target_column = c
            break
    for c in ID_CANDIDATES:
        if c in train_df.columns:
            id_column = c
            break

    if target_column is None:
        pytest.skip("Could not find target column")

    drop_cols = [target_column]
    if id_column:
        drop_cols.append(id_column)

    X_train = train_df.drop(drop_cols, axis=1)
    feature_column_names = X_train.columns.tolist()

    # Load metadata
    with open(METADATA_PATH, 'r') as f:
        preprocessing_metadata = json.load(f)

    # Create group manager
    group_manager = None
    collapsed_feature_names = feature_column_names
    if 'onehot_group_manager' in preprocessing_metadata:
        group_manager = OneHotGroupManager.from_preprocessing_metadata(preprocessing_metadata)
        _, collapsed_feature_names = collapse_onehot_features(
            X_train.values, group_manager, feature_column_names
        )

    # Load model
    checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    scaler = checkpoint.get('scaler')
    blackbox_model = checkpoint.get('model')
    if blackbox_model is None and 'model_state_dict' in checkpoint:
        blackbox_model = MaskedMLP(input_dim=len(feature_column_names), output_dim=1)
        blackbox_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    blackbox_model.eval()

    # Load LASSO
    lasso_data = torch.load(LASSO_PATH, map_location='cpu', weights_only=False)
    lasso_results = (
        lasso_data
        if isinstance(lasso_data, LassoResultsManager)
        else lasso_data.get('lasso_results')
    )
    lasso_results.select_lambda(62)

    # Load labels
    label_manager = FeatureLabelManager.from_csv(
        LABELS_PATH, column_name_col='processed_name', label_col='user_label'
    )

    # Create X tensor
    X_train_tensor = (
        scaler.to_tensor(X_train, device='cpu')
        if scaler
        else torch.tensor(X_train.values, dtype=torch.float32)
    )

    return {
        'X_train': X_train,
        'X_train_tensor': X_train_tensor,
        'feature_column_names': feature_column_names,
        'collapsed_feature_names': collapsed_feature_names,
        'group_manager': group_manager,
        'scaler': scaler,
        'blackbox_model': blackbox_model,
        'lasso_results': lasso_results,
        'label_manager': label_manager,
        'preprocessing_metadata': preprocessing_metadata,
    }


class TestBivariateCategoricalDetection:
    """Investigate categorical detection for bivariate pair components."""

    def test_identify_problematic_bivariate_pairs(self, production_data):
        """
        Step 1: Identify which bivariate pairs have incorrectly classified features.

        List all bivariate pairs and check if the categorical flags are correct.
        """
        lasso_results = production_data['lasso_results']
        collapsed_feature_names = production_data['collapsed_feature_names']

        # Get selected features
        selected_univariate = lasso_results.get_selected_univariate_indices()
        selected_bivariate = lasso_results.get_selected_bivariate_index_pairs()

        print("\n" + "=" * 80)
        print("SELECTED UNIVARIATE FEATURES")
        print("=" * 80)
        for idx in selected_univariate:
            name = collapsed_feature_names[idx]
            print(f"  [{idx}] {name}")

        print("\n" + "=" * 80)
        print("SELECTED BIVARIATE PAIRS")
        print("=" * 80)

        # Find features that are ONLY in bivariate pairs
        univariate_set = set(selected_univariate)
        bivariate_only_features = set()

        for i, j in selected_bivariate:
            name_i = collapsed_feature_names[i]
            name_j = collapsed_feature_names[j]

            in_univariate_i = i in univariate_set
            in_univariate_j = j in univariate_set

            print(f"  ({i}, {j}): {name_i} × {name_j}")
            print(f"    - {name_i}: in_univariate={in_univariate_i}")
            print(f"    - {name_j}: in_univariate={in_univariate_j}")

            if not in_univariate_i:
                bivariate_only_features.add((i, name_i))
            if not in_univariate_j:
                bivariate_only_features.add((j, name_j))

        print("\n" + "=" * 80)
        print("FEATURES ONLY IN BIVARIATE PAIRS (no univariate selection)")
        print("These features will have NO metadata entry!")
        print("=" * 80)

        if len(bivariate_only_features) == 0:
            print(
                "  (No bivariate-only features found - all bivariate features are also selected as univariate)"
            )
        else:
            for idx, name in sorted(bivariate_only_features):
                print(f"  [{idx}] {name}")

            # Check if rececmo and recinfections2weeks are in this set
            bivariate_only_names = {name for _, name in bivariate_only_features}
            problematic = {'rececmo', 'recinfections2weeks'}
            found_problematic = problematic.intersection(bivariate_only_names)
            print(f"\nProblematic binary features in bivariate-only set: {found_problematic}")

    def test_check_data_values_for_bivariate_only_features(self, production_data):
        """
        Step 2: Check the actual data values for bivariate-only features.

        Verify these are binary (0/1) and should be treated as categorical.
        """
        X_train = production_data['X_train']
        feature_column_names = production_data['feature_column_names']
        lasso_results = production_data['lasso_results']
        collapsed_feature_names = production_data['collapsed_feature_names']

        selected_univariate = lasso_results.get_selected_univariate_indices()
        selected_bivariate = lasso_results.get_selected_bivariate_index_pairs()

        univariate_set = set(selected_univariate)

        print("\n" + "=" * 80)
        print("DATA VALUE ANALYSIS FOR BIVARIATE-ONLY FEATURES")
        print("=" * 80)

        for i, j in selected_bivariate:
            for idx in [i, j]:
                if idx not in univariate_set:
                    name = collapsed_feature_names[idx]

                    # Get data for this feature
                    if name in X_train.columns:
                        data = X_train[name].values
                        unique_vals = np.unique(data)

                        print(f"\n  [{idx}] {name}:")
                        print(f"    Unique values: {unique_vals}")
                        print(f"    N unique: {len(unique_vals)}")
                        print(
                            f"    Is binary (0/1): {set(unique_vals).issubset({0, 1, 0.0, 1.0})}"
                        )
                        print(f"    Should be categorical: {len(unique_vals) < 15}")
                    else:
                        print(f"\n  [{idx}] {name}: NOT FOUND in X_train columns")

    def test_pipeline_categorical_flags(self, production_data):
        """
        Step 3: Check the categorical flags produced by PlottingPipeline.

        This shows what the current code produces vs what it should produce.
        """
        X_train_tensor = production_data['X_train_tensor']
        blackbox_model = production_data['blackbox_model']
        scaler = production_data['scaler']
        group_manager = production_data['group_manager']
        feature_column_names = production_data['feature_column_names']
        lasso_results = production_data['lasso_results']
        label_manager = production_data['label_manager']
        collapsed_feature_names = production_data['collapsed_feature_names']

        # Create pipeline
        pipeline = PlottingPipeline(
            lasso_results=lasso_results,
            group_manager=group_manager,
            label_manager=label_manager,
        )

        # Prepare bundle
        bundle = pipeline.prepare_plotting_bundle(
            x=X_train_tensor,
            model=blackbox_model,
            scaler=scaler,
            n_steps=50,
            method='lebesgue',
            x_train=X_train_tensor,
            device='cpu',
            categorical_threshold=15,
            subtract_univariate=True,
            feature_names=feature_column_names,
        )

        print("\n" + "=" * 80)
        print("BIVARIATE PAIR CATEGORICAL FLAGS (from bundle)")
        print("=" * 80)

        selected_univariate = lasso_results.get_selected_univariate_indices()
        univariate_set = set(selected_univariate)

        issues = []
        for info in bundle.bivariate_pairs():
            i, j = info.indices
            name_i, name_j = info.names
            label_i, label_j = info.labels
            is_cat_i, is_cat_j = info.is_categorical

            in_uni_i = i in univariate_set
            in_uni_j = j in univariate_set

            print(f"\n  ({i}, {j}): {name_i} × {name_j}")
            print(f"    Labels: {label_i} × {label_j}")
            print(f"    is_categorical: ({is_cat_i}, {is_cat_j})")
            print(f"    in_univariate: ({in_uni_i}, {in_uni_j})")

            # Check for issues: binary features marked as continuous
            # rececmo, recinfections2weeks are known binary features
            binary_features = {'rececmo', 'recinfections2weeks'}

            if name_i in binary_features and not is_cat_i:
                issues.append(f"{name_i} marked as continuous, should be categorical")
            if name_j in binary_features and not is_cat_j:
                issues.append(f"{name_j} marked as continuous, should be categorical")

        print("\n" + "=" * 80)
        print("ISSUES DETECTED")
        print("=" * 80)
        for issue in issues:
            print(f"  ⚠️  {issue}")

        # Don't fail - this is diagnostic
        if issues:
            print(f"\nFound {len(issues)} categorical detection issues")

    def test_proposed_fix_cardinality_based(self, production_data):
        """
        Test Proposed Fix #1: Use cardinality-based detection from x_data.

        For features without metadata, check unique values in x_data to determine
        if they should be categorical.
        """
        X_train = production_data['X_train']
        lasso_results = production_data['lasso_results']
        collapsed_feature_names = production_data['collapsed_feature_names']
        group_manager = production_data['group_manager']

        selected_univariate = lasso_results.get_selected_univariate_indices()
        selected_bivariate = lasso_results.get_selected_bivariate_index_pairs()

        univariate_set = set(selected_univariate)
        categorical_threshold = 15

        print("\n" + "=" * 80)
        print("PROPOSED FIX: Cardinality-based categorical detection")
        print("=" * 80)

        # Get x_data (need collapsed version if we have group_manager)
        if group_manager:
            X_collapsed, _ = collapse_onehot_features(
                X_train.values, group_manager, X_train.columns.tolist()
            )
        else:
            X_collapsed = X_train.values

        for i, j in selected_bivariate:
            name_i = collapsed_feature_names[i]
            name_j = collapsed_feature_names[j]

            print(f"\n  ({i}, {j}): {name_i} × {name_j}")

            for idx, name in [(i, name_i), (j, name_j)]:
                in_univariate = idx in univariate_set

                # Get data and check cardinality
                col_data = X_collapsed[:, idx]
                n_unique = len(np.unique(col_data))
                would_be_categorical = n_unique < categorical_threshold

                # Also check if it's a collapsed group
                is_collapsed = group_manager.is_categorical_group(name) if group_manager else False

                status = "✓" if in_univariate else "→ NEEDS FALLBACK"
                print(
                    f"    [{idx}] {name}: unique={n_unique}, is_collapsed={is_collapsed}, should_be_cat={would_be_categorical} {status}"
                )

    def test_check_metadata_registry_contents(self, production_data):
        """
        Step 4: Check what's in the metadata registry vs what's needed.
        """
        X_train_tensor = production_data['X_train_tensor']
        blackbox_model = production_data['blackbox_model']
        scaler = production_data['scaler']
        group_manager = production_data['group_manager']
        feature_column_names = production_data['feature_column_names']
        lasso_results = production_data['lasso_results']
        label_manager = production_data['label_manager']
        collapsed_feature_names = production_data['collapsed_feature_names']

        # Create pipeline
        pipeline = PlottingPipeline(
            lasso_results=lasso_results,
            group_manager=group_manager,
            label_manager=label_manager,
        )

        # Prepare bundle
        bundle = pipeline.prepare_plotting_bundle(
            x=X_train_tensor,
            model=blackbox_model,
            scaler=scaler,
            n_steps=50,
            method='lebesgue',
            x_train=X_train_tensor,
            device='cpu',
            categorical_threshold=15,
            subtract_univariate=True,
            feature_names=feature_column_names,
        )

        print("\n" + "=" * 80)
        print("METADATA REGISTRY CONTENTS")
        print("=" * 80)

        registry = bundle.metadata_registry

        print(f"Registry has {len(registry)} entries (for univariate features)")
        print("\nEntries:")
        for metadata in registry:
            print(
                f"  dense={metadata.dense_idx}, collapsed={metadata.collapsed_idx}: {metadata.column_name} (is_cat={metadata.is_categorical})"
            )

        print("\n" + "=" * 80)
        print("BIVARIATE PAIR COMPONENTS - REGISTRY LOOKUPS")
        print("=" * 80)

        for info in bundle.bivariate_pairs():
            i, j = info.indices
            metadata_i = registry.get_by_collapsed(i)
            metadata_j = registry.get_by_collapsed(j)

            print(f"\n  ({i}, {j}):")
            print(f"    metadata_i: {metadata_i}")
            print(f"    metadata_j: {metadata_j}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
