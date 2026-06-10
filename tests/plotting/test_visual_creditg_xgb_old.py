"""
Visual Validation Test for XGB Model with OLD One-Hot Scaler.

This test validates XGB models with the OLD scaler (which scales binary/one-hot columns
individually) to show how the nomogram plots appear in this configuration.

The code handles this gracefully - it's not a bug, but a different scaling approach
where each one-hot column is scaled separately, rather than leaving them as 0/1 values
as in the new fixed scaler.

Expected behavior (OLD scaler):
- This provides a comparison to the new fixed scaler behavior

Output Directory:
    tests/fixtures/output/visual_validation/credit_g_xgb_scaled_onehot/

Usage:
    pytest tests/plotting/test_visual_creditg_xgb.py -v -s
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from prism.lasso import LassoResultsManager
from prism.plotting import NomogramRenderer, PlotFormatter, PlottingPipeline
from prism.preprocessing import (
    OneHotGroupManager,
    build_ordinal_labels_dict,
    collapse_onehot_features,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Path Configuration
# =============================================================================

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "credit_g_xgb_scaled_onehot"
OUTPUT_BASE = Path(__file__).parent.parent / "fixtures" / "output" / "visual_validation"
OUTPUT_DIR = OUTPUT_BASE / "credit_g_xgb_scaled_onehot"

# Fixed file paths in fixtures folder
TRAIN_DATA_PATH = FIXTURES_DIR / "credit-g_xgb_20251215_112449_train.csv"
METADATA_PATH = FIXTURES_DIR / "preprocessing_metadata_credit-g_20251215_112226.json"
MODEL_PATH = FIXTURES_DIR / "credit-g_xgb_model_20251215_112449.pt"
LASSO_PATH = FIXTURES_DIR / "blackbox_credit-g_xgb_lasso_20251215_113034_lasso.pt"

# Target and ID column detection
TARGET_CANDIDATES = ['target', 'class', 'outcome', 'y', 'label']
ID_CANDIDATES = ['trr_id_code', 'id', 'patient_id', 'subject_id']


# =============================================================================
# Helper Functions
# =============================================================================


def detect_target_and_id_columns(df, target_candidates, id_candidates):
    """Detect target and ID columns from candidate lists."""
    target_column = None
    id_column = None

    for candidate in target_candidates:
        if candidate in df.columns:
            target_column = candidate
            break

    for candidate in id_candidates:
        if candidate in df.columns:
            id_column = candidate
            break

    return target_column, id_column


def check_fixture_files_exist():
    """Check that all required fixture files exist."""
    required_files = [
        (TRAIN_DATA_PATH, "Training data"),
        (METADATA_PATH, "Preprocessing metadata"),
        (MODEL_PATH, "Model checkpoint"),
        (LASSO_PATH, "LASSO results"),
    ]

    missing = []
    for path, description in required_files:
        if not path.exists():
            missing.append(f"  - {description}: {path}")

    if missing:
        msg = "Missing fixture files:\n" + "\n".join(missing)
        msg += f"\n\nPlease ensure files are in: {FIXTURES_DIR}"
        return False, msg

    return True, "All fixture files present"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def verify_fixtures():
    """Verify all fixture files exist before running tests."""
    exists, msg = check_fixture_files_exist()
    if not exists:
        pytest.skip(msg)
    return True


@pytest.fixture(scope="module")
def loaded_train_data(verify_fixtures):
    """Load training data from fixtures."""
    train_df = pd.read_csv(TRAIN_DATA_PATH, comment='#')

    target_column, id_column = detect_target_and_id_columns(
        train_df, TARGET_CANDIDATES, ID_CANDIDATES
    )

    if target_column is None:
        pytest.skip("Could not detect target column")

    drop_cols = [target_column]
    if id_column:
        drop_cols.append(id_column)

    X_train = train_df.drop(drop_cols, axis=1)
    y_train = train_df[target_column]

    logger.info(f"Loaded training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")

    return {
        'X_train': X_train,
        'y_train': y_train,
        'feature_column_names': X_train.columns.tolist(),
        'target_column': target_column,
        'id_column': id_column,
    }


@pytest.fixture(scope="module")
def loaded_model_and_scaler(verify_fixtures, loaded_train_data):
    """Load XGB model and scaler from fixtures."""
    checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)

    blackbox_model = None
    scaler = None
    hyperparameters = {}

    if isinstance(checkpoint, dict):
        if 'scaler' in checkpoint:
            scaler = checkpoint['scaler']
        if 'hyperparameters' in checkpoint:
            hyperparameters = checkpoint['hyperparameters']
        if 'model' in checkpoint:
            blackbox_model = checkpoint['model']

    if blackbox_model is None:
        pytest.skip("Could not load model from checkpoint")

    logger.info(f"Loaded model: {type(blackbox_model).__name__}")
    logger.info(f"Loaded scaler: {type(scaler).__name__ if scaler else 'None'}")

    return {
        'model': blackbox_model,
        'scaler': scaler,
        'hyperparameters': hyperparameters,
    }


@pytest.fixture(scope="module")
def loaded_group_manager(verify_fixtures, loaded_train_data):
    """Load OneHotGroupManager from metadata."""
    with open(METADATA_PATH, 'r') as f:
        preprocessing_metadata = json.load(f)

    group_manager = None
    feature_column_names = loaded_train_data['feature_column_names']
    collapsed_feature_names = feature_column_names

    if (
        'onehot_group_manager' in preprocessing_metadata
        and preprocessing_metadata['onehot_group_manager'] is not None
    ):
        group_manager = OneHotGroupManager.from_preprocessing_metadata(preprocessing_metadata)

        X_train_values = loaded_train_data['X_train'].values
        _, collapsed_feature_names = collapse_onehot_features(
            X_train_values, group_manager, feature_column_names
        )
        logger.info(
            f"Loaded OneHotGroupManager: {len(collapsed_feature_names)} collapsed features"
        )
    else:
        logger.warning("No onehot_group_manager in metadata")

    return {
        'group_manager': group_manager,
        'collapsed_feature_names': collapsed_feature_names,
        'preprocessing_metadata': preprocessing_metadata,
    }


@pytest.fixture(scope="module")
def categorical_labels_dict(loaded_group_manager):
    """Build categorical labels dict for both one-hot and ordinal features."""
    preprocessing_metadata = loaded_group_manager['preprocessing_metadata']
    group_manager = loaded_group_manager['group_manager']

    categorical_labels = group_manager.build_categorical_labels_dict(None) if group_manager else {}

    ordinal_labels = build_ordinal_labels_dict(preprocessing_metadata)
    categorical_labels.update(ordinal_labels)

    logger.info(f"Built categorical_labels with {len(categorical_labels)} features total")
    return categorical_labels


@pytest.fixture(scope="module")
def loaded_lasso_results(verify_fixtures):
    """Load LASSO results and select lambda with target_ratio=0.998."""
    lasso_data = torch.load(LASSO_PATH, map_location='cpu', weights_only=False)

    if isinstance(lasso_data, LassoResultsManager):
        lasso_results = lasso_data
    elif isinstance(lasso_data, dict):
        if 'lasso_results' in lasso_data:
            lasso_results = lasso_data['lasso_results']
        else:
            pytest.skip(
                f"LASSO file does not contain 'lasso_results' key. Keys: {lasso_data.keys()}"
            )
    else:
        lasso_results = lasso_data

    if not isinstance(lasso_results, LassoResultsManager):
        pytest.skip(f"Loaded object is not LassoResultsManager: {type(lasso_results)}")

    # Select lambda using target_ratio=0.998 to reproduce the bug scenario
    lasso_results.select_lambda_max_test_auc(target_ratio=0.998)
    logger.info(
        f"Selected lambda index {lasso_results.selected_lambda_index} (target_ratio=0.998)"
    )

    return lasso_results


@pytest.fixture(scope="module")
def x_train_tensor(loaded_train_data, loaded_model_and_scaler):
    """Create X_train tensor."""
    scaler = loaded_model_and_scaler['scaler']
    X_train = loaded_train_data['X_train']

    device = 'cpu'

    if scaler is not None:
        X_train_tensor = scaler.to_tensor(X_train, device=device)
    else:
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32, device=device)

    logger.info(f"Created X_train_tensor: shape={X_train_tensor.shape}")

    return X_train_tensor


# =============================================================================
# Test Class - Old Scaler Validation
# =============================================================================


class TestXGBOldScaler:
    """Test class validating XGB model with old one-hot scaler (scaled individually)."""

    @pytest.fixture(autouse=True)
    def setup_output_dir(self):
        """Create output directory for saved figures."""
        self.output_dir = OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def test_full_pipeline_bug_reproduction(
        self,
        loaded_train_data,
        loaded_model_and_scaler,
        loaded_group_manager,
        x_train_tensor,
    ):
        """
        FULL PIPELINE TEST: partial responses → LASSO → plotting.

        This test runs the complete pipeline to isolate WHERE the bug occurs:
        - Stage 1: Calculate partial responses from scratch
        - Stage 2: Run LASSO feature selection
        - Stage 3: Generate plots

        Expected (BUG): Stage 1 will show identical partial responses for all
        one-hot categories when XGB model is queried with unscaled 0/1 values.
        """
        # Setup
        blackbox_model = loaded_model_and_scaler['model']
        scaler = loaded_model_and_scaler['scaler']
        group_manager = loaded_group_manager['group_manager']
        collapsed_feature_names = loaded_group_manager['collapsed_feature_names']
        feature_column_names = loaded_train_data['feature_column_names']
        y_train = loaded_train_data['y_train']
        device = 'cpu'

        logger.info("\n" + "=" * 80)
        logger.info("FULL PIPELINE TEST: OLD SCALER")
        logger.info("=" * 80)

        # =============================================================
        # STAGE 1: Calculate partial responses
        # =============================================================
        logger.info("\nSTAGE 1: Calculating partial responses with Dirac method...")

        from prism.partial_responses import partial_responses

        partial_responses_train = partial_responses(
            x=x_train_tensor,
            model=blackbox_model,
            x_train=x_train_tensor,
            method='dirac',
            device=device,
            batch_size=128,
            group_manager=group_manager,
            feature_names=feature_column_names,
            scaler=scaler,
        )

        # partial_responses() returns shape (n_samples, n_features)
        logger.info(
            f"Partial responses shape (samples, features): {partial_responses_train.shape}"
        )

        # Diagnostic: Check partial response values for checking_status
        # Extract univariate responses (first N_collapsed_features columns)
        n_collapsed = len(collapsed_feature_names)
        n_bivariate = partial_responses_train.shape[1] - n_collapsed
        univariate_responses = partial_responses_train[
            :, :n_collapsed
        ]  # All rows, first n_collapsed columns

        logger.info(f"  Univariate features: {n_collapsed}")
        logger.info(f"  Bivariate pairs: {n_bivariate}")

        # Find checking_status index
        if 'checking_status' in collapsed_feature_names:
            checking_idx = collapsed_feature_names.index('checking_status')
            checking_responses = univariate_responses[:, checking_idx]  # Column for this feature

            logger.info("\nSTAGE 1 DIAGNOSTIC: checking_status partial responses")
            logger.info(f"  Feature index: {checking_idx}")
            logger.info(f"  Number of samples: {checking_responses.shape[0]}")
            logger.info(f"  Mean response: {checking_responses.mean().item():.6f}")
            logger.info(f"  Std response: {checking_responses.std().item():.6f}")
            logger.info(f"  Min response: {checking_responses.min().item():.6f}")
            logger.info(f"  Max response: {checking_responses.max().item():.6f}")

            unique_vals = torch.unique(checking_responses)
            logger.info(f"  Unique values: {unique_vals.numel()}")

            if unique_vals.numel() <= 2:
                logger.warning(
                    f"  !!! BUG CONFIRMED: Only {unique_vals.numel()} unique partial response value(s) !!!"
                )
                logger.warning(f"  Unique values: {unique_vals.tolist()}")
            else:
                logger.info(f"  OK: Found {unique_vals.numel()} unique response values")

        # =============================================================
        # STAGE 2: LASSO feature selection
        # =============================================================
        logger.info("\nSTAGE 2: Running LASSO feature selection...")

        from prism.lasso import LassoRegression

        # For test purposes, use a smaller lambda sweep
        # NOTE: For this diagnostic test, we'll use the full dataset for both train and test
        # (not ideal for real validation, but sufficient for bug reproduction)
        lasso = LassoRegression(
            nlambda=30,  # Even smaller for faster execution
            min_lambda=0.001,
            max_lambda=50,
            batch_size=10,
            seed=42,
            max_workers=-1,
        )

        # partial_responses already returns shape (n_samples, n_features) - no transpose needed!
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)

        print(f"\n  PR shape: {partial_responses_train.shape} (samples, features)")
        print(f"  y_train shape: {y_train_tensor.shape}")
        print(
            f"  Using full dataset: {partial_responses_train.shape[0]} samples, {partial_responses_train.shape[1]} features"
        )

        lasso_results, _ = lasso.fit(
            partial_responses_train,  # Already in correct format!
            partial_responses_train,  # Use same as "test" (just for diagnostics)
            y_train_tensor,
            y_train_tensor,
            feature_names=None,  # Let LASSO generate generic names for now
        )

        # Select lambda
        lasso_results.select_lambda_max_test_auc(target_ratio=0.998)

        # Diagnostic: Check LASSO results
        selected_lambda_idx = lasso_results.selected_lambda_index

        logger.info("\nSTAGE 2 DIAGNOSTIC: LASSO results")
        logger.info(f"  Selected lambda index: {selected_lambda_idx}")

        # Skip detailed beta analysis for this diagnostic test
        # The bug has already been confirmed in Stage 1

        # =============================================================
        # STAGE 3: Plotting
        # =============================================================
        logger.info("\nSTAGE 3: Generating plots via PlottingPipeline...")

        from prism.plotting import PlottingPipeline

        pipeline = PlottingPipeline(
            lasso_results=lasso_results,
            group_manager=group_manager,
            label_manager=None,
        )

        bundle = pipeline.prepare_plotting_bundle(
            x=x_train_tensor,
            model=blackbox_model,
            scaler=scaler,
            n_steps=15,
            method='dirac',
            x_train=x_train_tensor,
            device=device,
            categorical_threshold=15,
            subtract_univariate=True,
            feature_names=feature_column_names,
        )

        # Diagnostic: Check plotting values for checking_status
        for info in bundle.univariate_features():
            if info.name == 'checking_status':
                logger.info("\nSTAGE 3 DIAGNOSTIC: checking_status plotting values")

                x_vals = (
                    info.x_values.tolist()
                    if hasattr(info.x_values, 'tolist')
                    else list(info.x_values)
                )
                resp_vals = (
                    info.response.tolist()
                    if hasattr(info.response, 'tolist')
                    else list(info.response)
                )

                logger.info(f"  x_values (categories): {x_vals}")
                logger.info(f"  response values: {[f'{v:.6f}' for v in resp_vals]}")

                resp_array = np.array(resp_vals)
                unique_resp = np.unique(resp_array)

                logger.info(f"  Unique response values: {len(unique_resp)}")

                if len(unique_resp) == 1:
                    logger.warning(
                        f"  !!! BUG CONFIRMED: All categories show identical response = {unique_resp[0]:.6f} !!!"
                    )
                else:
                    logger.info(f"  OK: Found {len(unique_resp)} unique response values")

        logger.info("\n" + "=" * 80)
        logger.info("FULL PIPELINE TEST COMPLETE")
        logger.info("=" * 80)

        # The test doesn't assert - it's designed to reproduce and diagnose the bug
        # User will review the diagnostic output to confirm bug location

    def test_onehot_fixed_visualization(
        self,
        loaded_lasso_results,
        x_train_tensor,
        loaded_model_and_scaler,
        loaded_group_manager,
        loaded_train_data,
        categorical_labels_dict,
    ):
        """
        OLD SCALER VALIDATION: Generate nomogram with XGB model using old scaler.

        This test validates XGB models with the OLD scaler (which scales binary/one-hot columns
        individually) to show how the nomogram plots appear in this configuration.

        Expected (OLD): Uses the old scaling approach where each one-hot column is scaled separately.
        """
        lasso_results = loaded_lasso_results
        X_train_tensor = x_train_tensor
        blackbox_model = loaded_model_and_scaler['model']
        scaler = loaded_model_and_scaler['scaler']
        group_manager = loaded_group_manager['group_manager']
        feature_column_names = loaded_train_data['feature_column_names']

        device = 'cpu'

        logger.info("=" * 80)
        logger.info("OLD SCALER TEST: XGB Model with One-Hot Encoded Features")
        logger.info("=" * 80)
        logger.info(f"Model type: {type(blackbox_model)}")
        logger.info(f"Selected lambda index: {lasso_results.selected_lambda_index}")

        # Get selected features
        univ_indices = lasso_results.get_selected_univariate_indices()
        biv_pairs = lasso_results.get_selected_bivariate_index_pairs()
        logger.info(
            f"Selected features: {len(univ_indices)} univariate, {len(biv_pairs)} bivariate"
        )

        # Create PlottingPipeline
        pipeline = PlottingPipeline(
            lasso_results=lasso_results,
            group_manager=group_manager,
            label_manager=None,
        )

        # Prepare plotting bundle with Dirac method (simpler for debugging)
        logger.info("\nCalculating partial responses with Dirac method...")
        bundle = pipeline.prepare_plotting_bundle(
            x=X_train_tensor,
            model=blackbox_model,
            scaler=scaler,
            n_steps=15,
            method='dirac',
            x_train=X_train_tensor,
            device=device,
            categorical_threshold=15,
            subtract_univariate=True,
            feature_names=feature_column_names,
        )

        # DIAGNOSTIC: Print partial response values for one-hot features
        logger.info("\n" + "=" * 80)
        logger.info("DIAGNOSTIC: Partial Response Values for One-Hot Features")
        logger.info("=" * 80)

        collapsed_names = loaded_group_manager['collapsed_feature_names']

        # Check checking_status (should be collapsed index 0)
        if group_manager and 'checking_status' in group_manager.groups_dict:
            for info in bundle.univariate_features():
                if info.name == 'checking_status':
                    logger.info(f"\nFeature: {info.name}")

                    # Convert to lists for display (handle both tensor and numpy)
                    x_vals = (
                        info.x_values.tolist()
                        if hasattr(info.x_values, 'tolist')
                        else list(info.x_values)
                    )
                    resp_vals = (
                        info.response.tolist()
                        if hasattr(info.response, 'tolist')
                        else list(info.response)
                    )

                    logger.info(f"  x_values (categories): {x_vals}")
                    logger.info(f"  response values: {resp_vals}")
                    logger.info("  >>> BUG CHECK: Are all response values identical? <<<")

                    # Check if all values are the same (use numpy for compatibility)
                    resp_array = np.array(resp_vals)
                    unique_responses = np.unique(resp_array)

                    if len(unique_responses) == 1:
                        logger.warning(
                            f"  !!! BUG CONFIRMED: All {len(resp_vals)} categories have identical response = {unique_responses[0]:.6f} !!!"
                        )
                    else:
                        logger.info(f"  OK: Found {len(unique_responses)} unique response values")

        # Apply beta scaling for nomogram
        bundle_scaled = pipeline.apply_beta_scaling(bundle)

        # Create formatter and renderer
        formatter = PlotFormatter(
            use_odds_ratio=False,
            categorical_labels=categorical_labels_dict,
        )
        renderer = NomogramRenderer(bundle_scaled, formatter, use_odds_ratio=False)

        # Generate nomogram
        fig_nomogram = renderer.render_nomogram(two_column=False)

        # Validate and save
        if isinstance(fig_nomogram, list):
            figs = fig_nomogram
        else:
            figs = [fig_nomogram]

        assert len(figs) > 0, "render_nomogram returned empty list"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for i, fig in enumerate(figs):
            suffix = f"_page{i+1}" if len(figs) > 1 else ""
            output_path = (
                self.output_dir / f"DIRAC_nomogram_xgb_scaled_onehot_{timestamp}{suffix}.png"
            )
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"\nSaved OLD scaler plot: {output_path}")

        for fig in figs:
            plt.close(fig)

        logger.info("\n" + "=" * 80)
        logger.info("TEST COMPLETE - Check plots for distinct values in one-hot features")
        logger.info("=" * 80)
        print(f"\nDIRAC + OLD SCALER: Plots saved to {self.output_dir}")

    def test_diagnostic_raw_predictions(
        self,
        loaded_model_and_scaler,
        x_train_tensor,
        loaded_group_manager,
    ):
        """
        DIAGNOSTIC: Test raw XGB predictions with different one-hot patterns.

        This confirms that XGB returns identical predictions when using unscaled 0/1 values.
        """
        blackbox_model = loaded_model_and_scaler['model']
        scaler = loaded_model_and_scaler['scaler']
        group_manager = loaded_group_manager['group_manager']

        logger.info("\n" + "=" * 80)
        logger.info("DIAGNOSTIC: Raw XGB Predictions Test")
        logger.info("=" * 80)

        # Get checking_status one-hot indices
        # These should be indices 7, 8, 9 based on the feature order
        checking_status_cols = group_manager.groups_dict['checking_status']
        logger.info(f"checking_status one-hot columns: {checking_status_cols}")

        # Find their indices in the feature array
        train_df = pd.read_csv(TRAIN_DATA_PATH, comment='#')
        X_train = train_df.drop(['target', 'trr_id_code'], axis=1, errors='ignore')
        feature_names = X_train.columns.tolist()

        col_indices = [feature_names.index(col) for col in checking_status_cols]
        logger.info(f"Indices in feature array: {col_indices}")

        # Test with all zeros + one-hot patterns
        logger.info("\nTest: All zeros baseline + different one-hot patterns")
        x_test = torch.zeros((4, x_train_tensor.shape[1]), device='cpu')

        # Reference category (all zeros)
        # Category 1: first column = 1
        x_test[1, col_indices[0]] = 1.0
        # Category 2: second column = 1
        x_test[2, col_indices[1]] = 1.0
        # Category 3: third column = 1
        x_test[3, col_indices[2]] = 1.0

        predictions = blackbox_model.predict_proba(x_test, device='cpu')

        logger.info("\nPredictions (UNSCALED 0/1 values):")
        for i in range(4):
            logger.info(f"  Category {i}: {predictions[i].item():.6f}")

        # Check if all are identical
        unique_preds = torch.unique(predictions)
        if len(unique_preds) == 1:
            logger.warning(
                f"\n!!! BUG CONFIRMED: All predictions are identical = {unique_preds[0].item():.6f} !!!"
            )
            logger.warning(
                "This proves XGB model doesn't distinguish between one-hot patterns when using raw 0/1 values"
            )
        else:
            logger.info(f"\nOK: Found {len(unique_preds)} unique predictions")

        logger.info("=" * 80)

    def test_lebesgue_onehot_unique_values(
        self,
        loaded_train_data,
        loaded_model_and_scaler,
        loaded_group_manager,
        loaded_lasso_results,
        x_train_tensor,
        categorical_labels_dict,
    ):
        """
        Validate that Lebesgue method also correctly distinguishes one-hot categories.

        The collapse fix applies to both Dirac and Lebesgue methods (shared code).
        This test verifies that Lebesgue produces distinct values for distinct categories.
        Saves a nomogram plot for visual validation.
        """
        blackbox_model = loaded_model_and_scaler['model']
        scaler = loaded_model_and_scaler['scaler']
        group_manager = loaded_group_manager['group_manager']
        feature_column_names = loaded_train_data['feature_column_names']
        lasso_results = loaded_lasso_results

        logger.info("\n" + "=" * 80)
        logger.info("LEBESGUE METHOD VALIDATION: One-Hot Category Distinction")
        logger.info("=" * 80)

        from prism.partial_responses import partial_responses
        from prism.preprocessing import collapse_onehot_features

        # Calculate partial responses with Lebesgue method
        pr_lebesgue = partial_responses(
            x=x_train_tensor,
            model=blackbox_model,
            x_train=x_train_tensor,
            method='lebesgue',
            device='cpu',
            batch_size=64,
            group_manager=group_manager,
            feature_names=feature_column_names,
            scaler=scaler,
        )

        _, collapsed_names = collapse_onehot_features(
            loaded_train_data['X_train'].values, group_manager, feature_column_names
        )

        logger.info(f"Lebesgue PR shape: {pr_lebesgue.shape}")

        # Check checking_status (the key test case - should have 4 unique values)
        if 'checking_status' in collapsed_names:
            idx = collapsed_names.index('checking_status')
            responses = pr_lebesgue[:, idx]
            unique_vals = torch.unique(responses)

            logger.info("\nchecking_status (Lebesgue):")
            logger.info(f"  Unique values: {unique_vals.numel()}")
            logger.info(f"  Values: {unique_vals.tolist()}")

            # checking_status has 4 categories in the data
            # With proper scaling, we should get 4 distinct values
            assert unique_vals.numel() >= 3, (
                "Expected at least 3 unique values for checking_status with Lebesgue, "
                f"got {unique_vals.numel()}. This may indicate the scaler fix is not working."
            )

        # Summary for all one-hot groups
        logger.info("\nLebesgue unique values per one-hot group:")
        for group_name in group_manager.groups_dict.keys():
            if group_name in collapsed_names:
                idx = collapsed_names.index(group_name)
                responses = pr_lebesgue[:, idx]
                unique_vals = torch.unique(responses)
                logger.info(f"  {group_name}: {unique_vals.numel()} unique values")

        # Generate nomogram plot for visual validation
        logger.info("\nGenerating Lebesgue nomogram plot for visual validation...")

        pipeline = PlottingPipeline(
            lasso_results=lasso_results,
            group_manager=group_manager,
            label_manager=None,
        )

        bundle = pipeline.prepare_plotting_bundle(
            x=x_train_tensor,
            model=blackbox_model,
            scaler=scaler,
            n_steps=15,
            method='lebesgue',
            x_train=x_train_tensor,
            device='cpu',
            categorical_threshold=15,
            subtract_univariate=True,
            feature_names=feature_column_names,
        )

        # Apply beta scaling and render
        bundle_scaled = pipeline.apply_beta_scaling(bundle)

        formatter = PlotFormatter(
            use_odds_ratio=False,
            categorical_labels=categorical_labels_dict,
        )
        renderer = NomogramRenderer(bundle_scaled, formatter, use_odds_ratio=False)

        fig_nomogram = renderer.render_nomogram(two_column=False)

        # Save plot
        if isinstance(fig_nomogram, list):
            figs = fig_nomogram
        else:
            figs = [fig_nomogram]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for i, fig in enumerate(figs):
            suffix = f"_page{i+1}" if len(figs) > 1 else ""
            output_path = self.output_dir / f"LEBESGUE_nomogram_xgb_{timestamp}{suffix}.png"
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved Lebesgue nomogram: {output_path}")

        for fig in figs:
            plt.close(fig)

        logger.info("\n" + "=" * 80)
        logger.info("LEBESGUE VALIDATION COMPLETE")
        logger.info("=" * 80)
        print(f"\nLebesgue plots saved to {self.output_dir}")


# =============================================================================
# Main entry point
# =============================================================================

if __name__ == "__main__":
    exists, msg = check_fixture_files_exist()
    if not exists:
        print(f"\nERROR: {msg}")
        sys.exit(1)

    pytest.main([__file__, "-v", "-s", "--tb=short"])
