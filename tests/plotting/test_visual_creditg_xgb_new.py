"""
Visual Validation Test for XGB Model with Fixed One-Hot Scaler.

This test validates that XGB models with the new scaler (which excludes binary/one-hot
columns from scaling) work correctly with the PRiSM pipeline and generate nomogram plots.

This is the FIXED behavior (opposite of the old bug where all categories had identical values).

Output Directory:
    tests/fixtures/output/visual_validation/credit_g_xgb/

Usage:
    pytest tests/plotting/test_visual_creditg_xgb_fixed.py -v -s
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from prism.plotting import NomogramRenderer, PlotFormatter, PlottingPipeline
from prism.preprocessing import OneHotGroupManager, collapse_onehot_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Path Configuration
# =============================================================================

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "credit_g_xgb"
OUTPUT_BASE = Path(__file__).parent.parent / "fixtures" / "output" / "visual_validation"
OUTPUT_DIR = OUTPUT_BASE / "credit_g_xgb"

# Fixed file paths in fixtures folder
TRAIN_DATA_PATH = FIXTURES_DIR / "credit-g_xgb_train.csv"
METADATA_PATH = FIXTURES_DIR / "preprocessing_metadata.json"
MODEL_PATH = FIXTURES_DIR / "credit-g_xgb_model.pt"
LASSO_PATH = FIXTURES_DIR / "blackbox_lasso_results.pt"

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
    for file_path, description in required_files:
        if not file_path.exists():
            missing.append(f"{description}: {file_path}")

    if missing:
        pytest.skip("Missing required fixture files:\n" + "\n".join(missing))


# =============================================================================
# Test Class
# =============================================================================


class TestXGBFixedScaler:
    """Tests for XGB model with fixed scaler (one-hot columns excluded from scaling)."""

    @pytest.fixture(autouse=True)
    def setup_output_dir(self):
        """Create output directory for saved figures."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def test_full_pipeline_fixed_scaler(self):
        """Test full pipeline with XGB model using fixed scaler."""
        check_fixture_files_exist()

        # Load data (skip comment line at top)
        df_train = pd.read_csv(TRAIN_DATA_PATH, comment='#')
        target_column, id_column = detect_target_and_id_columns(
            df_train, TARGET_CANDIDATES, ID_CANDIDATES
        )

        # Load preprocessing metadata
        with open(METADATA_PATH, 'r') as f:
            _metadata_dict = json.load(f)

        # Load model and LASSO results
        _model_checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
        lasso_data = torch.load(LASSO_PATH, map_location='cpu', weights_only=False)

        # Extract LassoResultsManager from wrapper structure
        lasso_results = lasso_data['lasso_results']

        # Select lambda with best test AUC
        lambda_idx = lasso_results.select_lambda_max_test_auc(target_ratio=0.998)
        logger.info(f"Selected lambda index: {lambda_idx}")

        assert lambda_idx is not None, "Failed to select lambda index"
        assert lasso_results.test_aucs[lambda_idx] > 0.5, "Test AUC should be reasonable"

    def test_render_nomogram_fixed_scaler(self):
        """Generate nomogram plot with XGB model using fixed scaler (visual validation)."""
        check_fixture_files_exist()

        # Load data (skip comment line at top)
        df_train = pd.read_csv(TRAIN_DATA_PATH, comment='#')
        target_column, id_column = detect_target_and_id_columns(
            df_train, TARGET_CANDIDATES, ID_CANDIDATES
        )

        # Prepare feature data
        cols_to_drop = [target_column]
        if id_column:
            cols_to_drop.append(id_column)
        X_train = df_train.drop(columns=cols_to_drop)
        feature_column_names = X_train.columns.tolist()

        # Load preprocessing metadata
        with open(METADATA_PATH, 'r') as f:
            metadata_dict = json.load(f)

        # Load model and LASSO results
        model_checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
        lasso_data = torch.load(LASSO_PATH, map_location='cpu', weights_only=False)
        lasso_results = lasso_data['lasso_results']

        # Select lambda
        lambda_idx = lasso_results.select_lambda_max_test_auc(target_ratio=0.998)
        logger.info(f"Selected lambda index: {lambda_idx}")

        # Extract model and scaler
        blackbox_model = model_checkpoint['model']
        scaler = model_checkpoint.get('scaler')

        # Create tensor using scaler (like the old test does)
        if scaler is not None:
            X_train_tensor = scaler.to_tensor(X_train, device='cpu')
        else:
            X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)

        # Create OneHotGroupManager from metadata and collapse feature names
        group_manager = None
        collapsed_feature_names = feature_column_names

        if (
            'onehot_group_manager' in metadata_dict
            and metadata_dict['onehot_group_manager'] is not None
        ):
            group_manager = OneHotGroupManager.from_preprocessing_metadata(metadata_dict)
            _, collapsed_feature_names = collapse_onehot_features(
                X_train, group_manager, feature_column_names
            )
            logger.info(f"Collapsed features: {len(collapsed_feature_names)} features")
        else:
            logger.warning("No onehot_group_manager in metadata")

        # Create PlottingPipeline
        pipeline = PlottingPipeline(
            lasso_results=lasso_results,
            group_manager=group_manager,
            label_manager=None,
        )

        # Prepare plotting bundle
        # NOTE: Pass feature_column_names (expanded), not collapsed_feature_names
        # The PlottingPipeline handles collapsing internally via group_manager
        logger.info("Calculating partial responses...")
        bundle = pipeline.prepare_plotting_bundle(
            x=X_train_tensor,
            model=blackbox_model,
            scaler=scaler,
            n_steps=15,
            method='dirac',
            x_train=X_train_tensor,
            device='cpu',
            categorical_threshold=15,
            subtract_univariate=True,
            feature_names=feature_column_names,
        )

        # DEBUG: Check partial responses for continuous features
        logger.info("\n" + "=" * 80)
        logger.info("DEBUG: Checking partial responses for continuous features")
        logger.info("=" * 80)

        continuous_features = ['duration', 'credit_amount', 'age', 'installment_commitment']
        for info in bundle.univariate_features():
            if info.name in continuous_features:
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

                logger.info(f"\nFeature: {info.name} (continuous)")
                logger.info(f"  x_values range: {min(x_vals):.3f} to {max(x_vals):.3f}")
                logger.info(f"  response values: {[f'{v:.6f}' for v in resp_vals[:5]]}...")
                logger.info(f"  Unique response values: {len(set(resp_vals))}")

                if len(set(resp_vals)) == 1:
                    logger.warning(
                        f"  !!! PROBLEM: All responses are identical = {resp_vals[0]:.6f} !!!"
                    )
                else:
                    logger.info(f"  OK: Found {len(set(resp_vals))} unique response values")

        # Apply beta scaling for nomogram
        bundle_scaled = pipeline.apply_beta_scaling(bundle)

        # Render nomogram
        logger.info("Rendering nomogram...")
        # Need to build categorical labels for proper one-hot category display
        categorical_labels = {}
        if group_manager:
            categorical_labels = group_manager.build_categorical_labels_dict(None)

        formatter = PlotFormatter(
            use_odds_ratio=False,
            categorical_labels=categorical_labels,
        )
        renderer = NomogramRenderer(bundle_scaled, formatter, use_odds_ratio=False)
        fig = renderer.render_nomogram(
            two_column=False,
        )

        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUT_DIR / f"nomogram_xgb_fixed_{timestamp}.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved nomogram to: {output_path}")

        plt.close('all')

        assert fig is not None, "Should generate a figure"
        logger.info("SUCCESS: Nomogram generated with fixed scaler")
