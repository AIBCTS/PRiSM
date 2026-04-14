"""
Visual Validation Tests with Credit-G Production Data.

This test module validates the PlottingPipeline + NomogramRenderer architecture
using credit-g MLP data from tests/fixtures/credit_g_replica/.

The test generates plots for visual review, allowing systematic validation of:
- Nomogram plots (single-column, two-column, with/without surround_axes)
- Bivariate heatmaps (catxcat, contxcont)
- Response plots

Output Directory Structure:
    tests/fixtures/output/visual_validation/
    └── credit_g/              # Credit-g dataset tests
        └── no_labels/         # Tests without labels (no labels file available)

Usage:
    pytest tests/plotting/test_visual_creditg.py -v -s

Or run directly:
    python tests/plotting/test_visual_creditg.py
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend to avoid Tk errors on Windows
import matplotlib.pyplot as plt
import pandas as pd
import pytest
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prism.lasso import LassoResultsManager
from prism.maskedmlp import MaskedMLP
from prism.nomogram_plot import nomogram
from prism.plotting import NomogramRenderer, PlotFormatter, PlottingPipeline
from prism.preprocessing import (
    OneHotGroupManager,
    build_ordinal_labels_dict,
    collapse_onehot_features,
)
from prism.response_plot import plot_partial_responses

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Path Configuration
# =============================================================================

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "credit_g_replica"
OUTPUT_BASE = Path(__file__).parent.parent / "fixtures" / "output" / "visual_validation"
OUTPUT_DIR = OUTPUT_BASE / "credit_g" / "no_labels"

# Fixed file paths in fixtures folder
TRAIN_DATA_PATH = FIXTURES_DIR / "credit-g_mlp_train.csv"
METADATA_PATH = FIXTURES_DIR / "preprocessing_metadata.json"
MODEL_PATH = FIXTURES_DIR / "credit-g_mlp_model.pt"
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
    for path, description in required_files:
        if not path.exists():
            missing.append(f"  - {description}: {path}")

    if missing:
        msg = "Missing fixture files:\\n" + "\\n".join(missing)
        msg += f"\\n\\nPlease copy files to: {FIXTURES_DIR}"
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

    # Detect target and ID columns
    target_column, id_column = detect_target_and_id_columns(
        train_df, TARGET_CANDIDATES, ID_CANDIDATES
    )

    if target_column is None:
        pytest.skip("Could not detect target column")

    # Drop columns for X
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
    """Load model and scaler from fixtures."""
    checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)

    blackbox_model = None
    scaler = None
    hyperparameters = {}

    if isinstance(checkpoint, dict):
        if 'scaler' in checkpoint:
            scaler = checkpoint['scaler']
        if 'hyperparameters' in checkpoint:
            hyperparameters = checkpoint['hyperparameters']

        # Try different ways to get model
        if 'model' in checkpoint:
            blackbox_model = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            # Need to instantiate model first
            n_features = len(loaded_train_data['feature_column_names'])
            blackbox_model = MaskedMLP(input_dim=n_features, output_dim=1)
            blackbox_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    elif isinstance(checkpoint, torch.nn.Module):
        blackbox_model = checkpoint

    if blackbox_model is None:
        pytest.skip("Could not load model from checkpoint")

    # Set to eval mode if PyTorch model
    if isinstance(blackbox_model, torch.nn.Module):
        blackbox_model.eval()

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
        # Use the classmethod to load from metadata, exactly as notebook does
        group_manager = OneHotGroupManager.from_preprocessing_metadata(preprocessing_metadata)

        # Get collapsed feature names using collapse_onehot_features
        X_train_values = loaded_train_data['X_train'].values
        _, collapsed_feature_names = collapse_onehot_features(
            X_train_values, group_manager, feature_column_names
        )
        logger.info(
            f"Loaded OneHotGroupManager: {len(collapsed_feature_names)} collapsed features"
        )
    else:
        logger.warning("No onehot_group_manager in metadata - will be None")

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

    # Start with one-hot group labels (if group_manager exists)
    categorical_labels = group_manager.build_categorical_labels_dict(None) if group_manager else {}

    # Add ordinal feature labels from metadata
    ordinal_labels = build_ordinal_labels_dict(preprocessing_metadata)
    categorical_labels.update(ordinal_labels)

    logger.info(f"Built categorical_labels with {len(categorical_labels)} features total")
    return categorical_labels


@pytest.fixture(scope="module")
def loaded_lasso_results(verify_fixtures):
    """Load LASSO results from fixtures."""
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

    # Select lambda index 81 to ensure we have bivariate features for testing
    # (16 univariate + 15 bivariate features, test AUC=0.7348)
    lasso_results.select_lambda(81)
    logger.info("Selected lambda index 81 for testing (includes bivariate features)")

    return lasso_results


@pytest.fixture(scope="module")
def x_train_tensor(loaded_train_data, loaded_model_and_scaler):
    """Create X_train tensor exactly as notebook does."""
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
# Test Class
# =============================================================================


class TestVisualValidationCreditG:
    """Test class that validates the plotting architecture with credit-g production data."""

    @pytest.fixture(autouse=True)
    def setup_output_dir(self):
        """Create output directory for saved figures."""
        self.output_dir = OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def test_render_nomogram_single_column(
        self,
        loaded_lasso_results,
        x_train_tensor,
        loaded_model_and_scaler,
        loaded_group_manager,
        loaded_train_data,
        categorical_labels_dict,
    ):
        """
        Generate nomogram using PlottingPipeline + NomogramRenderer.

        This tests the single-column layout without odds ratio conversion.
        """
        # Extract components
        lasso_results = loaded_lasso_results
        X_train_tensor = x_train_tensor
        blackbox_model = loaded_model_and_scaler['model']
        scaler = loaded_model_and_scaler['scaler']
        group_manager = loaded_group_manager['group_manager']
        feature_column_names = loaded_train_data['feature_column_names']

        # Parameters
        device = 'cpu'

        # Log the call parameters
        logger.info("=" * 60)
        logger.info("Generating nomogram with PlottingPipeline:")
        logger.info(f"  lasso_results type: {type(lasso_results)}")
        logger.info(f"  X_train_tensor shape: {X_train_tensor.shape}")
        logger.info(f"  model type: {type(blackbox_model)}")
        logger.info(f"  group_manager type: {type(group_manager)}")
        logger.info(f"  n_features: {len(feature_column_names)}")
        logger.info("=" * 60)

        # Create PlottingPipeline (no label manager for credit-g)
        pipeline = PlottingPipeline(
            lasso_results=lasso_results,
            group_manager=group_manager,
            label_manager=None,
        )

        # Prepare plotting bundle
        bundle = pipeline.prepare_plotting_bundle(
            x=X_train_tensor,
            model=blackbox_model,
            scaler=scaler,
            n_steps=50,
            method='lebesgue',
            x_train=X_train_tensor,
            device=device,
            categorical_threshold=15,
            subtract_univariate=True,
            feature_names=feature_column_names,
        )

        # Apply beta scaling for nomogram
        bundle_scaled = pipeline.apply_beta_scaling(bundle)

        # Create formatter and renderer
        formatter = PlotFormatter(
            use_odds_ratio=False,
            categorical_labels=categorical_labels_dict,
        )
        renderer = NomogramRenderer(bundle_scaled, formatter, use_odds_ratio=False)

        # Generate nomogram (single column)
        fig_nomogram = renderer.render_nomogram(two_column=False)

        # Validate result
        if isinstance(fig_nomogram, list):
            figs = fig_nomogram
        else:
            figs = [fig_nomogram]

        assert len(figs) > 0, "render_nomogram returned empty list"
        assert all(isinstance(f, plt.Figure) for f in figs), "All items should be Figure objects"

        # Save figures for visual validation
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for i, fig in enumerate(figs):
            suffix = f"_page{i+1}" if len(figs) > 1 else ""
            output_path = self.output_dir / f"nomogram_single_column_{timestamp}{suffix}.png"
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved: {output_path}")

        # Close figures
        for fig in figs:
            plt.close(fig)

        print(f"\\n[OK] Single-column nomogram saved to: {self.output_dir}")

    def test_render_nomogram_two_column(
        self,
        loaded_lasso_results,
        x_train_tensor,
        loaded_model_and_scaler,
        loaded_group_manager,
        loaded_train_data,
        categorical_labels_dict,
    ):
        """
        Generate nomogram with two-column layout.
        """
        # Extract components
        lasso_results = loaded_lasso_results
        X_train_tensor = x_train_tensor
        blackbox_model = loaded_model_and_scaler['model']
        scaler = loaded_model_and_scaler['scaler']
        group_manager = loaded_group_manager['group_manager']
        feature_column_names = loaded_train_data['feature_column_names']

        device = 'cpu'

        # Create PlottingPipeline
        pipeline = PlottingPipeline(
            lasso_results=lasso_results,
            group_manager=group_manager,
            label_manager=None,
        )

        # Prepare and scale bundle
        bundle = pipeline.prepare_plotting_bundle(
            x=X_train_tensor,
            model=blackbox_model,
            scaler=scaler,
            n_steps=50,
            method='lebesgue',
            x_train=X_train_tensor,
            device=device,
            categorical_threshold=15,
            subtract_univariate=True,
            feature_names=feature_column_names,
        )
        bundle_scaled = pipeline.apply_beta_scaling(bundle)

        # Create renderer with odds ratio disabled
        formatter = PlotFormatter(
            use_odds_ratio=False,
            categorical_labels=categorical_labels_dict,
        )
        renderer = NomogramRenderer(bundle_scaled, formatter, use_odds_ratio=False)

        # Generate nomogram (two column)
        fig_nomogram = renderer.render_nomogram(two_column=True)

        # Validate and save
        if isinstance(fig_nomogram, list):
            figs = fig_nomogram
        else:
            figs = [fig_nomogram]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for i, fig in enumerate(figs):
            suffix = f"_page{i+1}" if len(figs) > 1 else ""
            output_path = self.output_dir / f"nomogram_two_column_{timestamp}{suffix}.png"
            fig.savefig(output_path, dpi=150, bbox_inches='tight')

        for fig in figs:
            plt.close(fig)

        print(f"\\n[OK] Two-column nomogram saved to: {self.output_dir}")

    def test_render_nomogram_single_column_surround(
        self,
        loaded_lasso_results,
        x_train_tensor,
        loaded_model_and_scaler,
        loaded_group_manager,
        loaded_train_data,
        categorical_labels_dict,
    ):
        """
        Generate nomogram with single-column layout and surround_axes=True.

        Tests nomogram rendering with surrounding axes to show full axis frames.
        """
        # Extract components
        lasso_results = loaded_lasso_results
        X_train_tensor = x_train_tensor
        blackbox_model = loaded_model_and_scaler['model']
        scaler = loaded_model_and_scaler['scaler']
        group_manager = loaded_group_manager['group_manager']
        feature_column_names = loaded_train_data['feature_column_names']

        device = 'cpu'

        # Create PlottingPipeline
        pipeline = PlottingPipeline(
            lasso_results=lasso_results,
            group_manager=group_manager,
            label_manager=None,
        )

        # Prepare and scale bundle
        bundle = pipeline.prepare_plotting_bundle(
            x=X_train_tensor,
            model=blackbox_model,
            scaler=scaler,
            n_steps=50,
            method='lebesgue',
            x_train=X_train_tensor,
            device=device,
            categorical_threshold=15,
            subtract_univariate=True,
            feature_names=feature_column_names,
        )
        bundle_scaled = pipeline.apply_beta_scaling(bundle)

        # Create renderer
        formatter = PlotFormatter(
            use_odds_ratio=False,
            categorical_labels=categorical_labels_dict,
        )
        renderer = NomogramRenderer(bundle_scaled, formatter, use_odds_ratio=False)

        # Generate nomogram (single column with surround axes)
        fig_nomogram = renderer.render_nomogram(two_column=False, surround_axes=True)

        # Validate and save
        if isinstance(fig_nomogram, list):
            figs = fig_nomogram
        else:
            figs = [fig_nomogram]

        assert len(figs) > 0, "render_nomogram returned empty list"
        assert all(isinstance(f, plt.Figure) for f in figs), "All items should be Figure objects"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for i, fig in enumerate(figs):
            suffix = f"_page{i+1}" if len(figs) > 1 else ""
            output_path = (
                self.output_dir / f"nomogram_single_column_surround_{timestamp}{suffix}.png"
            )
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved: {output_path}")

        for fig in figs:
            plt.close(fig)

        print(f"\\n[OK] Single-column nomogram (surround_axes) saved to: {self.output_dir}")

    def test_render_nomogram_two_column_surround(
        self,
        loaded_lasso_results,
        x_train_tensor,
        loaded_model_and_scaler,
        loaded_group_manager,
        loaded_train_data,
        categorical_labels_dict,
    ):
        """
        Generate nomogram with two-column layout and surround_axes=True.

        Tests nomogram rendering with surrounding axes to show full axis frames.
        """
        # Extract components
        lasso_results = loaded_lasso_results
        X_train_tensor = x_train_tensor
        blackbox_model = loaded_model_and_scaler['model']
        scaler = loaded_model_and_scaler['scaler']
        group_manager = loaded_group_manager['group_manager']
        feature_column_names = loaded_train_data['feature_column_names']

        device = 'cpu'

        # Create PlottingPipeline
        pipeline = PlottingPipeline(
            lasso_results=lasso_results,
            group_manager=group_manager,
            label_manager=None,
        )

        # Prepare and scale bundle
        bundle = pipeline.prepare_plotting_bundle(
            x=X_train_tensor,
            model=blackbox_model,
            scaler=scaler,
            n_steps=50,
            method='lebesgue',
            x_train=X_train_tensor,
            device=device,
            categorical_threshold=15,
            subtract_univariate=True,
            feature_names=feature_column_names,
        )
        bundle_scaled = pipeline.apply_beta_scaling(bundle)

        # Create renderer
        formatter = PlotFormatter(
            use_odds_ratio=False,
            categorical_labels=categorical_labels_dict,
        )
        renderer = NomogramRenderer(bundle_scaled, formatter, use_odds_ratio=False)

        # Generate nomogram (two column with surround axes)
        fig_nomogram = renderer.render_nomogram(two_column=True, surround_axes=True)

        # Validate and save
        if isinstance(fig_nomogram, list):
            figs = fig_nomogram
        else:
            figs = [fig_nomogram]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for i, fig in enumerate(figs):
            suffix = f"_page{i+1}" if len(figs) > 1 else ""
            output_path = self.output_dir / f"nomogram_two_column_surround_{timestamp}{suffix}.png"
            fig.savefig(output_path, dpi=150, bbox_inches='tight')

        for fig in figs:
            plt.close(fig)

        print(f"\\n[OK] Two-column nomogram (surround_axes) saved to: {self.output_dir}")

    def test_render_response_plots(
        self,
        loaded_lasso_results,
        x_train_tensor,
        loaded_model_and_scaler,
        loaded_group_manager,
        loaded_train_data,
        categorical_labels_dict,
    ):
        """
        Generate response plots.

        Note: Response plots use UN-scaled bundle (no beta scaling).
        """
        # Extract components
        lasso_results = loaded_lasso_results
        X_train_tensor = x_train_tensor
        blackbox_model = loaded_model_and_scaler['model']
        scaler = loaded_model_and_scaler['scaler']
        group_manager = loaded_group_manager['group_manager']
        feature_column_names = loaded_train_data['feature_column_names']

        device = 'cpu'

        # Create PlottingPipeline
        pipeline = PlottingPipeline(
            lasso_results=lasso_results,
            group_manager=group_manager,
            label_manager=None,
        )

        # Prepare bundle (NO beta scaling for response plots)
        bundle = pipeline.prepare_plotting_bundle(
            x=X_train_tensor,
            model=blackbox_model,
            scaler=scaler,
            n_steps=50,
            method='lebesgue',
            x_train=X_train_tensor,
            device=device,
            categorical_threshold=15,
            subtract_univariate=True,
            feature_names=feature_column_names,
        )

        # Create renderer (no beta scaling)
        formatter = PlotFormatter(
            use_odds_ratio=False,
            categorical_labels=categorical_labels_dict,
        )
        renderer = NomogramRenderer(bundle, formatter, use_odds_ratio=False)

        # Generate response plots
        fig_responses = renderer.render_response_plots()

        # Validate and save
        assert fig_responses is not None, "render_response_plots returned None"
        assert isinstance(fig_responses, plt.Figure), f"Expected Figure, got {type(fig_responses)}"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"response_plots_{timestamp}.png"
        fig_responses.savefig(output_path, dpi=150, bbox_inches='tight')

        plt.close(fig_responses)

        print(f"\\n[OK] Response plots saved to: {self.output_dir}")

    def test_render_bivariate_heatmaps(
        self,
        loaded_lasso_results,
        x_train_tensor,
        loaded_model_and_scaler,
        loaded_group_manager,
        loaded_train_data,
        categorical_labels_dict,
    ):
        """
        Generate bivariate heatmaps.

        Tests catxcat and contxcont interaction plots.
        """
        # Extract components
        lasso_results = loaded_lasso_results
        X_train_tensor = x_train_tensor
        blackbox_model = loaded_model_and_scaler['model']
        scaler = loaded_model_and_scaler['scaler']
        group_manager = loaded_group_manager['group_manager']
        feature_column_names = loaded_train_data['feature_column_names']

        device = 'cpu'

        # Create PlottingPipeline
        pipeline = PlottingPipeline(
            lasso_results=lasso_results,
            group_manager=group_manager,
            label_manager=None,
        )

        # Prepare and scale bundle
        bundle = pipeline.prepare_plotting_bundle(
            x=X_train_tensor,
            model=blackbox_model,
            scaler=scaler,
            n_steps=50,
            method='lebesgue',
            x_train=X_train_tensor,
            device=device,
            categorical_threshold=15,
            subtract_univariate=True,
            feature_names=feature_column_names,
        )
        bundle_scaled = pipeline.apply_beta_scaling(bundle)

        # Create renderer
        formatter = PlotFormatter(
            use_odds_ratio=False,
            categorical_labels=categorical_labels_dict,
        )
        renderer = NomogramRenderer(bundle_scaled, formatter, use_odds_ratio=False)

        # Generate bivariate heatmaps
        fig_heatmaps = renderer.render_bivariate_heatmaps()

        # Heatmaps may be None if no catxcat or contxcont pairs
        if fig_heatmaps is None:
            logger.info("No catxcat or contxcont bivariate pairs - skipping heatmap test")
            pytest.skip("No bivariate heatmaps to render")

        # Save figures
        if isinstance(fig_heatmaps, list):
            figs = fig_heatmaps
        else:
            figs = [fig_heatmaps]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for i, fig in enumerate(figs):
            suffix = f"_page{i+1}" if len(figs) > 1 else ""
            output_path = self.output_dir / f"bivariate_heatmaps_{timestamp}{suffix}.png"
            fig.savefig(output_path, dpi=150, bbox_inches='tight')

        for fig in figs:
            plt.close(fig)

        print(f"\\n[OK] Bivariate heatmaps saved to: {self.output_dir}")

    def test_fixture_diagnostics(
        self,
        loaded_train_data,
        loaded_model_and_scaler,
        loaded_group_manager,
        loaded_lasso_results,
    ):
        """Diagnostic test to verify fixtures and report summary."""
        print("\\n" + "=" * 60)
        print("VISUAL VALIDATION DIAGNOSTICS (CREDIT-G)")
        print("=" * 60)

        print(f"\\nFixtures directory: {FIXTURES_DIR}")
        print(f"Output directory: {OUTPUT_DIR}")

        print("\\nTrain data:")
        print(f"  X_train shape: {loaded_train_data['X_train'].shape}")
        print(f"  n_features: {len(loaded_train_data['feature_column_names'])}")

        print("\\nModel:")
        print(f"  model type: {type(loaded_model_and_scaler['model'])}")
        print(f"  scaler type: {type(loaded_model_and_scaler['scaler'])}")

        print("\\nGroup manager:")
        gm = loaded_group_manager['group_manager']
        if gm is not None:
            print(f"  n_groups: {len(gm.groups_dict)}")
            print(f"  collapsed features: {len(loaded_group_manager['collapsed_feature_names'])}")
        else:
            print("  group_manager is None (credit-g has no feature groups)")

        print("\\nLASSO results:")
        print(f"  selected_lambda_index: {loaded_lasso_results.selected_lambda_index}")

        if loaded_lasso_results.selected_lambda_index is not None:
            selected_univ = loaded_lasso_results.get_selected_univariate_indices()
            selected_biv = loaded_lasso_results.get_selected_bivariate_index_pairs()
            print(f"  selected univariate: {len(selected_univ)} features")
            print(f"  selected bivariate: {len(selected_biv)} pairs")

        print("=" * 60)

    def test_categorical_detection(
        self,
        loaded_lasso_results,
        x_train_tensor,
        loaded_model_and_scaler,
        loaded_group_manager,
        loaded_train_data,
    ):
        """
        Verify categorical detection works correctly.

        Binary features in bivariate-only pairs should be detected as categorical.
        """
        # Extract components
        lasso_results = loaded_lasso_results
        X_train_tensor = x_train_tensor
        blackbox_model = loaded_model_and_scaler['model']
        scaler = loaded_model_and_scaler['scaler']
        group_manager = loaded_group_manager['group_manager']
        feature_column_names = loaded_train_data['feature_column_names']

        device = 'cpu'

        # Create PlottingPipeline
        pipeline = PlottingPipeline(
            lasso_results=lasso_results,
            group_manager=group_manager,
            label_manager=None,
        )

        # Prepare bundle
        bundle = pipeline.prepare_plotting_bundle(
            x=X_train_tensor,
            model=blackbox_model,
            scaler=scaler,
            n_steps=50,
            method='lebesgue',
            x_train=X_train_tensor,
            device=device,
            categorical_threshold=15,
            subtract_univariate=True,
            feature_names=feature_column_names,
        )

        print("\\n" + "=" * 60)
        print("CATEGORICAL DETECTION TEST (CREDIT-G)")
        print("=" * 60)

        # Known binary features that should be detected as categorical
        known_binary = {
            'checking_status_<0',
            'checking_status_>=200',
            'checking_status_no checking',
            'purpose_domestic appliance',
            'purpose_education',
        }

        issues = []
        for info in bundle.bivariate_pairs():
            i, j = info.indices
            name_i, name_j = info.names
            is_cat_i, is_cat_j = info.is_categorical

            print(f"\\n  ({i}, {j}): {name_i} x {name_j}")
            print(f"    is_categorical: ({is_cat_i}, {is_cat_j})")

            # Check binary features are correctly detected
            if name_i in known_binary and not is_cat_i:
                issues.append(f"{name_i} should be categorical but got is_categorical=False")
            if name_j in known_binary and not is_cat_j:
                issues.append(f"{name_j} should be categorical but got is_categorical=False")

        print("\\n" + "=" * 60)
        if issues:
            print("ISSUES FOUND:")
            for issue in issues:
                print(f"  - {issue}")
            pytest.fail(f"Categorical detection failed: {issues}")
        else:
            print("All binary features correctly detected as categorical")
        print("=" * 60)


class TestHighLevelFunctionsCreditG:
    """
    Test class that validates the HIGH-LEVEL functions (nomogram, plot_partial_responses)
    using Credit-G production data.

    These tests ensure the migrated top-level API functions work correctly.
    """

    @pytest.fixture(autouse=True)
    def setup_output_dir(self):
        """Create output directory for saved figures."""
        self.output_dir = OUTPUT_BASE / "credit_g" / "high_level_api"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def test_nomogram_function(
        self,
        loaded_lasso_results,
        x_train_tensor,
        loaded_model_and_scaler,
        loaded_group_manager,
        loaded_train_data,
        categorical_labels_dict,
    ):
        """
        Test the top-level nomogram() function with Credit-G data.

        This validates the migrated nomogram() function works end-to-end.
        """
        lasso_results = loaded_lasso_results
        X_train_tensor = x_train_tensor
        blackbox_model = loaded_model_and_scaler['model']
        scaler = loaded_model_and_scaler['scaler']
        group_manager = loaded_group_manager['group_manager']
        feature_column_names = loaded_train_data['feature_column_names']

        logger.info("=" * 60)
        logger.info("Testing nomogram() high-level function with Credit-G")
        logger.info("=" * 60)

        # Call the high-level nomogram() function
        result = nomogram(
            lasso_results=lasso_results,
            x=X_train_tensor,
            model=blackbox_model,
            scaler=scaler,
            n_steps=50,
            method='lebesgue',
            x_train=X_train_tensor,
            device='cpu',
            categorical_threshold=15,
            subtract_univariate=True,
            show_fig=False,
            return_fig=True,
            use_odds_ratio=False,
            two_column=True,
            surround_axes=True,
            categorical_labels=categorical_labels_dict,
            onehot_group_manager=group_manager,
            label_manager=None,  # Credit-G has no labels file
            feature_names=feature_column_names,
        )

        # Extract values from NomogramResult dataclass
        univariate_responses = result.univariate_responses
        _bivariate_responses = result.bivariate_responses
        x_univariate = result.x_univariate
        _x_bivariate = result.x_bivariate
        _selected_univariate_indices = result.selected_univariate_indices
        _selected_bivariate_pairs = result.selected_bivariate_pairs
        fig_main = result.fig_main
        fig_non_mixed = result.fig_bivariate

        # Validate return values
        assert len(univariate_responses) > 0, "Should have univariate responses"
        assert len(x_univariate) > 0, "Should have x_univariate values"
        assert fig_main is not None, "Should return main figure"

        # Save figures
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if isinstance(fig_main, list):
            for i, fig in enumerate(fig_main):
                output_path = self.output_dir / f"nomogram_highlevel_main_{timestamp}_p{i+1}.png"
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                logger.info(f"Saved: {output_path}")
                plt.close(fig)
        else:
            output_path = self.output_dir / f"nomogram_highlevel_main_{timestamp}.png"
            fig_main.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved: {output_path}")
            plt.close(fig_main)

        if fig_non_mixed is not None:
            if isinstance(fig_non_mixed, list):
                for i, fig in enumerate(fig_non_mixed):
                    output_path = (
                        self.output_dir / f"nomogram_highlevel_nonmixed_{timestamp}_p{i+1}.png"
                    )
                    fig.savefig(output_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
            else:
                output_path = self.output_dir / f"nomogram_highlevel_nonmixed_{timestamp}.png"
                fig_non_mixed.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close(fig_non_mixed)

        print("\\n[OK] nomogram() high-level function test passed for Credit-G")
        print(f"     Output saved to: {self.output_dir}")

    def test_plot_partial_responses_function(
        self,
        loaded_lasso_results,
        x_train_tensor,
        loaded_model_and_scaler,
        loaded_group_manager,
        loaded_train_data,
        categorical_labels_dict,
    ):
        """
        Test the top-level plot_partial_responses() function with Credit-G data.

        This validates the migrated plot_partial_responses() function works end-to-end.
        Note: Response plots do NOT apply beta scaling (raw partial responses).
        """
        lasso_results = loaded_lasso_results
        X_train_tensor = x_train_tensor
        blackbox_model = loaded_model_and_scaler['model']
        scaler = loaded_model_and_scaler['scaler']
        group_manager = loaded_group_manager['group_manager']
        feature_column_names = loaded_train_data['feature_column_names']

        logger.info("=" * 60)
        logger.info("Testing plot_partial_responses() high-level function with Credit-G")
        logger.info("=" * 60)

        # Call the high-level plot_partial_responses() function
        result = plot_partial_responses(
            lasso_results=lasso_results,
            x=X_train_tensor,
            model=blackbox_model,
            scaler=scaler,
            n_steps=50,
            method='lebesgue',
            x_train=X_train_tensor,
            device='cpu',
            categorical_threshold=15,
            subtract_univariate=True,
            subfig_size=3.5,
            show_fig=False,
            return_fig=True,
            use_odds_ratio=False,
            categorical_labels=categorical_labels_dict,
            onehot_group_manager=group_manager,
            label_manager=None,  # Credit-G has no labels file
            feature_names=feature_column_names,
        )

        # Validate return value - now returns (fig_responses, fig_heatmaps) tuple
        assert result is not None, "Should return tuple when return_fig=True"
        fig_responses, fig_heatmaps = result
        assert fig_responses is not None, "Should return response plots figure"

        # Save response plots figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if isinstance(fig_responses, list):
            for i, f in enumerate(fig_responses):
                output_path = self.output_dir / f"response_plots_highlevel_{timestamp}_p{i+1}.png"
                f.savefig(output_path, dpi=150, bbox_inches='tight')
                logger.info(f"Saved: {output_path}")
                plt.close(f)
        else:
            output_path = self.output_dir / f"response_plots_highlevel_{timestamp}.png"
            fig_responses.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved: {output_path}")
            plt.close(fig_responses)

        # Save heatmaps figure (non-mixed bivariate, no beta scaling)
        if fig_heatmaps is not None:
            if isinstance(fig_heatmaps, list):
                for i, f in enumerate(fig_heatmaps):
                    output_path = (
                        self.output_dir / f"response_heatmaps_highlevel_{timestamp}_p{i+1}.png"
                    )
                    f.savefig(output_path, dpi=150, bbox_inches='tight')
                    logger.info(f"Saved: {output_path}")
                    plt.close(f)
            else:
                output_path = self.output_dir / f"response_heatmaps_highlevel_{timestamp}.png"
                fig_heatmaps.savefig(output_path, dpi=150, bbox_inches='tight')
                logger.info(f"Saved: {output_path}")
                plt.close(fig_heatmaps)

        print("\\n[OK] plot_partial_responses() high-level function test passed for Credit-G")
        print(f"     Output saved to: {self.output_dir}")

    def test_nomogram_function_with_trim_quantile(
        self,
        loaded_lasso_results,
        x_train_tensor,
        loaded_model_and_scaler,
        loaded_group_manager,
        loaded_train_data,
        categorical_labels_dict,
    ):
        """
        Test nomogram() with trim_quantile to verify outlier trimming for continuous variables.

        Uses trim_quantile=0.05 to trim 5th and 95th percentiles from continuous
        variable ranges. This should result in plots where histograms may extend
        beyond the axis limits set by the partial response curves.
        """
        lasso_results = loaded_lasso_results
        X_train_tensor = x_train_tensor
        blackbox_model = loaded_model_and_scaler['model']
        scaler = loaded_model_and_scaler['scaler']
        group_manager = loaded_group_manager['group_manager']
        feature_column_names = loaded_train_data['feature_column_names']

        logger.info("=" * 60)
        logger.info("Testing nomogram() with trim_quantile=0.05 on Credit-G")
        logger.info("=" * 60)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # =====================================================================
        # Part 1: Test nomogram() with trim_quantile
        # =====================================================================
        result = nomogram(
            lasso_results=lasso_results,
            x=X_train_tensor,
            model=blackbox_model,
            scaler=scaler,
            n_steps=50,
            method='lebesgue',
            x_train=X_train_tensor,
            device='cpu',
            categorical_threshold=15,
            show_fig=False,
            return_fig=True,
            two_column=False,
            categorical_labels=categorical_labels_dict,
            onehot_group_manager=group_manager,
            label_manager=None,
            feature_names=feature_column_names,
            trim_quantile=0.05,  # Trim to 5th-95th percentile (keep 90%)
        )

        univariate_responses = result.univariate_responses
        _x_univariate = result.x_univariate
        fig_main = result.fig_main
        fig_non_mixed = result.fig_bivariate

        assert fig_main is not None, "Should return main figure"
        assert len(univariate_responses) > 0, "Should have univariate responses"

        # Save main nomogram figure(s)
        if isinstance(fig_main, list):
            for i, fig in enumerate(fig_main):
                output_path = self.output_dir / f"nomogram_trim_q05_main_{timestamp}_p{i+1}.png"
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                logger.info(f"Saved: {output_path}")
                plt.close(fig)
        else:
            output_path = self.output_dir / f"nomogram_trim_q05_main_{timestamp}.png"
            fig_main.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved: {output_path}")
            plt.close(fig_main)

        # Save non-mixed nomogram figure(s) if present
        if fig_non_mixed is not None:
            if isinstance(fig_non_mixed, list):
                for i, fig in enumerate(fig_non_mixed):
                    output_path = (
                        self.output_dir / f"nomogram_trim_q05_nonmixed_{timestamp}_p{i+1}.png"
                    )
                    fig.savefig(output_path, dpi=150, bbox_inches='tight')
                    logger.info(f"Saved: {output_path}")
                    plt.close(fig)
            else:
                output_path = self.output_dir / f"nomogram_trim_q05_nonmixed_{timestamp}.png"
                fig_non_mixed.savefig(output_path, dpi=150, bbox_inches='tight')
                logger.info(f"Saved: {output_path}")
                plt.close(fig_non_mixed)

        # =====================================================================
        # Part 2: Test plot_partial_responses() with trim_quantile
        # =====================================================================
        logger.info("Testing plot_partial_responses() with trim_quantile=0.05")

        response_result = plot_partial_responses(
            lasso_results=lasso_results,
            x=X_train_tensor,
            model=blackbox_model,
            scaler=scaler,
            n_steps=50,
            method='lebesgue',
            x_train=X_train_tensor,
            device='cpu',
            categorical_threshold=15,
            subtract_univariate=True,
            subfig_size=3.5,
            show_fig=False,
            return_fig=True,
            use_odds_ratio=False,
            categorical_labels=categorical_labels_dict,
            onehot_group_manager=group_manager,
            label_manager=None,
            feature_names=feature_column_names,
            trim_quantile=0.05,  # Trim to 5th-95th percentile
        )

        assert response_result is not None, "Should return tuple when return_fig=True"
        fig_responses, fig_heatmaps = response_result
        assert fig_responses is not None, "Should return response plots figure"

        # Save response plots figure(s)
        if isinstance(fig_responses, list):
            for i, fig in enumerate(fig_responses):
                output_path = self.output_dir / f"response_trim_q05_{timestamp}_p{i+1}.png"
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                logger.info(f"Saved: {output_path}")
                plt.close(fig)
        else:
            output_path = self.output_dir / f"response_trim_q05_{timestamp}.png"
            fig_responses.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved: {output_path}")
            plt.close(fig_responses)

        # Save heatmaps figure(s) if present
        if fig_heatmaps is not None:
            if isinstance(fig_heatmaps, list):
                for i, fig in enumerate(fig_heatmaps):
                    output_path = self.output_dir / f"heatmaps_trim_q05_{timestamp}_p{i+1}.png"
                    fig.savefig(output_path, dpi=150, bbox_inches='tight')
                    logger.info(f"Saved: {output_path}")
                    plt.close(fig)
            else:
                output_path = self.output_dir / f"heatmaps_trim_q05_{timestamp}.png"
                fig_heatmaps.savefig(output_path, dpi=150, bbox_inches='tight')
                logger.info(f"Saved: {output_path}")
                plt.close(fig_heatmaps)

        print(
            "\\n[OK] nomogram() and plot_partial_responses() with trim_quantile test passed for Credit-G"
        )
        print(f"     Output saved to: {self.output_dir}")

    def test_nomogram_function_odds_ratio_scale(
        self,
        loaded_lasso_results,
        x_train_tensor,
        loaded_model_and_scaler,
        loaded_group_manager,
        loaded_train_data,
        categorical_labels_dict,
    ):
        """
        Test nomogram() with odds ratio scale (use_odds_ratio=True).

        This validates log-scale formatting for odds ratios works correctly.
        """
        lasso_results = loaded_lasso_results
        X_train_tensor = x_train_tensor
        blackbox_model = loaded_model_and_scaler['model']
        scaler = loaded_model_and_scaler['scaler']
        group_manager = loaded_group_manager['group_manager']
        feature_column_names = loaded_train_data['feature_column_names']

        logger.info("=" * 60)
        logger.info("Testing nomogram() with use_odds_ratio=True on Credit-G")
        logger.info("=" * 60)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Call nomogram with odds ratio scale
        result = nomogram(
            lasso_results=lasso_results,
            x=X_train_tensor,
            model=blackbox_model,
            scaler=scaler,
            n_steps=50,
            method='lebesgue',
            x_train=X_train_tensor,
            device='cpu',
            categorical_threshold=15,
            subtract_univariate=True,
            show_fig=False,
            return_fig=True,
            use_odds_ratio=True,  # Use odds ratio (log) scale
            two_column=True,
            surround_axes=True,
            categorical_labels=categorical_labels_dict,
            onehot_group_manager=group_manager,
            label_manager=None,
            feature_names=feature_column_names,
        )

        fig_main = result.fig_main
        fig_non_mixed = result.fig_bivariate

        assert fig_main is not None, "Should return main figure"

        # Save main nomogram figure(s)
        if isinstance(fig_main, list):
            for i, fig in enumerate(fig_main):
                output_path = self.output_dir / f"nomogram_odds_ratio_{timestamp}_p{i+1}.png"
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                logger.info(f"Saved: {output_path}")
                plt.close(fig)
        else:
            output_path = self.output_dir / f"nomogram_odds_ratio_{timestamp}.png"
            fig_main.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved: {output_path}")
            plt.close(fig_main)

        # Save non-mixed figure(s) if present
        if fig_non_mixed is not None:
            if isinstance(fig_non_mixed, list):
                for i, fig in enumerate(fig_non_mixed):
                    output_path = (
                        self.output_dir / f"nomogram_odds_ratio_nonmixed_{timestamp}_p{i+1}.png"
                    )
                    fig.savefig(output_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
            else:
                output_path = self.output_dir / f"nomogram_odds_ratio_nonmixed_{timestamp}.png"
                fig_non_mixed.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close(fig_non_mixed)

        print("\\n[OK] nomogram() with odds ratio scale test passed for Credit-G")
        print(f"     Output saved to: {self.output_dir}")

    def test_response_plots_odds_ratio_scale(
        self,
        loaded_lasso_results,
        x_train_tensor,
        loaded_model_and_scaler,
        loaded_group_manager,
        loaded_train_data,
        categorical_labels_dict,
    ):
        """
        Test plot_partial_responses() with odds ratio scale (use_odds_ratio=True).

        This validates log-scale formatting for response plots works correctly.
        """
        lasso_results = loaded_lasso_results
        X_train_tensor = x_train_tensor
        blackbox_model = loaded_model_and_scaler['model']
        scaler = loaded_model_and_scaler['scaler']
        group_manager = loaded_group_manager['group_manager']
        feature_column_names = loaded_train_data['feature_column_names']

        logger.info("=" * 60)
        logger.info("Testing plot_partial_responses() with use_odds_ratio=True on Credit-G")
        logger.info("=" * 60)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Call response plots with odds ratio scale
        result = plot_partial_responses(
            lasso_results=lasso_results,
            x=X_train_tensor,
            model=blackbox_model,
            scaler=scaler,
            n_steps=50,
            method='lebesgue',
            x_train=X_train_tensor,
            device='cpu',
            categorical_threshold=15,
            subtract_univariate=True,
            subfig_size=3.5,
            show_fig=False,
            return_fig=True,
            use_odds_ratio=True,  # Use odds ratio (log) scale
            categorical_labels=categorical_labels_dict,
            onehot_group_manager=group_manager,
            label_manager=None,
            feature_names=feature_column_names,
        )

        assert result is not None, "Should return tuple when return_fig=True"
        fig_responses, fig_heatmaps = result
        assert fig_responses is not None, "Should return response plots figure"

        # Save response plots figure(s)
        if isinstance(fig_responses, list):
            for i, fig in enumerate(fig_responses):
                output_path = self.output_dir / f"response_odds_ratio_{timestamp}_p{i+1}.png"
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                logger.info(f"Saved: {output_path}")
                plt.close(fig)
        else:
            output_path = self.output_dir / f"response_odds_ratio_{timestamp}.png"
            fig_responses.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved: {output_path}")
            plt.close(fig_responses)

        # Save heatmaps figure(s) if present
        if fig_heatmaps is not None:
            if isinstance(fig_heatmaps, list):
                for i, fig in enumerate(fig_heatmaps):
                    output_path = self.output_dir / f"heatmaps_odds_ratio_{timestamp}_p{i+1}.png"
                    fig.savefig(output_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
            else:
                output_path = self.output_dir / f"heatmaps_odds_ratio_{timestamp}.png"
                fig_heatmaps.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close(fig_heatmaps)

        print("\\n[OK] plot_partial_responses() with odds ratio scale test passed for Credit-G")
        print(f"     Output saved to: {self.output_dir}")


# =============================================================================
# Main entry point for direct execution
# =============================================================================

if __name__ == "__main__":
    # Check fixtures exist before running
    exists, msg = check_fixture_files_exist()
    if not exists:
        print(f"\\nERROR: {msg}")
        sys.exit(1)

    # Run with verbose output
    pytest.main([__file__, "-v", "-s", "--tb=short"])
