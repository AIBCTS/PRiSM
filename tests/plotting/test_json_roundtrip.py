"""
Tests for JSON roundtrip: save -> load -> plot.

Uses credit_g fixtures for visual validation of nomogram reconstruction from JSON.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prism.lasso import LassoResultsManager
from prism.maskedmlp import MaskedMLP
from prism.nomogram_plot import nomogram, nomogram_from_json
from prism.plotting import NomogramData, load_nomogram_json
from prism.preprocessing import (
    OneHotGroupManager,
    build_ordinal_labels_dict,
    collapse_onehot_features,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Path Configuration
# =============================================================================

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "credit_g_replica"
OUTPUT_BASE = Path(__file__).parent.parent / "fixtures" / "output" / "visual_validation"
OUTPUT_DIR = OUTPUT_BASE / "json_roundtrip"

# Fixed file paths in fixtures folder
TRAIN_DATA_PATH = FIXTURES_DIR / "credit-g_mlp_train.csv"
METADATA_PATH = FIXTURES_DIR / "preprocessing_metadata.json"
MODEL_PATH = FIXTURES_DIR / "credit-g_mlp_model.pt"
LASSO_PATH = FIXTURES_DIR / "blackbox_lasso_results.pt"

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

    return {
        'X_train': X_train,
        'y_train': y_train,
        'feature_column_names': X_train.columns.tolist(),
        'target_column': target_column,
    }


@pytest.fixture(scope="module")
def loaded_model_and_scaler(verify_fixtures, loaded_train_data):
    """Load model and scaler from fixtures."""
    checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)

    blackbox_model = None
    scaler = None

    if isinstance(checkpoint, dict):
        if 'scaler' in checkpoint:
            scaler = checkpoint['scaler']

        if 'model' in checkpoint:
            blackbox_model = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            n_features = len(loaded_train_data['feature_column_names'])
            blackbox_model = MaskedMLP(input_dim=n_features, output_dim=1)
            blackbox_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    elif isinstance(checkpoint, torch.nn.Module):
        blackbox_model = checkpoint

    if blackbox_model is None:
        pytest.skip("Could not load model from checkpoint")

    if isinstance(blackbox_model, torch.nn.Module):
        blackbox_model.eval()

    return {
        'model': blackbox_model,
        'scaler': scaler,
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
            pytest.skip("LASSO file does not contain 'lasso_results' key.")
    else:
        lasso_results = lasso_data

    if not isinstance(lasso_results, LassoResultsManager):
        pytest.skip(f"Loaded object is not LassoResultsManager: {type(lasso_results)}")

    # Select lambda index 81 for testing (includes bivariate features)
    lasso_results.select_lambda(81)

    return lasso_results


@pytest.fixture(scope="module")
def x_train_tensor(loaded_train_data, loaded_model_and_scaler):
    """Create X_train tensor."""
    scaler = loaded_model_and_scaler['scaler']
    X_train = loaded_train_data['X_train']

    if scaler is not None:
        X_train_tensor = scaler.to_tensor(X_train, device='cpu')
    else:
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32, device='cpu')

    return X_train_tensor


# =============================================================================
# Test Classes
# =============================================================================


class TestLoadNomogramJson:
    """Test the load_nomogram_json function."""

    def test_load_valid_json(self, tmp_path):
        """Test loading a valid JSON file returns NomogramData."""
        # Create a minimal valid JSON file
        json_data = {
            "version": "1.0",
            "metadata": {
                "base_model": "test_model",
                "method": "dirac",
                "n_steps": 50,
                "categorical_threshold": 15,
            },
            "model": {
                "intercept": -0.5,
                "selected_lambda": 0.001,
                "selected_lambda_index": 42,
            },
            "univariate": {
                "age": {
                    "index": 0,
                    "name": "age",
                    "label": "Age (years)",
                    "is_categorical": False,
                    "x_values": [20, 40, 60, 80],
                    "response": [-0.2, -0.1, 0.1, 0.2],
                    "beta": 0.5,
                    "histogram": {"bin_edges": [20, 40, 60, 80, 100], "counts": [10, 20, 15, 5]},
                }
            },
            "bivariate": {},
        }

        json_path = tmp_path / "test_nomogram.json"
        with open(json_path, 'w') as f:
            json.dump(json_data, f)

        # Load and verify
        data = load_nomogram_json(json_path)

        assert isinstance(data, NomogramData)
        assert data.version == "1.0"
        assert data.intercept == -0.5
        assert data.n_steps == 50
        assert data.base_model == "test_model"
        assert "age" in data.univariate

    def test_load_file_not_found(self, tmp_path):
        """Test FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_nomogram_json(tmp_path / "nonexistent.json")

    def test_get_category_labels(self, tmp_path):
        """Test extracting category labels from loaded data."""
        json_data = {
            "version": "1.0",
            "metadata": {},
            "model": {"intercept": 0.0},
            "univariate": {
                "gender": {
                    "index": 0,
                    "name": "gender",
                    "label": "Gender",
                    "is_categorical": True,
                    "x_values": [0, 1],
                    "response": [0.1, -0.1],
                    "beta": 0.3,
                    "histogram": {},
                    "category_labels": {"0": "Male", "1": "Female"},
                }
            },
            "bivariate": {},
        }

        json_path = tmp_path / "test_nomogram.json"
        with open(json_path, 'w') as f:
            json.dump(json_data, f)

        data = load_nomogram_json(json_path)
        labels = data.get_category_labels()

        assert "gender" in labels
        assert labels["gender"]["0"] == "Male"
        assert labels["gender"]["1"] == "Female"


class TestNomogramFromJson:
    """Test nomogram_from_json function with synthetic data."""

    def test_nomogram_from_json_renders(self, tmp_path):
        """Test that nomogram_from_json produces a valid figure."""
        # Create test JSON data
        json_data = {
            "version": "1.0",
            "metadata": {
                "base_model": "test",
                "method": "dirac",
                "n_steps": 10,
                "categorical_threshold": 15,
            },
            "model": {
                "intercept": -0.5,
                "selected_lambda": 0.001,
                "selected_lambda_index": 42,
            },
            "univariate": {
                "age": {
                    "index": 0,
                    "name": "age",
                    "label": "Age (years)",
                    "is_categorical": False,
                    "x_values": list(np.linspace(20, 80, 10)),
                    "response": list(np.linspace(-0.5, 0.5, 10)),
                    "beta": 0.5,
                    "histogram": {},
                },
                "income": {
                    "index": 1,
                    "name": "income",
                    "label": "Income ($)",
                    "is_categorical": False,
                    "x_values": list(np.linspace(30000, 100000, 10)),
                    "response": list(np.linspace(-0.3, 0.3, 10)),
                    "beta": 0.3,
                    "histogram": {},
                },
            },
            "bivariate": {},
        }

        json_path = tmp_path / "test_nomogram.json"
        with open(json_path, 'w') as f:
            json.dump(json_data, f)

        # Render from JSON
        result = nomogram_from_json(json_path, two_column=False)

        # Verify result
        assert result is not None
        assert result.fig_main is not None
        assert len(result.univariate_responses) == 2
        assert len(result.x_univariate) == 2

        # Close figure
        if isinstance(result.fig_main, list):
            for fig in result.fig_main:
                plt.close(fig)
        else:
            plt.close(result.fig_main)


class TestJsonRoundtrip:
    """Test save -> load -> plot roundtrip with credit_g data."""

    @pytest.fixture(autouse=True)
    def setup_output_dir(self):
        """Create output directory for saved figures."""
        self.output_dir = OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def test_roundtrip_visual_comparison(
        self,
        loaded_lasso_results,
        x_train_tensor,
        loaded_model_and_scaler,
        loaded_group_manager,
        loaded_train_data,
        categorical_labels_dict,
        tmp_path,
    ):
        """
        Full roundtrip test: generate nomogram, save to JSON, load, and render.

        Saves both original and reconstructed figures for visual comparison.
        """
        lasso_results = loaded_lasso_results
        X_train_tensor = x_train_tensor
        blackbox_model = loaded_model_and_scaler['model']
        scaler = loaded_model_and_scaler['scaler']
        group_manager = loaded_group_manager['group_manager']
        feature_column_names = loaded_train_data['feature_column_names']

        logger.info("=" * 60)
        logger.info("JSON Roundtrip Test: Generate -> Save -> Load -> Render")
        logger.info("=" * 60)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = tmp_path / f"nomogram_roundtrip_{timestamp}.json"

        # Step 1: Generate original nomogram and save to JSON
        logger.info("Step 1: Generating original nomogram with save_json=True...")
        original_result = nomogram(
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
            two_column=True,
            surround_axes=True,
            categorical_labels=categorical_labels_dict,
            onehot_group_manager=group_manager,
            label_manager=None,
            feature_names=feature_column_names,
            save_json=True,
            json_path=str(json_path),
            comment="Roundtrip test",
        )

        assert json_path.exists(), "JSON file was not created"
        logger.info(f"JSON saved to: {json_path}")

        # Save original figure
        original_path = self.output_dir / f"original_nomogram_{timestamp}.png"
        if isinstance(original_result.fig_main, list):
            original_result.fig_main[0].savefig(original_path, dpi=150, bbox_inches='tight')
        else:
            original_result.fig_main.savefig(original_path, dpi=150, bbox_inches='tight')
        logger.info(f"Original figure saved to: {original_path}")

        # Step 2: Load JSON and render
        logger.info("Step 2: Loading JSON and rendering nomogram_from_json()...")
        reconstructed_result = nomogram_from_json(
            json_path,
            two_column=True,
            surround_axes=True,
            show_fig=False,
            return_fig=True,
        )

        assert reconstructed_result.fig_main is not None, "Reconstructed figure is None"

        # Save reconstructed figure
        reconstructed_path = self.output_dir / f"reconstructed_nomogram_{timestamp}.png"
        if isinstance(reconstructed_result.fig_main, list):
            reconstructed_result.fig_main[0].savefig(
                reconstructed_path, dpi=150, bbox_inches='tight'
            )
        else:
            reconstructed_result.fig_main.savefig(reconstructed_path, dpi=150, bbox_inches='tight')
        logger.info(f"Reconstructed figure saved to: {reconstructed_path}")

        # Step 3: Verify data matches
        logger.info("Step 3: Verifying data consistency...")

        # Check same number of features
        assert len(original_result.univariate_responses) == len(
            reconstructed_result.univariate_responses
        ), "Number of univariate features mismatch"
        assert len(original_result.bivariate_responses) == len(
            reconstructed_result.bivariate_responses
        ), "Number of bivariate features mismatch"

        # Check response values are close (within floating point tolerance)
        for i, (orig, recon) in enumerate(
            zip(original_result.univariate_responses, reconstructed_result.univariate_responses)
        ):
            if orig is not None and recon is not None:
                np.testing.assert_allclose(
                    orig, recon, rtol=1e-5, err_msg=f"Univariate response {i} mismatch"
                )

        logger.info("Data verification passed!")

        # Clean up figures
        for result in [original_result, reconstructed_result]:
            if result.fig_main:
                if isinstance(result.fig_main, list):
                    for fig in result.fig_main:
                        plt.close(fig)
                else:
                    plt.close(result.fig_main)
            if result.fig_bivariate:
                if isinstance(result.fig_bivariate, list):
                    for fig in result.fig_bivariate:
                        plt.close(fig)
                else:
                    plt.close(result.fig_bivariate)

        print("\\n[OK] JSON roundtrip test passed!")
        print(f"     Original: {original_path}")
        print(f"     Reconstructed: {reconstructed_path}")

    def test_roundtrip_with_conversion_line(
        self,
        loaded_lasso_results,
        x_train_tensor,
        loaded_model_and_scaler,
        loaded_group_manager,
        loaded_train_data,
        categorical_labels_dict,
        tmp_path,
    ):
        """Test roundtrip with conversion line enabled."""
        lasso_results = loaded_lasso_results
        X_train_tensor = x_train_tensor
        blackbox_model = loaded_model_and_scaler['model']
        scaler = loaded_model_and_scaler['scaler']
        group_manager = loaded_group_manager['group_manager']
        feature_column_names = loaded_train_data['feature_column_names']

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = tmp_path / f"nomogram_conversion_{timestamp}.json"

        # Generate with conversion line and save
        original_result = nomogram(
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
            show_conversion_line=True,
            categorical_labels=categorical_labels_dict,
            onehot_group_manager=group_manager,
            feature_names=feature_column_names,
            save_json=True,
            json_path=str(json_path),
        )

        # Render from JSON with conversion line
        reconstructed_result = nomogram_from_json(
            json_path,
            two_column=False,
            show_conversion_line=True,
            show_fig=False,
        )

        assert reconstructed_result.fig_main is not None

        # Save for visual comparison
        reconstructed_path = self.output_dir / f"reconstructed_conversion_{timestamp}.png"
        if isinstance(reconstructed_result.fig_main, list):
            reconstructed_result.fig_main[0].savefig(
                reconstructed_path, dpi=150, bbox_inches='tight'
            )
        else:
            reconstructed_result.fig_main.savefig(reconstructed_path, dpi=150, bbox_inches='tight')

        # Clean up
        for result in [original_result, reconstructed_result]:
            if result.fig_main:
                if isinstance(result.fig_main, list):
                    for fig in result.fig_main:
                        plt.close(fig)
                else:
                    plt.close(result.fig_main)

        print("\\n[OK] Conversion line roundtrip test passed!")
        print(f"     Output: {reconstructed_path}")


# =============================================================================
# Main entry point
# =============================================================================

if __name__ == "__main__":
    exists, msg = check_fixture_files_exist()
    if not exists:
        print(f"\\nERROR: {msg}")
        sys.exit(1)

    pytest.main([__file__, "-v", "-s", "--tb=short"])
