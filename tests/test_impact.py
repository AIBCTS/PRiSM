"""Tests for the IMPACT model implementation."""

import numpy as np
import pandas as pd
import pytest
import torch

from prism.impact import IMPACTModel, apply_impact_model_torch, dataframe_to_tensor

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def impact_model():
    """Return an IMPACTModel instance on CPU."""
    return IMPACTModel()


@pytest.fixture
def valid_impact_df():
    """Minimal DataFrame with all 18 IMPACT features plus one-hot reference cols."""
    n = 20
    rng = np.random.RandomState(42)

    # Proper one-hot diagnosis
    diagn_cats = ["CAD", "Congenital", "Cardiomyopathy", "Valve", "Graftfailure", "Misc"]
    diagn_choice = rng.choice(diagn_cats, size=n)
    diagn_cols = {f"diagn_{c}": (diagn_choice == c).astype(int) for c in diagn_cats}

    # Proper one-hot ethnicity
    eth_cats = ["African American", "Hispanic", "Other", "Caucasian"]
    eth_choice = rng.choice(eth_cats, size=n)
    eth_cols = {f"recethcat_{c}": (eth_choice == c).astype(int) for c in eth_cats}

    df = pd.DataFrame(
        {
            "recageyear": rng.randint(20, 75, n).astype(float),
            "recbilirubin": rng.uniform(5, 80, n),
            "creatinine_clearance": rng.uniform(15, 120, n),
            "recsex": rng.randint(0, 2, n),
            "recdialysis": rng.randint(0, 2, n),
            **diagn_cols,
            "recinfections2weeks": rng.randint(0, 2, n),
            "reciabp": rng.randint(0, 2, n),
            "recventilator": rng.randint(0, 2, n),
            **eth_cols,
            "rececmo": rng.randint(0, 2, n),
            "recvad": rng.randint(0, 2, n),
        }
    )
    return df


# ---------------------------------------------------------------------------
# IMPACTModel basic tests
# ---------------------------------------------------------------------------


class TestIMPACTModel:
    """Core model tests."""

    def test_feature_names_length(self):
        assert len(IMPACTModel.FEATURE_NAMES) == 18

    def test_forward_output_shape(self, impact_model):
        x = torch.zeros(5, 18)
        out = impact_model(x)
        assert out["mortality_prob_logit"].shape == (5,)
        assert out["impact_score"] is None  # default: not calculated

    def test_forward_with_score(self, impact_model):
        x = torch.zeros(5, 18)
        out = impact_model(x, calculate_impact_score=True)
        assert out["impact_score"].shape == (5,)
        assert out["mortality_prob_points"].shape == (5,)

    def test_predict_returns_probabilities(self, impact_model):
        x = torch.rand(10, 18)
        probs = impact_model.predict_proba(x)
        assert probs.shape == (10,)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_zero_input_deterministic(self, impact_model):
        """All-zero input should produce a deterministic probability.

        Note: zero creatinine_clearance triggers creat_under_30 (0.8961),
        so the logit is intercept + creat_under_30 = -2.75 + 0.8961.
        """
        x = torch.zeros(1, 18)
        prob = impact_model.predict_proba(x).item()
        expected = torch.sigmoid(torch.tensor(-2.75 + 0.8961)).item()
        assert abs(prob - expected) < 1e-5


# ---------------------------------------------------------------------------
# dataframe_to_tensor tests
# ---------------------------------------------------------------------------


class TestDataframeToTensor:
    """Tests for DataFrame -> tensor conversion."""

    def test_valid_df(self, valid_impact_df):
        tensor = dataframe_to_tensor(valid_impact_df)
        assert tensor.shape == (len(valid_impact_df), 18)

    def test_missing_feature_raises(self, valid_impact_df):
        df = valid_impact_df.drop(columns=["recageyear"])
        with pytest.raises(ValueError, match="Missing required features"):
            dataframe_to_tensor(df)

    def test_extra_columns_ignored(self, valid_impact_df):
        """Columns not in FEATURE_NAMES should be silently ignored."""
        df = valid_impact_df.copy()
        df["donage"] = 40.0
        df["ischtime"] = 3.0
        tensor = dataframe_to_tensor(df)
        assert tensor.shape == (len(df), 18)

    def test_baseline_inferred_without_reference_cols(self, valid_impact_df):
        """Dropping diagn_Cardiomyopathy and recethcat_Caucasian should still work."""
        df = valid_impact_df.drop(
            columns=["diagn_Cardiomyopathy", "recethcat_Caucasian"], errors="ignore"
        )
        tensor = dataframe_to_tensor(df)
        assert tensor.shape == (len(df), 18)

    def test_invalid_diagnosis_encoding(self, valid_impact_df):
        """Multiple active diagnosis columns should raise."""
        df = valid_impact_df.copy()
        df["diagn_CAD"] = 1
        df["diagn_Congenital"] = 1
        with pytest.raises(ValueError, match="Diagnosis categories"):
            dataframe_to_tensor(df)


# ---------------------------------------------------------------------------
# apply_impact_model_torch tests
# ---------------------------------------------------------------------------


class TestApplyImpactModelTorch:

    def test_adds_prediction_column(self, valid_impact_df):
        result = apply_impact_model_torch(valid_impact_df)
        assert "pred_impact" in result.columns
        assert result["pred_impact"].between(0, 1).all()

    def test_full_mode_adds_score(self, valid_impact_df):
        result = apply_impact_model_torch(valid_impact_df, calculate_impact_score=True)
        assert "impact_score" in result.columns
        assert "pred_impact_points" in result.columns


# Integrity tests for the non-public IMPACT dataset live in
# tests/test_impact_data_integrity.py (dev-only; excluded from the publish whitelist).
