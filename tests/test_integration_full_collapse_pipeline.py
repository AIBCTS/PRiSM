"""
Full integration tests for collapsed space consistency across the PRiSM pipeline.

This test suite validates the complete pipeline:
    1. Partial Response Calculation (OHE input → collapsed output)
    2. LASSO Feature Selection (operates in collapsed space)
    3. Plotting (uses collapsed indices, auto-collapses scaler)

Test coverage aligns with Week 1 of the collapse validation plan.
"""

import numpy as np
import pytest
import torch

from prism.lasso import LassoComputation, LassoConfig
from prism.partial_responses import partial_responses, partial_responses_subset
from prism.plotting.pipeline import PlottingPipeline
from prism.preprocessing import MedianStdScaler, NoScaler, OneHotGroupManager, PRiSMScaler

# ============================================================================
# Fixtures
# ============================================================================


class LinearTestModel(torch.nn.Module):
    """Purely linear model with known coefficients for ground truth testing."""

    def __init__(self, coefficients, bias=0.0):
        super().__init__()
        self.coefficients = torch.nn.Parameter(
            torch.tensor(coefficients, dtype=torch.float32), requires_grad=False
        )
        self.bias = torch.nn.Parameter(
            torch.tensor(bias, dtype=torch.float32), requires_grad=False
        )

    def forward(self, x):
        logits = torch.matmul(x, self.coefficients) + self.bias
        return torch.sigmoid(logits)

    def predict_proba(self, x, device=None):
        if device is not None:
            x = x.to(device)
            self.to(device)
        return self.forward(x)


@pytest.fixture
def full_pipeline_data():
    """Create comprehensive test data for full pipeline integration.

    OHE Space (8 features):
        - age (continuous, index 0)
        - bmi (continuous, index 1)
        - diagn_CAD (index 2)
        - diagn_Valve (index 3)
        - diagn_Other (index 4)
        - urgency_Elective (index 5)
        - urgency_Urgent (index 6)
        - urgency_Emergent (index 7)

    Collapsed Space (4 features):
        - age (continuous, index 0)
        - bmi (continuous, index 1)
        - diagn (categorical, index 2)
        - urgency (categorical, index 3)
    """
    torch.manual_seed(42)
    np.random.seed(42)

    n_train = 150
    n_test = 50
    n_samples = n_train + n_test

    # Continuous features (normalized)
    age = torch.randn(n_samples, 1) * 0.5
    bmi = torch.randn(n_samples, 1) * 0.3

    # Categorical: diagn (3 OHE columns + implicit reference)
    diagn = torch.zeros(n_samples, 3)
    for i in range(n_samples):
        cat = np.random.randint(0, 4)  # 0=ref, 1-3=OHE columns
        if cat > 0:
            diagn[i, cat - 1] = 1.0

    # Categorical: urgency (3 OHE columns + implicit reference)
    urgency = torch.zeros(n_samples, 3)
    for i in range(n_samples):
        cat = np.random.randint(0, 4)
        if cat > 0:
            urgency[i, cat - 1] = 1.0

    # Combine into OHE tensor
    X_ohe = torch.cat([age, bmi, diagn, urgency], dim=1)

    # Generate target based on model
    coefficients = [0.5, -0.3, 0.2, 0.4, -0.1, 0.1, 0.3, 0.6]
    model = LinearTestModel(coefficients, bias=-0.5)
    probs = model(X_ohe)
    y = (probs > 0.5).float()

    # Split train/test
    X_train = X_ohe[:n_train]
    X_test = X_ohe[n_train:]
    y_train = y[:n_train]
    y_test = y[n_train:]

    feature_names_ohe = [
        'age',
        'bmi',
        'diagn_CAD',
        'diagn_Valve',
        'diagn_Other',
        'urgency_Elective',
        'urgency_Urgent',
        'urgency_Emergent',
    ]

    feature_names_collapsed = ['age', 'bmi', 'diagn', 'urgency']

    groups_dict = {
        'diagn': ['diagn_CAD', 'diagn_Valve', 'diagn_Other'],
        'urgency': ['urgency_Elective', 'urgency_Urgent', 'urgency_Emergent'],
    }

    group_manager = OneHotGroupManager(groups_dict)

    # Create a scaler in OHE space
    scaler_inner = MedianStdScaler(sd_scale=1.0)
    scaler_inner.fit(X_train.numpy())
    scaler_ohe = PRiSMScaler(scaler_inner)

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'model': model,
        'feature_names_ohe': feature_names_ohe,
        'feature_names_collapsed': feature_names_collapsed,
        'groups_dict': groups_dict,
        'group_manager': group_manager,
        'scaler_ohe': scaler_ohe,
        'n_ohe_features': 8,
        'n_collapsed_features': 4,
    }


# ============================================================================
# Test: Full Pipeline OHE → Collapsed → LASSO → Plotting
# ============================================================================


@pytest.mark.integration
class TestFullCollapsePipeline:
    """End-to-end tests for PR → LASSO → Plotting with collapse."""

    def test_partial_responses_collapsed_dimensions(self, full_pipeline_data):
        """Step 1: Verify PR calculation outputs collapsed dimensions."""
        data = full_pipeline_data

        pr_train = partial_responses(
            x=data['X_train'],
            model=data['model'],
            method='lebesgue',
            x_train=data['X_train'],
            group_manager=data['group_manager'],
            feature_names=data['feature_names_ohe'],
            scaler=NoScaler(),
        )

        # 4 univariate + C(4,2)=6 bivariate = 10 columns
        assert (
            pr_train.shape[1] == 10
        ), f"Expected 10 collapsed PR columns, got {pr_train.shape[1]}"

    def test_lasso_operates_in_collapsed_space(self, full_pipeline_data):
        """Step 2: Verify LASSO operates in collapsed space."""
        data = full_pipeline_data

        # Calculate PR in collapsed space
        pr_train = partial_responses(
            x=data['X_train'],
            model=data['model'],
            method='lebesgue',
            x_train=data['X_train'],
            group_manager=data['group_manager'],
            feature_names=data['feature_names_ohe'],
            scaler=NoScaler(),
        )

        pr_test = partial_responses(
            x=data['X_test'],
            model=data['model'],
            method='lebesgue',
            x_train=data['X_train'],
            group_manager=data['group_manager'],
            feature_names=data['feature_names_ohe'],
            scaler=NoScaler(),
        )

        # Fit LASSO
        config = LassoConfig(
            max_lambda=5.0,
            min_lambda=0.01,
            nlambda=10,
            batch_size=10,
            max_workers=1,
            real_time_plot=False,
        )

        lasso = LassoComputation(config)
        results, _ = lasso.fit(
            partial_responses_train=pr_train,
            partial_responses_test=pr_test,
            y_train=data['y_train'].numpy(),
            y_test=data['y_test'].numpy(),
            feature_names=data['feature_names_collapsed'],
        )

        # Select a lambda
        results.select_lambda(lambda_index=5)

        # Get selected univariate indices - should be in [0, 3] (collapsed space)
        selected = results.get_selected_univariate_indices()

        for idx in selected:
            assert 0 <= idx < 4, f"Selected index {idx} is out of collapsed space bounds [0, 3]"

    def test_full_pipeline_pr_lasso_plotting(self, full_pipeline_data):
        """Step 3: Full pipeline from PR → LASSO → Plotting."""
        data = full_pipeline_data

        # === Step 1: Calculate PR (collapsed) ===
        pr_train = partial_responses(
            x=data['X_train'],
            model=data['model'],
            method='lebesgue',
            x_train=data['X_train'],
            group_manager=data['group_manager'],
            feature_names=data['feature_names_ohe'],
            scaler=NoScaler(),
        )

        pr_test = partial_responses(
            x=data['X_test'],
            model=data['model'],
            method='lebesgue',
            x_train=data['X_train'],
            group_manager=data['group_manager'],
            feature_names=data['feature_names_ohe'],
            scaler=NoScaler(),
        )

        # Verify collapsed dimensions
        assert pr_train.shape[1] == 10

        # === Step 2: Run LASSO ===
        config = LassoConfig(
            max_lambda=5.0,
            min_lambda=0.01,
            nlambda=10,
            batch_size=10,
            max_workers=1,
            real_time_plot=False,
        )

        lasso = LassoComputation(config)
        results, _ = lasso.fit(
            partial_responses_train=pr_train,
            partial_responses_test=pr_test,
            y_train=data['y_train'].numpy(),
            y_test=data['y_test'].numpy(),
            feature_names=data['feature_names_collapsed'],
        )

        results.select_lambda(lambda_index=5)

        # Verify LASSO indices are in collapsed space
        selected_univ = results.get_selected_univariate_indices()
        assert all(0 <= idx < 4 for idx in selected_univ)

        # === Step 3: Plotting Pipeline ===
        pipeline = PlottingPipeline(
            lasso_results=results,
            group_manager=data['group_manager'],
        )

        # This should auto-collapse the scaler
        bundle = pipeline.prepare_plotting_bundle(
            x=data['X_test'],
            model=data['model'],
            scaler=data['scaler_ohe'],  # OHE space scaler
            n_steps=10,  # Small for speed
            method='lebesgue',
            x_train=data['X_train'],
            device='cpu',
            feature_names=data['feature_names_ohe'],
        )

        # Verify bundle was created
        assert bundle is not None
        assert bundle.n_univariate >= 0

        # Apply beta scaling
        bundle = pipeline.apply_beta_scaling(bundle)

        # Verify we can iterate features
        for info in bundle.univariate_features():
            # Index should be valid (in collapsed space)
            assert info.index >= 0
            assert info.name is not None

    def test_scaler_auto_collapse(self, full_pipeline_data):
        """Test that scaler is auto-collapsed from OHE to collapsed space."""
        data = full_pipeline_data

        # OHE scaler has 8 dimensions
        assert len(data['scaler_ohe'].scaler.median_) == 8

        # Collapse the scaler
        collapsed_scaler = data['group_manager'].create_collapsed_scaler(
            data['scaler_ohe'],
            data['feature_names_ohe'],
        )

        # Collapsed scaler should have 4 dimensions
        assert len(collapsed_scaler.scaler.median_) == 4

    def test_scaler_collapse_raises_on_mismatch(self, full_pipeline_data):
        """Test that scaler collapse raises error on dimension mismatch."""
        data = full_pipeline_data

        # Create a scaler with wrong dimensions
        wrong_scaler_inner = MedianStdScaler(sd_scale=1.0)
        wrong_scaler_inner.fit(np.random.randn(50, 5))  # 5 features, not 8
        wrong_scaler = PRiSMScaler(wrong_scaler_inner)

        with pytest.raises(ValueError, match="Scaler has 5 dimensions"):
            data['group_manager'].create_collapsed_scaler(
                wrong_scaler,
                data['feature_names_ohe'],
            )


# ============================================================================
# Test: Cross-Space Index Consistency
# ============================================================================


@pytest.mark.integration
class TestCrossSpaceConsistency:
    """Tests for index space consistency across pipeline stages."""

    def test_partial_responses_subset_uses_collapsed_indices(self, full_pipeline_data):
        """Test that partial_responses_subset correctly uses collapsed indices."""
        data = full_pipeline_data

        # Select collapsed indices 0 and 2 (age and diagn)
        selected_features = [0, 2]
        selected_pairs = [(0, 2)]

        univariate, bivariate, x_univ, x_biv = partial_responses_subset(
            data['X_test'],
            data['model'],
            method='lebesgue',
            x_train=data['X_train'],
            selected_features=selected_features,
            selected_feature_pairs=selected_pairs,
            group_manager=data['group_manager'],
            feature_names=data['feature_names_ohe'],
            scaler=NoScaler(),
        )

        # Should have 2 univariate and 1 bivariate response
        assert len(univariate) == 2
        assert len(bivariate) == 1

    def test_lasso_feature_names_match_collapsed_space(self, full_pipeline_data):
        """Test that LASSO feature names are in collapsed space."""
        data = full_pipeline_data

        pr_train = partial_responses(
            x=data['X_train'],
            model=data['model'],
            method='lebesgue',
            x_train=data['X_train'],
            group_manager=data['group_manager'],
            feature_names=data['feature_names_ohe'],
            scaler=NoScaler(),
        )

        pr_test = partial_responses(
            x=data['X_test'],
            model=data['model'],
            method='lebesgue',
            x_train=data['X_train'],
            group_manager=data['group_manager'],
            feature_names=data['feature_names_ohe'],
            scaler=NoScaler(),
        )

        config = LassoConfig(
            max_lambda=5.0,
            min_lambda=0.01,
            nlambda=10,
            batch_size=10,
            max_workers=1,
            real_time_plot=False,
        )

        lasso = LassoComputation(config)
        results, _ = lasso.fit(
            partial_responses_train=pr_train,
            partial_responses_test=pr_test,
            y_train=data['y_train'].numpy(),
            y_test=data['y_test'].numpy(),
            feature_names=data['feature_names_collapsed'],
        )

        # Univariate feature names should be collapsed names
        assert results.univariate_feature_names == data['feature_names_collapsed']

    def test_plotting_pipeline_validates_index_bounds(self, full_pipeline_data):
        """Test that PlottingPipeline validates index bounds."""
        data = full_pipeline_data

        # Create minimal LASSO results
        pr_train = partial_responses(
            x=data['X_train'],
            model=data['model'],
            method='lebesgue',
            x_train=data['X_train'],
            group_manager=data['group_manager'],
            feature_names=data['feature_names_ohe'],
            scaler=NoScaler(),
        )

        pr_test = partial_responses(
            x=data['X_test'],
            model=data['model'],
            method='lebesgue',
            x_train=data['X_train'],
            group_manager=data['group_manager'],
            feature_names=data['feature_names_ohe'],
            scaler=NoScaler(),
        )

        config = LassoConfig(
            max_lambda=5.0,
            min_lambda=0.01,
            nlambda=10,
            batch_size=10,
            max_workers=1,
            real_time_plot=False,
        )

        lasso = LassoComputation(config)
        results, _ = lasso.fit(
            partial_responses_train=pr_train,
            partial_responses_test=pr_test,
            y_train=data['y_train'].numpy(),
            y_test=data['y_test'].numpy(),
            feature_names=data['feature_names_collapsed'],
        )
        results.select_lambda(lambda_index=5)

        # Create pipeline
        pipeline = PlottingPipeline(
            lasso_results=results,
            group_manager=data['group_manager'],
        )

        # Prepare bundle with valid data
        bundle = pipeline.prepare_plotting_bundle(
            x=data['X_test'],
            model=data['model'],
            scaler=data['scaler_ohe'],
            n_steps=10,
            method='lebesgue',
            x_train=data['X_train'],
            device='cpu',
            feature_names=data['feature_names_ohe'],
        )

        # Bundle should be created successfully
        assert bundle is not None


# ============================================================================
# Test: Dirac and Lebesgue Methods
# ============================================================================


@pytest.mark.integration
class TestBothMethods:
    """Test collapse works with both Dirac and Lebesgue methods."""

    def test_dirac_method_collapse(self, full_pipeline_data):
        """Test that Dirac method works with collapse."""
        data = full_pipeline_data

        pr_train = partial_responses(
            x=data['X_train'],
            model=data['model'],
            method='dirac',
            group_manager=data['group_manager'],
            feature_names=data['feature_names_ohe'],
            scaler=NoScaler(),
        )

        # Should have 10 collapsed columns
        assert pr_train.shape[1] == 10

    def test_lebesgue_method_collapse(self, full_pipeline_data):
        """Test that Lebesgue method works with collapse."""
        data = full_pipeline_data

        pr_train = partial_responses(
            x=data['X_train'],
            model=data['model'],
            method='lebesgue',
            x_train=data['X_train'],
            group_manager=data['group_manager'],
            feature_names=data['feature_names_ohe'],
            scaler=NoScaler(),
        )

        # Should have 10 collapsed columns
        assert pr_train.shape[1] == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
