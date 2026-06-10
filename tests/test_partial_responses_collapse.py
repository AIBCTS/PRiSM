"""
Comprehensive tests for one-hot encoding collapse in partial response calculations.

These tests validate that the collapsed space operations are consistent:
1. PartialResponseCalculator properly validates input dimensions
2. One-hot groups are correctly collapsed to single categorical features
3. Partial responses output is in collapsed space
4. Reconstruction works correctly with collapsed features

Test coverage aligns with Week 1 of the collapse validation plan.
"""

import numpy as np
import pytest
import torch

from prism.partial_responses import (
    PartialResponseCalculator,
    partial_responses,
    partial_responses_subset,
    stable_logit,
)
from prism.preprocessing import NoScaler, OneHotGroupManager, collapse_onehot_features

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
def simple_ohe_data():
    """Create simple OHE data with 8 features: 2 continuous + 2 categorical groups.

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
        - diagn (categorical, index 2)  # 4 categories: 0=Ref, 1=CAD, 2=Valve, 3=Other
        - urgency (categorical, index 3)  # 4 categories: 0=Ref, 1=Elective, 2=Urgent, 3=Emergent
    """
    torch.manual_seed(42)
    np.random.seed(42)

    n_samples = 100

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

    groups_dict = {
        'diagn': ['diagn_CAD', 'diagn_Valve', 'diagn_Other'],
        'urgency': ['urgency_Elective', 'urgency_Urgent', 'urgency_Emergent'],
    }

    group_manager = OneHotGroupManager(groups_dict)

    return {
        'X_ohe': X_ohe,
        'feature_names_ohe': feature_names_ohe,
        'groups_dict': groups_dict,
        'group_manager': group_manager,
        'n_ohe_features': 8,
        'n_collapsed_features': 4,
    }


@pytest.fixture
def simple_model():
    """Create a simple linear model for 8 OHE features."""
    # Coefficients for: age, bmi, diagn_CAD, diagn_Valve, diagn_Other,
    #                   urgency_Elective, urgency_Urgent, urgency_Emergent
    coefficients = [0.5, -0.3, 0.2, 0.4, -0.1, 0.1, 0.3, 0.6]
    return LinearTestModel(coefficients, bias=-0.5)


# ============================================================================
# Test: Input Dimension Validation
# ============================================================================


class TestInputDimensionValidation:
    """Tests for runtime dimension validation in PartialResponseCalculator."""

    def test_x_train_dimension_mismatch_raises_error(self, simple_model, simple_ohe_data):
        """Test that x_train with wrong dimensions raises ValueError."""
        X_ohe = simple_ohe_data['X_ohe']

        # Create x_train with wrong number of features
        x_train_wrong = torch.randn(50, 5)  # 5 features instead of 8

        with pytest.raises(ValueError, match="x_train has 5 features, but input_dim=8"):
            PartialResponseCalculator(
                simple_model,
                method='lebesgue',
                x_train=x_train_wrong,
                input_dim=8,
            )

    def test_feature_names_length_mismatch_raises_error(self, simple_model, simple_ohe_data):
        """Test that feature_names with wrong length raises ValueError."""
        X_ohe = simple_ohe_data['X_ohe']
        group_manager = simple_ohe_data['group_manager']

        # Wrong number of feature names
        wrong_names = ['a', 'b', 'c', 'd', 'e']  # 5 instead of 8

        with pytest.raises(ValueError, match="feature_names has 5 names, but input_dim=8"):
            PartialResponseCalculator(
                simple_model,
                method='dirac',
                input_dim=8,
                feature_names=wrong_names,
            )

    def test_valid_dimensions_pass(self, simple_model, simple_ohe_data):
        """Test that correct dimensions don't raise errors."""
        X_ohe = simple_ohe_data['X_ohe']
        feature_names = simple_ohe_data['feature_names_ohe']
        group_manager = simple_ohe_data['group_manager']

        # This should not raise
        calculator = PartialResponseCalculator(
            simple_model,
            method='lebesgue',
            x_train=X_ohe,
            input_dim=8,
            group_manager=group_manager,
            feature_names=feature_names,
            scaler=NoScaler(),
        )

        assert calculator.input_dim == 8
        assert calculator.n_collapsed_features == 4


# ============================================================================
# Test: Calculator Accepts OHE Input
# ============================================================================


class TestCalculatorAcceptsOHEInput:
    """Tests that PartialResponseCalculator correctly processes OHE input."""

    def test_calculator_input_dim_matches_ohe(self, simple_model, simple_ohe_data):
        """Test that calculator input_dim equals number of OHE features."""
        X_ohe = simple_ohe_data['X_ohe']
        feature_names = simple_ohe_data['feature_names_ohe']
        group_manager = simple_ohe_data['group_manager']

        calculator = PartialResponseCalculator(
            simple_model,
            method='lebesgue',
            x_train=X_ohe,
            input_dim=8,
            group_manager=group_manager,
            feature_names=feature_names,
            scaler=NoScaler(),
        )

        assert calculator.input_dim == 8
        assert X_ohe.shape[1] == 8

    def test_calculator_collapsed_features_correct(self, simple_model, simple_ohe_data):
        """Test that calculator correctly counts collapsed features."""
        X_ohe = simple_ohe_data['X_ohe']
        feature_names = simple_ohe_data['feature_names_ohe']
        group_manager = simple_ohe_data['group_manager']

        calculator = PartialResponseCalculator(
            simple_model,
            method='lebesgue',
            x_train=X_ohe,
            input_dim=8,
            group_manager=group_manager,
            feature_names=feature_names,
            scaler=NoScaler(),
        )

        # 2 continuous + 2 collapsed groups = 4 features
        assert calculator.n_collapsed_features == 4

    def test_collapsed_feature_names_correct(self, simple_model, simple_ohe_data):
        """Test that collapsed feature names are derived correctly."""
        X_ohe = simple_ohe_data['X_ohe']
        feature_names = simple_ohe_data['feature_names_ohe']
        group_manager = simple_ohe_data['group_manager']

        calculator = PartialResponseCalculator(
            simple_model,
            method='lebesgue',
            x_train=X_ohe,
            input_dim=8,
            group_manager=group_manager,
            feature_names=feature_names,
            scaler=NoScaler(),
        )

        # Check collapsed names - groups first, then continuous
        # Order depends on groups_dict ordering + remaining features
        collapsed_names = calculator.collapsed_feature_names
        assert len(collapsed_names) == 4
        # Should contain all these names (order may vary)
        assert set(collapsed_names) == {'age', 'bmi', 'diagn', 'urgency'}


# ============================================================================
# Test: Calculator Outputs Collapsed PR
# ============================================================================


class TestCalculatorOutputsCollapsedPR:
    """Tests that partial responses are output in collapsed space."""

    def test_univariate_shape_is_collapsed(self, simple_model, simple_ohe_data):
        """Test that univariate PR has shape (n_samples, n_collapsed_features)."""
        X_ohe = simple_ohe_data['X_ohe']
        feature_names = simple_ohe_data['feature_names_ohe']
        group_manager = simple_ohe_data['group_manager']

        calculator = PartialResponseCalculator(
            simple_model,
            method='lebesgue',
            x_train=X_ohe,
            input_dim=8,
            group_manager=group_manager,
            feature_names=feature_names,
            scaler=NoScaler(),
        )

        univariate, bivariate = calculator.calculate(X_ohe[:10])

        # 4 collapsed features
        assert univariate.shape[1] == 4

    def test_bivariate_shape_is_collapsed(self, simple_model, simple_ohe_data):
        """Test that bivariate PR has correct shape for collapsed features."""
        X_ohe = simple_ohe_data['X_ohe']
        feature_names = simple_ohe_data['feature_names_ohe']
        group_manager = simple_ohe_data['group_manager']

        calculator = PartialResponseCalculator(
            simple_model,
            method='lebesgue',
            x_train=X_ohe,
            input_dim=8,
            group_manager=group_manager,
            feature_names=feature_names,
            scaler=NoScaler(),
        )

        univariate, bivariate = calculator.calculate(X_ohe[:10])

        # 4 collapsed features → C(4,2) = 6 bivariate pairs
        assert bivariate.shape[1] == 6

    def test_univariate_not_in_ohe_space(self, simple_model, simple_ohe_data):
        """Test that univariate PR is NOT in OHE space (8 features)."""
        X_ohe = simple_ohe_data['X_ohe']
        feature_names = simple_ohe_data['feature_names_ohe']
        group_manager = simple_ohe_data['group_manager']

        calculator = PartialResponseCalculator(
            simple_model,
            method='lebesgue',
            x_train=X_ohe,
            input_dim=8,
            group_manager=group_manager,
            feature_names=feature_names,
            scaler=NoScaler(),
        )

        univariate, _ = calculator.calculate(X_ohe[:10])

        # Should NOT be 8 (OHE space), should be 4 (collapsed)
        assert univariate.shape[1] != 8
        assert univariate.shape[1] == 4


# ============================================================================
# Test: Reconstruction with Collapsed Features
# ============================================================================


class TestReconstructionWithCollapse:
    """Tests that reconstruction works correctly with collapsed features."""

    def test_reconstruction_error_small_lebesgue(self, simple_model, simple_ohe_data):
        """Test that reconstruction error is small for Lebesgue method."""
        X_ohe = simple_ohe_data['X_ohe']
        feature_names = simple_ohe_data['feature_names_ohe']
        group_manager = simple_ohe_data['group_manager']

        calculator = PartialResponseCalculator(
            simple_model,
            method='lebesgue',
            x_train=X_ohe,
            input_dim=8,
            group_manager=group_manager,
            feature_names=feature_names,
            scaler=NoScaler(),
        )

        x_test = X_ohe[:20]
        univariate, bivariate = calculator.calculate(x_test)

        # Reconstruct
        y_pred = simple_model(x_test)
        logit_pred = stable_logit(y_pred)

        logit_reconstructed = calculator.logit_y0 + univariate.sum(dim=1) + bivariate.sum(dim=1)

        error = (logit_pred - logit_reconstructed).abs().max().item()

        # Linear model should reconstruct almost exactly
        # Tightened from 1e-3 to 1e-4 after recent corrections
        assert error < 1e-4, f"Reconstruction error {error:.6e} exceeds threshold"

    def test_reconstruction_error_small_dirac(self, simple_model, simple_ohe_data):
        """Test that reconstruction error is small for Dirac method."""
        X_ohe = simple_ohe_data['X_ohe']
        feature_names = simple_ohe_data['feature_names_ohe']
        group_manager = simple_ohe_data['group_manager']

        calculator = PartialResponseCalculator(
            simple_model,
            method='dirac',
            input_dim=8,
            group_manager=group_manager,
            feature_names=feature_names,
            scaler=NoScaler(),
        )

        x_test = X_ohe[:20]
        univariate, bivariate = calculator.calculate(x_test)

        # Reconstruct
        y_pred = simple_model(x_test)
        logit_pred = stable_logit(y_pred)

        logit_reconstructed = calculator.logit_y0 + univariate.sum(dim=1) + bivariate.sum(dim=1)

        error = (logit_pred - logit_reconstructed).abs().max().item()

        # Linear model should reconstruct almost exactly
        # Tightened from 1e-3 to 1e-4 after recent corrections
        assert error < 1e-4, f"Reconstruction error {error:.6e} exceeds threshold"


# ============================================================================
# Test: partial_responses Function
# ============================================================================


class TestPartialResponsesFunction:
    """Tests for the partial_responses convenience function."""

    def test_partial_responses_with_group_manager(self, simple_model, simple_ohe_data):
        """Test partial_responses() with group_manager outputs collapsed PR."""
        X_ohe = simple_ohe_data['X_ohe']
        feature_names = simple_ohe_data['feature_names_ohe']
        group_manager = simple_ohe_data['group_manager']

        pr_train = partial_responses(
            X_ohe,
            simple_model,
            method='lebesgue',
            x_train=X_ohe,
            group_manager=group_manager,
            feature_names=feature_names,
            scaler=NoScaler(),
        )

        # 4 univariate + 6 bivariate = 10 columns
        assert pr_train.shape[1] == 10

    def test_partial_responses_without_group_manager(self, simple_model, simple_ohe_data):
        """Test partial_responses() without group_manager outputs OHE space."""
        X_ohe = simple_ohe_data['X_ohe']
        feature_names = simple_ohe_data['feature_names_ohe']

        pr_train = partial_responses(
            X_ohe,
            simple_model,
            method='lebesgue',
            x_train=X_ohe,
        )

        # 8 univariate + C(8,2)=28 bivariate = 36 columns
        assert pr_train.shape[1] == 36


# ============================================================================
# Test: partial_responses_subset Function
# ============================================================================


class TestPartialResponsesSubset:
    """Tests for the partial_responses_subset function."""

    def test_subset_returns_collapsed_dimensions(self, simple_model, simple_ohe_data):
        """Test that subset returns responses in collapsed dimensions."""
        X_ohe = simple_ohe_data['X_ohe']
        feature_names = simple_ohe_data['feature_names_ohe']
        group_manager = simple_ohe_data['group_manager']

        # Select collapsed indices 0 and 2 (age and diagn)
        selected_features = [0, 2]
        selected_pairs = [(0, 2)]  # age x diagn

        univariate, bivariate, x_univ, x_biv = partial_responses_subset(
            X_ohe,
            simple_model,
            method='lebesgue',
            x_train=X_ohe,
            selected_features=selected_features,
            selected_feature_pairs=selected_pairs,
            group_manager=group_manager,
            feature_names=feature_names,
            scaler=NoScaler(),
        )

        # Should have 2 univariate responses (for selected features)
        assert len(univariate) == 2
        # Should have 1 bivariate response
        assert len(bivariate) == 1


# ============================================================================
# Test: Edge Cases
# ============================================================================


class TestCollapseEdgeCases:
    """Tests for edge cases in collapse handling."""

    def test_single_group_collapses_correctly(self):
        """Test with only one categorical group."""
        torch.manual_seed(42)

        # 3 OHE features: 1 continuous + 1 group (2 columns)
        X = torch.zeros(50, 3)
        X[:, 0] = torch.randn(50)  # continuous
        for i in range(50):
            cat = np.random.randint(0, 3)  # 0=ref, 1-2=OHE
            if cat > 0:
                X[i, cat] = 1.0

        feature_names = ['cont', 'cat_A', 'cat_B']
        groups_dict = {'cat': ['cat_A', 'cat_B']}
        group_manager = OneHotGroupManager(groups_dict)

        model = LinearTestModel([0.5, 0.2, -0.3], bias=0.0)

        calculator = PartialResponseCalculator(
            model,
            method='lebesgue',
            x_train=X,
            input_dim=3,
            group_manager=group_manager,
            feature_names=feature_names,
            scaler=NoScaler(),
        )

        # Should collapse to 2 features: cont, cat
        assert calculator.n_collapsed_features == 2

        univariate, bivariate = calculator.calculate(X[:10])
        assert univariate.shape[1] == 2
        assert bivariate.shape[1] == 1  # C(2,2) = 1

    def test_all_continuous_no_collapse(self, simple_model):
        """Test that all-continuous features work without collapse."""
        torch.manual_seed(42)

        X = torch.randn(50, 4)
        feature_names = ['f0', 'f1', 'f2', 'f3']

        model = LinearTestModel([0.5, 0.2, -0.3, 0.1], bias=0.0)

        calculator = PartialResponseCalculator(
            model,
            method='dirac',
            input_dim=4,
            feature_names=feature_names,
            # No group_manager - no collapse
        )

        # Should remain at 4 features
        assert calculator.n_collapsed_features is None  # No collapse

        univariate, bivariate = calculator.calculate(X[:10])
        assert univariate.shape[1] == 4
        assert bivariate.shape[1] == 6  # C(4,2) = 6

    def test_many_groups_collapse(self):
        """Test with multiple groups (>2)."""
        torch.manual_seed(42)

        # 9 features: 3 continuous + 3 groups (2 OHE columns each)
        n_samples = 100
        X = torch.zeros(n_samples, 9)

        # Continuous
        X[:, 0] = torch.randn(n_samples)
        X[:, 1] = torch.randn(n_samples)
        X[:, 2] = torch.randn(n_samples)

        # Group 1 (indices 3, 4)
        for i in range(n_samples):
            cat = np.random.randint(0, 3)
            if cat > 0:
                X[i, 2 + cat] = 1.0

        # Group 2 (indices 5, 6)
        for i in range(n_samples):
            cat = np.random.randint(0, 3)
            if cat > 0:
                X[i, 4 + cat] = 1.0

        # Group 3 (indices 7, 8)
        for i in range(n_samples):
            cat = np.random.randint(0, 3)
            if cat > 0:
                X[i, 6 + cat] = 1.0

        feature_names = ['c0', 'c1', 'c2', 'g1_A', 'g1_B', 'g2_A', 'g2_B', 'g3_A', 'g3_B']
        groups_dict = {
            'g1': ['g1_A', 'g1_B'],
            'g2': ['g2_A', 'g2_B'],
            'g3': ['g3_A', 'g3_B'],
        }
        group_manager = OneHotGroupManager(groups_dict)

        model = LinearTestModel([0.1] * 9, bias=0.0)

        calculator = PartialResponseCalculator(
            model,
            method='lebesgue',
            x_train=X,
            input_dim=9,
            group_manager=group_manager,
            feature_names=feature_names,
            scaler=NoScaler(),
        )

        # Should collapse to 6 features: 3 continuous + 3 groups
        assert calculator.n_collapsed_features == 6

        univariate, bivariate = calculator.calculate(X[:10])
        assert univariate.shape[1] == 6
        assert bivariate.shape[1] == 15  # C(6,2) = 15


# ============================================================================
# Test: Index Mapping Correctness
# ============================================================================


class TestIndexMapping:
    """Tests for correct index mapping in collapse."""

    def test_index_mapping_structure(self, simple_model, simple_ohe_data):
        """Test that index_mapping correctly maps collapsed -> original indices."""
        X_ohe = simple_ohe_data['X_ohe']
        feature_names = simple_ohe_data['feature_names_ohe']
        group_manager = simple_ohe_data['group_manager']

        calculator = PartialResponseCalculator(
            simple_model,
            method='dirac',
            input_dim=8,
            group_manager=group_manager,
            feature_names=feature_names,
            scaler=NoScaler(),
        )

        # Verify the mapping covers all original indices
        all_original_indices = set()
        for collapsed_idx, original_indices in calculator.index_mapping.items():
            all_original_indices.update(original_indices)

        # Should cover all 8 original indices
        assert all_original_indices == {0, 1, 2, 3, 4, 5, 6, 7}

        # Verify group mapping - diagn group should have 3 members
        diagn_found = False
        urgency_found = False
        for collapsed_idx, original_indices in calculator.index_mapping.items():
            if set(original_indices) == {2, 3, 4}:
                diagn_found = True
            if set(original_indices) == {5, 6, 7}:
                urgency_found = True

        assert diagn_found, "diagn group (indices 2,3,4) not found in mapping"
        assert urgency_found, "urgency group (indices 5,6,7) not found in mapping"


# ============================================================================
# Test: Collapse with OneHotGroupManager Directly
# ============================================================================


class TestCollapseWithGroupManager:
    """Tests for OneHotGroupManager collapse functionality."""

    def test_collapse_data_shape(self, simple_ohe_data):
        """Test that collapse_onehot_features produces correct shape."""
        X_ohe = simple_ohe_data['X_ohe']
        feature_names = simple_ohe_data['feature_names_ohe']
        group_manager = simple_ohe_data['group_manager']

        X_collapsed, collapsed_names = collapse_onehot_features(
            X_ohe.numpy(), group_manager, feature_names
        )

        assert X_collapsed.shape[1] == 4
        assert len(collapsed_names) == 4

    def test_collapse_categorical_values(self, simple_ohe_data):
        """Test that collapsed categorical values are integers 0..n_categories."""
        X_ohe = simple_ohe_data['X_ohe']
        feature_names = simple_ohe_data['feature_names_ohe']
        group_manager = simple_ohe_data['group_manager']

        X_collapsed, collapsed_names = collapse_onehot_features(
            X_ohe.numpy(), group_manager, feature_names
        )

        # Find which collapsed index corresponds to diagn
        diagn_idx = collapsed_names.index('diagn')
        urgency_idx = collapsed_names.index('urgency')

        # diagn should have values in {0, 1, 2, 3} (ref + 3 categories)
        diagn_values = set(int(v) for v in np.unique(X_collapsed[:, diagn_idx]))
        assert diagn_values <= {
            0,
            1,
            2,
            3,
        }, f"diagn values {diagn_values} not in expected range [0, 3]"

        # urgency should have values in {0, 1, 2, 3} (ref + 3 categories)
        urgency_values = set(int(v) for v in np.unique(X_collapsed[:, urgency_idx]))
        assert urgency_values <= {
            0,
            1,
            2,
            3,
        }, f"urgency values {urgency_values} not in expected range [0, 3]"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
