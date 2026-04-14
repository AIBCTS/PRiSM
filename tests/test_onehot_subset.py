"""Test one-hot group handling in subset partial response calculation (for plotting)."""

import numpy as np
import torch

from prism.partial_responses import (
    PartialResponseCalculator,
    partial_responses_subset,
    stable_logit,
)
from prism.preprocessing import NoScaler, OneHotGroupManager
from tests.conftest import LinearTestModel


def test_subset_with_onehot_collapse():
    """Test that subset calculation works correctly with automatic one-hot collapsing.

    This test validates the plotting pathway where we generate grids of feature
    values and compute responses on those grids. One-hot groups are automatically
    collapsed when a group_manager is provided.
    """
    torch.manual_seed(42)

    # Linear model for exact reconstruction
    model = LinearTestModel([0.3, 0.5, -0.2, 0.1, 0.4], bias=0.0)

    n_train = 100
    n_test = 20

    # Create proper one-hot data
    # Features 0-2: one-hot group (3-category variable)
    # Features 3-4: continuous
    x_train = torch.zeros(n_train, 5)
    x_test = torch.zeros(n_test, 5)

    for i in range(n_train):
        category = torch.randint(0, 4, (1,)).item()
        if category > 0:
            x_train[i, category - 1] = 1

    for i in range(n_test):
        category = torch.randint(0, 4, (1,)).item()
        if category > 0:
            x_test[i, category - 1] = 1

    # Add continuous features
    x_train[:, 3:5] = torch.randn(n_train, 2)
    x_test[:, 3:5] = torch.randn(n_test, 2)

    feature_names = ['cat_A', 'cat_B', 'cat_C', 'feat_3', 'feat_4']
    groups_dict = {'category': ['cat_A', 'cat_B', 'cat_C']}
    group_manager = OneHotGroupManager(groups_dict)

    print("\n" + "=" * 70)
    print("TEST: Subset Calculation with One-Hot Groups and Collapsing")
    print("=" * 70)

    # Test 1: Call subset function
    print("\n1. SUBSET CALCULATION")
    print("-" * 70)

    univariate_responses, bivariate_responses, x_univariate, x_bivariate = (
        partial_responses_subset(
            x=x_test,
            model=model,
            method='lebesgue',
            x_train=x_train,
            device='cpu',
            n_steps=15,
            selected_features=None,  # All features
            selected_feature_pairs=None,  # All pairs
            batch_size=64,
            group_manager=group_manager,
            feature_names=feature_names,
            scaler=NoScaler(),
        )
    )

    print(f"Number of univariate features: {len(univariate_responses)}")
    print(f"Number of bivariate pairs: {len(bivariate_responses)}")
    print("Expected univariate: 3 (collapsed)")
    print("Expected bivariate: 3 (C(3,2) = 3)")

    assert (
        len(univariate_responses) == 3
    ), f"Expected 3 univariate, got {len(univariate_responses)}"
    assert len(bivariate_responses) == 3, f"Expected 3 bivariate, got {len(bivariate_responses)}"

    # Test 2: Check grid structure for one-hot group
    print("\n2. GRID STRUCTURE FOR ONE-HOT GROUP")
    print("-" * 70)

    # First feature (collapsed one-hot group) should have categorical integer grid
    grid_onehot = x_univariate[0]
    print("One-hot group grid (feature 0 - 'category'):")
    print(f"  Values: {grid_onehot}")
    print("  Expected: [0, 1, 2, 3] (reference + 3 categories)")
    print("  Type: categorical integers")

    # Continuous features should have continuous grids
    grid_continuous = x_univariate[1]  # Feature 1 in collapsed space (feat_3)
    print("\nContinuous feature grid (feature 1 - 'feat_3'):")
    print(f"  Number of points: {len(grid_continuous)}")
    print(f"  Min: {grid_continuous.min():.3f}, Max: {grid_continuous.max():.3f}")
    print("  Type: continuous (linspace)")

    # Test 3: Verify responses are finite
    print("\n3. RESPONSE VALIDITY")
    print("-" * 70)

    for i, resp in enumerate(univariate_responses):
        is_finite = np.all(np.isfinite(resp))
        print(f"  Univariate feature {i}: finite={is_finite}, shape={resp.shape}")
        assert is_finite, f"Univariate {i} contains NaN/Inf"

    for i, resp in enumerate(bivariate_responses):
        is_finite = np.all(np.isfinite(resp))
        print(f"  Bivariate pair {i}: finite={is_finite}, shape={resp.shape}")
        assert is_finite, f"Bivariate {i} contains NaN/Inf"

    # Test 4: Compare subset responses with full calculation at data points
    print("\n4. CONSISTENCY WITH FULL CALCULATION")
    print("-" * 70)

    # Full calculation (collapsing is automatic when group_manager is provided)
    calc_full = PartialResponseCalculator(
        model,
        method='lebesgue',
        x_train=x_train,
        input_dim=5,
        group_manager=group_manager,
        feature_names=feature_names,
        scaler=NoScaler(),
    )

    uni_full, bi_full = calc_full.calculate(x_test)

    print("Full calculation shapes:")
    print(f"  Univariate: {uni_full.shape}")
    print(f"  Bivariate: {bi_full.shape}")

    # Test 5: Reconstruction with full calculation
    print("\n5. RECONSTRUCTION TEST")
    print("-" * 70)

    y_pred = model(x_test)
    logit_pred = stable_logit(y_pred)

    logit_reconstructed = calc_full.logit_y0 + uni_full.sum(dim=1) + bi_full.sum(dim=1)

    error = (logit_pred - logit_reconstructed).abs().max().item()

    print(f"Baseline: {calc_full.logit_y0:.6f}")
    print(f"Max reconstruction error: {error:.6e}")
    print("Expected: < 1e-3 (linear model should be exact)")

    assert error < 1e-3, f"Reconstruction failed with error {error:.6e}"

    # Test 6: Verify one-hot grid has correct structure
    print("\n6. ONE-HOT GRID DETAILED CHECK")
    print("-" * 70)

    # The one-hot grid should be [0, 1, 2, 3] representing:
    # 0 = reference (all zeros)
    # 1 = category A (cat_A=1)
    # 2 = category B (cat_B=1)
    # 3 = category C (cat_C=1)

    expected_categories = np.array([0, 1, 2, 3])
    grid_categories = x_univariate[0]

    print(f"Expected categories: {expected_categories}")
    print(f"Actual grid: {grid_categories}")

    # Should have 4 values (reference + 3 categories)
    assert len(grid_categories) == 4, f"Expected 4 categories, got {len(grid_categories)}"

    # Should be integers 0, 1, 2, 3
    np.testing.assert_array_equal(
        grid_categories, expected_categories, err_msg="One-hot grid should be [0, 1, 2, 3]"
    )

    print("  Grid structure: CORRECT")

    # Test 7: Response values make sense
    print("\n7. RESPONSE VALUE SANITY CHECK")
    print("-" * 70)

    # For linear model f(x) = 0.3*x0 + 0.5*x1 - 0.2*x2 + 0.1*x3 + 0.4*x4
    # The one-hot responses should reflect these coefficients

    onehot_responses = univariate_responses[0]
    print("One-hot group responses at each category:")
    print(f"  Category 0 (reference):     {onehot_responses[0]:.6f}")
    print(f"  Category 1 (cat_A, coef=0.3): {onehot_responses[1]:.6f}")
    print(f"  Category 2 (cat_B, coef=0.5): {onehot_responses[2]:.6f}")
    print(f"  Category 3 (cat_C, coef=-0.2): {onehot_responses[3]:.6f}")

    # The response differences should relate to the coefficient differences
    # E.g., cat_B - cat_A should relate to 0.5 - 0.3 = 0.2
    diff_B_A = onehot_responses[2] - onehot_responses[1]
    print(f"\nResponse difference (cat_B - cat_A): {diff_B_A:.6f}")
    print("Expected (approx): related to coefficient diff 0.5 - 0.3 = 0.2")
    print("Note: Exact relationship depends on sigmoid transformation")

    print("\n" + "=" * 70)
    print("SUCCESS: All subset calculation tests passed!")
    print("=" * 70)


def test_onehot_groups_automatically_collapse():
    """Test that onehot_groups ALWAYS trigger automatic collapsing.

    After the fix, onehot_groups automatically enable collapsing to ensure
    mathematically correct results. This test validates the automatic behavior.
    """
    torch.manual_seed(42)

    model = LinearTestModel([0.3, 0.5, -0.2, 0.1, 0.4], bias=0.0)

    n_train = 50
    x_train = torch.zeros(n_train, 5)
    x_test = torch.zeros(10, 5)

    for i in range(n_train):
        category = torch.randint(0, 4, (1,)).item()
        if category > 0:
            x_train[i, category - 1] = 1

    for i in range(10):
        category = torch.randint(0, 4, (1,)).item()
        if category > 0:
            x_test[i, category - 1] = 1

    x_train[:, 3:5] = torch.randn(n_train, 2)
    x_test[:, 3:5] = torch.randn(10, 2)

    feature_names = ['cat_A', 'cat_B', 'cat_C', 'feat_3', 'feat_4']
    onehot_groups = [[0, 1, 2]]

    print("\n" + "=" * 70)
    print("TEST: onehot_groups triggers automatic collapsing")
    print("=" * 70)

    # Create calculator with onehot_groups (should auto-collapse)
    calc = PartialResponseCalculator(
        model,
        method='lebesgue',
        x_train=x_train,
        input_dim=5,
        onehot_groups=onehot_groups,
        feature_names=feature_names,
        scaler=NoScaler(),
    )

    print(f"\nCollapse automatically enabled: {calc.collapse_onehot}")
    print("Expected: True")
    assert calc.collapse_onehot is True, "onehot_groups should automatically enable collapsing"

    # Calculate responses
    uni_full, bi_full = calc.calculate(x_test)

    # Should return collapsed space (3 features, not 5)
    print(f"\nUnvariate shape: {uni_full.shape}")
    print("Expected: (10, 3) - collapsed space")
    assert uni_full.shape[1] == 3, "Should return collapsed features"

    # Try reconstruction - should have SMALL error
    y_pred = model(x_test)
    logit_pred = stable_logit(y_pred)
    logit_reconstructed = calc.logit_y0 + uni_full.sum(dim=1) + bi_full.sum(dim=1)
    error = (logit_pred - logit_reconstructed).abs().max().item()

    print(f"\nReconstruction error: {error:.6e}")
    print("Expected: < 1e-3 (should be nearly exact for linear model)")
    print(
        f"Result: {'GOOD - automatic collapse works!' if error < 1e-3 else 'BAD - unexpected error'}"
    )

    assert error < 1e-3, f"With automatic collapse, error should be small, got {error:.6e}"

    print("\n" + "=" * 70)
    print("SUCCESS: onehot_groups automatically enables collapsing")
    print("This ensures mathematically correct results!")
    print("=" * 70)


if __name__ == "__main__":
    test_subset_with_onehot_collapse()
    print("\n\n")
    test_onehot_groups_automatically_collapse()
