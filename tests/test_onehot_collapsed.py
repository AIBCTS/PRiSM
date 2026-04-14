"""Test one-hot groups with proper collapsing enabled."""

import torch

from prism.partial_responses import PartialResponseCalculator, stable_logit
from prism.preprocessing import NoScaler
from tests.conftest import LinearTestModel


def test_onehot_with_collapse():
    """Test one-hot groups with collapse_onehot=True."""
    torch.manual_seed(42)

    model = LinearTestModel([0.3, 0.5, -0.2, 0.1, 0.4], bias=0.0)

    n_train = 50
    n_test = 5

    # Create proper one-hot data
    x_train = torch.zeros(n_train, 5)
    x_test = torch.zeros(n_test, 5)

    for i in range(n_train):
        category = torch.randint(0, 4, (1,)).item()
        if category > 0:
            x_train[i, category - 1] = 1

    # Test samples
    x_test[0, :] = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
    x_test[1, :] = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0])
    x_test[2, :] = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0])
    x_test[3, :] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
    x_test[4, :] = torch.tensor([1.0, 0.0, 0.0, 0.5, 0.3])

    onehot_groups = [[0, 1, 2]]
    feature_names = ['cat_A', 'cat_B', 'cat_C', 'feat_3', 'feat_4']

    print("\n" + "=" * 70)
    print("TEST: One-Hot Groups with collapse_onehot=True")
    print("=" * 70)

    # Create calculator WITH collapsing
    from prism.preprocessing import OneHotGroupManager

    groups_dict = {'category': ['cat_A', 'cat_B', 'cat_C']}
    group_manager = OneHotGroupManager(groups_dict)

    calculator = PartialResponseCalculator(
        model,
        method='lebesgue',
        x_train=x_train,
        input_dim=5,
        group_manager=group_manager,
        feature_names=feature_names,
        scaler=NoScaler(),
    )

    print(f"\nOriginal feature space: {5} features")
    print(f"Collapsed feature space: {calculator.n_collapsed_features} features")
    print(f"Index mapping: {calculator.index_mapping}")

    univariate, bivariate = calculator.calculate(x_test)

    print(f"\nUnivariate shape: {univariate.shape}")
    print(f"Bivariate shape: {bivariate.shape}")

    print("\nCollapsed univariate responses (sample 0, x=[1,0,0,0,0]):")
    for i in range(univariate.shape[1]):
        print(f"  Collapsed feature {i}: {univariate[0, i].item():.6f}")

    # Test reconstruction
    y_pred = model(x_test)
    logit_pred = stable_logit(y_pred)

    logit_reconstructed = calculator.logit_y0 + univariate.sum(dim=1) + bivariate.sum(dim=1)

    error = (logit_pred - logit_reconstructed).abs().max().item()

    print("\nReconstruction:")
    print(f"  True logit (sample 0):  {logit_pred[0].item():.6f}")
    print(f"  Reconstructed:          {logit_reconstructed[0].item():.6f}")
    print(f"  Error:                  {error:.6e}")
    print("  Expected:               < 1e-3 for linear model")

    if error < 1e-3:
        print("\n  SUCCESS: Collapsed one-hot groups work correctly!")
    else:
        print("\n  FAILURE: Collapsed one-hot groups still have bug")

    print("=" * 70)

    assert error < 1e-3, f"Reconstruction failed with collapse_onehot=True: {error:.6e}"


if __name__ == "__main__":
    test_onehot_with_collapse()
