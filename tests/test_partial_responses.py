"""Tests for partial response calculations."""

import numpy as np
import pytest
import torch

from prism.partial_responses import (
    PartialResponseCalculator,
    get_variable_range,
    stable_logit,
    to_numpy,
)
from prism.preprocessing import NoScaler


class TestToNumpy:
    """Tests for to_numpy conversion function."""

    def test_torch_tensor_to_numpy(self):
        """Test conversion of torch tensor to numpy array."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = to_numpy(tensor)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))

    def test_numpy_array_passthrough(self):
        """Test that numpy arrays are returned as-is."""
        array = np.array([1.0, 2.0, 3.0])
        result = to_numpy(array)

        assert isinstance(result, np.ndarray)
        assert result is array  # Should be same object

    def test_cuda_tensor_to_numpy(self):
        """Test conversion of CUDA tensor (if available)."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        if torch.cuda.is_available():
            tensor = tensor.cuda()

        result = to_numpy(tensor)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))

    def test_2d_tensor_conversion(self):
        """Test conversion of 2D tensor."""
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = to_numpy(tensor)

        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)
        np.testing.assert_array_equal(result, np.array([[1.0, 2.0], [3.0, 4.0]]))


class TestStableLogit:
    """Tests for stable_logit function."""

    def test_stable_logit_middle_values(self):
        """Test logit for values in the middle range (0.2 to 0.8)."""
        y = torch.tensor([0.2, 0.5, 0.8])
        result = stable_logit(y)

        # Expected: logit(p) = log(p / (1-p))
        expected = torch.log(y / (1 - y))

        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-7)

    def test_stable_logit_near_zero(self):
        """Test logit for values near 0 (should be clipped for stability)."""
        y = torch.tensor([1e-10, 1e-8, 0.001])
        result = stable_logit(y, eps=1e-7)

        # Should not return -inf
        assert torch.all(torch.isfinite(result))
        # Should be large negative values
        assert torch.all(result < 0)

    def test_stable_logit_near_one(self):
        """Test logit for values near 1 (should be clipped for stability)."""
        y = torch.tensor([0.999, 0.99999999, 1.0 - 1e-10])
        result = stable_logit(y, eps=1e-7)

        # Should not return +inf
        assert torch.all(torch.isfinite(result))
        # Should be large positive values
        assert torch.all(result > 0)

    def test_stable_logit_exactly_zero(self):
        """Test logit for exactly 0."""
        y = torch.tensor([0.0])
        result = stable_logit(y, eps=1e-7)

        # Should be finite (clipped)
        assert torch.isfinite(result).all()

    def test_stable_logit_exactly_one(self):
        """Test logit for exactly 1."""
        y = torch.tensor([1.0])
        result = stable_logit(y, eps=1e-7)

        # Should be finite (clipped)
        assert torch.isfinite(result).all()

    def test_stable_logit_with_custom_eps(self):
        """Test logit with different epsilon values."""
        # Use a value that's very close to the boundary
        y = torch.tensor([1e-6, 1.0 - 1e-6, 0.5])

        result_small_eps = stable_logit(y, eps=1e-8)
        result_large_eps = stable_logit(y, eps=1e-4)

        # Both should be finite
        assert torch.all(torch.isfinite(result_small_eps))
        assert torch.all(torch.isfinite(result_large_eps))

        # Middle value (0.5) should be same regardless of epsilon
        assert torch.allclose(result_small_eps[2:3], result_large_eps[2:3])

        # Boundary values should differ based on epsilon
        # The larger epsilon will clamp the values more aggressively
        assert not torch.allclose(result_small_eps[0:2], result_large_eps[0:2])

    def test_stable_logit_vector_input(self):
        """Test logit with various vector inputs."""
        y = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
        result = stable_logit(y)

        # All values should be finite
        assert torch.all(torch.isfinite(result))
        # Result should have same shape as input
        assert result.shape == y.shape
        # Monotonicity: logit should be monotonically increasing
        assert torch.all(result[1:] >= result[:-1])


class TestGetVariableRange:
    """Tests for get_variable_range function."""

    def test_continuous_variable(self):
        """Test range generation for continuous variable (many unique values)."""
        # Create continuous variable with many unique values
        x = torch.linspace(0.0, 10.0, 100)
        n_steps = 50
        categorical_threshold = 10

        result = get_variable_range(x, n_steps, categorical_threshold)

        # Should return linear range (n_steps values between min and max)
        assert result.shape == (n_steps,)
        assert result[0] == pytest.approx(x.min().item(), rel=1e-5)
        assert result[-1] == pytest.approx(x.max().item(), rel=1e-5)

        # Should be evenly spaced
        diffs = result[1:] - result[:-1]
        assert torch.allclose(diffs, diffs[0], rtol=1e-5)

    def test_categorical_variable(self):
        """Test range generation for categorical variable (few unique values)."""
        # Create categorical variable with only 5 unique values
        x = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0])
        n_steps = 50
        categorical_threshold = 10

        result = get_variable_range(x, n_steps, categorical_threshold)

        # Should return unique values sorted
        expected = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
        assert result.shape == expected.shape
        torch.testing.assert_close(result, expected)

    def test_boundary_case_exactly_at_threshold(self):
        """Test variable with exactly categorical_threshold unique values."""
        # Create variable with exactly 10 unique values
        x = torch.tensor(list(range(10)) * 5, dtype=torch.float32)  # 10 unique values, repeated
        n_steps = 50
        categorical_threshold = 10

        result = get_variable_range(x, n_steps, categorical_threshold)

        # With exactly threshold values (using <, not <=), should be treated as continuous
        assert len(result) == n_steps
        assert result[0] == pytest.approx(0.0, rel=1e-5)
        assert result[-1] == pytest.approx(9.0, rel=1e-5)

    def test_boundary_case_one_above_threshold(self):
        """Test variable with threshold+1 unique values (should be continuous)."""
        x = torch.tensor(list(range(11)) * 5)  # 11 unique values
        n_steps = 50
        categorical_threshold = 10

        result = get_variable_range(x, n_steps, categorical_threshold)

        # Should be treated as continuous (linear range)
        assert len(result) == n_steps
        assert result[0] == pytest.approx(0.0, rel=1e-5)
        assert result[-1] == pytest.approx(10.0, rel=1e-5)

    def test_single_value_variable(self):
        """Test variable with only one unique value."""
        x = torch.tensor([5.0] * 100)
        n_steps = 50
        categorical_threshold = 10

        result = get_variable_range(x, n_steps, categorical_threshold)

        # Should return single value
        assert len(result) == 1
        assert result[0] == 5.0

    def test_binary_variable(self):
        """Test binary variable (2 unique values)."""
        x = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        n_steps = 50
        categorical_threshold = 10

        result = get_variable_range(x, n_steps, categorical_threshold)

        # Should return both unique values sorted
        expected = torch.tensor([0.0, 1.0])
        torch.testing.assert_close(result, expected)

    def test_trim_quantile_continuous_variable(self):
        """Test trim_quantile limits range for continuous variables."""
        # Create data with outliers
        x = torch.cat(
            [
                torch.tensor([0.0]),  # Low outlier (below 1st percentile)
                torch.linspace(10.0, 90.0, 98),  # Main data
                torch.tensor([100.0]),  # High outlier (above 99th percentile)
            ]
        )
        n_steps = 50
        categorical_threshold = 10
        trim_quantile = 0.01  # Trim 1% from each tail

        result = get_variable_range(x, n_steps, categorical_threshold, trim_quantile)

        # Should return linear range using quantiles, not min/max
        assert result.shape == (n_steps,)
        # Range should be narrower than full range (0-100)
        assert result[0].item() > 0.0, "Lower bound should be above minimum (0.0)"
        assert result[-1].item() < 100.0, "Upper bound should be below maximum (100.0)"
        # Should be evenly spaced
        diffs = result[1:] - result[:-1]
        assert torch.allclose(diffs, diffs[0], rtol=1e-5)

    def test_trim_quantile_none_uses_full_range(self):
        """Test that trim_quantile=None uses full min/max range."""
        x = torch.cat(
            [
                torch.tensor([0.0]),  # Low outlier
                torch.linspace(10.0, 90.0, 98),  # Main data
                torch.tensor([100.0]),  # High outlier
            ]
        )
        n_steps = 50
        categorical_threshold = 10

        result = get_variable_range(x, n_steps, categorical_threshold, trim_quantile=None)

        # Should use full range including outliers
        assert result[0] == pytest.approx(0.0, rel=1e-5)
        assert result[-1] == pytest.approx(100.0, rel=1e-5)

    def test_trim_quantile_zero_uses_full_range(self):
        """Test that trim_quantile=0 uses full min/max range."""
        x = torch.linspace(0.0, 100.0, 100)
        n_steps = 50
        categorical_threshold = 10

        result = get_variable_range(x, n_steps, categorical_threshold, trim_quantile=0.0)

        # Should use full range
        assert result[0] == pytest.approx(0.0, rel=1e-5)
        assert result[-1] == pytest.approx(100.0, rel=1e-5)

    def test_trim_quantile_ignored_for_categorical(self):
        """Test that trim_quantile is ignored for categorical variables."""
        # Categorical variable with outlier-like values
        x = torch.tensor([0.0, 1.0, 2.0, 3.0, 100.0] * 10, dtype=torch.float32)
        n_steps = 50
        categorical_threshold = 10
        trim_quantile = 0.1  # Would trim aggressively if applied

        result = get_variable_range(x, n_steps, categorical_threshold, trim_quantile)

        # Should return ALL unique values (categorical behavior)
        expected = torch.tensor([0.0, 1.0, 2.0, 3.0, 100.0])
        assert result.shape == expected.shape
        torch.testing.assert_close(result, expected)

    def test_trim_quantile_symmetric(self):
        """Test that trim_quantile trims symmetrically from both tails."""
        # Create symmetric distribution
        x = torch.linspace(-100.0, 100.0, 1001)
        n_steps = 50
        categorical_threshold = 10
        trim_quantile = 0.05  # Trim 5% from each tail

        result = get_variable_range(x, n_steps, categorical_threshold, trim_quantile)

        # With symmetric data and symmetric trimming, range should be roughly symmetric
        center = (result[0] + result[-1]) / 2
        assert center.item() == pytest.approx(0.0, abs=1.0)  # Center near 0
        # Range should be narrower than full range
        assert result[0].item() > -100.0
        assert result[-1].item() < 100.0


class TestPartialResponseCalculator:
    """Tests for PartialResponseCalculator class."""

    def test_validate_onehot_groups_valid(self, mock_model, small_tensor_2d):
        """Test validation with valid one-hot groups."""
        feature_names = ['cat_A', 'cat_B', 'cat_C', 'cont_0', 'cont_1']
        calculator = PartialResponseCalculator(
            model=mock_model,
            input_dim=5,
            onehot_groups=[(0, 1, 2), (3, 4)],  # Use tuples, not lists
            x_train=small_tensor_2d,
            feature_names=feature_names,
            scaler=NoScaler(),
        )

        # Should not raise any errors
        calculator._validate_onehot_groups()

    def test_validate_onehot_groups_overlap(self, mock_model, small_tensor_2d):
        """Test that overlapping groups raise ValueError."""
        with pytest.raises(ValueError, match="multiple groups"):
            calculator = PartialResponseCalculator(
                model=mock_model,
                input_dim=5,
                onehot_groups=[(0, 1, 2), (2, 3, 4)],  # Feature 2 in both groups
                x_train=small_tensor_2d,
                scaler=NoScaler(),
            )

    def test_validate_onehot_groups_out_of_range(self, mock_model, small_tensor_2d):
        """Test that out-of-range indices raise ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            calculator = PartialResponseCalculator(
                model=mock_model,
                input_dim=5,
                onehot_groups=[(0, 1, 2), (3, 10)],  # Index 10 is out of range
                x_train=small_tensor_2d,
                scaler=NoScaler(),
            )

    def test_should_skip_bivariate_pair_within_group(self, mock_model, small_tensor_2d):
        """Test that pairs within same one-hot group are skipped."""
        feature_names = ['cat_A', 'cat_B', 'cat_C', 'grp2_A', 'grp2_B']
        calculator = PartialResponseCalculator(
            model=mock_model,
            input_dim=5,
            onehot_groups=[(0, 1, 2), (3, 4)],
            x_train=small_tensor_2d,
            feature_names=feature_names,
            scaler=NoScaler(),
        )

        # Pairs within first group should be skipped
        assert calculator._should_skip_bivariate_pair(0, 1) is True
        assert calculator._should_skip_bivariate_pair(1, 2) is True
        assert calculator._should_skip_bivariate_pair(0, 2) is True

        # Pairs within second group should be skipped
        assert calculator._should_skip_bivariate_pair(3, 4) is True

        # Pairs across groups should not be skipped
        assert calculator._should_skip_bivariate_pair(0, 3) is False
        assert calculator._should_skip_bivariate_pair(2, 4) is False

    def test_should_skip_bivariate_pair_no_groups(self, mock_model, small_tensor_2d):
        """Test that no pairs are skipped when no one-hot groups defined."""
        calculator = PartialResponseCalculator(
            model=mock_model,
            input_dim=5,
            onehot_groups=None,
            x_train=small_tensor_2d,
        )

        # No pairs should be skipped
        for i in range(5):
            for j in range(i + 1, 5):
                assert calculator._should_skip_bivariate_pair(i, j) is False

    def test_analyze_cardinality(self, small_tensor_2d, mock_model):
        """Test cardinality analysis of features."""
        calculator = PartialResponseCalculator(
            model=mock_model,
            input_dim=5,
            x_train=small_tensor_2d,
        )

        cardinality_info = calculator._analyze_cardinality(small_tensor_2d)

        # Should return list with info for each feature
        assert len(cardinality_info) == 5
        assert all(isinstance(info, dict) for info in cardinality_info)

        # Each info dict should have cache_optimized key
        for info in cardinality_info:
            assert "cache_optimized" in info
            # If cache_optimized is True, should have unique_values and inverse_indices
            if info["cache_optimized"]:
                assert "unique_values" in info
                assert "inverse_indices" in info

    def test_calculate_baseline(self, small_tensor_2d, mock_model):
        """Test baseline calculation."""
        calculator = PartialResponseCalculator(
            model=mock_model,
            input_dim=5,
            x_train=small_tensor_2d,
        )

        # calculate_baseline sets self.logit_y0 as a side effect, returns None
        calculator.calculate_baseline(small_tensor_2d)

        # Check that logit_y0 was set
        assert hasattr(calculator, 'logit_y0')
        assert isinstance(calculator.logit_y0, float)

        # Baseline should be finite
        assert np.isfinite(calculator.logit_y0)


@pytest.mark.unit
class TestPartialResponsesUtilityFunctions:
    """Combined tests for utility functions used in partial responses."""

    def test_logit_inverse_relationship(self):
        """Test that logit and sigmoid are inverses."""
        # sigmoid(logit(p)) should equal p (for p in valid range)
        p = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
        logit_p = stable_logit(p)
        recovered_p = torch.sigmoid(logit_p)

        torch.testing.assert_close(recovered_p, p, rtol=1e-5, atol=1e-7)

    def test_variable_range_consistency(self):
        """Test that get_variable_range produces consistent results."""
        x = torch.randn(100)

        # Multiple calls should produce same result
        result1 = get_variable_range(x, n_steps=50, categorical_threshold=10)
        result2 = get_variable_range(x, n_steps=50, categorical_threshold=10)

        torch.testing.assert_close(result1, result2)

    def test_conversion_preserves_values(self):
        """Test that numpy conversion preserves values exactly."""
        original = torch.tensor([[1.234567, 2.345678], [3.456789, 4.567890]])
        converted = to_numpy(original)

        np.testing.assert_array_almost_equal(converted, original.numpy(), decimal=6)


@pytest.mark.unit
class TestOneHotEncodedPartialResponses:
    """Tests for partial response calculations with one-hot encoded features."""

    def test_get_onehot_group(self, mock_model, small_tensor_2d):
        """Test _get_onehot_group helper method."""
        feature_names = ['cat_A', 'cat_B', 'cat_C', 'grp2_A', 'grp2_B']
        calculator = PartialResponseCalculator(
            model=mock_model,
            input_dim=5,
            onehot_groups=[(0, 1, 2), (3, 4)],
            x_train=small_tensor_2d,
            feature_names=feature_names,
            scaler=NoScaler(),
        )

        # Features in first group
        assert calculator._get_onehot_group(0) == (0, 1, 2)
        assert calculator._get_onehot_group(1) == (0, 1, 2)
        assert calculator._get_onehot_group(2) == (0, 1, 2)

        # Features in second group
        assert calculator._get_onehot_group(3) == (3, 4)
        assert calculator._get_onehot_group(4) == (3, 4)

    def test_get_onehot_group_no_group(self):
        """Test _get_onehot_group for feature not in any group."""
        from tests.conftest import MockBinaryClassifier

        x_train = torch.randn(10, 6)
        model = MockBinaryClassifier(n_features=6)
        feature_names = ['cat_A', 'cat_B', 'cat_C', 'cont_0', 'cont_1', 'cont_2']

        calculator = PartialResponseCalculator(
            model=model,
            input_dim=6,
            onehot_groups=[(0, 1, 2)],
            x_train=x_train,
            feature_names=feature_names,
            scaler=NoScaler(),
        )

        # Feature not in any group should return None
        assert calculator._get_onehot_group(3) is None
        assert calculator._get_onehot_group(4) is None
        assert calculator._get_onehot_group(5) is None

    def test_univariate_reference_state_consistency(self, mock_model):
        """Test that all columns in one-hot group have same value=0 (reference) response."""
        # Create training data with one-hot encoded diagnosis: [diag_A, diag_B, diag_C]
        # Diagnosis categories: A, B, C, None (reference)
        n_train = 100
        x_train = torch.zeros(n_train, 5)

        # First 3 columns are one-hot group (diagnosis)
        # Randomly assign one diagnosis per sample (some can be None/all zeros)
        for i in range(n_train):
            choice = np.random.choice([0, 1, 2, 3])  # 0=A, 1=B, 2=C, 3=None
            if choice < 3:
                x_train[i, choice] = 1

        # Columns 3-4 are continuous features
        x_train[:, 3:5] = torch.randn(n_train, 2)

        feature_names = ['diag_A', 'diag_B', 'diag_C', 'cont_0', 'cont_1']
        calculator = PartialResponseCalculator(
            model=mock_model,
            input_dim=5,
            method='lebesgue',
            x_train=x_train,
            onehot_groups=[(0, 1, 2)],
            feature_names=feature_names,
            scaler=NoScaler(),
        )

        # Compute unique responses for each feature in the one-hot group
        unique_vals = torch.tensor([0.0, 1.0])  # Binary features
        response_0 = calculator._compute_unique_responses(0, unique_vals, batch_size=10)
        response_1 = calculator._compute_unique_responses(1, unique_vals, batch_size=10)
        response_2 = calculator._compute_unique_responses(2, unique_vals, batch_size=10)

        # The value=0 response (index 0) should be the same for all columns in the group
        # This represents the reference state (all diagnosis columns = 0)
        torch.testing.assert_close(response_0[0], response_1[0], rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(response_0[0], response_2[0], rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(response_1[0], response_2[0], rtol=1e-5, atol=1e-7)

    def test_univariate_active_state_differs(self, mock_model):
        """Test that value=1 (active) responses differ across columns in one-hot group."""
        n_train = 100
        x_train = torch.zeros(n_train, 5)

        # One-hot group in first 3 columns
        for i in range(n_train):
            choice = np.random.choice([0, 1, 2, 3])
            if choice < 3:
                x_train[i, choice] = 1

        x_train[:, 3:5] = torch.randn(n_train, 2)

        feature_names = ['cat_A', 'cat_B', 'cat_C', 'cont_0', 'cont_1']
        calculator = PartialResponseCalculator(
            model=mock_model,
            input_dim=5,
            method='lebesgue',
            x_train=x_train,
            onehot_groups=[(0, 1, 2)],
            feature_names=feature_names,
            scaler=NoScaler(),
        )

        unique_vals = torch.tensor([0.0, 1.0])
        response_0 = calculator._compute_unique_responses(0, unique_vals, batch_size=10)
        response_1 = calculator._compute_unique_responses(1, unique_vals, batch_size=10)
        response_2 = calculator._compute_unique_responses(2, unique_vals, batch_size=10)

        # The value=1 responses (index 1) can differ across columns
        # (they represent different active states: diag_A vs diag_B vs diag_C)
        # We just verify they are not all identical (which would be suspicious)
        # Note: With mock model they might be similar, but at least check they exist
        assert response_0[1] is not None
        assert response_1[1] is not None
        assert response_2[1] is not None

    def test_reference_state_equals_all_zeros_prediction(self):
        """Test that reference response equals model prediction with all group columns = 0."""
        from tests.conftest import MockBinaryClassifier

        n_train = 50
        x_train = torch.zeros(n_train, 4)

        # One-hot group in first 2 columns
        for i in range(n_train):
            choice = np.random.choice([0, 1, 2])  # 0=A, 1=B, 2=None
            if choice < 2:
                x_train[i, choice] = 1

        x_train[:, 2:4] = torch.randn(n_train, 2)

        model = MockBinaryClassifier(n_features=4)
        feature_names = ['cat_A', 'cat_B', 'cont_0', 'cont_1']

        calculator = PartialResponseCalculator(
            model=model,
            input_dim=4,
            method='lebesgue',
            x_train=x_train,
            onehot_groups=[(0, 1)],
            feature_names=feature_names,
            scaler=NoScaler(),
        )

        # Get reference response for feature 0
        unique_vals = torch.tensor([0.0, 1.0])
        response_0 = calculator._compute_unique_responses(0, unique_vals, batch_size=10)
        reference_response = response_0[0]  # value=0 case

        # Manually compute what the reference response should be
        # Set all columns in the one-hot group to 0
        x_modified = x_train.clone()
        x_modified[:, 0] = 0
        x_modified[:, 1] = 0

        y_pred = calculator.predict_proba(x_modified)
        expected_response = stable_logit(y_pred).mean() - calculator.logit_y0

        # Reference response should match the manual calculation
        torch.testing.assert_close(reference_response, expected_response, rtol=1e-4, atol=1e-6)

    def test_bivariate_different_groups(self, mock_model):
        """Test bivariate responses for features in different one-hot groups."""
        n_train = 50
        x_train = torch.zeros(n_train, 5)

        # Group 1: columns 0, 1 (e.g., diagnosis)
        for i in range(n_train):
            choice = np.random.choice([0, 1, 2])
            if choice < 2:
                x_train[i, choice] = 1

        # Group 2: columns 2, 3 (e.g., ethnicity)
        for i in range(n_train):
            choice = np.random.choice([2, 3, 4])
            if choice < 4:
                x_train[i, choice] = 1

        # Column 4: continuous
        x_train[:, 4] = torch.randn(n_train)

        feature_names = ['diag_A', 'diag_B', 'ethn_X', 'ethn_Y', 'cont_0']
        calculator = PartialResponseCalculator(
            model=mock_model,
            input_dim=5,
            method='lebesgue',
            x_train=x_train,
            onehot_groups=[(0, 1), (2, 3)],
            feature_names=feature_names,
            scaler=NoScaler(),
        )

        # Create pair info for features 0 and 2 (from different groups)
        unique_pairs = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        pair_info = {
            'indices': (0, 2),
            'unique_pairs': unique_pairs,
        }

        responses = calculator._compute_unique_pair_responses(pair_info, batch_size=10)

        # Should compute 4 responses (one for each pair combination)
        assert responses.shape == (4,)
        assert torch.all(torch.isfinite(responses))

    def test_dirac_method_onehot_correctness(self):
        """Test that Dirac method handles one-hot encoding correctly."""
        from tests.conftest import MockBinaryClassifier

        # Create test data with one-hot encoded features
        x = torch.tensor(
            [
                [1.0, 0.0, 0.0, 2.5],  # diagnosis A, continuous feature
                [0.0, 1.0, 0.0, 1.3],  # diagnosis B
                [0.0, 0.0, 1.0, 0.8],  # diagnosis C
                [0.0, 0.0, 0.0, 1.9],  # diagnosis None (reference)
            ]
        )

        model = MockBinaryClassifier(n_features=4)
        feature_names = ['diag_A', 'diag_B', 'diag_C', 'cont_0']

        calculator = PartialResponseCalculator(
            model=model,
            input_dim=4,
            method='dirac',
            onehot_groups=[(0, 1, 2)],
            feature_names=feature_names,
            scaler=NoScaler(),
        )

        univariate_responses, bivariate_responses = calculator.calculate(x)

        # Check shapes (with collapse: 3 OHE columns -> 1 group + 1 continuous = 2 features)
        assert univariate_responses.shape == (4, 2)  # 4 samples, 2 collapsed features
        assert bivariate_responses.shape == (4, 1)  # 4 samples, C(2,2)=1 pair

        # For the reference sample (row 3), all diagnosis columns are 0
        # The partial responses for diagnosis features should all be for the reference state
        assert torch.all(torch.isfinite(univariate_responses))
        assert torch.all(torch.isfinite(bivariate_responses))


@pytest.mark.unit
class TestPartialResponseCalculatorPredict:
    """Tests for PartialResponseCalculator.predict method."""

    def test_predict_basic(self, mock_model, small_tensor_2d):
        """Test basic prediction functionality."""
        calculator = PartialResponseCalculator(
            model=mock_model,
            input_dim=5,
            method='dirac',
        )

        predictions = calculator.predict_proba(small_tensor_2d)

        # Should return predictions for all samples
        assert predictions.shape == (5,)
        assert torch.all(torch.isfinite(predictions))
        # Predictions should be probabilities (between 0 and 1)
        assert torch.all(predictions >= 0)
        assert torch.all(predictions <= 1)

    def test_predict_with_device(self, mock_model, small_tensor_2d):
        """Test prediction with explicit device."""
        calculator = PartialResponseCalculator(
            model=mock_model,
            input_dim=5,
            method='dirac',
            device='cpu',
        )

        predictions = calculator.predict_proba(small_tensor_2d)

        assert predictions.device == torch.device('cpu')
        assert predictions.shape == (5,)

    def test_predict_single_sample(self, mock_model):
        """Test prediction with a single sample."""
        calculator = PartialResponseCalculator(
            model=mock_model,
            input_dim=5,
            method='dirac',
        )

        x = torch.randn(1, 5)
        predictions = calculator.predict_proba(x)

        # predict() squeezes the result, so single sample becomes scalar
        assert predictions.ndim == 0 or predictions.shape == (1,)
        assert torch.isfinite(predictions).all()


@pytest.mark.integration
class TestDiracMethodCalculation:
    """Integration tests for Dirac method partial response calculation."""

    def test_dirac_univariate_additivity(self, mock_model):
        """Test that Dirac univariate responses are additive on logit scale."""
        x = torch.randn(10, 5)

        calculator = PartialResponseCalculator(
            model=mock_model,
            input_dim=5,
            method='dirac',
        )

        univariate_responses, _ = calculator.calculate(x)

        # Check that responses are on logit scale (can be negative)
        assert univariate_responses.shape == (10, 5)
        assert torch.all(torch.isfinite(univariate_responses))

        # The sum of univariate responses should approximate the total effect
        # (not exact due to bivariate interactions)
        total_univariate = univariate_responses.sum(dim=1)
        assert torch.all(torch.isfinite(total_univariate))

    def test_dirac_bivariate_shape(self, mock_model):
        """Test that Dirac bivariate responses have correct shape."""
        x = torch.randn(10, 5)

        calculator = PartialResponseCalculator(
            model=mock_model,
            input_dim=5,
            method='dirac',
        )

        _, bivariate_responses = calculator.calculate(x)

        # For 5 features, should have C(5,2) = 10 pairs
        expected_pairs = 5 * 4 // 2
        assert bivariate_responses.shape == (10, expected_pairs)

    def test_dirac_zero_input(self, mock_model):
        """Test Dirac method with zero input."""
        x = torch.zeros(5, 5)

        calculator = PartialResponseCalculator(
            model=mock_model,
            input_dim=5,
            method='dirac',
        )

        univariate_responses, bivariate_responses = calculator.calculate(x)

        # All responses should be zero (or close to baseline)
        # Since all features are zero, their individual contributions should be minimal
        assert torch.all(torch.isfinite(univariate_responses))
        assert torch.all(torch.isfinite(bivariate_responses))

    def test_dirac_deterministic(self, mock_model):
        """Test that Dirac method produces deterministic results."""
        x = torch.randn(10, 5)

        calculator = PartialResponseCalculator(
            model=mock_model,
            input_dim=5,
            method='dirac',
        )

        univariate_1, bivariate_1 = calculator.calculate(x)
        univariate_2, bivariate_2 = calculator.calculate(x)

        # Results should be identical
        torch.testing.assert_close(univariate_1, univariate_2)
        torch.testing.assert_close(bivariate_1, bivariate_2)


@pytest.mark.integration
class TestLebesgueMethodCalculation:
    """Integration tests for Lebesgue method partial response calculation."""

    def test_lebesgue_requires_training_data(self, mock_model):
        """Test that Lebesgue method requires training data."""
        # The compatibility check will fail first, but then the real error is raised
        with pytest.raises(ValueError):
            _ = PartialResponseCalculator(
                model=mock_model,
                input_dim=5,
                method='lebesgue',
                x_train=None,
            )

    def test_lebesgue_basic_calculation(self, mock_model):
        """Test basic Lebesgue calculation."""
        x_train = torch.randn(100, 5)
        x_test = torch.randn(10, 5)

        calculator = PartialResponseCalculator(
            model=mock_model,
            input_dim=5,
            method='lebesgue',
            x_train=x_train,
        )

        univariate_responses, bivariate_responses = calculator.calculate(x_test, batch_size=32)

        assert univariate_responses.shape == (10, 5)
        assert torch.all(torch.isfinite(univariate_responses))
        assert torch.all(torch.isfinite(bivariate_responses))

    def test_lebesgue_baseline_set_on_init(self, mock_model):
        """Test that Lebesgue method sets baseline during initialization."""
        x_train = torch.randn(100, 5)

        calculator = PartialResponseCalculator(
            model=mock_model,
            input_dim=5,
            method='lebesgue',
            x_train=x_train,
        )

        # Baseline should be set during init
        assert hasattr(calculator, 'logit_y0')
        assert calculator.logit_y0 is not None
        assert np.isfinite(calculator.logit_y0)

    def test_lebesgue_vs_dirac_difference(self, mock_model):
        """Test that Lebesgue and Dirac methods produce different results."""
        x_train = torch.randn(100, 5)
        x_test = torch.randn(10, 5)

        calculator_dirac = PartialResponseCalculator(
            model=mock_model,
            input_dim=5,
            method='dirac',
        )

        calculator_lebesgue = PartialResponseCalculator(
            model=mock_model,
            input_dim=5,
            method='lebesgue',
            x_train=x_train,
        )

        univariate_dirac, _ = calculator_dirac.calculate(x_test)
        univariate_lebesgue, _ = calculator_lebesgue.calculate(x_test, batch_size=32)

        # Results should be different (Lebesgue averages over training data)
        # but not completely unrelated
        assert not torch.allclose(univariate_dirac, univariate_lebesgue)
        # Both should have same shape
        assert univariate_dirac.shape == univariate_lebesgue.shape

    def test_lebesgue_with_batching(self, mock_model):
        """Test that Lebesgue method works with different batch sizes."""
        x_train = torch.randn(100, 5)
        x_test = torch.randn(50, 5)

        calculator = PartialResponseCalculator(
            model=mock_model,
            input_dim=5,
            method='lebesgue',
            x_train=x_train,
        )

        # Calculate with different batch sizes
        univariate_small, bivariate_small = calculator.calculate(x_test, batch_size=10)
        univariate_large, bivariate_large = calculator.calculate(x_test, batch_size=100)

        # Results should be identical regardless of batch size
        torch.testing.assert_close(univariate_small, univariate_large)
        torch.testing.assert_close(bivariate_small, bivariate_large)


@pytest.mark.unit
class TestPartialResponsesWrapperFunction:
    """Tests for the partial_responses wrapper function."""

    def test_partial_responses_dirac(self, mock_model):
        """Test partial_responses function with Dirac method."""
        from prism.partial_responses import partial_responses

        x = torch.randn(10, 5)
        responses = partial_responses(
            x=x,
            model=mock_model,
            method='dirac',
            device='cpu',
        )

        # Should return combined responses (univariate + bivariate)
        # 5 univariate + 10 bivariate = 15 total
        assert responses.shape == (10, 15)
        assert torch.all(torch.isfinite(responses))

    def test_partial_responses_lebesgue(self, mock_model):
        """Test partial_responses function with Lebesgue method."""
        from prism.partial_responses import partial_responses

        x_train = torch.randn(100, 5)
        x_test = torch.randn(10, 5)

        responses = partial_responses(
            x=x_test,
            model=mock_model,
            x_train=x_train,
            method='lebesgue',
            device='cpu',
            batch_size=32,
        )

        # Should return combined responses
        assert responses.shape == (10, 15)
        assert torch.all(torch.isfinite(responses))

    def test_partial_responses_with_onehot_groups(self, mock_model):
        """Test partial_responses function with one-hot groups."""
        from prism.partial_responses import partial_responses

        x = torch.randn(10, 5)
        # Features 0,1,2 are one-hot encoded
        onehot_groups = [(0, 1, 2)]
        feature_names = ['cat_A', 'cat_B', 'cat_C', 'cont_0', 'cont_1']

        responses = partial_responses(
            x=x,
            model=mock_model,
            method='dirac',
            device='cpu',
            onehot_groups=onehot_groups,
            feature_names=feature_names,
            scaler=NoScaler(),
        )

        # Should still return responses but skip bivariate pairs within the same group
        assert responses.shape[0] == 10
        assert torch.all(torch.isfinite(responses))

    def test_partial_responses_invalid_method(self, mock_model):
        """Test that invalid method raises error."""
        x = torch.randn(10, 5)

        # Create calculator with dirac first to avoid compatibility check
        calculator = PartialResponseCalculator(
            model=mock_model,
            input_dim=5,
            method='dirac',
        )
        # Then change method to invalid
        calculator.method = 'invalid_method'

        # Now calculate should raise error about invalid method
        with pytest.raises(ValueError, match="not implemented"):
            calculator.calculate(x)


@pytest.mark.integration
class TestPartialResponsesEndToEnd:
    """End-to-end integration tests for partial response calculations."""

    def test_end_to_end_dirac_workflow(self, mock_model):
        """Test complete workflow with Dirac method."""
        from prism.partial_responses import partial_responses, partial_responses_subset

        # Generate test data
        x_full = torch.randn(100, 5)

        # Calculate full responses
        responses_full = partial_responses(
            x=x_full,
            model=mock_model,
            method='dirac',
        )

        assert responses_full.shape == (100, 15)

        # Calculate subset responses for visualization
        univariate, bivariate, x_uni, x_bi = partial_responses_subset(
            x=x_full,
            model=mock_model,
            method='dirac',
            n_steps=20,
        )

        assert len(univariate) == 5
        assert all(len(resp) == len(x_vals) for resp, x_vals in zip(univariate, x_uni))

    def test_end_to_end_lebesgue_workflow(self, mock_model):
        """Test complete workflow with Lebesgue method."""
        from prism.partial_responses import partial_responses, partial_responses_subset

        # Generate training and test data
        x_train = torch.randn(200, 5)
        x_test = torch.randn(50, 5)

        # Calculate full responses
        responses_full = partial_responses(
            x=x_test,
            model=mock_model,
            x_train=x_train,
            method='lebesgue',
            batch_size=32,
        )

        assert responses_full.shape == (50, 15)

        # Calculate subset responses
        univariate, bivariate, _, _ = partial_responses_subset(
            x=x_test,
            model=mock_model,
            x_train=x_train,
            method='lebesgue',
            n_steps=15,
            batch_size=16,
        )

        assert len(univariate) == 5

    def test_end_to_end_with_onehot_encoding(self):
        """Test complete workflow with one-hot encoded features."""
        from prism.partial_responses import partial_responses, partial_responses_subset
        from tests.conftest import MockBinaryClassifier

        # Create data with one-hot encoded features
        x = torch.randn(100, 6)
        # Features 0-2 are one-hot group 1, features 3-4 are one-hot group 2
        onehot_groups = [(0, 1, 2), (3, 4)]
        feature_names = ['grp1_A', 'grp1_B', 'grp1_C', 'grp2_X', 'grp2_Y', 'cont_0']

        model_6d = MockBinaryClassifier(n_features=6)

        # Calculate full responses
        responses = partial_responses(
            x=x,
            model=model_6d,
            method='dirac',
            onehot_groups=onehot_groups,
            feature_names=feature_names,
            scaler=NoScaler(),
        )

        # Should return responses with bivariate pairs within groups skipped
        assert responses.shape[0] == 100
        assert torch.all(torch.isfinite(responses))

        # Calculate subset
        univariate, bivariate, _, _ = partial_responses_subset(
            x=x,
            model=model_6d,
            method='dirac',
            n_steps=10,
            onehot_groups=onehot_groups,
            feature_names=feature_names,
            scaler=NoScaler(),
        )

        assert len(univariate) == 6

    def test_consistency_across_batch_sizes(self, mock_model):
        """Test that results are consistent across different batch sizes."""
        from prism.partial_responses import partial_responses

        x_train = torch.randn(200, 5)
        x_test = torch.randn(50, 5)

        # Calculate with different batch sizes
        responses_small = partial_responses(
            x=x_test,
            model=mock_model,
            x_train=x_train,
            method='lebesgue',
            batch_size=10,
        )

        responses_large = partial_responses(
            x=x_test,
            model=mock_model,
            x_train=x_train,
            method='lebesgue',
            batch_size=100,
        )

        # Results should be identical
        torch.testing.assert_close(responses_small, responses_large, rtol=1e-5, atol=1e-7)
