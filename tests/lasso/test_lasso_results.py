"""Tests for LASSO results management and lambda selection."""

import inspect

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from prism.lasso.lasso_results import LassoResultsManager


class TestLassoResultsManager:
    """Tests for LassoResultsManager class."""

    @pytest.fixture
    def sample_lasso_results(self):
        """Create sample LASSO results for testing."""
        # Simulate LASSO sweep results
        n_lambda = 10
        n_features = 5

        lambdas = np.logspace(-3, 1, n_lambda)
        betas = []
        models = []
        train_aucs = []
        test_aucs = []
        train_losses = []
        test_losses = []

        # Simulate results where fewer features are selected as lambda increases
        for i, lam in enumerate(lambdas):
            # Coefficients decay with increasing lambda
            coef = np.random.randn(n_features) * np.exp(-lam)
            # Zero out some coefficients based on lambda
            threshold = 0.1 * lam
            coef[np.abs(coef) < threshold] = 0.0

            betas.append(coef)

            # Create mock model
            mock_model = LogisticRegression()
            models.append(mock_model)

            # Simulate decreasing performance with too much regularization
            train_aucs.append(0.9 - 0.1 * (i / n_lambda))
            test_aucs.append(0.85 - 0.15 * (i / n_lambda))
            train_losses.append(0.1 + 0.1 * (i / n_lambda))
            test_losses.append(0.15 + 0.15 * (i / n_lambda))

        feature_names = [f'feature_{i}' for i in range(n_features)]

        return {
            'lambdas': lambdas,
            'betas': np.array(betas),
            'models': models,
            'feature_names': feature_names,
            'train_aucs': np.array(train_aucs),
            'test_aucs': np.array(test_aucs),
            'train_losses': np.array(train_losses),
            'test_losses': np.array(test_losses),
        }

    def test_selection_workflow_yields_features_and_mask(self, sample_lasso_results):
        """Selecting a lambda exposes consistent feature names and a PRN mask.

        Covers the basic manager lifecycle: construct from sweep results,
        select the max-test-AUC lambda, then read the selected feature names
        (a subset of the sweep's feature vocabulary) and a PRN connectivity
        mask sized to the input features.
        """
        manager = LassoResultsManager(
            lambdas=sample_lasso_results['lambdas'],
            betas=sample_lasso_results['betas'],
            models=sample_lasso_results['models'],
            feature_names=sample_lasso_results['feature_names'],
            train_losses=sample_lasso_results['train_losses'],
            test_losses=sample_lasso_results['test_losses'],
            train_aucs=sample_lasso_results['train_aucs'],
            test_aucs=sample_lasso_results['test_aucs'],
        )
        assert len(manager.lambdas) == len(sample_lasso_results['lambdas'])

        manager.select_lambda_max_test_auc()
        assert manager.selected_lambda_index == int(np.argmax(manager.test_aucs))

        selected = manager.get_selected_feature_names()
        expected = [manager.all_feature_names[i] for i in manager.get_selected_feature_indicies()]
        assert selected == expected

        mask, _ = manager.get_mask(verbose=False)
        assert mask.shape[0] == len(sample_lasso_results['feature_names'])


class TestBetaNoiseFilter:
    """|beta| > 0.1 is a fixed noise filter applied at selection time.

    It is a fixed threshold (not a tuned hyperparameter) and uses strict greater-than,
    so a coefficient of exactly 0.1 is excluded. lambda is the real complexity control.
    """

    def _manager(self, beta_vector):
        """Build a manager with 3 univariate features (=> 3 bivariate pairs, 6 total).

        beta_vector must have length 6 = n_univ(3) + n_biv(C(3,2)=3).
        """
        n_lambda = 1
        betas = np.asarray(beta_vector, dtype=float).reshape(-1, n_lambda)
        mgr = LassoResultsManager(
            lambdas=np.array([1.0]),
            betas=betas,
            models=[LogisticRegression()],
            feature_names=["f0", "f1", "f2"],
            train_losses=np.array([0.1]),
            test_losses=np.array([0.15]),
            train_aucs=np.array([0.9]),
            test_aucs=np.array([0.85]),
        )
        mgr.selected_lambda_index = 0  # bypass console-heavy select_lambda()
        return mgr

    def test_boundary_inclusion_exclusion(self):
        # univariate: 0.09 (drop), 0.11 (keep), 0.10 (drop, strict >)
        # bivariate:  0.5 (keep), 0.0 (drop), 0.2 (keep)
        mgr = self._manager([0.09, 0.11, 0.10, 0.5, 0.0, 0.2])
        selected = set(int(i) for i in mgr.get_selected_feature_indicies())
        assert selected == {1, 3, 5}, f"|beta|>0.1 filter misapplied at the boundary: {selected}"

    def test_exactly_threshold_is_excluded(self):
        mgr = self._manager([0.10, 0.10, 0.10, 0.10, 0.10, 0.10])
        assert (
            list(mgr.get_selected_feature_indicies()) == []
        ), "coefficients of exactly 0.1 must be excluded (strict greater-than)"

    def test_threshold_default_is_point_one(self):
        for method_name in (
            "get_selected_feature_indicies",
            "get_selected_feature_names",
            "select_lambda",
            "get_mask",
        ):
            sig = inspect.signature(getattr(LassoResultsManager, method_name))
            assert (
                sig.parameters["threshold"].default == 0.1
            ), f"{method_name} default beta threshold is not the fixed 0.1"


class TestPRNScaffold:
    """The PRN mask encodes the selected-effect scaffold.

    Each univariate subnet may see only its single input; each bivariate subnet
    only its two inputs; and a selected one-hot member pulls in the whole group
    (variable-level selection). Higher-order coupling is excluded by construction.

    n_univ=3 -> bivariate_inputs = [(0,1), (0,2), (1,2)], beta indices 3, 4, 5.
    """

    def _manager(self, beta_vector):
        betas = np.asarray(beta_vector, dtype=float).reshape(-1, 1)
        mgr = LassoResultsManager(
            lambdas=np.array([1.0]),
            betas=betas,
            models=[LogisticRegression()],
            feature_names=["f0", "f1", "f2"],
            train_losses=np.array([0.1]),
            test_losses=np.array([0.15]),
            train_aucs=np.array([0.9]),
            test_aucs=np.array([0.85]),
        )
        mgr.selected_lambda_index = 0
        return mgr

    def test_univariate_subnet_touches_one_input(self):
        # select univariate f0 and f2 only
        mgr = self._manager([1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        mask, _ = mgr.get_mask(subnet_nodes=5, verbose=False)

        # mask: (n_univ=3 input rows, subnet_nodes * n_selected_effects)
        # the two univariate subnets occupy the first two 5-column blocks.
        for block, feat in enumerate([0, 2]):
            cols = slice(block * 5, (block + 1) * 5)
            active_rows = list(np.where(mask[:, cols].any(axis=1))[0])
            assert active_rows == [feat], (
                f"univariate subnet block {block} touches inputs {active_rows}, "
                f"expected only [{feat}]"
            )

    def test_bivariate_subnet_touches_exactly_two_inputs(self):
        # select only bivariate pair (0, 2) -> beta index 3 + 1 = 4
        mgr = self._manager([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        mask, _ = mgr.get_mask(subnet_nodes=5, verbose=False)

        active_rows = set(int(r) for r in np.where(mask.any(axis=1))[0])
        assert active_rows == {0, 2}, (
            f"bivariate subnet for pair (0, 2) connects to {active_rows}, "
            f"expected exactly {{0, 2}}"
        )

    def test_onehot_group_selected_as_a_unit(self):
        # only f1 clears the 0.1 filter, but (0, 1, 2) is one one-hot variable
        mgr = self._manager([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        mask, _ = mgr.get_mask(subnet_nodes=5, onehot_groups=[(0, 1, 2)], verbose=False)

        active_rows = set(int(r) for r in np.where(mask.any(axis=1))[0])
        assert {0, 1, 2}.issubset(
            active_rows
        ), f"one-hot group not expanded to the variable level: active rows {active_rows}"


@pytest.mark.integration
class TestLassoResultsIntegration:
    """Integration tests for LASSO results workflow."""

    def test_full_lambda_selection_workflow(self):
        """Test complete workflow: results -> selection -> feature extraction."""
        # Create synthetic results
        n_lambda = 15
        n_features = 10

        lambdas = np.logspace(-2, 1, n_lambda)
        betas = []
        models = []
        train_aucs = []
        test_aucs = []
        train_losses = []
        test_losses = []

        for i, lam in enumerate(lambdas):
            # Simulate coefficient shrinkage
            coef = np.random.randn(n_features) / (1 + lam)
            coef[np.abs(coef) < 0.1 * lam] = 0.0
            betas.append(coef)

            # Mock model
            models.append(LogisticRegression())

            # Simulate performance
            n_selected = np.sum(coef != 0)
            train_aucs.append(min(0.95, 0.7 + 0.05 * n_selected / n_features))
            test_aucs.append(min(0.90, 0.65 + 0.05 * n_selected / n_features))
            train_losses.append(0.1)
            test_losses.append(0.15)

        # Create manager
        feature_names = [f'f{i}' for i in range(n_features)]
        manager = LassoResultsManager(
            lambdas=lambdas,
            betas=np.array(betas).T,  # Transpose to shape (n_features, n_lambda)
            models=models,
            feature_names=feature_names,
            train_losses=np.array(train_losses),
            test_losses=np.array(test_losses),
            train_aucs=np.array(train_aucs),
            test_aucs=np.array(test_aucs),
        )

        # Select lambda
        manager.select_lambda_max_test_auc()

        # Get selected features
        selected_features = manager.get_selected_feature_names()

        # Should have selected some features
        assert len(selected_features) > 0
        assert len(selected_features) <= n_features

        # All selected features should be from original set
        assert all(f in feature_names for f in selected_features)


class TestSelectLambdaMinTestAuc:
    """Tests for select_lambda_min_test_auc method."""

    @pytest.fixture
    def lasso_results_with_ordered_aucs(self):
        """Create LASSO results with ordered test AUCs for predictable selection."""
        n_lambda = 10
        n_features = 5

        lambdas = np.logspace(-3, 1, n_lambda)

        # Test AUCs: increasing from 0.60 to 0.85 as lambda decreases
        # Index:      0     1     2     3     4     5     6     7     8     9
        # test_aucs: 0.60, 0.65, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.85
        test_aucs = np.array([0.60, 0.65, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.85])
        train_aucs = test_aucs + 0.05
        train_losses = np.linspace(0.4, 0.1, n_lambda)
        test_losses = np.linspace(0.45, 0.15, n_lambda)

        # Create betas array: shape (n_features, n_lambda)
        # Each column represents coefficients for one lambda value
        betas = np.zeros((n_features, n_lambda))
        for i in range(n_lambda):
            # Create coefficients with non-zero values that increase with index
            # to ensure we have features selected at various thresholds
            coef = np.array([0.5, 0.3, 0.2, 0.15, 0.12]) * (i + 1) / n_lambda
            betas[:, i] = coef

        models = [LogisticRegression() for _ in range(n_lambda)]
        feature_names = [f'feature_{i}' for i in range(n_features)]

        return LassoResultsManager(
            lambdas=lambdas,
            betas=betas,
            models=models,
            feature_names=feature_names,
            train_losses=train_losses,
            test_losses=test_losses,
            train_aucs=train_aucs,
            test_aucs=test_aucs,
        )

    def test_selects_first_lambda_exceeding_threshold(self, lasso_results_with_ordered_aucs):
        """Should select first lambda that exceeds min_auc."""
        manager = lasso_results_with_ordered_aucs

        # Request min_auc=0.71, first to exceed is index 3 (0.72)
        result = manager.select_lambda_min_test_auc(min_auc=0.71)

        assert result == 3
        assert manager.selected_lambda_index == 3
        assert manager.test_aucs[result] >= 0.71

    def test_selects_exact_match(self, lasso_results_with_ordered_aucs):
        """Should select lambda with exact AUC match."""
        manager = lasso_results_with_ordered_aucs

        # Request min_auc=0.70, first to match is index 2 (0.70)
        result = manager.select_lambda_min_test_auc(min_auc=0.70)

        assert result == 2
        assert manager.test_aucs[result] == 0.70

    def test_fallback_to_max_when_no_lambda_exceeds(self, lasso_results_with_ordered_aucs, caplog):
        """Should fallback to max_test_auc with warning if no lambda exceeds."""
        manager = lasso_results_with_ordered_aucs

        # Request higher than any available (max is 0.85)
        result = manager.select_lambda_min_test_auc(min_auc=0.90)

        # Should have issued a warning
        assert "Falling back to max test AUC" in caplog.text

        # Should return index of max AUC (index 9 with 0.85)
        assert result == np.argmax(manager.test_aucs)
        assert result == 9

    def test_invalid_min_auc_zero_raises_error(self, lasso_results_with_ordered_aucs):
        """Should raise ValueError for min_auc=0."""
        manager = lasso_results_with_ordered_aucs

        with pytest.raises(ValueError, match="must be between 0 and 1"):
            manager.select_lambda_min_test_auc(min_auc=0)

    def test_invalid_min_auc_negative_raises_error(self, lasso_results_with_ordered_aucs):
        """Should raise ValueError for negative min_auc."""
        manager = lasso_results_with_ordered_aucs

        with pytest.raises(ValueError, match="must be between 0 and 1"):
            manager.select_lambda_min_test_auc(min_auc=-0.5)

    def test_invalid_min_auc_greater_than_one_raises_error(self, lasso_results_with_ordered_aucs):
        """Should raise ValueError for min_auc > 1."""
        manager = lasso_results_with_ordered_aucs

        with pytest.raises(ValueError, match="must be between 0 and 1"):
            manager.select_lambda_min_test_auc(min_auc=1.5)

    def test_custom_threshold(self, lasso_results_with_ordered_aucs):
        """Should accept custom threshold parameter."""
        manager = lasso_results_with_ordered_aucs

        # Should work with custom threshold
        result = manager.select_lambda_min_test_auc(min_auc=0.75, threshold=0.05)

        assert manager.selected_lambda_index is not None
        assert manager.test_aucs[result] >= 0.75


class TestSelectLambdaNonInferiority:
    """Tests for select_lambda_non_inferiority method."""

    @pytest.fixture
    def lasso_results_for_ni(self):
        """Create LASSO results with known AUCs for non-inferiority testing."""
        n_lambda = 10
        n_features = 5

        lambdas = np.logspace(-3, 1, n_lambda)

        # Test AUCs: increasing from 0.60 to 0.80 as lambda decreases
        # With max_auc=0.80:
        #   - ni_level=0.1: threshold = 0.80 - 0.1*(0.80-0.50) = 0.77
        #   - ni_level=0.2: threshold = 0.80 - 0.2*(0.80-0.50) = 0.74
        #   - ni_level=0.5: threshold = 0.80 - 0.5*(0.80-0.50) = 0.65
        #   - ni_level=1.0: threshold = 0.80 - 1.0*(0.80-0.50) = 0.50
        # Index:      0     1     2     3     4     5     6     7     8     9
        # test_aucs: 0.55, 0.60, 0.65, 0.68, 0.72, 0.74, 0.76, 0.78, 0.79, 0.80
        test_aucs = np.array([0.55, 0.60, 0.65, 0.68, 0.72, 0.74, 0.76, 0.78, 0.79, 0.80])
        train_aucs = test_aucs + 0.05
        train_losses = np.linspace(0.4, 0.1, n_lambda)
        test_losses = np.linspace(0.45, 0.15, n_lambda)

        # Create betas array: shape (n_features, n_lambda)
        betas = np.zeros((n_features, n_lambda))
        for i in range(n_lambda):
            coef = np.array([0.5, 0.3, 0.2, 0.15, 0.12]) * (i + 1) / n_lambda
            betas[:, i] = coef

        models = [LogisticRegression() for _ in range(n_lambda)]
        feature_names = [f'feature_{i}' for i in range(n_features)]

        return LassoResultsManager(
            lambdas=lambdas,
            betas=betas,
            models=models,
            feature_names=feature_names,
            train_losses=train_losses,
            test_losses=test_losses,
            train_aucs=train_aucs,
            test_aucs=test_aucs,
        )

    def test_computes_correct_threshold(self, lasso_results_for_ni):
        """Should compute correct non-inferiority threshold."""
        manager = lasso_results_for_ni

        # With max_auc=0.80 and ni_level=0.1:
        # threshold = 0.80 - 0.1 * (0.80 - 0.50) = 0.80 - 0.03 = 0.77
        # First lambda with AUC >= 0.77 is index 7 (0.78)
        result = manager.select_lambda_non_inferiority(ni_level=0.1)

        assert result == 7
        assert manager.test_aucs[result] >= 0.77

    def test_higher_ni_level_selects_sparser_model(self, lasso_results_for_ni):
        """Higher ni_level should select earlier (sparser) lambda."""
        manager = lasso_results_for_ni

        # ni_level=0.1: threshold=0.77, selects index 7
        result_strict = manager.select_lambda_non_inferiority(ni_level=0.1)

        # ni_level=0.2: threshold=0.74, selects index 5
        result_relaxed = manager.select_lambda_non_inferiority(ni_level=0.2)

        # More relaxed should select earlier (sparser) model
        assert result_relaxed < result_strict
        assert result_strict == 7
        assert result_relaxed == 5

    def test_ni_level_one_accepts_down_to_random(self, lasso_results_for_ni):
        """ni_level=1.0 should set threshold to 0.5 (random chance)."""
        manager = lasso_results_for_ni

        # ni_level=1.0: threshold = 0.80 - 1.0 * (0.80 - 0.50) = 0.50
        # First lambda with AUC >= 0.50 would be index 0 (0.55), but if it has
        # no features above beta_threshold, fallback occurs to first lambda
        # with features (index 2 in this test data).
        result = manager.select_lambda_non_inferiority(ni_level=1.0)

        # The selected lambda should meet the threshold (0.5)
        assert manager.test_aucs[result] >= 0.50
        # With ni_level=1.0, we should get an earlier (sparser) lambda than
        # with stricter ni_levels
        assert result < 5  # Should be earlier than ni_level=0.2 which selects index 5

    def test_fallback_to_max_auc_when_threshold_unreachable(self, lasso_results_for_ni, caplog):
        """When no lambda meets the AUC threshold, fall back to max-test-AUC selection.

        A reference_auc above any achievable AUC pushes the non-inferiority threshold
        past the best lambda, deterministically triggering the documented fallback.
        max test AUC here is 0.80; reference_auc=0.99, ni_level=0.05 gives
        threshold = 0.99 - 0.05*(0.99-0.5) = 0.9655, which no lambda reaches.
        """
        manager = lasso_results_for_ni

        result = manager.select_lambda_non_inferiority(ni_level=0.05, reference_auc=0.99)

        # Fallback path was taken...
        assert "Falling back to max test AUC" in caplog.text
        # ...and it lands on the max-test-AUC region (within the 99% target ratio).
        assert manager.selected_lambda_index is not None
        assert manager.test_aucs[result] >= 0.99 * np.max(manager.test_aucs)

    def test_invalid_ni_level_zero_raises_error(self, lasso_results_for_ni):
        """Should raise ValueError for ni_level=0."""
        manager = lasso_results_for_ni

        with pytest.raises(ValueError, match="must be in range"):
            manager.select_lambda_non_inferiority(ni_level=0)

    def test_invalid_ni_level_negative_raises_error(self, lasso_results_for_ni):
        """Should raise ValueError for negative ni_level."""
        manager = lasso_results_for_ni

        with pytest.raises(ValueError, match="must be in range"):
            manager.select_lambda_non_inferiority(ni_level=-0.1)

    def test_invalid_ni_level_greater_than_one_raises_error(self, lasso_results_for_ni):
        """Should raise ValueError for ni_level > 1."""
        manager = lasso_results_for_ni

        with pytest.raises(ValueError, match="must be in range"):
            manager.select_lambda_non_inferiority(ni_level=1.5)

    def test_custom_threshold_parameter(self, lasso_results_for_ni):
        """Should accept custom threshold parameter for beta coefficients."""
        manager = lasso_results_for_ni

        result = manager.select_lambda_non_inferiority(ni_level=0.1, threshold=0.05)

        assert manager.selected_lambda_index is not None
        assert result >= 0

    def test_selected_lambda_meets_threshold(self, lasso_results_for_ni):
        """Selected lambda should always meet the computed threshold."""
        manager = lasso_results_for_ni

        for ni_level in [0.05, 0.1, 0.2, 0.3, 0.5]:
            manager.select_lambda_non_inferiority(ni_level=ni_level)

            # Compute expected threshold
            max_auc = np.max(manager.test_aucs)
            threshold = max_auc - ni_level * (max_auc - 0.5)

            # Selected lambda should meet threshold
            assert manager.test_aucs[manager.selected_lambda_index] >= threshold

    def test_reference_auc_uses_provided_value(self, lasso_results_for_ni):
        """When reference_auc is provided, threshold should be based on it."""
        manager = lasso_results_for_ni

        # With reference_auc=0.90 and ni_level=0.1:
        # useful_auc = 0.90 - 0.50 = 0.40
        # allowed_loss = 0.1 * 0.40 = 0.04
        # threshold = 0.90 - 0.04 = 0.86
        # No lambda has AUC >= 0.86 (max is 0.80), so fallback to max_test_auc
        result = manager.select_lambda_non_inferiority(ni_level=0.1, reference_auc=0.90)
        # Fallback selects max test AUC index
        assert manager.selected_lambda_index is not None

        # With reference_auc=0.85 and ni_level=0.2:
        # useful_auc = 0.85 - 0.50 = 0.35
        # allowed_loss = 0.2 * 0.35 = 0.07
        # threshold = 0.85 - 0.07 = 0.78
        # First lambda with AUC >= 0.78 is index 7 (0.78)
        result = manager.select_lambda_non_inferiority(ni_level=0.2, reference_auc=0.85)
        assert result == 7

    def test_reference_auc_none_uses_max_lasso_auc(self, lasso_results_for_ni):
        """When reference_auc is None, should use max LASSO test AUC (backward compat)."""
        manager = lasso_results_for_ni

        # Without reference_auc: max_auc=0.80, ni_level=0.1
        # threshold = 0.80 - 0.1*(0.80-0.50) = 0.77
        # First with AUC >= 0.77 is index 7 (0.78)
        result_default = manager.select_lambda_non_inferiority(ni_level=0.1)
        assert result_default == 7

        result_explicit_none = manager.select_lambda_non_inferiority(
            ni_level=0.1,
            reference_auc=None,
        )
        assert result_explicit_none == 7

    def test_reference_auc_lower_than_some_lasso_aucs(self, lasso_results_for_ni):
        """reference_auc lower than max LASSO AUC should produce a more relaxed threshold."""
        manager = lasso_results_for_ni

        # reference_auc=0.70, ni_level=0.1:
        # useful_auc = 0.70 - 0.50 = 0.20
        # allowed_loss = 0.1 * 0.20 = 0.02
        # threshold = 0.70 - 0.02 = 0.68
        # First lambda with AUC >= 0.68 is index 3 (0.68)
        result = manager.select_lambda_non_inferiority(ni_level=0.1, reference_auc=0.70)
        assert result == 3

        # Without reference_auc (max=0.80): threshold=0.77, selects index 7
        result_default = manager.select_lambda_non_inferiority(ni_level=0.1)
        assert result_default == 7

        # Lower reference -> sparser model (earlier index)
        assert result < result_default
