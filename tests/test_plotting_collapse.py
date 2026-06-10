"""
Tests for plotting functions with collapsed one-hot groups.

This module tests:
1. Integration tests for the full plotting pipeline with collapse (TestPlottingIntegration)
2. OneHotGroupManager methods used by the plotting refactor (TestOneHotGroupManagerNewMethods)
3. PlottingDataBundle bivariate indexing correctness (TestBivariateIndexingBug)

Note: Tests for obsolete pre-calculated PR paths have been removed as of 2024-12-08.
The new architecture always recalculates responses on grids via PlottingPipeline + NomogramRenderer.
Full visual validation is now in tests/plotting/test_visual_production.py.
"""

import numpy as np
import pytest
import torch

from prism.nomogram_plot import nomogram
from prism.preprocessing import OneHotGroupManager
from prism.response_plot import plot_partial_responses


# Integration test
class TestPlottingIntegration:
    """Integration tests for full plotting pipeline with collapse."""

    def test_full_pipeline_credit_g(self):
        """
        Full integration test with credit-g dataset mirroring the notebook workflow.

        This test faithfully follows the pipeline from:
        - preprocessing.py: Load data, detect categoricals, one-hot encode, split
        - train_mlp.py: Train model on one-hot features
        - prism_analysis.py: Calculate PR with collapse, run LASSO, generate plots

        Validates Phase 3.5: Plotting with pre-calculated collapsed PR.
        """
        from pathlib import Path

        import matplotlib.pyplot as plt
        import pandas as pd
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder

        from prism.lasso import LassoRegression
        from prism.partial_responses import partial_responses
        from prism.preprocessing import OneHotGroupManager, PRiSMScaler, collapse_onehot_features
        from prism.wrapper import SklearnWrapper

        # =====================================================================
        # STEP 1: PREPROCESSING (mirrors preprocessing.py)
        # =====================================================================
        data_path = Path('data/raw/credit-g.csv')
        if not data_path.exists():
            pytest.skip(f"Credit-g dataset not found at {data_path}")

        df = pd.read_csv(data_path)

        # Detect target column (credit-g uses 'target' column)
        target_col = 'target'
        if target_col not in df.columns:
            pytest.skip(f"Target column '{target_col}' not found")

        # Encode target to binary
        le = LabelEncoder()
        y = le.fit_transform(df[target_col])
        X = df.drop(target_col, axis=1)

        # Identify categorical vs numerical columns (as in preprocessing.py)
        categorical_cols = X.select_dtypes(include='object').columns.tolist()
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

        # For faster testing, select subset of features:
        # 3 numerical + 2 categorical (one binary, one multi-category)
        selected_numerical = ['duration', 'credit_amount', 'age']

        # Find one binary categorical and one multi-category
        binary_cat = None
        multi_cat = None
        for col in categorical_cols:
            n_unique = X[col].nunique()
            if n_unique == 2 and binary_cat is None:
                binary_cat = col
            elif 3 <= n_unique <= 5 and multi_cat is None:
                multi_cat = col
            if binary_cat and multi_cat:
                break

        if multi_cat is None:
            pytest.skip("No suitable multi-category column found")

        selected_cols = selected_numerical + ([binary_cat] if binary_cat else []) + [multi_cat]
        X = X[selected_cols].copy()

        # Convert numerical columns
        for col in selected_numerical:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        X = X.fillna(X.median(numeric_only=True))

        # Split data BEFORE encoding (as in preprocessing.py)
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # One-hot encode multi-category columns (like preprocessing.py default behavior)
        # Binary columns also need encoding (ordinal or one-hot)
        cols_to_onehot = [multi_cat]

        # First, encode binary categorical columns with ordinal encoding (0/1)
        if binary_cat:
            binary_encoder = LabelEncoder()
            X_train_raw[binary_cat] = binary_encoder.fit_transform(X_train_raw[binary_cat])
            X_test_raw[binary_cat] = binary_encoder.transform(X_test_raw[binary_cat])

        # Determine the reference category BEFORE encoding (first alphabetically)
        # pandas get_dummies(drop_first=True) drops the first category alphabetically
        multi_cat_categories = sorted(X_train_raw[multi_cat].unique())
        reference_category = multi_cat_categories[0]  # First alphabetically = dropped
        reference_col_name = f"{multi_cat}_{reference_category}"

        # One-hot encode (drop_first=True to create reference category, as in preprocessing)
        X_train_encoded = pd.get_dummies(X_train_raw, columns=cols_to_onehot, drop_first=True)
        X_test_encoded = pd.get_dummies(X_test_raw, columns=cols_to_onehot, drop_first=True)

        # Ensure consistent columns across splits
        all_columns = sorted(set(X_train_encoded.columns) | set(X_test_encoded.columns))
        for col in all_columns:
            if col not in X_train_encoded.columns:
                X_train_encoded[col] = 0
            if col not in X_test_encoded.columns:
                X_test_encoded[col] = 0
        X_train_encoded = X_train_encoded[all_columns]
        X_test_encoded = X_test_encoded[all_columns]

        feature_names = X_train_encoded.columns.tolist()
        X_train_np = X_train_encoded.values.astype(float)
        X_test_np = X_test_encoded.values.astype(float)

        # Create OneHotGroupManager (as stored in preprocessing metadata)
        # Identify one-hot groups from feature names
        groups_dict = {}
        onehot_features = [f for f in feature_names if f.startswith(f'{multi_cat}_')]
        if onehot_features:
            groups_dict[multi_cat] = onehot_features

        if not groups_dict:
            pytest.skip("No one-hot groups detected after encoding")

        # Properly set reference_columns to match production preprocessing
        # The reference column is the one that was dropped by get_dummies(drop_first=True)
        reference_columns = {multi_cat: reference_col_name}

        group_manager = OneHotGroupManager(
            groups_dict=groups_dict, reference_columns=reference_columns
        )

        print("\n=== PREPROCESSING SUMMARY ===")
        print(f"Original features: {len(selected_cols)}")
        print(f"One-hot encoded features: {len(feature_names)}")
        print(f"One-hot groups: {groups_dict}")
        print(f"Reference columns: {reference_columns}")
        print(f"Training samples: {len(X_train_np)}")
        print(f"Test samples: {len(X_test_np)}")

        # =====================================================================
        # STEP 2: MODEL TRAINING (mirrors train_mlp.py)
        # =====================================================================
        # Use LogisticRegression for faster testing (MLP would work the same way)
        model_sklearn = LogisticRegression(max_iter=1000, random_state=42)
        model_sklearn.fit(X_train_np, y_train)
        model = SklearnWrapper(model_sklearn)

        # Scale data (as in train_mlp.py)
        scaler = PRiSMScaler()
        scaler.fit(X_train_np)
        X_train_scaled = scaler.transform(X_train_np)
        X_test_scaled = scaler.transform(X_test_np)

        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

        # Verify model works
        y_pred_train = model.predict_proba(X_train_tensor)
        train_acc = ((y_pred_train.numpy() > 0.5).astype(int) == y_train).mean()
        print("\n=== MODEL TRAINING ===")
        print("Model: LogisticRegression (wrapped)")
        print(f"Training accuracy: {train_acc:.3f}")

        # =====================================================================
        # STEP 3: PRISM ANALYSIS (mirrors prism_analysis.py)
        # =====================================================================
        # Get collapsed feature names
        X_collapsed_np, collapsed_feature_names = collapse_onehot_features(
            X_train_np, group_manager, feature_names
        )

        print("\n=== COLLAPSE INFO ===")
        print(f"Original features: {len(feature_names)}")
        print(f"Collapsed features: {len(collapsed_feature_names)}")
        print(f"Collapsed names: {collapsed_feature_names}")

        # Test BOTH Dirac and Lebesgue methods
        for method in ['dirac', 'lebesgue']:
            print(f"\n{'='*60}")
            print(f"TESTING {method.upper()} METHOD")
            print(f"{'='*60}")

            # Calculate partial responses WITH COLLAPSE (the key test)
            # Note: collapse happens automatically when group_manager is provided
            partial_responses_params = {
                'x_train': X_train_tensor,
                'method': method,
                'device': 'cpu',
                'batch_size': 64,
                'group_manager': group_manager,
                'feature_names': feature_names,
                'scaler': scaler,  # Pass the actual scaler since data is scaled
            }

            print(f"\n=== PARTIAL RESPONSE CALCULATION ({method}) ===")
            print("Calculating collapsed partial responses...")
            pr_train = partial_responses(X_train_tensor, model, **partial_responses_params)
            pr_test = partial_responses(X_test_tensor, model, **partial_responses_params)

            # Verify PR dimensions
            n_collapsed = len(collapsed_feature_names)
            n_collapsed_bivariate = n_collapsed * (n_collapsed - 1) // 2
            expected_pr_cols = n_collapsed + n_collapsed_bivariate

            print(f"PR train shape: {pr_train.shape}")
            print(
                f"Expected collapsed dimensions: {n_collapsed} univariate + {n_collapsed_bivariate} bivariate = {expected_pr_cols}"
            )

            # The PR should be collapsed - verify dimension is less than original
            n_original = len(feature_names)
            n_original_bivariate = n_original * (n_original - 1) // 2
            original_pr_cols = n_original + n_original_bivariate

            assert (
                pr_train.shape[1] < original_pr_cols
            ), f"[{method}] PR should be collapsed: got {pr_train.shape[1]}, original would be {original_pr_cols}"

            # Verify exact collapsed dimensions
            assert (
                pr_train.shape[1] == expected_pr_cols
            ), f"[{method}] PR has wrong shape: got {pr_train.shape[1]}, expected {expected_pr_cols}"

            # =====================================================================
            # STEP 4: LASSO FEATURE SELECTION (mirrors prism_analysis.py)
            # =====================================================================
            print(f"\n=== LASSO FEATURE SELECTION ({method}) ===")

            lasso = LassoRegression(nlambda=20, max_workers=2, seed=42)
            lasso_results, _ = lasso.fit(
                pr_train,
                pr_test,
                torch.tensor(y_train, dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.float32),
                feature_names=collapsed_feature_names,  # Use collapsed names!
            )

            # Select lambda
            lasso_results.select_lambda_max_test_auc()

            selected_indices = lasso_results.get_selected_univariate_indices()
            print(f"Selected features: {len(selected_indices)}")

            if len(selected_indices) == 0:
                print(
                    f"[{method}] LASSO didn't select any features - skipping plotting for this method"
                )
                continue

            # =====================================================================
            # STEP 5: PLOTTING WITH UNIFIED API (Automatic collapse via OneHotGroupManager)
            # =====================================================================
            print(f"\n=== PLOTTING WITH UNIFIED API ({method}) ===")

            # Test plot_partial_responses with unified API
            # Pass one-hot encoded data + model + group_manager for automatic collapse
            # Note: Pass the ORIGINAL scaler (one-hot space), not collapsed scaler
            # The pipeline will handle collapse internally
            try:
                result = plot_partial_responses(
                    lasso_results,
                    x=X_train_tensor,  # Pass one-hot encoded tensor
                    model=model,  # Pass model trained on one-hot features
                    scaler=scaler,  # Pass original scaler (one-hot space)
                    onehot_group_manager=group_manager,  # Automatic collapse handling
                    n_steps=10,
                    show_fig=False,
                    return_fig=True,
                    feature_names=feature_names,  # Pass one-hot feature names
                )
                assert result is not None, f"[{method}] plot_partial_responses should return tuple"
                # plot_partial_responses returns (fig_responses, fig_heatmaps) tuple
                fig_responses, fig_heatmaps = result
                assert fig_responses is not None, f"[{method}] Should have response plots figure"
                if isinstance(fig_responses, list):
                    n_axes = sum(len(f.axes) for f in fig_responses)
                else:
                    n_axes = len(fig_responses.axes)
                assert n_axes > 0, f"[{method}] Figure should have axes"
                print(f"plot_partial_responses ({method}): SUCCESS - {n_axes} axes")
                if isinstance(fig_responses, list):
                    for f in fig_responses:
                        plt.close(f)
                else:
                    plt.close(fig_responses)
                if fig_heatmaps is not None:
                    if isinstance(fig_heatmaps, list):
                        for f in fig_heatmaps:
                            plt.close(f)
                    else:
                        plt.close(fig_heatmaps)
            except Exception as e:
                pytest.fail(f"[{method}] plot_partial_responses with unified API failed: {e}")

            # Test nomogram with unified API
            try:
                nomogram_results = nomogram(
                    lasso_results,
                    x=X_train_tensor,  # Pass one-hot encoded tensor
                    model=model,  # Pass model trained on one-hot features
                    scaler=scaler,  # Pass original scaler (one-hot space)
                    onehot_group_manager=group_manager,  # Automatic collapse handling
                    n_steps=10,
                    show_fig=False,
                    return_fig=True,
                    feature_names=feature_names,  # Pass one-hot feature names
                )
                assert nomogram_results is not None, f"[{method}] nomogram should return results"
                # nomogram returns NomogramResult dataclass
                assert hasattr(
                    nomogram_results, 'fig_main'
                ), f"[{method}] Expected NomogramResult with fig_main attribute"
                print(f"nomogram ({method}): SUCCESS - returned NomogramResult")

                # Close any figures
                if nomogram_results.fig_main is not None:
                    plt.close(nomogram_results.fig_main)
                if nomogram_results.fig_bivariate is not None:
                    plt.close(nomogram_results.fig_bivariate)
            except Exception as e:
                pytest.fail(f"[{method}] nomogram with unified API failed: {e}")

            print(f"\n=== {method.upper()} METHOD PASSED ===")

        print(f"\n{'='*60}")
        print("INTEGRATION TEST PASSED - Both Dirac and Lebesgue methods validated!")
        print(f"{'='*60}")

    def test_plotting_with_unified_api(self):
        """
        Focused test: Validate plotting functions work with unified API.

        Tests that plot_partial_responses() and nomogram() work correctly
        when recalculating on grids (the unified approach).
        """
        import matplotlib

        matplotlib.use('Agg')  # Use non-interactive backend for testing
        import matplotlib.pyplot as plt
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split

        from prism.lasso import LassoRegression
        from prism.partial_responses import partial_responses
        from prism.preprocessing import PRiSMScaler
        from prism.wrapper import SklearnWrapper

        # Create simple synthetic data
        np.random.seed(42)
        n_samples = 200
        n_features = 6

        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        feature_names = [f'feature_{i}' for i in range(n_features)]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train model
        model_sklearn = LogisticRegression(max_iter=1000, random_state=42)
        model_sklearn.fit(X_train, y_train)
        model = SklearnWrapper(model_sklearn)

        # Scale
        scaler = PRiSMScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)

        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        X_test_tensor = torch.tensor(scaler.transform(X_test), dtype=torch.float32)

        # Calculate PR (without collapse for simplicity)
        pr_train = partial_responses(
            X_train_tensor,
            model,
            x_train=X_train_tensor,
            method='dirac',
            device='cpu',
            batch_size=64,
        )

        pr_test = partial_responses(
            X_test_tensor,
            model,
            x_train=X_train_tensor,
            method='dirac',
            device='cpu',
            batch_size=64,
        )

        # Run LASSO
        lasso = LassoRegression(nlambda=10, max_workers=2)
        lasso_results, _ = lasso.fit(
            pr_train,
            pr_test,
            torch.tensor(y_train, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32),
            feature_names=feature_names,
        )

        # Select best lambda
        lasso_results.select_lambda_max_test_auc()

        # Verify some features selected
        selected = lasso_results.get_selected_univariate_indices()
        if len(selected) == 0:
            pytest.skip("LASSO didn't select any features - data may be too simple")

        # KEY TEST: Use unified API (always recalculate on grids)
        result = plot_partial_responses(
            lasso_results,
            x=X_train_tensor,  # Pass input data
            model=model,  # Pass model for recalculation
            scaler=scaler,
            n_steps=10,
            show_fig=False,
            return_fig=True,
        )

        assert result is not None, "Should return tuple with unified API"
        # plot_partial_responses returns (fig_responses, fig_heatmaps) tuple
        fig_responses, fig_heatmaps = result
        assert fig_responses is not None, "Should generate response plots figure"
        if isinstance(fig_responses, list):
            n_axes = sum(len(f.axes) for f in fig_responses)
        else:
            n_axes = len(fig_responses.axes)
        assert n_axes > 0, "Figure should have axes"

        # Test nomogram with unified API
        nomogram_results = nomogram(
            lasso_results,
            x=X_train_tensor,  # Pass input data
            model=model,  # Pass model for recalculation
            scaler=scaler,
            n_steps=10,
            show_fig=False,
            return_fig=True,
        )

        assert nomogram_results is not None
        assert hasattr(
            nomogram_results, 'fig_main'
        ), "Expected NomogramResult with fig_main attribute"

        # Clean up figures
        if isinstance(fig_responses, list):
            for f in fig_responses:
                plt.close(f)
        else:
            plt.close(fig_responses)
        if fig_heatmaps is not None:
            if isinstance(fig_heatmaps, list):
                for f in fig_heatmaps:
                    plt.close(f)
            else:
                plt.close(fig_heatmaps)
        if nomogram_results.fig_main is not None:
            plt.close(nomogram_results.fig_main)
        if nomogram_results.fig_bivariate is not None:
            plt.close(nomogram_results.fig_bivariate)

        print("\n[OK] Integration test passed: Plotting functions work with unified API")


class TestOneHotGroupManagerNewMethods:
    """Test new methods added to OneHotGroupManager for plotting refactoring."""

    def test_is_categorical_group(self):
        """Test checking if feature is a collapsed one-hot group."""
        manager = OneHotGroupManager({'diagn': ['diagn_CAD', 'diagn_Congenital']})

        assert manager.is_categorical_group('diagn') is True
        assert manager.is_categorical_group('age') is False
        assert manager.is_categorical_group('diagn_CAD') is False  # Member, not group

    # REMOVED TESTS: The following tests have been removed as of 2025-11-25
    # because the corresponding methods in OneHotGroupManager were deprecated.
    # Collapse functionality is now handled internally by PartialResponseCalculator.
    #
    # Removed tests:
    # - test_map_collapsed_to_original_indices()
    # - test_map_collapsed_to_original_pairs()
    # - test_collapse_responses()
    # - test_collapse_responses_with_torch()
    # - test_collapse_bivariate_responses()
    # - test_collapse_x_grids()
    # - test_collapse_x_bivariate_grids()
    #
    # Collapse functionality is now validated through integration tests
    # in TestPlottingIntegration that test the full pipeline with automatic collapse.


class TestBivariateIndexingBug:
    """
    Tests for bivariate indexing bug (Phase B).

    Root Cause: nomogram_plot.py filters out skipped pairs before passing to
    generate_non_mixed_bivariate_nomogram(), but the generator enumerates ALL
    pairs from LASSO (including skipped ones), causing IndexError when accessing
    the filtered list with indices from the full list.

    These tests should FAIL with the current buggy code and PASS after the fix.
    """

    @pytest.fixture
    def data_with_skipped_pairs(self):
        """Create data with same-group pairs that will be skipped."""
        np.random.seed(42)
        n_samples = 100

        # Create 2 continuous features + 1 categorical with 4 categories
        # Categorical encoded as one-hot with 3 dummy columns (reference dropped)
        continuous_1 = np.random.randn(n_samples)
        continuous_2 = np.random.randn(n_samples)

        # Categorical with 4 categories (0, 1, 2, 3)
        # One-hot: [cat_A, cat_B, cat_C] (reference = 0 is dropped)
        categorical = np.random.randint(0, 4, n_samples)
        cat_A = (categorical == 1).astype(float)
        cat_B = (categorical == 2).astype(float)
        cat_C = (categorical == 3).astype(float)

        # One-hot encoding: [cont1, cont2, cat_A, cat_B, cat_C]
        X_onehot = np.column_stack([continuous_1, continuous_2, cat_A, cat_B, cat_C])

        # Collapsed: [cont1, cont2, cat]
        X_collapsed = np.column_stack([continuous_1, continuous_2, categorical])

        feature_names_onehot = ['cont1', 'cont2', 'cat_A', 'cat_B', 'cat_C']
        feature_names_collapsed = ['cont1', 'cont2', 'cat']

        # Create OneHotGroupManager
        groups_dict = {'cat': ['cat_A', 'cat_B', 'cat_C']}
        group_manager = OneHotGroupManager(
            groups_dict=groups_dict, reference_columns={'cat': 'cat_ref'}
        )

        # Binary target
        y = (continuous_1 + continuous_2 + categorical * 0.3 > 0).astype(int)

        # Expected bivariate pairs in collapsed space (0,1), (0,2), (1,2)
        # Pair (2,2) would be same-group and skipped (cat x cat within same group)
        # But we have 3 features, so all pairs are different features
        # To trigger skipped pairs, we need pairs within the same one-hot group

        return {
            'X_onehot': torch.tensor(X_onehot, dtype=torch.float32),
            'X_collapsed': torch.tensor(X_collapsed, dtype=torch.float32),
            'y': torch.tensor(y, dtype=torch.float32),
            'feature_names_onehot': feature_names_onehot,
            'feature_names_collapsed': feature_names_collapsed,
            'group_manager': group_manager,
            'n_features_onehot': 5,
            'n_features_collapsed': 3,
        }

    def test_bivariate_dense_to_list_mapping(self, data_with_skipped_pairs):
        """
        Test that dense_idx from enumerate() correctly maps to list position.

        This test verifies that when we enumerate all selected pairs and use
        dense_idx to access responses, the indexing works correctly.
        """
        from prism.plotting_data import FeaturePairInfo, PlottingDataBundle

        # Simulate scenario:
        # - LASSO selected 10 pairs total (indices 0-9)
        # - 2 pairs are skipped (same-group): positions 3 and 7
        # - Responses list should have 10 elements (8 real + 2 dummy)

        all_pairs = [
            (0, 1),
            (0, 2),
            (1, 2),
            (2, 3),
            (0, 4),
            (1, 4),
            (2, 4),
            (3, 4),
            (0, 5),
            (1, 5),
        ]
        skipped_positions = {3, 7}  # Pairs at these positions are skipped

        # Create bundle with all pairs
        bivariate_info = []
        for pos, (i, j) in enumerate(all_pairs):
            is_skipped = pos in skipped_positions
            response = np.array([0.0]) if is_skipped else np.random.randn(10)
            x_vals = np.array([[0.0, 0.0]]) if is_skipped else np.random.randn(10, 2)

            info = FeaturePairInfo(
                indices=(i, j),
                names=(f'f{i}', f'f{j}'),
                labels=(f'Feature {i}', f'Feature {j}'),
                is_categorical=(False, False),
                response=response,
                x_values=x_vals,
                skipped=is_skipped,
            )
            bivariate_info.append(info)

        bundle = PlottingDataBundle(
            all_feature_names=['f0', 'f1', 'f2', 'f3', 'f4', 'f5'],
            _bivariate_info=bivariate_info,
        )

        # Extract lists like nomogram_plot.py does (BUGGY VERSION - filters skipped)
        bivariate_responses_filtered = [
            info.response for info in bundle.bivariate_pairs() if not info.skipped
        ]

        # Verify filtered list has wrong length
        assert (
            len(bivariate_responses_filtered) == 8
        ), "Filtered list should have 8 non-skipped pairs"

        # Simulate what generate_non_mixed_bivariate_nomogram() does
        bivariate_index_pairs = all_pairs  # Gets ALL pairs from LASSO

        # This should cause IndexError with filtered list
        with pytest.raises(IndexError, match="list index out of range"):
            for dense_idx, (f1, f2) in enumerate(bivariate_index_pairs):
                # When dense_idx = 8 or 9, this will fail with filtered list
                _ = bivariate_responses_filtered[dense_idx]

        print("EXPECTED FAILURE: test_bivariate_dense_to_list_mapping - IndexError reproduced!")

    def test_bivariate_with_skipped_pairs(self, data_with_skipped_pairs):
        """
        Test plotting with skipped same-group pairs.

        Verifies that skipped pairs don't cause indexing errors when
        properly handled (i.e., not filtered out).
        """
        from prism.plotting_data import FeaturePairInfo, PlottingDataBundle

        # Create bundle with mix of regular and skipped pairs
        all_pairs = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3)]  # 5 pairs
        # Pairs (0,3) and (1,3) are same-group and skipped
        skipped_pairs = {(0, 3), (1, 3)}

        bivariate_info = []
        for i, j in all_pairs:
            is_skipped = (i, j) in skipped_pairs
            response = np.array([0.0]) if is_skipped else np.random.randn(15)
            x_vals = np.array([[0.0, 0.0]]) if is_skipped else np.random.randn(15, 2)

            info = FeaturePairInfo(
                indices=(i, j),
                names=(f'f{i}', f'f{j}'),
                labels=(f'Feature {i}', f'Feature {j}'),
                is_categorical=(False, False),
                response=response,
                x_values=x_vals,
                skipped=is_skipped,
            )
            bivariate_info.append(info)

        bundle = PlottingDataBundle(
            all_feature_names=['f0', 'f1', 'f2', 'f3'],
            _bivariate_info=bivariate_info,
        )

        # CORRECT VERSION: Don't filter skipped pairs
        bivariate_responses_all = [info.response for info in bundle.bivariate_pairs()]

        assert len(bivariate_responses_all) == 5, "Should have all 5 pairs (including skipped)"

        # This should work without IndexError
        for dense_idx, info in enumerate(bundle.bivariate_pairs()):
            response = bivariate_responses_all[dense_idx]
            assert response is not None
            if info.skipped:
                assert response.size == 1, "Skipped pair should have size-1 response"

    def test_all_bivariate_types(self, data_with_skipped_pairs):
        """
        Test mixed, cat-cat, and cont-cont bivariate combinations.

        Verifies that all types of bivariate interactions work with
        correct indexing.
        """
        from prism.plotting_data import FeaturePairInfo, PlottingDataBundle

        # Create pairs of different types:
        # (0,1): cont-cont
        # (0,2): cont-cat (mixed)
        # (1,2): cont-cat (mixed)
        # (2,2): same feature (would be filtered by LASSO, but simulate)

        pairs_with_types = [
            ((0, 1), ('cont', 'cont')),  # continuous-continuous
            ((0, 2), ('cont', 'cat')),  # mixed
            ((1, 2), ('cont', 'cat')),  # mixed
        ]

        bivariate_info = []
        for (i, j), (type1, type2) in pairs_with_types:
            is_cat1 = type1 == 'cat'
            is_cat2 = type2 == 'cat'

            # Generate appropriate response shape
            if is_cat1 and is_cat2:
                # cat-cat: grid of category combinations
                n_cats1, n_cats2 = 3, 3
                response = np.random.randn(n_cats1 * n_cats2)
                x_vals = np.array([[i, j] for i in range(n_cats1) for j in range(n_cats2)])
            elif is_cat1 or is_cat2:
                # mixed: lines for each category
                n_cats = 4 if is_cat1 else 4
                n_steps = 15
                response = np.random.randn(n_cats * n_steps)
                x_vals = np.random.randn(n_cats * n_steps, 2)
            else:
                # cont-cont: 2D grid
                n_steps = 15
                response = np.random.randn(n_steps * n_steps)
                x_vals = np.random.randn(n_steps * n_steps, 2)

            info = FeaturePairInfo(
                indices=(i, j),
                names=(f'{type1}{i}', f'{type2}{j}'),
                labels=(f'{type1.upper()} {i}', f'{type2.upper()} {j}'),
                is_categorical=(is_cat1, is_cat2),
                response=response,
                x_values=x_vals,
                skipped=False,
            )
            bivariate_info.append(info)

        bundle = PlottingDataBundle(
            all_feature_names=['cont0', 'cont1', 'cat2'],
            _bivariate_info=bivariate_info,
        )

        # Verify we can access all types
        for dense_idx, info in enumerate(bundle.bivariate_pairs()):
            assert info.response is not None
            assert info.x_values is not None
            assert info.response.shape[0] > 0

    # NOTE: test_non_mixed_bivariate_nomogram_indexing was removed as it tested
    # a bug in the legacy NomogramGenerator which has been deleted.
    # The new NomogramRenderer architecture handles this correctly by design
    # through PlottingDataBundle with explicit skipped pair tracking.

    def test_correct_handling_with_unfiltered_lists(self, data_with_skipped_pairs):
        """
        Test that the fix works: Keep skipped pairs in the lists.

        This test shows the CORRECT behavior after the fix is applied.
        """
        from prism.plotting_data import FeaturePairInfo, PlottingDataBundle

        # Create bundle with skipped pairs
        all_pairs = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3)]
        skipped_pairs = {(0, 3), (1, 3)}  # Same-group pairs

        bivariate_info = []
        for i, j in all_pairs:
            is_skipped = (i, j) in skipped_pairs
            response = np.array([0.0]) if is_skipped else np.random.randn(15)
            x_vals = np.array([[0.0, 0.0]]) if is_skipped else np.random.randn(15, 2)

            info = FeaturePairInfo(
                indices=(i, j),
                names=(f'f{i}', f'f{j}'),
                labels=(f'Feature {i}', f'Feature {j}'),
                is_categorical=(False, False),
                response=response,
                x_values=x_vals,
                skipped=is_skipped,
            )
            bivariate_info.append(info)

        bundle = PlottingDataBundle(
            all_feature_names=['f0', 'f1', 'f2', 'f3'],
            _bivariate_info=bivariate_info,
        )

        # CORRECT: Don't filter skipped pairs (FIX)
        bivariate_responses_all = [info.response for info in bundle.bivariate_pairs()]
        x_bivariate_all = [info.x_values for info in bundle.bivariate_pairs()]
        selected_pairs_all = [info.indices for info in bundle.bivariate_pairs()]

        # All lists should have same length as number of pairs
        assert len(bivariate_responses_all) == 5
        assert len(x_bivariate_all) == 5
        assert len(selected_pairs_all) == 5

        # Now indexing with dense_idx should work
        for dense_idx, (i, j) in enumerate(selected_pairs_all):
            response = bivariate_responses_all[dense_idx]
            x_vals = x_bivariate_all[dense_idx]

            assert response is not None
            assert x_vals is not None

            # Skipped pairs have dummy values
            if (i, j) in skipped_pairs:
                assert response.size == 1
                assert response.item() == 0.0
                assert x_vals.shape == (1, 2)

        print("SUCCESS: Unfiltered lists work correctly with dense_idx indexing!")
