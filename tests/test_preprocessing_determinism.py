"""Tests for preprocessing determinism using checksums.

These tests verify that the preprocessing pipeline produces identical results
across multiple runs when using the same random seed. This is critical for
reproducibility of machine learning experiments.

Uses the credit-g dataset (OpenML 31) with artificially introduced missing values
to test imputation determinism.
"""

import hashlib
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from prism.data_utilities import (
    analyze_categorical_columns,
    convert_to_categorical,
    create_binary_encoding,
    enforce_binary_target_encoding,
    ensure_consistent_categories,
    split_data_temporal_or_random,
)
from prism.preprocessing import impute_categorical_values, impute_numerical_values, preprocess_data

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def credit_g_path() -> Path:
    """Path to credit-g dataset."""
    return Path('data/raw/credit-g.csv')


@pytest.fixture
def credit_g_df(credit_g_path: Path) -> pd.DataFrame:
    """Load credit-g dataset."""
    if not credit_g_path.exists():
        pytest.skip(f"Credit-g dataset not found at {credit_g_path}")
    return pd.read_csv(credit_g_path)


@pytest.fixture
def credit_g_config() -> Dict[str, Any]:
    """Configuration for credit-g preprocessing (from credit-g.yaml)."""
    return {
        'integer_encoding': {
            'savings_status': ['no known savings', '<100', '100<=X<500', '500<=X<1000', '>=1000'],
            'employment': ['unemployed', '<1', '1<=X<4', '4<=X<7', '>=7'],
            'own_telephone': ['none', 'yes'],
            'foreign_worker': ['no', 'yes'],
        },
        'splitting_method': 'random',
        'split_ratios': [0.6, 0.2, 0.2],
        'invert_target': False,
        'target_variable': 'target',
        'random_seed': 257,
    }


@pytest.fixture
def credit_g_with_missing(credit_g_df: pd.DataFrame) -> pd.DataFrame:
    """Credit-g dataset with artificially introduced missing values.

    Introduces missing values in a deterministic manner:
    - 5% missing in 'duration' (numerical)
    - 5% missing in 'credit_amount' (numerical)
    - 5% missing in 'age' (numerical)
    - 5% missing in 'checking_status' (categorical)
    - 5% missing in 'purpose' (categorical)
    - 5% missing in 'employment' (categorical/ordinal)
    """
    df = credit_g_df.copy()
    np.random.seed(42)  # Fixed seed for deterministic missing pattern

    n_rows = len(df)
    missing_rate = 0.05
    n_missing = int(n_rows * missing_rate)

    # Introduce missing values in numerical columns
    numerical_cols = ['duration', 'credit_amount', 'age']
    for col in numerical_cols:
        missing_indices = np.random.choice(n_rows, size=n_missing, replace=False)
        df.loc[missing_indices, col] = np.nan

    # Introduce missing values in categorical columns
    categorical_cols = ['checking_status', 'purpose', 'employment']
    for col in categorical_cols:
        missing_indices = np.random.choice(n_rows, size=n_missing, replace=False)
        df.loc[missing_indices, col] = np.nan

    return df


# ============================================================================
# Helper Functions
# ============================================================================


def compute_dataframe_checksum(df: pd.DataFrame) -> str:
    """Compute a SHA256 checksum of a DataFrame.

    Handles floats with fixed precision to avoid floating-point comparison issues.
    """
    # Sort by all columns to ensure consistent ordering
    df_sorted = df.sort_values(by=list(df.columns)).reset_index(drop=True)

    # Round floats to avoid floating-point comparison issues
    df_rounded = df_sorted.copy()
    for col in df_rounded.select_dtypes(include=[np.number]).columns:
        df_rounded[col] = df_rounded[col].round(10)

    # Convert to string representation for hashing
    data_str = df_rounded.to_csv(index=False, float_format='%.10f')
    return hashlib.sha256(data_str.encode()).hexdigest()


def compute_split_checksums(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> Dict[str, str]:
    """Compute checksums for train/test/val splits."""
    return {
        'train': compute_dataframe_checksum(train_df),
        'test': compute_dataframe_checksum(test_df),
        'val': compute_dataframe_checksum(val_df),
    }


# ============================================================================
# Test Classes
# ============================================================================


class TestSplitDeterminism:
    """Tests for data splitting determinism."""

    def test_random_split_determinism(self, credit_g_df: pd.DataFrame):
        """Verify that random splitting is deterministic with same seed."""
        target = credit_g_df['target']

        # First run
        train1, test1, val1 = split_data_temporal_or_random(
            credit_g_df,
            temporal_column=None,
            train_size=0.6,
            test_size=0.2,
            val_size=0.2,
            random_state=257,
            stratify=target,
        )
        checksums1 = compute_split_checksums(train1, test1, val1)

        # Second run with same seed
        train2, test2, val2 = split_data_temporal_or_random(
            credit_g_df,
            temporal_column=None,
            train_size=0.6,
            test_size=0.2,
            val_size=0.2,
            random_state=257,
            stratify=target,
        )
        checksums2 = compute_split_checksums(train2, test2, val2)

        # Verify checksums match
        assert checksums1['train'] == checksums2['train'], "Train split checksum mismatch"
        assert checksums1['test'] == checksums2['test'], "Test split checksum mismatch"
        assert checksums1['val'] == checksums2['val'], "Validation split checksum mismatch"

    def test_different_seeds_produce_different_splits(self, credit_g_df: pd.DataFrame):
        """Verify that different seeds produce different splits."""
        target = credit_g_df['target']

        # First run with seed 257
        train1, test1, val1 = split_data_temporal_or_random(
            credit_g_df,
            temporal_column=None,
            train_size=0.6,
            test_size=0.2,
            val_size=0.2,
            random_state=257,
            stratify=target,
        )
        checksums1 = compute_split_checksums(train1, test1, val1)

        # Second run with different seed
        train2, test2, val2 = split_data_temporal_or_random(
            credit_g_df,
            temporal_column=None,
            train_size=0.6,
            test_size=0.2,
            val_size=0.2,
            random_state=999,
            stratify=target,
        )
        checksums2 = compute_split_checksums(train2, test2, val2)

        # Verify checksums are different
        assert (
            checksums1['train'] != checksums2['train']
        ), "Different seeds should produce different train splits"

    def test_split_sizes_correct(self, credit_g_df: pd.DataFrame):
        """Verify that split sizes are approximately correct."""
        target = credit_g_df['target']
        n_total = len(credit_g_df)

        train, test, val = split_data_temporal_or_random(
            credit_g_df,
            temporal_column=None,
            train_size=0.6,
            test_size=0.2,
            val_size=0.2,
            random_state=257,
            stratify=target,
        )

        # Allow 5% tolerance due to stratification constraints
        assert abs(len(train) / n_total - 0.6) < 0.05, "Train size outside tolerance"
        assert abs(len(test) / n_total - 0.2) < 0.05, "Test size outside tolerance"

    def test_temporal_column_excluded_from_categorical_conversion(self, credit_g_df: pd.DataFrame):
        """Verify that exclude_columns parameter prevents temporal column conversion.

        This tests the fix for the bug where temporal columns with few unique values
        (e.g., years 2010-2020) would be incorrectly converted to categorical,
        causing temporal split comparisons to fail.
        """
        df = credit_g_df.copy()

        # Add a temporal column with few unique values (would normally be converted)
        df['tx_year'] = [2010 + (i % 10) for i in range(len(df))]  # 10 unique values

        # Without exclude_columns, tx_year would become categorical
        df_with_cat = convert_to_categorical(
            df.copy(), categorical_threshold=12, convert_numeric=True
        )
        assert (
            df_with_cat['tx_year'].dtype.name == 'category'
        ), "tx_year should become categorical without exclude_columns"

        # With exclude_columns, tx_year should remain numeric
        df_excluded = convert_to_categorical(
            df.copy(),
            categorical_threshold=12,
            convert_numeric=True,
            exclude_columns=['tx_year'],
        )
        assert (
            df_excluded['tx_year'].dtype.name != 'category'
        ), "tx_year should remain numeric when excluded"
        assert pd.api.types.is_numeric_dtype(
            df_excluded['tx_year']
        ), "tx_year should be numeric dtype"

        # Verify temporal split works with excluded column
        train, test, val = split_data_temporal_or_random(
            df_excluded,
            temporal_column='tx_year',
            train_size=0.6,
            test_size=0.2,
            val_size=0.2,
            random_state=257,
        )

        # Should complete without error and have correct structure
        assert len(train) + len(test) + len(val) == len(df)

    def test_temporal_split_rejects_categorical_column(self, credit_g_df: pd.DataFrame):
        """Verify that temporal split raises TypeError for categorical columns.

        This tests that the validation catches categorical temporal columns early
        with a helpful error message, rather than failing with a cryptic pandas error.
        """
        df = credit_g_df.copy()

        # Add a temporal column and convert it to categorical
        df['tx_year'] = [2010 + (i % 10) for i in range(len(df))]
        df['tx_year'] = df['tx_year'].astype('category')

        # Should raise TypeError with helpful message
        with pytest.raises(TypeError, match="categorical dtype"):
            split_data_temporal_or_random(
                df,
                temporal_column='tx_year',
                train_size=0.6,
                test_size=0.2,
                val_size=0.2,
            )

    def test_temporal_split_rejects_string_column(self, credit_g_df: pd.DataFrame):
        """Verify that temporal split raises TypeError for string columns."""
        df = credit_g_df.copy()

        # Add a string temporal column
        df['tx_period'] = ['period_' + str(i % 10) for i in range(len(df))]

        # Should raise TypeError
        with pytest.raises(TypeError, match="must be numeric or datetime"):
            split_data_temporal_or_random(
                df,
                temporal_column='tx_period',
                train_size=0.6,
                test_size=0.2,
                val_size=0.2,
            )

    def test_temporal_split_accepts_datetime_column(self, credit_g_df: pd.DataFrame):
        """Verify that temporal split works with datetime columns."""
        df = credit_g_df.copy()

        # Add a datetime temporal column
        df['tx_date'] = pd.date_range('2010-01-01', periods=len(df), freq='D')

        # Should work without error
        train, test, val = split_data_temporal_or_random(
            df,
            temporal_column='tx_date',
            train_size=0.6,
            test_size=0.2,
            val_size=0.2,
        )

        assert len(train) + len(test) + len(val) == len(df)


class TestNumericalImputationDeterminism:
    """Tests for numerical imputation determinism."""

    @pytest.fixture
    def numerical_columns_only(self, credit_g_with_missing: pd.DataFrame) -> pd.DataFrame:
        """Extract only numerical columns from credit-g with missing values."""
        numerical_cols = credit_g_with_missing.select_dtypes(include=[np.number]).columns
        return credit_g_with_missing[numerical_cols].copy()

    def test_random_imputation_determinism(self, numerical_columns_only: pd.DataFrame):
        """Verify that random imputation is deterministic with same seed."""
        df = numerical_columns_only.copy()

        # First run
        result1 = impute_numerical_values(df.copy(), method='random', random_seed=257)
        checksum1 = compute_dataframe_checksum(result1)

        # Second run with same seed
        result2 = impute_numerical_values(df.copy(), method='random', random_seed=257)
        checksum2 = compute_dataframe_checksum(result2)

        # Verify checksums match
        assert checksum1 == checksum2, "Random imputation checksum mismatch"

    def test_median_imputation_determinism(self, numerical_columns_only: pd.DataFrame):
        """Verify that median imputation is deterministic."""
        df = numerical_columns_only.copy()

        # First run
        result1 = impute_numerical_values(df.copy(), method='median')
        checksum1 = compute_dataframe_checksum(result1)

        # Second run
        result2 = impute_numerical_values(df.copy(), method='median')
        checksum2 = compute_dataframe_checksum(result2)

        # Verify checksums match (median is deterministic by nature)
        assert checksum1 == checksum2, "Median imputation checksum mismatch"

    def test_zeros_imputation_determinism(self, numerical_columns_only: pd.DataFrame):
        """Verify that zeros imputation is deterministic."""
        df = numerical_columns_only.copy()

        # First run
        result1 = impute_numerical_values(df.copy(), method='zeros')
        checksum1 = compute_dataframe_checksum(result1)

        # Second run
        result2 = impute_numerical_values(df.copy(), method='zeros')
        checksum2 = compute_dataframe_checksum(result2)

        # Verify checksums match
        assert checksum1 == checksum2, "Zeros imputation checksum mismatch"

    def test_different_seeds_produce_different_imputations(
        self, numerical_columns_only: pd.DataFrame
    ):
        """Verify that different seeds produce different random imputations."""
        df = numerical_columns_only.copy()

        # First run with seed 257
        result1 = impute_numerical_values(df.copy(), method='random', random_seed=257)
        checksum1 = compute_dataframe_checksum(result1)

        # Second run with different seed
        result2 = impute_numerical_values(df.copy(), method='random', random_seed=999)
        checksum2 = compute_dataframe_checksum(result2)

        # Verify checksums are different
        assert checksum1 != checksum2, "Different seeds should produce different imputations"


class TestCategoricalImputationDeterminism:
    """Tests for categorical imputation determinism."""

    def test_mode_imputation_determinism(self, credit_g_with_missing: pd.DataFrame):
        """Verify that mode imputation is deterministic."""
        df = credit_g_with_missing.copy()
        categorical_cols = ['checking_status', 'purpose', 'employment']

        # First run
        result1 = impute_categorical_values(df.copy(), categorical_cols, method='mode')
        checksum1 = compute_dataframe_checksum(result1)

        # Second run
        result2 = impute_categorical_values(df.copy(), categorical_cols, method='mode')
        checksum2 = compute_dataframe_checksum(result2)

        # Verify checksums match (mode is deterministic)
        assert checksum1 == checksum2, "Mode imputation checksum mismatch"

    def test_random_imputation_determinism(self, credit_g_with_missing: pd.DataFrame):
        """Verify that random categorical imputation is deterministic with same seed."""
        df = credit_g_with_missing.copy()
        categorical_cols = ['checking_status', 'purpose', 'employment']

        # First run
        result1 = impute_categorical_values(
            df.copy(), categorical_cols, method='random', random_state=257
        )
        checksum1 = compute_dataframe_checksum(result1)

        # Second run with same seed
        result2 = impute_categorical_values(
            df.copy(), categorical_cols, method='random', random_state=257
        )
        checksum2 = compute_dataframe_checksum(result2)

        # Verify checksums match
        assert checksum1 == checksum2, "Random categorical imputation checksum mismatch"

    def test_different_seeds_produce_different_imputations(
        self, credit_g_with_missing: pd.DataFrame
    ):
        """Verify that different seeds produce different random imputations."""
        df = credit_g_with_missing.copy()
        categorical_cols = ['checking_status', 'purpose', 'employment']

        # First run with seed 257
        result1 = impute_categorical_values(
            df.copy(), categorical_cols, method='random', random_state=257
        )
        checksum1 = compute_dataframe_checksum(result1)

        # Second run with different seed
        result2 = impute_categorical_values(
            df.copy(), categorical_cols, method='random', random_state=999
        )
        checksum2 = compute_dataframe_checksum(result2)

        # Verify checksums are different
        assert (
            checksum1 != checksum2
        ), "Different seeds should produce different categorical imputations"


class TestEncodingDeterminism:
    """Tests for categorical encoding determinism."""

    def test_binary_ordinal_mapping_determinism(self, credit_g_df: pd.DataFrame):
        """Verify that binary ordinal mapping is deterministic."""
        df = convert_to_categorical(credit_g_df, categorical_threshold=12)
        cat_info = analyze_categorical_columns(df)
        binary_vars = cat_info['binary']

        # First run
        mapping1 = create_binary_encoding(df, binary_vars)

        # Second run
        mapping2 = create_binary_encoding(df, binary_vars)

        # Verify mappings are identical
        assert mapping1 == mapping2, "Binary ordinal mapping mismatch"

    def test_ordinal_encoding_determinism(
        self, credit_g_df: pd.DataFrame, credit_g_config: Dict[str, Any]
    ):
        """Verify that ordinal encoding is deterministic."""
        integer_encoding = credit_g_config['integer_encoding']

        # First run
        result1, meta1 = preprocess_data(
            credit_g_df.copy(),
            integer_encoding=integer_encoding,
            categorical_imputation_method='mode',
            numerical_imputation_method='median',
            random_state=257,
            drop_reference_columns=True,
        )
        checksum1 = compute_dataframe_checksum(result1)

        # Second run with same seed
        result2, meta2 = preprocess_data(
            credit_g_df.copy(),
            integer_encoding=integer_encoding,
            categorical_imputation_method='mode',
            numerical_imputation_method='median',
            random_state=257,
            drop_reference_columns=True,
        )
        checksum2 = compute_dataframe_checksum(result2)

        # Verify checksums match
        assert checksum1 == checksum2, "Ordinal encoding checksum mismatch"


class TestFullPreprocessingPipelineDeterminism:
    """Integration tests for full preprocessing pipeline determinism."""

    def test_full_pipeline_determinism(
        self, credit_g_with_missing: pd.DataFrame, credit_g_config: Dict[str, Any]
    ):
        """Verify that the full preprocessing pipeline is deterministic."""
        config = credit_g_config
        random_seed = config['random_seed']

        # Run pipeline twice
        checksums1 = self._run_preprocessing_pipeline(
            credit_g_with_missing.copy(), config, random_seed
        )
        checksums2 = self._run_preprocessing_pipeline(
            credit_g_with_missing.copy(), config, random_seed
        )

        # Verify all checksums match
        for split in ['train', 'test', 'val']:
            assert (
                checksums1[split] == checksums2[split]
            ), f"{split.capitalize()} split checksum mismatch in full pipeline"

    def test_pipeline_produces_different_results_with_different_seeds(
        self, credit_g_with_missing: pd.DataFrame, credit_g_config: Dict[str, Any]
    ):
        """Verify that different seeds produce different results."""
        config = credit_g_config

        checksums1 = self._run_preprocessing_pipeline(
            credit_g_with_missing.copy(), config, random_seed=257
        )
        checksums2 = self._run_preprocessing_pipeline(
            credit_g_with_missing.copy(), config, random_seed=999
        )

        # Verify checksums are different
        assert (
            checksums1['train'] != checksums2['train']
        ), "Different seeds should produce different results"

    def test_pipeline_consistency_across_multiple_runs(
        self, credit_g_with_missing: pd.DataFrame, credit_g_config: Dict[str, Any]
    ):
        """Verify pipeline consistency across 5 runs."""
        config = credit_g_config
        random_seed = config['random_seed']

        # Run pipeline 5 times
        all_checksums = []
        for _ in range(5):
            checksums = self._run_preprocessing_pipeline(
                credit_g_with_missing.copy(), config, random_seed
            )
            all_checksums.append(checksums)

        # Verify all runs produce identical results
        for i in range(1, 5):
            for split in ['train', 'test', 'val']:
                assert (
                    all_checksums[0][split] == all_checksums[i][split]
                ), f"Run {i+1} {split} checksum differs from run 1"

    def _run_preprocessing_pipeline(
        self, df: pd.DataFrame, config: Dict[str, Any], random_seed: int
    ) -> Dict[str, str]:
        """Run the preprocessing pipeline and return checksums.

        Mimics the preprocessing.py notebook workflow.
        """
        # Step 1: Enforce binary target encoding
        target_variable = config['target_variable']
        df = enforce_binary_target_encoding(df, target_variable)

        # Step 2: Convert to categorical
        df_processed = convert_to_categorical(df, categorical_threshold=12)

        # Step 3: Split data
        train_df_raw, test_df_raw, val_df_raw = split_data_temporal_or_random(
            df_processed,
            temporal_column=None,
            train_size=config['split_ratios'][0],
            test_size=config['split_ratios'][1],
            val_size=config['split_ratios'][2],
            stratify=df_processed[target_variable],
            random_state=random_seed,
        )

        # Step 4: Ensure consistent categories
        integer_encoding = config['integer_encoding']
        train_df_consistent = ensure_consistent_categories(train_df_raw, {}, integer_encoding)
        test_df_consistent = ensure_consistent_categories(test_df_raw, {}, integer_encoding)
        val_df_consistent = ensure_consistent_categories(val_df_raw, {}, integer_encoding)

        # Step 5: Apply preprocessing
        train_processed, _ = preprocess_data(
            train_df_consistent,
            integer_encoding=integer_encoding,
            categorical_imputation_method='random',
            numerical_imputation_method='random',
            random_state=random_seed,
            drop_reference_columns=True,
        )

        test_processed, _ = preprocess_data(
            test_df_consistent,
            integer_encoding=integer_encoding,
            categorical_imputation_method='random',
            numerical_imputation_method='random',
            random_state=random_seed,
            drop_reference_columns=True,
        )

        val_processed, _ = preprocess_data(
            val_df_consistent,
            integer_encoding=integer_encoding,
            categorical_imputation_method='random',
            numerical_imputation_method='random',
            random_state=random_seed,
            drop_reference_columns=True,
        )

        return compute_split_checksums(train_processed, test_processed, val_processed)

    def test_temporal_split_pipeline_determinism(
        self, credit_g_with_missing: pd.DataFrame, credit_g_config: Dict[str, Any]
    ):
        """Verify that temporal splitting is deterministic across runs.

        Adds an artificial 'tx_year' column to simulate temporal data,
        then verifies the temporal split produces identical results.
        """
        config = credit_g_config
        random_seed = config['random_seed']

        # Run temporal pipeline twice
        checksums1 = self._run_temporal_preprocessing_pipeline(
            credit_g_with_missing.copy(), config, random_seed
        )
        checksums2 = self._run_temporal_preprocessing_pipeline(
            credit_g_with_missing.copy(), config, random_seed
        )

        # Verify all checksums match
        for split in ['train', 'test', 'val']:
            assert (
                checksums1[split] == checksums2[split]
            ), f"{split.capitalize()} split checksum mismatch in temporal pipeline"

    def test_temporal_split_consistency_across_multiple_runs(
        self, credit_g_with_missing: pd.DataFrame, credit_g_config: Dict[str, Any]
    ):
        """Verify temporal pipeline consistency across 5 runs."""
        config = credit_g_config
        random_seed = config['random_seed']

        # Run pipeline 5 times
        all_checksums = []
        for _ in range(5):
            checksums = self._run_temporal_preprocessing_pipeline(
                credit_g_with_missing.copy(), config, random_seed
            )
            all_checksums.append(checksums)

        # Verify all runs produce identical results
        for i in range(1, 5):
            for split in ['train', 'test', 'val']:
                assert (
                    all_checksums[0][split] == all_checksums[i][split]
                ), f"Run {i+1} {split} checksum differs from run 1 in temporal split"

    def _run_temporal_preprocessing_pipeline(
        self, df: pd.DataFrame, config: Dict[str, Any], random_seed: int
    ) -> Dict[str, str]:
        """Run the preprocessing pipeline with temporal splitting.

        Adds an artificial temporal column and uses temporal split.
        """
        # Step 1: Enforce binary target encoding first
        target_variable = config['target_variable']
        df = enforce_binary_target_encoding(df, target_variable)

        # Step 2: Convert to categorical BEFORE adding temporal column
        # This prevents tx_year from being converted to categorical
        df_processed = convert_to_categorical(df, categorical_threshold=12)

        # Add artificial temporal column AFTER categorical conversion
        # Use a deterministic assignment based on row index
        n_rows = len(df_processed)
        # Assign years roughly proportionally: 60% early (train), 20% mid (test), 20% late (val)
        years = []
        for i in range(n_rows):
            if i < n_rows * 0.6:
                years.append(2010 + (i % 4))  # 2010-2013 for training
            elif i < n_rows * 0.8:
                years.append(2014 + (i % 2))  # 2014-2015 for test
            else:
                years.append(2016 + (i % 4))  # 2016-2019 for validation
        df_processed['tx_year'] = years

        # Step 3: Split data temporally
        train_df_raw, test_df_raw, val_df_raw = split_data_temporal_or_random(
            df_processed,
            temporal_column='tx_year',
            train_size=config['split_ratios'][0],
            test_size=config['split_ratios'][1],
            val_size=config['split_ratios'][2],
            random_state=random_seed,
        )

        # Drop the artificial temporal column before further processing
        train_df_raw = train_df_raw.drop(columns=['tx_year'])
        test_df_raw = test_df_raw.drop(columns=['tx_year'])
        val_df_raw = val_df_raw.drop(columns=['tx_year'])

        # Step 4: Ensure consistent categories
        integer_encoding = config['integer_encoding']
        train_df_consistent = ensure_consistent_categories(train_df_raw, {}, integer_encoding)
        test_df_consistent = ensure_consistent_categories(test_df_raw, {}, integer_encoding)
        val_df_consistent = ensure_consistent_categories(val_df_raw, {}, integer_encoding)

        # Step 5: Apply preprocessing
        train_processed, _ = preprocess_data(
            train_df_consistent,
            integer_encoding=integer_encoding,
            categorical_imputation_method='random',
            numerical_imputation_method='random',
            random_state=random_seed,
            drop_reference_columns=True,
        )

        test_processed, _ = preprocess_data(
            test_df_consistent,
            integer_encoding=integer_encoding,
            categorical_imputation_method='random',
            numerical_imputation_method='random',
            random_state=random_seed,
            drop_reference_columns=True,
        )

        val_processed, _ = preprocess_data(
            val_df_consistent,
            integer_encoding=integer_encoding,
            categorical_imputation_method='random',
            numerical_imputation_method='random',
            random_state=random_seed,
            drop_reference_columns=True,
        )

        return compute_split_checksums(train_processed, test_processed, val_processed)


class TestChecksumUtility:
    """Tests for the checksum utility function itself."""

    def test_checksum_different_for_different_data(self):
        """Verify that different data produces different checksums."""
        df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        df2 = pd.DataFrame({'a': [1, 2, 4], 'b': [4, 5, 6]})  # Different value

        assert compute_dataframe_checksum(df1) != compute_dataframe_checksum(df2)

    def test_checksum_same_for_same_data(self):
        """Verify that identical data produces identical checksums."""
        df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        df2 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

        assert compute_dataframe_checksum(df1) == compute_dataframe_checksum(df2)

    def test_checksum_independent_of_row_order(self):
        """Verify that row order doesn't affect checksum."""
        df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        df2 = pd.DataFrame({'a': [3, 1, 2], 'b': [6, 4, 5]})  # Different row order

        assert compute_dataframe_checksum(df1) == compute_dataframe_checksum(df2)

    def test_checksum_handles_floats(self):
        """Verify that floating-point values are handled correctly."""
        df1 = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [4.5, 5.5, 6.5]})
        df2 = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [4.5, 5.5, 6.5]})

        assert compute_dataframe_checksum(df1) == compute_dataframe_checksum(df2)

    def test_checksum_handles_nan(self):
        """Verify that NaN values are handled consistently."""
        df1 = pd.DataFrame({'a': [1.0, np.nan, 3.0], 'b': [4.5, 5.5, np.nan]})
        df2 = pd.DataFrame({'a': [1.0, np.nan, 3.0], 'b': [4.5, 5.5, np.nan]})

        assert compute_dataframe_checksum(df1) == compute_dataframe_checksum(df2)

    def test_checksum_handles_mixed_types(self):
        """Verify that mixed types are handled correctly."""
        df1 = pd.DataFrame({'num': [1, 2, 3], 'str': ['a', 'b', 'c']})
        df2 = pd.DataFrame({'num': [1, 2, 3], 'str': ['a', 'b', 'c']})

        assert compute_dataframe_checksum(df1) == compute_dataframe_checksum(df2)
