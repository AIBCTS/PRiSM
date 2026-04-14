import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from prism.config import PROCESSED_DATA_DIR

if TYPE_CHECKING:
    from prism.feature_labels import FeatureLabelManager

logger = logging.getLogger(__name__)


class NoScaler:
    """
    No-op scaler that returns data unchanged.
    Compatible with sklearn API.
    """

    def fit(self, X, y=None):
        """Fit scaler (no-op)."""
        return self

    def transform(self, X):
        """Transform data (returns unchanged)."""
        return np.asarray(X)

    def inverse_transform(self, X_scaled):
        """Inverse transform data (returns unchanged)."""
        return np.asarray(X_scaled)

    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        return self.transform(X)


class MedianStdScaler:
    """
    Custom scaler that uses median centering and std scaling with optional sd_scale.
    Compatible with sklearn API.

    Supports excluding binary/one-hot columns from scaling via `exclude_cols` parameter.
    """

    def __init__(self, sd_scale=1.0, exclude_cols=None):
        """
        Initialize scaler.

        Parameters
        ----------
        sd_scale : float
            Scale factor for standard deviation (default 1.0)
        exclude_cols : list of int, optional
            Column indices to exclude from scaling (will remain unchanged).
            Use this for binary/one-hot encoded columns that should stay as 0/1.
        """
        self.sd_scale = sd_scale
        self.exclude_cols = set(exclude_cols) if exclude_cols is not None else set()
        self.median_ = None
        self.std_ = None

    def fit(self, X, y=None):
        """Fit scaler parameters."""
        X = np.asarray(X)
        self.median_ = np.median(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # Avoid division by zero
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)

        # For excluded columns, set median=0 and std=1 so transform is identity
        for col_idx in self.exclude_cols:
            if col_idx < len(self.median_):
                self.median_[col_idx] = 0.0
                self.std_[col_idx] = 1.0 / self.sd_scale  # So (x - 0) / (std * sd_scale) = x

        return self

    def transform(self, X):
        """Transform data."""
        X = np.asarray(X)
        return (X - self.median_) / (self.std_ * self.sd_scale)

    def inverse_transform(self, X_scaled):
        """Inverse transform data."""
        X_scaled = np.asarray(X_scaled)
        return X_scaled * (self.std_ * self.sd_scale) + self.median_

    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)


def detect_binary_columns(X, feature_names=None):
    """
    Detect columns that appear to be binary (only 0 and 1 values).

    Parameters
    ----------
    X : array-like
        Data matrix
    feature_names : list of str, optional
        Feature names for logging

    Returns
    -------
    list of int
        Indices of binary columns
    """
    X = np.asarray(X)
    binary_cols = []

    for i in range(X.shape[1]):
        col = X[:, i]
        unique_vals = np.unique(col[~np.isnan(col)])

        # Check if column contains only 0 and 1 (or subset)
        if len(unique_vals) <= 2 and np.all(np.isin(unique_vals, [0, 1])):
            binary_cols.append(i)

    if binary_cols and feature_names:
        logger.debug(
            f"Detected {len(binary_cols)} binary columns: "
            f"{[feature_names[i] for i in binary_cols[:5]]}..."
        )

    return binary_cols


def detect_scaled_onehot_columns(X, onehot_indices, tolerance=0.1):
    """
    Detect if one-hot encoded columns appear to have been scaled.

    One-hot columns should contain only 0s and 1s. If they contain other values,
    they may have been incorrectly scaled.

    Parameters
    ----------
    X : array-like
        Data matrix (potentially scaled)
    onehot_indices : list of int
        Indices of columns that should be one-hot encoded (0/1 values)
    tolerance : float
        Tolerance for detecting non-binary values

    Returns
    -------
    bool
        True if columns appear to be scaled (contain non-0/1 values)
    list of int
        Indices of columns that appear scaled
    """
    X = np.asarray(X)
    scaled_cols = []

    for col_idx in onehot_indices:
        if col_idx >= X.shape[1]:
            continue

        col = X[:, col_idx]
        unique_vals = np.unique(col[~np.isnan(col)])

        # Check if any value is not close to 0 or 1
        is_zero = np.abs(unique_vals) < tolerance
        is_one = np.abs(unique_vals - 1.0) < tolerance

        if not np.all(is_zero | is_one):
            scaled_cols.append(col_idx)

    return len(scaled_cols) > 0, scaled_cols


class PRiSMScaler:
    """
    sklearn-compatible scaler wrapper for PRiSM models.

    Handles PRiSM-specific requirements while maintaining sklearn API.
    Supports excluding binary/one-hot columns from scaling.
    """

    def __init__(self, scaler='median_std', onehot_columns=None):
        """
        Initialize scaler.

        Parameters
        ----------
        scaler : str, sklearn scaler, or None
            - sklearn scaler: StandardScaler(), RobustScaler(), etc.
            - 'median_std': MedianStdScaler (default for compatibility)
            - None: NoScaler (no transformation)
        onehot_columns : list of int, optional
            Column indices of one-hot encoded features to exclude from scaling.
            These columns will remain as 0/1 values after transformation.
            If None, binary columns are auto-detected during fit().
        """
        self.feature_names_ = None
        self.onehot_columns = onehot_columns  # User-specified or auto-detected
        self._onehot_columns_fitted = None  # Actual columns used after fit

        if scaler is None:
            self.scaler = NoScaler()
            self._scaler_type = 'none'
        elif scaler == 'median_std':
            # Will be initialized in fit() with exclude_cols
            self._scaler_type = 'median_std'
            self.scaler = None
        else:
            self.scaler = scaler
            self._scaler_type = 'custom'

    def fit(self, X, y=None, auto_detect_binary=True):
        """
        Fit the scaler to training data.

        Parameters
        ----------
        X : array-like or DataFrame
            Training data
        y : ignored
        auto_detect_binary : bool
            If True and onehot_columns was not specified, automatically detect
            binary columns and exclude them from scaling.

        Returns
        -------
        self
        """
        if hasattr(X, 'columns'):
            self.feature_names_ = X.columns.tolist()
            X_values = X.values
        else:
            X_values = np.asarray(X)
            self.feature_names_ = None

        # Determine which columns to exclude from scaling
        if self.onehot_columns is not None:
            # User specified columns
            self._onehot_columns_fitted = list(self.onehot_columns)
        elif auto_detect_binary:
            # Auto-detect binary columns
            self._onehot_columns_fitted = detect_binary_columns(X_values, self.feature_names_)
            if self._onehot_columns_fitted:
                logger.info(
                    f"Auto-detected {len(self._onehot_columns_fitted)} binary columns "
                    f"to exclude from scaling"
                )
        else:
            self._onehot_columns_fitted = []

        # Initialize and fit the underlying scaler
        if self._scaler_type == 'median_std':
            self.scaler = MedianStdScaler(sd_scale=2.0, exclude_cols=self._onehot_columns_fitted)
        elif self._scaler_type == 'none':
            pass  # Already initialized
        # For custom scalers, we can't easily exclude columns

        self.scaler.fit(X_values)
        return self

    def transform(self, X):
        """Transform data using fitted scaler."""
        if hasattr(X, 'values'):
            X_values = X.values
        else:
            X_values = np.asarray(X)

        return self.scaler.transform(X_values)

    def fit_transform(self, X, y=None, auto_detect_binary=True):
        """Fit and transform in one step."""
        return self.fit(X, y, auto_detect_binary=auto_detect_binary).transform(X)

    def inverse_transform(self, X_scaled):
        """Transform scaled data back to original scale."""
        return self.scaler.inverse_transform(X_scaled)

    def to_tensor(self, X, device='cpu'):
        """Transform data and convert to PyTorch tensor."""
        X_scaled = self.transform(X)
        return torch.tensor(X_scaled, dtype=torch.float32, device=device)

    def get_excluded_columns(self):
        """Return the list of columns excluded from scaling."""
        return self._onehot_columns_fitted or []


def impute_numerical_values(
    data: pd.DataFrame,
    method: str = 'zeros',
    test: Optional[pd.DataFrame] = None,
    random_seed: Optional[int] = None,
    verbose: bool = False,
    empty_values: Optional[List[Any]] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Impute missing and empty values in the dataset using specified method.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset containing missing values to impute.
    method : str, optional
        Imputation method to use, one of:
        - 'zeros': Replace missing values with zeros (default)
        - 'random': Sample randomly from non-missing values
        - 'median': Replace missing values with median of non-missing values
    test : pd.DataFrame, optional
        Test dataset to impute using the same parameters as `data`.
        When provided, uses training data statistics for test imputation.
    random_seed : int, optional
        Random seed for reproducibility when using 'random' method.
    verbose : bool, optional
        If True, print detailed information about the imputation process.
    empty_values : List[Any], optional
        Additional values to treat as missing. Common examples include:
        ['', ' ', 'NA', 'N/A', 'none', 'None', 'null', 'NULL', '?']

    Returns
    -------
    Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]
        Imputed training dataset or tuple of imputed training and testing datasets.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> # Create sample data with missing values
    >>> data = pd.DataFrame({
    ...     'A': [1, np.nan, 3, 4, 5],
    ...     'B': [10, 20, 'NA', 40, 50]
    ... })
    >>> # Basic imputation with zeros
    >>> imputed = impute_numerical_values(data, method='zeros')
    >>> # Impute both train and test data using median
    >>> test_data = pd.DataFrame({
    ...     'A': [np.nan, 2, 3],
    ...     'B': [10, 'NULL', 30]
    ... })
    >>> train_imp, test_imp = impute_numerical_values(
    ...     data, test=test_data, method='median',
    ...     empty_values=['NA', 'NULL']
    ... )
    """
    if method not in ['zeros', 'random', 'median']:
        raise ValueError("Method must be one of: 'zeros', 'random', 'median'")

    # Default empty values to check
    if empty_values is None:
        empty_values = ['', ' ', 'NA', 'N/A', 'none', 'None', 'null', 'NULL', '?']

    # Create copies to avoid modifying original data
    x_train = data.copy()
    x_test = test.copy() if test is not None else None

    if random_seed is not None:
        np.random.seed(random_seed)

    # Replace empty values with np.nan for consistent processing
    for val in empty_values:
        x_train = x_train.replace(val, np.nan)
        if x_test is not None:
            x_test = x_test.replace(val, np.nan)

    # Convert string columns that should be numeric
    for col in x_train.columns:
        try:
            # Check if column contains any numeric values
            if x_train[col].dropna().astype(str).str.match(r'^[-+]?[0-9]*\.?[0-9]+$').any():
                x_train[col] = pd.to_numeric(x_train[col], errors='coerce')
                if x_test is not None:
                    x_test[col] = pd.to_numeric(x_test[col], errors='coerce')
        except (ValueError, TypeError):
            continue

    # Print initial statistics if verbose
    if verbose:
        print(f"\nImputation Method: {method.upper()}")
        print("-" * 50)
        print("Empty values being treated as missing:", empty_values)

        total_missing_train = x_train.isna().sum().sum()
        print(f"\nTraining set ({x_train.shape[0]} samples, {x_train.shape[1]} features):")
        print(f"Total missing values: {total_missing_train}")
        print(f"Missing percentage: {(total_missing_train / x_train.size * 100):.2f}%")

        if test is not None:
            total_missing_test = x_test.isna().sum().sum()
            print(f"\nTest set ({x_test.shape[0]} samples, {x_test.shape[1]} features):")
            print(f"Total missing values: {total_missing_test}")
            print(f"Missing percentage: {(total_missing_test / x_test.size * 100):.2f}%")

    # Process each column
    for column in x_train.columns:
        missing_train = x_train[column].isna().sum()
        missing_test = x_test[column].isna().sum() if x_test is not None else 0

        if missing_train == 0 and missing_test == 0:
            if verbose:
                print(f"{column}: No missing values - skipped")
            continue

        if method == 'zeros':
            x_train.loc[x_train[column].isna(), column] = 0
            if x_test is not None:
                x_test.loc[x_test[column].isna(), column] = 0

            if verbose:
                print(f"{column}: Replaced {missing_train} missing values with zeros")
                if test is not None:
                    print(f"        Replaced another {missing_test} missing values in test set")

        elif method == 'random':
            non_missing = x_train[column].dropna().values
            if len(non_missing) > 0:
                if missing_train > 0:
                    x_train.loc[x_train[column].isna(), column] = np.random.choice(
                        non_missing, size=missing_train
                    )

                if x_test is not None and missing_test > 0:
                    x_test.loc[x_test[column].isna(), column] = np.random.choice(
                        non_missing, size=missing_test
                    )

                if verbose:
                    if missing_train > 0:
                        print(
                            f"{column}: Sampled {missing_train} values from {len(non_missing)} non-missing values"
                        )
                        print(
                            f"        Sample range: [{non_missing.min():.3g}, {non_missing.max():.3g}]"
                        )
                    if test is not None and missing_test > 0:
                        print(f"        Sampled another {missing_test} values for test set")

        elif method == 'median':
            median_value = x_train[column].median()
            if pd.notnull(median_value):
                x_train.loc[x_train[column].isna(), column] = median_value
                if x_test is not None:
                    x_test.loc[x_test[column].isna(), column] = median_value

                if verbose:
                    print(
                        f"{column}: Replaced {missing_train} missing values with median ({median_value:.3g})"
                    )
                    if test is not None:
                        print(
                            f"        Replaced another {missing_test} missing values in test set"
                        )
            else:
                # If median is null (e.g., for string columns), use mode instead
                mode_value = x_train[column].mode().iloc[0] if not x_train[column].empty else 0
                x_train.loc[x_train[column].isna(), column] = mode_value
                if x_test is not None:
                    x_test.loc[x_test[column].isna(), column] = mode_value

                if verbose:
                    print(
                        f"{column}: Using mode imputation instead of median due to non-numeric data"
                    )
                    print(
                        f"        Replaced {missing_train} missing values with mode value: {mode_value}"
                    )
                    if test is not None:
                        print(
                            f"        Replaced another {missing_test} missing values in test set"
                        )

    if test is not None:
        return x_train, x_test
    else:
        return x_train


def impute_categorical_values(
    df: pd.DataFrame,
    categorical_cols: List[str],
    method: str = 'mode',
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """
    Impute missing values in categorical columns before encoding.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing categorical columns to impute
    categorical_cols : list
        List of categorical column names to impute
    method : str, optional
        Imputation method, either 'mode' or 'random'
    random_state : int, optional
        Random seed for reproducibility when using random imputation

    Returns:
    --------
    pd.DataFrame
        DataFrame with imputed categorical values
    """
    df_imputed = df.copy()

    if random_state is not None:
        np.random.seed(random_state)

    for col in categorical_cols:
        if col not in df_imputed.columns:
            logger.warning(f"Column '{col}' not found in DataFrame")
            continue

        missing_count = df_imputed[col].isna().sum()
        if missing_count == 0:
            logger.debug(f"No missing values in column '{col}'")
            continue

        if method == 'mode':
            mode_value = df_imputed[col].mode().iloc[0]
            df_imputed[col] = df_imputed[col].fillna(mode_value)
            logger.info(
                f"Imputed {missing_count} missing values in '{col}' with mode: {mode_value}"
            )

        elif method == 'random':
            non_missing = df_imputed[col].dropna().values
            if len(non_missing) > 0:
                missing_mask = df_imputed[col].isna()
                df_imputed.loc[missing_mask, col] = np.random.choice(
                    non_missing, size=missing_count
                )
                logger.info(
                    f"Imputed {missing_count} missing values in '{col}' with random sampling"
                )
            else:
                logger.warning(
                    f"Cannot perform random imputation for '{col}' - no non-missing values"
                )

        else:
            logger.error(f"Unknown imputation method: {method}")
            raise ValueError("Method must be one of: 'mode', 'random'")

    return df_imputed


def encode_categorical_features(
    df: pd.DataFrame,
    categorical_cols: Optional[List[str]] = None,
    integer_encoding: Optional[Dict[str, List[str]]] = None,
    imputation_method: str = 'mode',
    drop_original: bool = True,
    random_state: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Encode categorical features with one-hot by default, with option for ordinal encoding.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing categorical columns to encode
    categorical_cols : list, optional
        List of categorical columns to encode. If None, will use all object/string columns
    integer_encoding : dict, optional
        Dictionary mapping column names to lists of categories in desired order
        Example: {'recmedcond': ['Home', 'Hospital', 'ICU']}
    imputation_method : str, optional
        Method to use for imputing missing categorical values ('mode' or 'random')
    drop_original : bool, optional
        Whether to drop original categorical columns after encoding
    random_state : int, optional
        Random seed for reproducibility when using random imputation

    Returns:
    --------
    Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]
        DataFrame with encoded categorical features and detailed encoding metadata
    """
    df_encoded = df.copy()

    # If categorical_cols not specified, use all object columns
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if not categorical_cols:
        logger.info("No categorical columns to encode")
        return df_encoded, {}

    logger.info(f"Encoding {len(categorical_cols)} categorical features")

    # First impute all categorical columns before encoding
    df_encoded = impute_categorical_values(
        df_encoded, categorical_cols, method=imputation_method, random_state=random_state
    )

    # Initialize dictionary to track encoding metadata
    encoding_metadata = {}

    # Process each categorical column
    for col in categorical_cols:
        if col not in df_encoded.columns:
            logger.warning(f"Column '{col}' not found in DataFrame")
            continue

        # Create metadata entry for this column
        encoding_metadata[col] = {
            'encoding_type': None,
            'created_columns': [],
            'mapping': {},
            'reverse_mapping': {},
            'original_categories': sorted(df_encoded[col].unique().tolist()),
        }

        # Apply ordinal encoding if specified for this column
        if integer_encoding is not None and col in integer_encoding:
            ordered_categories = integer_encoding[col]

            # Create the ordinal mapping
            category_map = {cat: i for i, cat in enumerate(ordered_categories)}
            reverse_map = {i: cat for i, cat in enumerate(ordered_categories)}

            # Handle values not in the ordinal mapping
            missing_categories = set(df_encoded[col].unique()) - set(ordered_categories)
            if missing_categories:
                logger.warning(
                    f"Column '{col}' contains values not in ordinal mapping: {missing_categories}"
                )
                # Add missing categories to the end of the mapping
                for i, cat in enumerate(missing_categories):
                    idx = len(ordered_categories) + i
                    category_map[cat] = idx
                    reverse_map[idx] = cat

            # Apply the mapping (replaces original column in-place)
            df_encoded[col] = df_encoded[col].map(category_map)

            # Update metadata
            encoding_metadata[col]['encoding_type'] = 'ordinal'
            encoding_metadata[col]['created_columns'] = [col]
            encoding_metadata[col]['mapping'] = category_map
            encoding_metadata[col]['reverse_mapping'] = reverse_map
            encoding_metadata[col]['original_column'] = col

            logger.info(f"Applied ordinal encoding to '{col}' using specified order")

        # Check if this is a binary variable (exactly 2 categories) - auto-encode as integer
        elif len(df_encoded[col].unique()) == 2:
            # Automatically apply integer encoding for binary variables
            from prism.data_utilities import order_binary_categories

            categories = sorted(df_encoded[col].unique().tolist())

            # Use shared ordering logic
            ordered_categories = order_binary_categories(categories)

            # Create the integer mapping
            category_map = {cat: i for i, cat in enumerate(ordered_categories)}
            reverse_map = {i: cat for i, cat in enumerate(ordered_categories)}

            # Apply the mapping (replaces original column in-place)
            df_encoded[col] = df_encoded[col].map(category_map)

            # Update metadata
            encoding_metadata[col]['encoding_type'] = 'ordinal'
            encoding_metadata[col]['created_columns'] = [col]
            encoding_metadata[col]['mapping'] = category_map
            encoding_metadata[col]['reverse_mapping'] = reverse_map
            encoding_metadata[col]['original_column'] = col

            logger.info(
                f"Applied automatic integer encoding to binary variable '{col}': {ordered_categories[0]}=0, {ordered_categories[1]}=1"
            )

        # Apply one-hot encoding for multi-category columns
        else:
            # Get dummies and add prefix - explicitly setting dtype=int to get 1/0 instead of True/False
            dummies = pd.get_dummies(df_encoded[col], prefix=col, dtype=int)

            # Create mapping from original categories to dummy columns
            unique_values = df_encoded[col].dropna().unique()
            one_hot_mapping = {}
            reverse_mapping = {}

            for value in unique_values:
                column_name = f"{col}_{value}"
                one_hot_mapping[value] = column_name
                reverse_mapping[column_name] = value

            # Add dummy columns to the DataFrame
            df_encoded = pd.concat([df_encoded, dummies], axis=1)

            # Update metadata
            encoding_metadata[col]['encoding_type'] = 'one-hot'
            encoding_metadata[col]['created_columns'] = dummies.columns.tolist()
            encoding_metadata[col]['mapping'] = one_hot_mapping
            encoding_metadata[col]['reverse_mapping'] = reverse_mapping
            encoding_metadata[col]['original_column'] = col

            logger.info(
                f"Applied one-hot encoding to '{col}', created {len(dummies.columns)} binary features as integers (1/0)"
            )

    # Drop original categorical columns if requested
    # Note: Ordinal-encoded columns are already replaced in-place, so only drop one-hot encoded columns
    if drop_original:
        ordinal_encoded_cols = [
            col
            for col in categorical_cols
            if col in encoding_metadata and encoding_metadata[col]['encoding_type'] == 'ordinal'
        ]
        cols_to_drop = [col for col in categorical_cols if col not in ordinal_encoded_cols]
        if cols_to_drop:
            df_encoded = df_encoded.drop(columns=cols_to_drop)
            logger.info(
                f"Dropped {len(cols_to_drop)} original categorical columns after encoding (ordinal-encoded columns retained in-place)"
            )

    return df_encoded, encoding_metadata


def preprocess_data(
    df: pd.DataFrame,
    integer_encoding: Optional[Dict[str, List[str]]] = None,
    categorical_imputation_method: str = 'mode',
    numerical_imputation_method: str = 'median',
    drop_original_categorical: bool = True,
    random_state: Optional[int] = 257,
    save_metadata_path: Optional[Union[str, Path]] = None,
    reference_column_strategy: str = 'alphabetical',
    manual_reference_columns: Optional[Dict[str, str]] = None,
    drop_reference_columns: bool = True,
    dataset_prefix: Optional[str] = None,
    input_file_path: Optional[Union[str, Path]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Preprocess data by handling both numerical and categorical features appropriately.
    Categorical (non-numerical) features are one-hot encoded by default, with an option
    for ordinal encoding when specified in integer_encoding.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to preprocess, containing both numerical and categorical features.
    integer_encoding : Dict[str, List[str]], optional
        Dictionary mapping column names to lists of categories in desired order.
        Example: {'severity': ['Mild', 'Moderate', 'Severe']}
    categorical_imputation_method : str, optional
        Method to use for imputing missing categorical values ('mode' or 'random').
        Defaults to 'mode'.
    numerical_imputation_method : str, optional
        Method to use for imputing missing numerical values, one of:
        - 'zeros': Replace missing values with zeros
        - 'random': Sample randomly from non-missing values
        - 'median': Replace missing values with median of non-missing values (default)
    drop_original_categorical : bool, optional
        Whether to drop original categorical columns after encoding.
        Defaults to True.
    random_state : int, optional
        Random seed for reproducibility when using random imputation.
        Defaults to 257 for consistent results. Set to None for truly random behavior.
    save_metadata_path : Union[str, Path], optional
        Path to save preprocessing metadata as JSON file.
        Includes encoding mappings, imputation values, and column statistics.
    reference_column_strategy : str, optional
        Strategy for selecting reference columns for one-hot encoded variables.
        Options: 'alphabetical', 'frequency', 'manual'. Defaults to 'alphabetical'.
    manual_reference_columns : Dict[str, str], optional
        Manual specification of reference columns for one-hot encoded variables.
        Example: {'diagn': 'diagn_Cardiomyopathy', 'recethcat': 'recethcat_Caucasian'}
        Only used when reference_column_strategy='manual'.
    drop_reference_columns : bool, optional
        Whether to drop reference columns from one-hot encoded variables.
        Defaults to True to avoid multicollinearity.
    dataset_prefix : str, optional
        Dataset identifier to include in metadata filename for dataset-specific tracking.
        If provided and save_metadata_path is None, metadata will be saved as
        'preprocessing_metadata_{dataset_prefix}_{timestamp}.json'.
        If None, saved as 'preprocessing_metadata_{timestamp}.json'.
    input_file_path : Union[str, Path], optional
        Path to the raw input data file for hash computation and provenance tracking.
        If provided, SHA256 hash and file size will be computed and included in metadata
        under 'data_provenance' section. This enables data integrity verification and
        reproducibility tracking. If None, data_provenance.input will be None.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]
        - Preprocessed DataFrame with imputed and encoded features
        - Dictionary containing preprocessing metadata:
            - numerical_imputation_method: Method used for numerical imputation
            - categorical_imputation_method: Method used for categorical imputation
            - encoding: Mapping of categorical encodings
            - numerical_stats: Statistics for numerical columns
            - imputation_values: Values used for imputation
            - column_types: Original data types of columns
            - onehot_group_manager: OneHotGroupManager instance (if one-hot encoding used)
            - data_provenance: Data integrity tracking with input file hash and metadata

    Examples
    --------
    >>> import pandas as pd
    >>> # Create sample data
    >>> data = pd.DataFrame({
    ...     'age': [25, None, 35],
    ...     'severity': ['Mild', None, 'Severe']
    ... })
    >>> # Define ordinal features
    >>> ordinal_map = {'severity': ['Mild', 'Moderate', 'Severe']}
    >>> # Preprocess data
    >>> processed_df, metadata = preprocess_data(
    ...     data,
    ...     integer_encoding=ordinal_map,
    ...     numerical_imputation_method='median',
    ...     save_metadata_path='preprocessing_metadata.json'
    ... )
    """
    logger.info("Starting data preprocessing")

    # Make a copy to avoid modifying the original
    processed_df = df.copy()

    preprocessing_metadata = {
        'numerical_imputation_method': numerical_imputation_method,
        'categorical_imputation_method': categorical_imputation_method,
        'encoding': {},
        'numerical_stats': {},
        'imputation_values': {},
        'column_types': {},
        'data_provenance': {'hash_algorithm': 'sha256', 'input': None, 'output': {}},
    }

    # Compute hash for input raw data file if path provided
    if input_file_path:
        from prism.data_utilities import compute_file_hash_and_size

        try:
            input_hash_info = compute_file_hash_and_size(input_file_path)
            preprocessing_metadata['data_provenance']['input'] = input_hash_info
            logger.info(f"Computed hash for input file: {input_file_path}")
        except Exception as e:
            logger.warning(f"Could not compute hash for input file: {str(e)}")

    # Identify non-numerical columns that need encoding
    non_num_cols = processed_df.select_dtypes(exclude=['number']).columns.tolist()

    # Record original column types
    for col in df.columns:
        if col in non_num_cols:
            preprocessing_metadata['column_types'][col] = 'categorical'
        else:
            preprocessing_metadata['column_types'][col] = 'numeric'

    if non_num_cols:
        logger.info(f"Found {len(non_num_cols)} non-numerical columns to encode")

        # Impute then encode categorical features
        processed_df, encoding_metadata = encode_categorical_features(
            processed_df,
            categorical_cols=non_num_cols,
            integer_encoding=integer_encoding,
            imputation_method=categorical_imputation_method,
            drop_original=drop_original_categorical,
            random_state=random_state,
        )

        # Add encoding metadata to our overall preprocessing metadata
        preprocessing_metadata['encoding'] = encoding_metadata

        for original_col, meta in encoding_metadata.items():
            logger.info(
                f"Encoded '{original_col}' ({meta['encoding_type']}) into: {', '.join(meta['created_columns'])}"
            )

        # Handle one-hot encoded reference columns
        if drop_reference_columns:
            logger.info("Processing one-hot encoded reference columns")

            # Detect one-hot groups and determine reference columns using encoding metadata
            categorical_groups, reference_columns_dict = detect_reference_columns(
                processed_df,
                encoding_metadata=encoding_metadata,
                reference_column_strategy=reference_column_strategy,
                manual_reference_columns=manual_reference_columns,
                min_group_size=2,
            )

            if categorical_groups:
                logger.info(f"Detected {len(categorical_groups)} one-hot groups")

                # Create OneHotGroupManager
                groups_dict = {}
                reference_columns = {}

                for base_name, dummy_cols in categorical_groups.items():
                    ref_col = reference_columns_dict[base_name]
                    # Only include non-reference columns in groups_dict
                    non_ref_cols = [col for col in dummy_cols if col != ref_col]
                    groups_dict[base_name] = non_ref_cols
                    reference_columns[base_name] = ref_col

                    logger.info(f"  {base_name}: dropping reference column '{ref_col}'")

                # Create OneHotGroupManager and store in metadata
                group_manager = OneHotGroupManager(groups_dict, reference_columns)
                preprocessing_metadata['onehot_group_manager'] = group_manager

                # Drop reference columns from the dataframe
                columns_to_drop = list(reference_columns.values())
                processed_df = processed_df.drop(columns=columns_to_drop)
                logger.info(f"Dropped {len(columns_to_drop)} reference columns")

                # Store reference column info in metadata for serialization
                preprocessing_metadata['reference_columns'] = {
                    'dropped_columns': columns_to_drop,
                    'groups': groups_dict,
                    'references': reference_columns,
                    'strategy': reference_column_strategy,
                }
            else:
                logger.info(
                    "No one-hot encoded groups detected (all categorical variables are ordinal)"
                )
                preprocessing_metadata['onehot_group_manager'] = None
        else:
            logger.info("Skipping reference column dropping (drop_reference_columns=False)")
            preprocessing_metadata['onehot_group_manager'] = None
    else:
        logger.info("No categorical columns to encode")
        preprocessing_metadata['onehot_group_manager'] = None

    # Impute missing values for numerical features
    numerical_cols = processed_df.select_dtypes(include=['number']).columns.tolist()
    logger.info(f"Processing {len(numerical_cols)} numerical features")

    # Log detailed information about numerical columns before imputation
    total_missing = 0
    cols_with_missing = 0

    for col in numerical_cols:
        missing_count = processed_df[col].isna().sum()
        total_missing += missing_count

        if missing_count > 0:
            cols_with_missing += 1

        # Calculate statistics before imputation
        stats = {
            'mean': (
                float(processed_df[col].mean()) if not pd.isna(processed_df[col].mean()) else None
            ),
            'median': (
                float(processed_df[col].median())
                if not pd.isna(processed_df[col].median())
                else None
            ),
            'std': (
                float(processed_df[col].std()) if not pd.isna(processed_df[col].std()) else None
            ),
            'min': (
                float(processed_df[col].min()) if not pd.isna(processed_df[col].min()) else None
            ),
            'max': (
                float(processed_df[col].max()) if not pd.isna(processed_df[col].max()) else None
            ),
            'missing_count': int(missing_count),
            'missing_pct': float(missing_count / len(processed_df) * 100),
        }

        preprocessing_metadata['numerical_stats'][col] = stats

        # Determine imputation value based on method
        imputation_value = None
        if numerical_imputation_method not in ['zeros', 'random', 'median']:
            raise ValueError("Method must be one of: 'zeros', 'random', 'median'")

        if numerical_imputation_method == 'median' and not pd.isna(processed_df[col].median()):
            imputation_value = float(processed_df[col].median())
        elif numerical_imputation_method == 'zeros':
            imputation_value = 0.0
        elif numerical_imputation_method == 'random':
            non_missing = processed_df[col].dropna()
            if not non_missing.empty:
                imputation_value = "random sampling from non-missing values"

        preprocessing_metadata['imputation_values'][col] = imputation_value

        # Log information about the column
        if missing_count > 0:
            logger.info(
                f"Column '{col}': {missing_count} missing values ({stats['missing_pct']:.2f}%)"
            )
            if imputation_value is not None:
                logger.info(
                    f"  - Will impute with {numerical_imputation_method}: {imputation_value if isinstance(imputation_value, str) else f'{imputation_value:.4g}'}"
                )
                logger.debug(
                    f"  - Before imputation: mean={stats['mean']:.4g}, median={stats['median']:.4g}, std={stats['std']:.4g}, range=[{stats['min']:.4g}, {stats['max']:.4g}]"
                )
        else:
            logger.debug(f"Column '{col}': No missing values")

    # Log summary information
    if total_missing > 0:
        logger.info(
            f"Total missing numerical values: {total_missing} in {cols_with_missing}/{len(numerical_cols)} columns ({total_missing/len(processed_df)/len(numerical_cols)*100:.2f}% overall)"
        )
        logger.info(
            f"Imputing missing numerical values using '{numerical_imputation_method}' method"
        )
    else:
        logger.info("No missing values found in numerical columns")

    # Use impute_numerical_values function for the actual imputation
    processed_df = impute_numerical_values(
        processed_df,
        method=numerical_imputation_method,
        verbose=False,  # Disable verbose output since we're using our own logging
        empty_values=['', ' ', 'NA', 'N/A', 'none', 'None', 'null', 'NULL', '?'],
        random_seed=random_state,
    )

    # Log statistics after imputation for columns that had missing values
    for col in numerical_cols:
        if preprocessing_metadata['numerical_stats'][col]['missing_count'] > 0:
            after_stats = {
                'mean': float(processed_df[col].mean()),
                'median': float(processed_df[col].median()),
                'std': float(processed_df[col].std()),
                'min': float(processed_df[col].min()),
                'max': float(processed_df[col].max()),
            }

            logger.debug(
                f"Column '{col}' after imputation: mean={after_stats['mean']:.4g}, median={after_stats['median']:.4g}, std={after_stats['std']:.4g}, range=[{after_stats['min']:.4g}, {after_stats['max']:.4g}]"
            )

            # Store after-imputation stats in metadata
            preprocessing_metadata['numerical_stats'][col]['after_imputation'] = after_stats

    logger.info("Numerical imputation completed")

    # Save metadata to a timestamped file in PROCESSED_DATA_DIR by default
    if save_metadata_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if dataset_prefix:
            filename = f"preprocessing_metadata_{dataset_prefix}_{timestamp}.json"
        else:
            filename = f"preprocessing_metadata_{timestamp}.json"
        save_path = Path(PROCESSED_DATA_DIR) / filename
    else:
        save_path = Path(save_metadata_path)
    try:
        # Convert metadata to JSON-serializable format (handle numpy types)
        json_compatible_metadata = _convert_to_json_serializable(preprocessing_metadata)

        # Save to file
        with open(save_path, 'w') as f:
            json.dump(json_compatible_metadata, f, indent=2)
        logger.info(f"Saved preprocessing metadata to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save metadata: {str(e)}")

    logger.info("Data preprocessing completed successfully")
    return processed_df, preprocessing_metadata


def _convert_to_json_serializable(obj):
    """Convert numpy and other non-serializable types to standard Python types."""
    if isinstance(obj, OneHotGroupManager):
        # Use the manager's to_dict method for serialization
        # This includes the explicit category_integer_mapping
        return obj.to_dict()
    elif isinstance(obj, dict):
        return {k: _convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return _convert_to_json_serializable(obj.tolist())
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def get_original_category(encoded_value, metadata, column_name):
    """
    Convert encoded values back to original categories

    Parameters:
    -----------
    encoded_value : int or float
        The encoded value to convert back
    metadata : dict
        The metadata dictionary returned from preprocess_data
    column_name : str
        The name of the original column

    Returns:
    --------
    The original category value
    """
    if column_name not in metadata['encoding']:
        raise ValueError(f"No encoding metadata found for column '{column_name}'")

    encoding_info = metadata['encoding'][column_name]

    if encoding_info['encoding_type'] == 'ordinal':
        if encoded_value in encoding_info['reverse_mapping']:
            return encoding_info['reverse_mapping'][encoded_value]
        else:
            return None
    else:
        raise ValueError(
            f"Function only works for ordinal encoded columns. '{column_name}' uses {encoding_info['encoding_type']} encoding"
        )


def create_label_map_for_plotting(metadata, column_name):
    """
    Create a label map for plotting ordinal features

    Parameters:
    -----------
    metadata : dict
        The metadata dictionary returned from preprocess_data
    column_name : str
        The name of the original column

    Returns:
    --------
    Dict mapping encoded values to original category labels
    """
    if column_name not in metadata['encoding']:
        raise ValueError(f"No encoding metadata found for column '{column_name}'")

    encoding_info = metadata['encoding'][column_name]

    if encoding_info['encoding_type'] == 'ordinal':
        return encoding_info['reverse_mapping']
    else:
        raise ValueError(
            f"Function only works for ordinal encoded columns. '{column_name}' uses {encoding_info['encoding_type']} encoding"
        )


def build_ordinal_labels_dict(
    metadata: Dict[str, Any],
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Dict[int, str]]:
    """
    Extract ordinal feature labels from preprocessing metadata for PlotFormatter.

    Converts ordinal feature reverse_mappings to format expected by PlotFormatter:
    {
        'employment_ordinal': {0: 'unemployed', 1: '<1', 2: '1<=X<4', ...},
        'savings_status_ordinal': {0: 'no known savings', 1: '<100', ...},
    }

    Parameters
    ----------
    metadata : Dict[str, Any]
        Preprocessing metadata containing 'encoding' key
    feature_names : Optional[List[str]]
        If provided, only extract labels for ordinal features in this list

    Returns
    -------
    Dict[str, Dict[int, str]]
        Mapping ordinal feature names to {int_value: label} dicts.
        Ready to be passed to PlotFormatter(categorical_labels=...).

    Examples
    --------
    >>> metadata = {'encoding': {'employment': {
    ...     'encoding_type': 'ordinal',
    ...     'created_columns': ['employment'],
    ...     'reverse_mapping': {'0': 'unemployed', '1': '<1'}
    ... }}}
    >>> build_ordinal_labels_dict(metadata)
    {'employment': {0: 'unemployed', 1: '<1'}}
    """
    if 'encoding' not in metadata:
        logger.warning("No 'encoding' key found in metadata")
        return {}

    ordinal_labels = {}

    for original_col, encoding_info in metadata['encoding'].items():
        # Skip non-ordinal encodings
        if encoding_info.get('encoding_type') != 'ordinal':
            continue

        # Get created column name (e.g., 'employment_ordinal')
        created_columns = encoding_info.get('created_columns', [])
        if not created_columns:
            logger.warning(f"Ordinal feature '{original_col}' has no created_columns")
            continue

        ordinal_col_name = created_columns[0]

        # Filter by feature_names if provided
        if feature_names is not None and ordinal_col_name not in feature_names:
            continue

        # Convert {"0": "label", "1": "label"} to {0: "label", 1: "label"}
        reverse_mapping = encoding_info.get('reverse_mapping', {})
        if not reverse_mapping:
            logger.warning(f"Ordinal feature '{original_col}' has no reverse_mapping")
            continue

        int_mapping = {}
        for str_key, label in reverse_mapping.items():
            try:
                int_key = int(str_key)
                int_mapping[int_key] = label
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Could not convert key '{str_key}' to int for '{ordinal_col_name}': {e}"
                )

        if int_mapping:
            ordinal_labels[ordinal_col_name] = int_mapping
            logger.debug(
                f"Extracted {len(int_mapping)} labels for ordinal feature '{ordinal_col_name}'"
            )

    logger.info(f"Built ordinal labels dict with {len(ordinal_labels)} features")
    return ordinal_labels


class OneHotGroupManager:
    """
    Manages one-hot encoded groups throughout the PRiSM pipeline.

    Provides robust group tracking using feature names as the source of truth,
    with conversion to indices for computational efficiency.

    This class is specifically designed to support partial response collapsing,
    where one-hot encoded categorical variables need to be collapsed back to
    single categorical features.

    Attributes
    ----------
    groups_dict : Dict[str, List[str]]
        Dictionary mapping group names to lists of feature names
    reference_columns : Dict[str, str]
        Dictionary mapping group names to reference column names
    """

    def __init__(
        self, groups_dict: Dict[str, List[str]], reference_columns: Optional[Dict[str, str]] = None
    ):
        """
        Initialize OneHotGroupManager.

        Parameters
        ----------
        groups_dict : Dict[str, List[str]]
            Dictionary mapping group names to lists of feature names.
            Example: {'diagn': ['diagn_CAD', 'diagn_Congenital', ...]}
        reference_columns : Optional[Dict[str, str]]
            Dictionary mapping group names to reference column names.
            Example: {'diagn': 'diagn_Cardiomyopathy'}
        """
        self.groups_dict = groups_dict
        self.reference_columns = reference_columns or {}

    @classmethod
    def from_preprocessing_metadata(cls, metadata: Dict) -> 'OneHotGroupManager':
        """
        Create manager from preprocessing metadata.

        Parameters
        ----------
        metadata : Dict
            Preprocessing metadata dictionary containing either:
            - 'onehot_group_manager' key with serialized manager data
            - Legacy 'onehot_groups' and 'reference_columns' keys

        Returns
        -------
        OneHotGroupManager
            Initialized manager instance
        """
        # Check for new serialized format first
        if 'onehot_group_manager' in metadata and metadata['onehot_group_manager'] is not None:
            manager_data = metadata['onehot_group_manager']
            if isinstance(manager_data, dict) and '_type' in manager_data:
                # Serialized format from JSON
                groups_dict = manager_data.get('groups_dict', {})
                reference_columns = manager_data.get('reference_columns', {})
            else:
                # Already a manager instance
                return manager_data
        elif (
            'reference_columns' in metadata
            and isinstance(metadata['reference_columns'], dict)
            and 'groups' in metadata['reference_columns']
        ):
            # New reference_columns structure (from preprocess_data with drop_reference_columns=True)
            ref_cols_data = metadata['reference_columns']
            groups_dict = ref_cols_data.get('groups', {})
            reference_columns = ref_cols_data.get('references', {})
        else:
            # Legacy format: direct 'onehot_groups' and 'reference_columns' keys
            groups_dict = metadata.get('onehot_groups', {})
            reference_columns = metadata.get('reference_columns', {})

        return cls(groups_dict, reference_columns)

    def to_indices(self, feature_names: List[str]) -> Optional[List[Tuple[int, ...]]]:
        """
        Convert group structure from feature names to indices.

        Parameters
        ----------
        feature_names : List[str]
            Current feature names in order

        Returns
        -------
        Optional[List[Tuple[int, ...]]]
            Groups as tuples of feature indices, or None if no groups
        """
        groups_indices = []

        for group_name, group_features in self.groups_dict.items():
            indices = []
            for feature_name in group_features:
                try:
                    idx = feature_names.index(feature_name)
                    indices.append(idx)
                except ValueError:
                    logger.warning(
                        f"Feature '{feature_name}' from group '{group_name}' "
                        f"not found in current feature list. Skipping."
                    )

            if len(indices) >= 2:  # Only include if at least 2 members remain
                groups_indices.append(tuple(sorted(indices)))
            else:
                logger.warning(
                    f"Group '{group_name}' has fewer than 2 members after filtering. "
                    f"Excluding from groups."
                )

        return groups_indices if groups_indices else None

    def create_collapsed_scaler(
        self, original_scaler: 'PRiSMScaler', feature_names: List[str]
    ) -> 'PRiSMScaler':
        """
        Create a collapsed scaler for use with collapsed features.

        When features are collapsed, the scaler needs to match the new feature dimensions.
        For collapsed categorical features, we use identity scaling (median=0, std=1) since
        they don't need denormalization. For continuous features, we copy the original
        scaling parameters.

        Parameters
        ----------
        original_scaler : PRiSMScaler
            Original scaler fit on uncollapsed features
        feature_names : List[str]
            Original feature names (before collapse)

        Returns
        -------
        PRiSMScaler
            New scaler matching collapsed feature dimensions

        Raises
        ------
        ValueError
            If scaler dimensions don't match feature_names length
        """
        from prism.preprocessing import MedianStdScaler, NoScaler

        # Validate scaler dimensions before proceeding
        scaler_dim = None
        if hasattr(original_scaler, 'scaler') and original_scaler.scaler is not None:
            if hasattr(original_scaler.scaler, 'median_'):
                scaler_dim = len(original_scaler.scaler.median_)
            elif hasattr(original_scaler.scaler, 'mean_'):
                scaler_dim = len(original_scaler.scaler.mean_)

        if scaler_dim is not None and scaler_dim != len(feature_names):
            raise ValueError(
                f"Scaler has {scaler_dim} dimensions, but feature_names has "
                f"{len(feature_names)} names. Ensure the scaler was fit on the same "
                f"features (in OHE space) before calling create_collapsed_scaler()."
            )

        # Get collapsed feature names and create index mapping
        _, collapsed_names = collapse_onehot_features(
            np.zeros((1, len(feature_names))), self, feature_names
        )
        n_collapsed = len(collapsed_names)

        # If original scaler is NoScaler, return NoScaler
        if isinstance(original_scaler.scaler, NoScaler):
            return original_scaler

        # Create new scaler with collapsed dimensions
        if isinstance(original_scaler.scaler, MedianStdScaler):
            # Extract original parameters
            orig_median = original_scaler.scaler.median_
            orig_std = original_scaler.scaler.std_
            orig_sd_scale = original_scaler.scaler.sd_scale

            # Initialize collapsed parameters
            collapsed_median = np.zeros(n_collapsed)
            # For identity scaling, std should account for sd_scale
            # inverse_transform: x * (std * sd_scale) + median = x requires std = 1/sd_scale
            identity_std = 1.0 / orig_sd_scale if orig_sd_scale != 0 else 1.0
            collapsed_std = np.full(n_collapsed, identity_std)

            # Map features from original to collapsed
            collapsed_idx = 0
            for name in collapsed_names:
                if name in self.groups_dict:
                    # This is a collapsed categorical feature - use identity scaling
                    # Keep the identity_std value already set
                    collapsed_median[collapsed_idx] = 0.0
                else:
                    # This is a continuous feature - copy original scaling
                    try:
                        orig_idx = feature_names.index(name)
                        collapsed_median[collapsed_idx] = orig_median[orig_idx]
                        collapsed_std[collapsed_idx] = orig_std[orig_idx]
                    except ValueError:
                        # Feature not found, use identity
                        collapsed_median[collapsed_idx] = 0.0
                        collapsed_std[collapsed_idx] = 1.0
                collapsed_idx += 1

            # Create new scaler with collapsed parameters
            new_scaler = MedianStdScaler(sd_scale=orig_sd_scale)
            new_scaler.median_ = collapsed_median
            new_scaler.std_ = collapsed_std
            new_scaler.sd_scale = orig_sd_scale

            return PRiSMScaler(new_scaler)
        elif hasattr(original_scaler.scaler, 'mean_') and hasattr(
            original_scaler.scaler, 'scale_'
        ):
            # StandardScaler or similar sklearn scaler
            from copy import deepcopy

            # Get original parameters
            orig_mean = original_scaler.scaler.mean_
            orig_scale = original_scaler.scaler.scale_

            # Initialize collapsed parameters
            collapsed_mean = np.zeros(n_collapsed)
            collapsed_scale = np.ones(n_collapsed)

            # Map features from original to collapsed
            collapsed_idx = 0
            for name in collapsed_names:
                if name in self.groups_dict:
                    # This is a collapsed categorical feature - use identity scaling
                    collapsed_mean[collapsed_idx] = 0.0
                    collapsed_scale[collapsed_idx] = 1.0
                else:
                    # This is a continuous feature - copy original scaling
                    try:
                        orig_idx = feature_names.index(name)
                        collapsed_mean[collapsed_idx] = orig_mean[orig_idx]
                        collapsed_scale[collapsed_idx] = orig_scale[orig_idx]
                    except ValueError:
                        # Feature not found, use identity
                        collapsed_mean[collapsed_idx] = 0.0
                        collapsed_scale[collapsed_idx] = 1.0
                collapsed_idx += 1

            # Create new scaler with collapsed parameters
            new_sklearn_scaler = deepcopy(original_scaler.scaler)
            new_sklearn_scaler.mean_ = collapsed_mean
            new_sklearn_scaler.scale_ = collapsed_scale
            new_sklearn_scaler.n_features_in_ = n_collapsed

            return PRiSMScaler(new_sklearn_scaler)
        elif hasattr(original_scaler.scaler, 'center_') and hasattr(
            original_scaler.scaler, 'scale_'
        ):
            # RobustScaler or similar sklearn scaler (uses median_ in some versions, center_ in others)
            from copy import deepcopy

            # Get original parameters
            if hasattr(original_scaler.scaler, 'median_'):
                orig_center = original_scaler.scaler.median_
            else:
                orig_center = original_scaler.scaler.center_
            orig_scale = original_scaler.scaler.scale_

            # Initialize collapsed parameters
            collapsed_center = np.zeros(n_collapsed)
            collapsed_scale = np.ones(n_collapsed)

            # Map features from original to collapsed
            collapsed_idx = 0
            for name in collapsed_names:
                if name in self.groups_dict:
                    # This is a collapsed categorical feature - use identity scaling
                    collapsed_center[collapsed_idx] = 0.0
                    collapsed_scale[collapsed_idx] = 1.0
                else:
                    # This is a continuous feature - copy original scaling
                    try:
                        orig_idx = feature_names.index(name)
                        collapsed_center[collapsed_idx] = orig_center[orig_idx]
                        collapsed_scale[collapsed_idx] = orig_scale[orig_idx]
                    except ValueError:
                        # Feature not found, use identity
                        collapsed_center[collapsed_idx] = 0.0
                        collapsed_scale[collapsed_idx] = 1.0
                collapsed_idx += 1

            # Create new scaler with collapsed parameters
            new_sklearn_scaler = deepcopy(original_scaler.scaler)
            if hasattr(new_sklearn_scaler, 'median_'):
                new_sklearn_scaler.median_ = collapsed_center
            new_sklearn_scaler.center_ = collapsed_center
            new_sklearn_scaler.scale_ = collapsed_scale
            new_sklearn_scaler.n_features_in_ = n_collapsed

            return PRiSMScaler(new_sklearn_scaler)
        else:
            # Unknown scaler type, return a NoScaler
            logger.warning(
                f"Unknown scaler type {type(original_scaler.scaler)}, returning NoScaler"
            )
            return PRiSMScaler(NoScaler())

    def expand_mask_to_original_features(
        self, collapsed_mask: Union[np.ndarray, torch.Tensor], feature_names: List[str]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Expand a mask from collapsed feature space back to original one-hot feature space.

        When LASSO is performed on collapsed features, the mask is (n_collapsed, n_selected).
        To train PRN on original one-hot data, we need to expand it to (n_original, n_selected).

        Parameters
        ----------
        collapsed_mask : np.ndarray or torch.Tensor
            Mask with shape (n_collapsed_features, n_selected_responses)
        feature_names : List[str]
            Original feature names (before collapse)

        Returns
        -------
        np.ndarray or torch.Tensor
            Expanded mask with shape (n_original_features, n_selected_responses)
        """
        import torch

        # Convert to numpy for processing
        is_torch = isinstance(collapsed_mask, torch.Tensor)
        if is_torch:
            mask_np = collapsed_mask.cpu().numpy()
            device = collapsed_mask.device
        else:
            mask_np = np.asarray(collapsed_mask)

        # Get collapsed feature names
        _, collapsed_names = collapse_onehot_features(
            np.zeros((1, len(feature_names))), self, feature_names
        )

        n_original = len(feature_names)
        n_selected = mask_np.shape[1]
        expanded_mask = np.zeros((n_original, n_selected), dtype=mask_np.dtype)

        # Map each original feature to its collapsed counterpart
        for orig_idx, orig_name in enumerate(feature_names):
            # Check if this feature is part of a one-hot group
            group_name = None
            for gname, members in self.groups_dict.items():
                if orig_name in members:
                    group_name = gname
                    break

            if group_name:
                # This is a one-hot feature - map to the collapsed group
                collapsed_idx = collapsed_names.index(group_name)
            else:
                # This is a regular feature - map directly
                collapsed_idx = collapsed_names.index(orig_name)

            # Copy the mask row from collapsed to original
            expanded_mask[orig_idx, :] = mask_np[collapsed_idx, :]

        # Convert back to torch if needed
        if is_torch:
            return torch.tensor(expanded_mask, dtype=collapsed_mask.dtype, device=device)
        else:
            return expanded_mask

    def is_categorical_group(self, feature_name: str) -> bool:
        """
        Check if a collapsed feature name is a one-hot group (categorical by definition).

        Parameters
        ----------
        feature_name : str
            Feature name to check (should be a collapsed/group name)

        Returns
        -------
        bool
            True if feature_name is a one-hot group, False otherwise

        Examples
        --------
        >>> manager = OneHotGroupManager({'diagn': ['diagn_CAD', 'diagn_Congenital']})
        >>> manager.is_categorical_group('diagn')
        True
        >>> manager.is_categorical_group('age')
        False
        """
        return feature_name in self.groups_dict

    def get_category_integer_mapping(self, group_name: str) -> Dict[int, str]:
        """
        Get the mapping from category integer to column name for a group.

        This is the SINGLE SOURCE OF TRUTH for how categories are encoded as integers.
        The encoding follows this convention:
        - Category 0: Reference column (from reference_columns)
        - Category 1: First member of groups_dict[group_name]
        - Category 2: Second member of groups_dict[group_name]
        - etc.

        This ordering is used by:
        - collapse_onehot_features() for encoding during preprocessing
        - PartialResponseCalculator for expanding during analysis
        - get_category_labels() for display

        Parameters
        ----------
        group_name : str
            Name of the one-hot group (e.g., 'diagn', 'race')

        Returns
        -------
        Dict[int, str]
            Mapping from category integer to column name.
            Example: {0: 'diagn_Cardiomyopathy', 1: 'diagn_CAD', 2: 'diagn_Congenital'}

        Raises
        ------
        KeyError
            If group_name is not in groups_dict

        Examples
        --------
        >>> manager = OneHotGroupManager(
        ...     {'diagn': ['diagn_CAD', 'diagn_Congenital']},
        ...     {'diagn': 'diagn_Cardiomyopathy'}
        ... )
        >>> manager.get_category_integer_mapping('diagn')
        {0: 'diagn_Cardiomyopathy', 1: 'diagn_CAD', 2: 'diagn_Congenital'}
        """
        if group_name not in self.groups_dict:
            raise KeyError(f"Group '{group_name}' not found in groups_dict")

        mapping = {}

        # Category 0: Reference column (if available)
        ref_col = self.reference_columns.get(group_name)
        if ref_col:
            mapping[0] = ref_col

        # Categories 1-N: Members in groups_dict order
        for i, member_col in enumerate(self.groups_dict[group_name]):
            mapping[i + 1] = member_col

        return mapping

    def get_category_labels(
        self,
        group_name: str,
        label_manager: Optional['FeatureLabelManager'] = None,
    ) -> Dict[int, str]:
        """
        Get mapping from category integer to display label.

        This method provides the category labels for plotting and display,
        using the same integer encoding as get_category_integer_mapping().

        Label priority (fallback hierarchy):
        1. User-provided label from label_manager (if available)
        2. Category suffix extracted from column name (e.g., 'CAD' from 'diagn_CAD')
        3. Column name itself (fallback)

        Parameters
        ----------
        group_name : str
            Name of the one-hot group
        label_manager : Optional[FeatureLabelManager]
            Label manager for looking up user-defined labels

        Returns
        -------
        Dict[int, str]
            Mapping from category integer to display label.
            Example: {0: 'Cardiomyopathy', 1: 'CAD', 2: 'Congenital'}

        Examples
        --------
        >>> manager = OneHotGroupManager(
        ...     {'diagn': ['diagn_CAD', 'diagn_Congenital']},
        ...     {'diagn': 'diagn_Cardiomyopathy'}
        ... )
        >>> manager.get_category_labels('diagn')
        {0: 'Cardiomyopathy', 1: 'CAD', 2: 'Congenital'}
        """
        int_to_col = self.get_category_integer_mapping(group_name)
        labels = {}

        for cat_int, col_name in int_to_col.items():
            if label_manager is not None:
                # Try to get user-defined label
                user_label = label_manager.get_label(col_name)
                if user_label != col_name:  # get_label returns col_name if not found
                    labels[cat_int] = user_label
                    continue

            # Fall back to extracting suffix from column name
            labels[cat_int] = self._extract_category_suffix(col_name, group_name)

        return labels

    def _extract_category_suffix(self, column_name: str, group_name: str) -> str:
        """
        Extract category name from column name.

        Parameters
        ----------
        column_name : str
            Full column name (e.g., 'diagn_CAD')
        group_name : str
            Group name prefix (e.g., 'diagn')

        Returns
        -------
        str
            Category suffix (e.g., 'CAD')
        """
        prefix = f"{group_name}_"
        if column_name.startswith(prefix):
            return column_name[len(prefix) :]
        return column_name

    def build_categorical_labels_dict(
        self,
        label_manager: Optional['FeatureLabelManager'] = None,
    ) -> Dict[str, Dict[int, str]]:
        """
        Build categorical_labels dict for all groups for use with PlotFormatter.

        This method generates the categorical labels dictionary in the format
        expected by PlotFormatter.categorical_labels:

        {
            'diagn': {0: 'Cardiomyopathy', 1: 'CAD', 2: 'Congenital', ...},
            'recethcat': {0: 'Caucasian', 1: 'African American', ...},
        }

        Parameters
        ----------
        label_manager : Optional[FeatureLabelManager]
            Label manager for looking up user-defined labels.
            If None, uses extracted suffixes from column names.

        Returns
        -------
        Dict[str, Dict[int, str]]
            Dictionary mapping group names to {int: label} mappings.
            Ready to be passed to PlotFormatter(categorical_labels=...).

        Examples
        --------
        >>> manager = OneHotGroupManager(
        ...     {'diagn': ['diagn_CAD', 'diagn_Congenital']},
        ...     {'diagn': 'diagn_Cardiomyopathy'}
        ... )
        >>> cat_labels = manager.build_categorical_labels_dict()
        >>> cat_labels
        {'diagn': {0: 'Cardiomyopathy', 1: 'CAD', 2: 'Congenital'}}
        >>>
        >>> # Use with PlotFormatter:
        >>> formatter = PlotFormatter(categorical_labels=cat_labels)
        """
        return {
            group_name: self.get_category_labels(group_name, label_manager)
            for group_name in self.groups_dict
        }

    def to_dict(self) -> Dict:
        """
        Serialize the manager to a dictionary for JSON storage.

        The output includes:
        - groups_dict: The one-hot group structure
        - reference_columns: Reference column for each group
        - category_integer_mapping: Explicit integer-to-column mapping for each group

        The category_integer_mapping is the single source of truth for how
        categories are encoded as integers throughout the pipeline.

        Returns
        -------
        Dict
            Dictionary suitable for JSON serialization

        Examples
        --------
        >>> manager = OneHotGroupManager(
        ...     {'diagn': ['diagn_CAD', 'diagn_Congenital']},
        ...     {'diagn': 'diagn_Cardiomyopathy'}
        ... )
        >>> manager.to_dict()
        {
            '_type': 'OneHotGroupManager',
            'groups_dict': {'diagn': ['diagn_CAD', 'diagn_Congenital']},
            'reference_columns': {'diagn': 'diagn_Cardiomyopathy'},
            'category_integer_mapping': {
                'diagn': {0: 'diagn_Cardiomyopathy', 1: 'diagn_CAD', 2: 'diagn_Congenital'}
            }
        }
        """
        return {
            '_type': 'OneHotGroupManager',
            'groups_dict': self.groups_dict,
            'reference_columns': self.reference_columns,
            'category_integer_mapping': {
                group_name: self.get_category_integer_mapping(group_name)
                for group_name in self.groups_dict
            },
        }


def collapse_onehot_features(
    X: Union[np.ndarray, torch.Tensor, pd.DataFrame],
    group_manager: OneHotGroupManager,
    feature_names: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """
    Collapse one-hot encoded features to categorical integer encoding.

    Converts one-hot groups from N binary columns to 1 integer column per group.
    Non-grouped features are copied unchanged.

    IMPORTANT: This function expects UNSCALED data. When data is scaled
    (especially with MedianStdScaler), majority categories (>50% prevalence)
    can have their "active" value scaled to 0, making it impossible to
    distinguish from reference categories using argmax.

    For plotting pipelines, ensure you call scaler.inverse_transform() on
    your data before passing it to this function.

    Parameters
    ----------
    X : array-like
        Input data with one-hot encoding, shape (n_samples, n_features).
        MUST be in unscaled (original) form.
    group_manager : OneHotGroupManager
        Manager containing group structure and reference info
    feature_names : List[str]
        Feature names matching X columns

    Returns
    -------
    X_collapsed : np.ndarray
        Collapsed data, shape (n_samples, n_collapsed_features)
    collapsed_feature_names : List[str]
        Names of collapsed features

    Raises
    ------
    ValueError
        If data appears to be scaled (contains negative values in one-hot columns)

    Examples
    --------
    >>> # One-hot: [1,0,0] for group becomes categorical: 1
    >>> # Reference: [0,0,0] for group becomes categorical: 0
    >>> manager = OneHotGroupManager(
    ...     {'diagn': ['diagn_CAD', 'diagn_Congenital']},
    ...     {'diagn': 'diagn_Cardiomyopathy'}
    ... )
    >>> X = np.array([[1, 0], [0, 1], [0, 0]])
    >>> X_collapsed, names = collapse_onehot_features(X, manager, ['diagn_CAD', 'diagn_Congenital'])
    >>> X_collapsed
    array([[1], [2], [0]])
    >>> names
    ['diagn']
    """
    # Convert to numpy for easier manipulation
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    elif isinstance(X, pd.DataFrame):
        X = X.values

    n_samples = X.shape[0]
    collapsed_features = []
    collapsed_names = []

    grouped_indices = set()

    # Collapse one-hot groups
    for group_name, group_features in group_manager.groups_dict.items():
        indices = [feature_names.index(f) for f in group_features]
        grouped_indices.update(indices)

        # Get group data
        group_data = X[:, indices]  # (n_samples, len(indices))

        # Validate: one-hot data should only contain 0s and 1s (or very close to it)
        # This catches cases where scaled data was accidentally passed
        unique_vals = np.unique(group_data)
        if np.any(unique_vals < -0.01):
            raise ValueError(
                f"collapse_onehot_features received data with negative values in "
                f"one-hot group '{group_name}'. This suggests the data is scaled. "
                f"One-hot columns should only contain 0 and 1 values. "
                f"Call scaler.inverse_transform() before using this function."
            )

        # Create categorical column using simple argmax approach
        # For unscaled one-hot data:
        # - Active column has value 1, others have 0
        # - argmax returns the index of the 1 (or 0 if all zeros = reference)
        # We add 1 to make categories 1-indexed, with 0 reserved for reference
        categorical_col = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            row = group_data[i]
            max_val = np.max(row)

            # If max value is close to 1, we have an active category
            # If max value is close to 0, this is the reference category
            if max_val > 0.5:  # Threshold for detecting active column
                active_idx = np.argmax(row)
                categorical_col[i] = active_idx + 1
            # else: stays 0 (reference category)

        collapsed_features.append(categorical_col)
        collapsed_names.append(group_name)

    # Add non-grouped features
    for orig_idx, fname in enumerate(feature_names):
        if orig_idx not in grouped_indices:
            collapsed_features.append(X[:, orig_idx])
            collapsed_names.append(fname)

    X_collapsed = np.column_stack(collapsed_features)
    return X_collapsed, collapsed_names


def detect_reference_columns(
    df,
    encoding_metadata,
    reference_column_strategy='alphabetical',
    manual_reference_columns=None,
    min_group_size=2,
):
    """
    Detect one-hot encoded categorical variable groups and identify reference columns to drop.

    Uses encoding_metadata from encode_categorical_features() as the authoritative source
    of one-hot groups. This ensures correct group detection based on actual categorical
    encoding, avoiding false positives from features that happen to share a prefix.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the one-hot encoded columns
    encoding_metadata : dict
        Encoding metadata from encode_categorical_features(). This is the authoritative
        source for determining which columns are one-hot encoded.
    reference_column_strategy : str, optional
        Method for selecting reference columns. Options:
        - 'alphabetical': Select alphabetically first column as reference (default)
        - 'most_common': Select the most frequent category in training data as reference
        - 'manual': Use manually specified reference columns from manual_reference_columns
    manual_reference_columns : dict, optional
        Dictionary mapping base variable names to reference column names.
        Only used when reference_column_strategy='manual'.
        Example: {'diagn': 'diagn_Cardiomyopathy', 'recethcat': 'recethcat_Caucasian'}
    min_group_size : int, default=2
        Minimum number of features required to form a one-hot group

    Returns
    -------
    categorical_groups : dict
        Dictionary mapping original column names to lists of one-hot encoded column names
    reference_columns : dict
        Dictionary mapping original column names to selected reference column names

    Examples
    --------
    >>> # First encode categorical features
    >>> df_encoded, encoding_metadata = encode_categorical_features(df)

    >>> # Then detect reference columns using the metadata
    >>> groups, refs = detect_reference_columns(df_encoded, encoding_metadata)

    >>> # Most common category selection
    >>> groups, refs = detect_reference_columns(
    ...     df_encoded, encoding_metadata, reference_column_strategy='most_common'
    ... )

    >>> # Manual specification
    >>> manual_refs = {'diagn': 'diagn_Cardiomyopathy', 'recethcat': 'recethcat_Caucasian'}
    >>> groups, refs = detect_reference_columns(
    ...     df_encoded,
    ...     encoding_metadata,
    ...     reference_column_strategy='manual',
    ...     manual_reference_columns=manual_refs
    ... )
    """
    if reference_column_strategy not in ['alphabetical', 'most_common', 'manual']:
        raise ValueError(
            f"reference_column_strategy must be 'alphabetical', 'most_common', or 'manual', "
            f"got '{reference_column_strategy}'"
        )

    if reference_column_strategy == 'manual' and manual_reference_columns is None:
        raise ValueError(
            "manual_reference_columns must be provided when reference_column_strategy='manual'"
        )

    categorical_groups = {}
    reference_columns = {}

    # Extract one-hot groups from encoding metadata (the authoritative source)
    for original_col, meta in encoding_metadata.items():
        # Skip non-one-hot encoded columns (e.g., ordinal)
        if meta.get('encoding_type') != 'one-hot':
            continue

        created_cols = meta.get('created_columns', [])
        if len(created_cols) < min_group_size:
            continue

        # Store the group using original column name as the key
        categorical_groups[original_col] = created_cols

        # Determine reference column based on selection method
        if reference_column_strategy == 'alphabetical':
            # Select alphabetically first column
            reference_column = sorted(created_cols)[0]

        elif reference_column_strategy == 'most_common':
            # Find the most frequent category in the data
            category_counts = {}
            for col in created_cols:
                # Count how many times this category (column) is active (equals 1)
                if col in df.columns:
                    category_counts[col] = (df[col] == 1).sum()

            if not category_counts:
                # Fallback to alphabetical if no columns found in df
                reference_column = sorted(created_cols)[0]
            else:
                # Select the column with the highest count
                reference_column = max(category_counts, key=category_counts.get)

        elif reference_column_strategy == 'manual':
            # Use manually specified reference column
            if original_col not in manual_reference_columns:
                raise ValueError(
                    f"Manual reference column not specified for '{original_col}'. "
                    f"Available columns: {created_cols}"
                )
            reference_column = manual_reference_columns[original_col]

            # Validate the specified reference column
            if reference_column not in created_cols:
                raise ValueError(
                    f"Specified reference column '{reference_column}' not in group '{original_col}'. "
                    f"Available columns: {created_cols}"
                )

        reference_columns[original_col] = reference_column
        logger.debug(
            f"Selected reference column for '{original_col}': {reference_column} "
            f"(strategy: {reference_column_strategy})"
        )

    if not categorical_groups:
        logger.info("No one-hot encoded groups found in encoding metadata")

    return categorical_groups, reference_columns
