import hashlib
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from tableone import TableOne

from prism.config import PROJ_ROOT, RAW_DATA_DIR

logger = logging.getLogger(__name__)


# Statistical overview of features
def feature_summary(data: pd.DataFrame, categorical_threshold: int = 15) -> pd.DataFrame:
    """
    Generate a statistical summary of features in a DataFrame. All columns must be numerical.

    Args:
        data: Input DataFrame where all columns must be numerical
        categorical_threshold: Maximum number of unique values for a column to be considered categorical

    Returns:
        DataFrame containing statistical summaries for all columns

    Raises:
        ValueError: If any non-numerical columns are present in the DataFrame
    """
    # Check for non-numerical columns
    non_numerical = data.select_dtypes(exclude=['number']).columns
    if len(non_numerical) > 0:
        logging.warning(f"Found non-numerical columns: {list(non_numerical)}")
        raise ValueError(
            "All columns must be numerical. Please convert or remove non-numerical columns before proceeding."
        )

    summary = pd.DataFrame(
        {
            'Data Type': data.dtypes,
            'Non-Null Count': data.count(),
            'Null Count': data.isnull().sum(),
            'Mean': data.mean(),
            'Median': data.median(),
            'Std Dev': data.std(),
            'IQR': data.quantile(0.75) - data.quantile(0.25),
            'Min': data.min(),
            'Max': data.max(),
            'Unique Values': data.nunique(),
        }
    )
    summary['Is Categorical'] = summary['Unique Values'] < categorical_threshold
    return summary


def plot_feature_histograms(X, feature_stats, feature_names=None, figsize=(20, 20), bins="auto"):
    """
    Plot histograms for all features in X, distinguishing between categorical and continuous features.

    Parameters:
    X (pd.DataFrame): The dataset containing the features
    feature_stats (pd.DataFrame): DataFrame containing feature statistics including 'Is Categorical'
    feature_names (list): Optional list of feature names to use as labels
    figsize (tuple): Figure size (width, height) in inches
    """
    num_features = X.shape[1]
    num_cols = min(3, num_features)  # Maximum 3 columns
    num_rows = math.ceil(num_features / num_cols)

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=figsize)

    if num_features == 1:
        axes = np.array([axes])  # Ensure axes is always a 2D array
    axes = axes.flatten()  # Flatten axes array for easy indexing

    # Use feature_names if provided, otherwise use column names
    if feature_names is None:
        feature_names = X.columns

    for idx, (column, is_categorical, feature_name) in enumerate(
        zip(X.columns, feature_stats['Is Categorical'], feature_names)
    ):
        ax = axes[idx]
        if is_categorical:
            sns.countplot(x=column, data=X, ax=ax, color='lightgreen')
            feature_name = f'{feature_name} (Categorical)'
        else:
            sns.histplot(X[column], ax=ax, kde=True, color='skyblue', bins=bins)
            feature_name = f'{feature_name} (Continuous)'

        ax.set_xlabel(feature_name, fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=12)

        # Rotate x-axis labels if they are too long
        if max([len(str(item.get_text())) for item in ax.get_xticklabels()]) > 6:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    # Remove any unused subplots
    for idx in range(num_features, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.show()
    return plt


def find_categorical_variables(df: pd.DataFrame, threshold: int = 15) -> List[str]:
    """
    Identify categorical variables in a DataFrame based on the number of unique values.

    Args:
    df (pd.DataFrame): The input DataFrame
    threshold (int): The maximum number of unique values for a variable to be considered categorical

    Returns:
    List[str]: A list of column names identified as categorical variables
    """
    categorical_vars = []
    for column in df.columns:
        if df[column].nunique() <= threshold:
            categorical_vars.append(column)
    return categorical_vars


def format_label(label, sig_digits=3):
    try:
        # Try to convert to float and format
        num = float(label)
        return f'{num:.{sig_digits}g}'
    except ValueError:
        # If it's not a number, return the original label
        return str(label)


def plot_distributions(
    X,
    feature_names=None,
    categorical_threshold=15,
    subfigsize=(3, 4),
    plot_cols=4,
    violin_kwargs={
        'split': True,
        'inner': 'quartile',
        'cut': 0,
    },
):
    if feature_names is None:
        feature_names = X.columns.to_list()
    n_features = len(feature_names)
    n_cols = min(plot_cols, n_features)
    n_rows = math.ceil(n_features / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(subfigsize[0] * n_cols, subfigsize[1] * n_rows)
    )
    if n_features == 1:
        axes = np.array([axes])
    axes = axes.flatten()  # Flatten axes array for easy indexing

    for i, feature in enumerate(feature_names):
        data = X[feature]

        # Count missing values
        missing_count = data.isna().sum()
        total_count = len(data)
        missing_percentage = (missing_count / total_count) * 100

        # Remove missing values for analysis
        data_clean = data.dropna()

        is_categorical = len(data_clean.unique()) < categorical_threshold

        if is_categorical:
            # For categorical variables, use a bar plot
            counts = data_clean.value_counts(normalize=True).sort_index()

            # Format the index labels
            formatted_index = [format_label(idx) for idx in counts.index]

            axes[i].bar(formatted_index, counts.values, color='skyblue')

            axes[i].set_xticks(range(len(formatted_index)))
            axes[i].set_xticklabels(formatted_index, rotation=45, ha='right')
            axes[i].set_ylabel('Density')
            axes[i].set_xlabel('')
        else:
            # For continuous variables, use a violin plot
            sns.violinplot(y=data_clean, ax=axes[i], color='skyblue', **violin_kwargs)
            axes[i].set_ylabel('Value')
            axes[i].set_xlabel('Density')

        if missing_count > 0:
            axes[i].set_title(f"{feature}\n{missing_percentage:.1f}% missing")
        else:
            axes[i].set_title(f"{feature}")

    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_train_test_distributions(
    X_train,
    X_test,
    feature_names=None,
    categorical_threshold=15,
    subfigsize=(3, 4),
    plot_cols=4,
    violin_kwargs={
        'inner': 'quartile',
        'cut': 0,
        'gap': 0,
    },
):
    # Check for non-numerical columns in both datasets
    non_numerical_train = X_train.select_dtypes(exclude=['number']).columns
    non_numerical_test = X_test.select_dtypes(exclude=['number']).columns

    if len(non_numerical_train) > 0 or len(non_numerical_test) > 0:
        all_non_numerical = set(non_numerical_train) | set(non_numerical_test)
        logging.warning(f"Found non-numerical columns: {list(all_non_numerical)}")
        raise ValueError(
            "All columns must be numerical. Please convert or remove non-numerical columns before proceeding."
        )

    if feature_names is None:
        feature_names = X_train.columns.to_list()
    n_features = len(feature_names)
    n_cols = min(plot_cols, n_features)
    n_rows = math.ceil(n_features / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(subfigsize[0] * n_cols, subfigsize[1] * n_rows)
    )
    if n_features == 1:
        axes = np.array([axes])
    axes = axes.flatten()  # Flatten axes array for easy indexing

    legend_handles = []
    legend_labels = []

    for i, feature in enumerate(feature_names):
        data = pd.DataFrame(
            {
                'value': pd.concat([X_train[feature], X_test[feature]]),
                'set': ['Train'] * len(X_train) + ['Test'] * len(X_test),
            }
        )

        # Calculate missing percentages
        train_missing = X_train[feature].isna().mean() * 100
        test_missing = X_test[feature].isna().mean() * 100

        # Remove missing values for analysis
        data_clean = data.dropna()

        is_categorical = len(np.unique(data_clean['value'])) < categorical_threshold

        if is_categorical:
            # For categorical variables, use a bar plot
            train_counts = X_train[feature].value_counts(normalize=True).sort_index()
            test_counts = X_test[feature].value_counts(normalize=True).sort_index()

            # Ensure both train and test have the same categories
            all_categories = sorted(set(train_counts.index) | set(test_counts.index))
            train_counts = train_counts.reindex(all_categories, fill_value=0)
            test_counts = test_counts.reindex(all_categories, fill_value=0)

            bar_width = 0.35
            index = np.arange(len(all_categories))

            train_bars = axes[i].bar(
                index - bar_width / 2, train_counts, bar_width, label='Train', color='skyblue'
            )
            test_bars = axes[i].bar(
                index + bar_width / 2, test_counts, bar_width, label='Test', color='lightgreen'
            )

            formatted_categories = [format_label(cat) for cat in all_categories]
            axes[i].set_xticks(index)
            axes[i].set_xticklabels(formatted_categories, rotation=45, ha='right')
            axes[i].set_ylabel('Density')
            axes[i].set_xlabel('')

            if i == 0:
                legend_handles.extend([train_bars, test_bars])
                legend_labels.extend(['Train', 'Test'])
        else:
            # For continuous variables, use a violin plot
            sns.violinplot(
                x='set',
                y='value',
                hue='set',
                data=data_clean,
                ax=axes[i],
                split=True,
                palette={'Train': 'skyblue', 'Test': 'lightgreen'},
                **violin_kwargs,
            )
            axes[i].set_xticks([])
            axes[i].set_ylabel('Value')
            axes[i].set_xlabel('Density')

            if i == 0:
                legend_handles.extend(axes[i].collections)
                legend_labels.extend(['Train', 'Test'])

        # Set title with missing percentages
        if train_missing > 0 or test_missing > 0:
            axes[i].set_title(
                f"{feature}\nTrain: {train_missing:.1f}% missing\nTest: {test_missing:.1f}% missing"
            )
        else:
            axes[i].set_title(f"{feature}")

        # Add a legend
        if i == 0:  # Only add legend to the first subplot
            axes[i].legend(legend_handles, legend_labels, title='Dataset', loc='upper right')

    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def load_data(filename, sample_size=None, random_seed=42):
    """
    Load data from a CSV file and optionally select a random sample

    Parameters:
    -----------
    filename : str
        Name of the CSV file to load from RAW_DATA_DIR
    sample_size : int, optional
        Number of samples to randomly select. If None, use all data
    random_seed : int, optional
        Random seed for reproducibility when sampling

    Returns:
    --------
    pandas.DataFrame
        The loaded (and possibly sampled) data
    """
    file_path = RAW_DATA_DIR.joinpath(filename)
    print(f"Loading data from {file_path}")

    data = pd.read_csv(file_path)
    print(f"Loaded {len(data)} records with {len(data.columns)} columns")

    # If sample_size is specified and smaller than the total data size, take a random sample
    if sample_size is not None and sample_size < len(data):
        np.random.seed(random_seed)
        sampled_data = data.sample(n=sample_size, random_state=random_seed)
        print(f"Randomly selected {sample_size} samples from the original dataset")
        return sampled_data

    return data


def compute_file_hash_and_size(
    file_path: Union[str, Path], algorithm: str = 'sha256', buffer_size: int = 65536
) -> Dict[str, Any]:
    """
    Compute cryptographic hash and file size for a given file.

    This function provides data provenance tracking by computing file hashes
    and sizes, enabling verification of data integrity and reproducibility.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the file to hash. Can be absolute or relative.
    algorithm : str, optional
        Hash algorithm to use (default: 'sha256').
        Must be a valid algorithm supported by hashlib.
    buffer_size : int, optional
        Buffer size in bytes for reading file in chunks (default: 65536 = 64KB).
        Using buffered reading ensures efficient handling of large files.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'file': Relative path to file from project root (for portability)
        - 'hash': Hexadecimal hash digest (or None if error occurred)
        - 'size_bytes': File size in bytes (or None if error occurred)
        - 'computed_at': ISO format timestamp (UTC)
        - 'algorithm': Hash algorithm used
        - 'error': Error message if computation failed (only present on error)

    Examples
    --------
    >>> result = compute_file_hash_and_size('data/raw/htx_example.csv')
    >>> print(result['hash'])
    '7fe37db7d7d0cf575933936c4516cca9f1118f9cdb9068889a0270d1616ce18b'
    >>> print(result['size_bytes'])
    123456

    >>> # Verify hash matches expected value
    >>> expected_hash = '7fe37...'
    >>> if result['hash'] == expected_hash:
    ...     print("Data integrity verified")

    Notes
    -----
    - File paths are stored relative to PROJ_ROOT for portability across systems
    - Errors are handled gracefully - hash/size will be None with error message
    - Uses buffered reading to efficiently handle large files
    - Timestamps are in UTC for consistency across time zones
    """
    file_path = Path(file_path)

    # Initialize result with timestamp and algorithm
    result = {
        'computed_at': datetime.now(timezone.utc).isoformat(),
        'algorithm': algorithm,
    }

    try:
        # Compute relative path from project root for portability
        try:
            relative_path = file_path.relative_to(PROJ_ROOT)
            result['file'] = str(relative_path).replace(
                '\\', '/'
            )  # Use forward slashes for portability
        except ValueError:
            # File is outside project root, use absolute path
            result['file'] = str(file_path).replace('\\', '/')

        # Get file size
        result['size_bytes'] = file_path.stat().st_size

        # Compute hash using buffered reading for memory efficiency
        hash_obj = hashlib.new(algorithm)
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(buffer_size), b''):
                hash_obj.update(chunk)

        result['hash'] = hash_obj.hexdigest()

        logger.debug(
            f"Computed {algorithm} hash for {result['file']}: "
            f"{result['hash']} ({result['size_bytes']} bytes)"
        )

    except FileNotFoundError:
        logger.warning(f"File not found: {file_path}")
        result['file'] = str(file_path).replace('\\', '/')
        result['hash'] = None
        result['size_bytes'] = None
        result['error'] = f"File not found: {file_path}"

    except PermissionError:
        logger.warning(f"Permission denied reading file: {file_path}")
        result['file'] = str(file_path).replace('\\', '/')
        result['hash'] = None
        result['size_bytes'] = None
        result['error'] = f"Permission denied: {file_path}"

    except Exception as e:
        logger.error(f"Error computing hash for {file_path}: {str(e)}")
        result['file'] = str(file_path).replace('\\', '/')
        result['hash'] = None
        result['size_bytes'] = None
        result['error'] = str(e)

    return result


def generate_data_overview(df):
    """Generate comprehensive overview of the data"""
    # Get feature summary from PRiSM
    feat_summary = feature_summary(df)
    print("\nFeature Summary:")
    print(feat_summary)

    # Create TableOne summary (all columns should be numeric at this point)
    continuous_cols = df.columns.tolist()

    # Create TableOne summary
    tableone = TableOne(
        df,
        columns=continuous_cols,
        categorical=[],  # No categorical columns as we've dropped them
        groupby=None,
        missing=True,
        pval=False,
    )

    print("\nTableOne Summary:")
    print(tableone)

    # Generate PRiSM distribution plots
    print("\nGenerating distribution plots...")
    plot_distributions(df, categorical_threshold=15)

    return feat_summary, tableone


def split_data_temporal_or_random(
    df,
    temporal_column=None,
    train_size=0.6,
    test_size=0.2,
    val_size=0.2,
    plot_distributions=False,
    random_state=42,
    stratify=None,
    max_ratio_deviation=0.10,
):
    """
    Split data into train, validation and test sets based on specified temporal column.

    If temporal_column is None, performs random split. If temporal_column is provided,
    splits at temporal bin boundaries (e.g., year boundaries) to ensure clean splits.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to split
    temporal_column : str, optional
        Column name to use for temporal splitting. If None, performs random split instead.
        Must be numeric (int, float) or datetime dtype to support ordering comparisons.
        Categorical columns are not supported - use exclude_columns in convert_to_categorical()
        to prevent temporal columns from being converted to categorical.
    train_size : float, optional
        Fraction of data for training set (default: 0.6)
    test_size : float, optional
        Fraction of data for test set (default: 0.2)
    val_size : float, optional
        Fraction of data for validation set (default: 0.2)
    plot_distributions : bool, optional
        Whether to plot distributions between train and test sets (default: False)
    random_state : int, optional
        Random seed for reproducibility (default: 42)
    stratify : array-like, optional
        If not None, data is split in a stratified fashion, using this as the class labels (default: None)
    max_ratio_deviation : float, optional
        Maximum allowed deviation between target and actual split ratios (default: 0.10 = 10%)
        Warning is displayed if actual ratios deviate more than this threshold

    Returns:
    --------
    train_df, test_df, val_df : pandas.DataFrame
        The split datasets
    """
    # If no column specified, do a random split
    if temporal_column is None:
        print("No temporal column specified. Using random split.")
        train_df, temp_df = train_test_split(
            df, test_size=(1 - train_size), random_state=random_state, stratify=stratify
        )
        # For the second split, we need to extract the stratify values for temp_df if provided
        temp_stratify = None if stratify is None else stratify.iloc[temp_df.index]
        val_df, test_df = train_test_split(
            temp_df,
            test_size=test_size / (test_size + (1 - train_size - test_size)),
            random_state=random_state,
            stratify=temp_stratify,
        )
        return train_df, test_df, val_df

    # Verify the column exists
    if temporal_column not in df.columns:
        raise ValueError(f"Column '{temporal_column}' not found in DataFrame")

    # Verify the column is numeric or datetime (orderable types that support <= comparisons)
    col_dtype = df[temporal_column].dtype
    if isinstance(col_dtype, pd.CategoricalDtype):
        raise TypeError(
            f"Temporal column '{temporal_column}' is categorical dtype. "
            f"Temporal columns must be numeric or datetime to support ordering comparisons. "
            f"Use exclude_columns parameter in convert_to_categorical() to prevent conversion."
        )
    if not (
        pd.api.types.is_numeric_dtype(col_dtype) or pd.api.types.is_datetime64_any_dtype(col_dtype)
    ):
        raise TypeError(
            f"Temporal column '{temporal_column}' has dtype '{col_dtype}'. "
            f"Temporal columns must be numeric or datetime for proper ordering."
        )

    print(f"Splitting by temporal bins: {temporal_column}")
    print(f"Target ratios - train: {train_size:.1%}, test: {test_size:.1%}, val: {val_size:.1%}")

    # Set random seed for reproducibility
    np.random.seed(random_state)

    # Sort by specified column
    sorted_df = df.sort_values(by=temporal_column)

    # Get unique temporal bin values
    unique_bins = sorted(sorted_df[temporal_column].unique())
    n = len(sorted_df)

    # Find optimal bin boundary for train/test split
    train_target_size = train_size
    best_train_split_bin = None
    best_train_deviation = float('inf')

    for bin_val in unique_bins:
        # Calculate how many samples would be in train if we split at this bin
        train_count = (sorted_df[temporal_column] <= bin_val).sum()
        actual_ratio = train_count / n
        deviation = abs(actual_ratio - train_target_size)

        if deviation < best_train_deviation:
            best_train_deviation = deviation
            best_train_split_bin = bin_val

    # Find optimal bin boundary for test/val split
    test_target_size = train_size + test_size
    best_test_split_bin = None
    best_test_deviation = float('inf')

    for bin_val in unique_bins:
        # Calculate how many samples would be in train+test if we split at this bin
        train_test_count = (sorted_df[temporal_column] <= bin_val).sum()
        actual_ratio = train_test_count / n
        deviation = abs(actual_ratio - test_target_size)

        if deviation < best_test_deviation:
            best_test_deviation = deviation
            best_test_split_bin = bin_val

    # Perform the splits at the optimal bin boundaries
    train_df = sorted_df[sorted_df[temporal_column] <= best_train_split_bin]
    test_df = sorted_df[
        (sorted_df[temporal_column] > best_train_split_bin)
        & (sorted_df[temporal_column] <= best_test_split_bin)
    ]
    val_df = sorted_df[sorted_df[temporal_column] > best_test_split_bin]

    # Calculate actual ratios
    actual_train_ratio = len(train_df) / n
    actual_test_ratio = len(test_df) / n
    actual_val_ratio = len(val_df) / n

    print(
        f"Actual ratios - train: {actual_train_ratio:.1%}, test: {actual_test_ratio:.1%}, val: {actual_val_ratio:.1%}"
    )

    # Check for deviations and warn if necessary
    train_deviation = abs(actual_train_ratio - train_size)
    test_deviation = abs(actual_test_ratio - test_size)
    val_deviation = abs(actual_val_ratio - val_size)

    if train_deviation > max_ratio_deviation:
        print(
            f"WARNING: Split ratio deviation > {max_ratio_deviation:.1%} for train: "
            f"target {train_size:.1%}, actual {actual_train_ratio:.1%}"
        )
    if test_deviation > max_ratio_deviation:
        print(
            f"WARNING: Split ratio deviation > {max_ratio_deviation:.1%} for test: "
            f"target {test_size:.1%}, actual {actual_test_ratio:.1%}"
        )
    if val_deviation > max_ratio_deviation:
        print(
            f"WARNING: Split ratio deviation > {max_ratio_deviation:.1%} for val: "
            f"target {val_size:.1%}, actual {actual_val_ratio:.1%}"
        )

    # Show the value range in each split
    for name, split_df in [('Train', train_df), ('Test', test_df), ('Val', val_df)]:
        min_val = split_df[temporal_column].min()
        max_val = split_df[temporal_column].max()
        print(f"{name} set: {len(split_df)} samples, {temporal_column} range {min_val}-{max_val}")

    # Visualize distributions between train and test sets only if requested
    if plot_distributions:
        print("\nComparing distributions between train and test sets:")
        plot_train_test_distributions(train_df, test_df)

    return train_df, test_df, val_df


def split_data_predefined(
    df: pd.DataFrame,
    split_column: str,
    split_labels: Optional[Dict[str, List[str]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data using a predefined split column that specifies train/test/val assignments.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to split
    split_column : str
        Column name containing split assignments (e.g., 'train', 'test', 'val')
    split_labels : dict, optional
        Dictionary mapping target splits to source values. Format:
        {
            'train': ['train', 'Training', 'Tr'],
            'test': ['test', 'Testing', 'Te'],
            'val': ['val', 'validation', 'Val']
        }
        If None, uses default mapping with common variations.

    Returns
    -------
    train_df, test_df, val_df : pd.DataFrame
        The split datasets. val_df may be empty if no validation split exists.
        The split column is preserved in the output (drop it separately if needed).

    Raises
    ------
    ValueError
        If split_column not found in DataFrame
        If required splits (train, test) are missing
        If any row has an unrecognized split value
    """
    # Default mapping if not provided
    if split_labels is None:
        split_labels = {
            'train': ['train', 'Train', 'TRAIN', 'training', 'Training', 'TRAINING', 'Tr', 'tr'],
            'test': ['test', 'Test', 'TEST', 'testing', 'Testing', 'TESTING', 'Te', 'te'],
            'val': [
                'val',
                'Val',
                'VAL',
                'validation',
                'Validation',
                'VALIDATION',
                'valid',
                'Valid',
                'VALID',
                'Va',
                'va',
            ],
        }

    # Verify split column exists
    if split_column not in df.columns:
        raise ValueError(
            f"Split column '{split_column}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    # Create reverse mapping: source_value -> target_split
    reverse_mapping = {}
    for target_split, source_values in split_labels.items():
        for source_value in source_values:
            reverse_mapping[source_value] = target_split

    # Get unique values in split column
    unique_values = df[split_column].dropna().unique()

    # Check for unrecognized values
    unrecognized = [v for v in unique_values if v not in reverse_mapping]
    if unrecognized:
        raise ValueError(
            f"Unrecognized split values in column '{split_column}': {unrecognized}\n"
            f"Expected values: {list(reverse_mapping.keys())}\n"
            f"Configure 'split_labels' in config to map these values."
        )

    # Map split values to normalized names
    normalized_splits = df[split_column].map(reverse_mapping)

    # Perform the splits
    train_df = df[normalized_splits == 'train'].copy()
    test_df = df[normalized_splits == 'test'].copy()
    val_df = df[normalized_splits == 'val'].copy()

    # Validate required splits
    if len(train_df) == 0:
        raise ValueError(
            f"No training samples found. Check split_labels for 'train' values. "
            f"Values in column: {list(unique_values)}"
        )
    if len(test_df) == 0:
        raise ValueError(
            f"No test samples found. Check split_labels for 'test' values. "
            f"Values in column: {list(unique_values)}"
        )

    # Log results
    total = len(df)
    print(f"Predefined split using column: {split_column}")
    print(f"  Training: {len(train_df)} samples ({len(train_df)/total*100:.1f}%)")
    print(f"  Test: {len(test_df)} samples ({len(test_df)/total*100:.1f}%)")
    if len(val_df) > 0:
        print(f"  Validation: {len(val_df)} samples ({len(val_df)/total*100:.1f}%)")
    else:
        print("  Validation: 0 samples (empty DataFrame created for compatibility)")

    return train_df, test_df, val_df


def save_split_datasets(
    train_df, test_df, val_df, base_filename, save_dir, include_timestamp=True, header_comment=None
):
    """
    Save train, validation, and test datasets to CSV files in the specified directory

    Parameters:
    -----------
    train_df, test_df, val_df : pandas.DataFrame
        The datasets to save
    base_filename : str
        Base name for the saved files (e.g., 'unos_data')
    save_dir : str or pathlib.Path
        Directory where the CSV files will be saved
    include_timestamp : bool, optional
        Whether to include a timestamp in filenames (default: True)
    header_comment : str, optional
        Optional comment to include at the top of each CSV file

    Returns:
    --------
    dict
        Dictionary with the paths to the saved files
    """
    # Create save_dir if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Add timestamp to filename if requested
    if include_timestamp:
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_prefix = f"{base_filename}_{timestamp}"
    else:
        file_prefix = base_filename

    # Define file paths
    train_path = os.path.join(save_dir, f"{file_prefix}_train.csv")
    test_path = os.path.join(save_dir, f"{file_prefix}_test.csv")
    val_path = os.path.join(save_dir, f"{file_prefix}_val.csv")

    # Save files with optional header comment
    if header_comment:
        for df, path in [(train_df, train_path), (test_df, test_path), (val_df, val_path)]:
            # First write the comment line
            with open(path, 'w') as f:
                f.write(f"# {header_comment}\n")

            # Then append the CSV data without writing the header again
            df.to_csv(path, mode='a', index=False)
    else:
        # Save files normally
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        val_df.to_csv(val_path, index=False)

    print(f"Saved train dataset ({len(train_df)} samples) to: {train_path}")
    print(f"Saved test dataset ({len(test_df)} samples) to: {test_path}")
    print(f"Saved validation dataset ({len(val_df)} samples) to: {val_path}")

    # Compute hashes for data provenance tracking
    logger.debug("Computing hashes for saved split datasets...")
    train_hash_info = compute_file_hash_and_size(train_path)
    test_hash_info = compute_file_hash_and_size(test_path)
    val_hash_info = compute_file_hash_and_size(val_path)

    return {
        'train_path': train_path,
        'test_path': test_path,
        'val_path': val_path,
        'train_hash_info': train_hash_info,
        'test_hash_info': test_hash_info,
        'val_hash_info': val_hash_info,
    }


def enforce_binary_target_encoding(df, target_variable):
    """Enforce binary encoding for target variable (e.g., true/false -> 1/0, cat/dog -> 0/1)."""
    if target_variable not in df.columns:
        print(f"Warning: Target variable '{target_variable}' not found in dataset")
        return df

    print(f"Enforcing binary encoding for target variable: {target_variable}")
    print(f"Original unique values: {sorted(df[target_variable].dropna().unique())}")

    # Check current data type and values
    original_type = df[target_variable].dtype
    unique_values = sorted(df[target_variable].dropna().unique())

    # Handle case where there are not exactly 2 unique values
    if len(unique_values) != 2:
        if len(unique_values) < 2:
            print(
                f"Warning: Target variable has only {len(unique_values)} unique value(s): {unique_values}"
            )
            print("Cannot create binary encoding - need exactly 2 unique values")
            return df
        else:
            print(
                f"Warning: Target variable has {len(unique_values)} unique values: {unique_values}"
            )
            print("Cannot create binary encoding - need exactly 2 unique values")
            print("Consider creating a binary version of this variable first")
            return df

    # Check if values are already numeric 0.0 and 1.0
    if set(unique_values) == {0.0, 1.0}:
        print("Values are already binary (0.0, 1.0) - converting to integer")
        # Handle missing values before conversion
        if df[target_variable].isna().any():
            print(f"Found {df[target_variable].isna().sum()} missing values")
            # Use nullable integer type to preserve missing values
            df[target_variable] = df[target_variable].astype('Int64')
        else:
            # No missing values, safe to convert to regular int
            df[target_variable] = df[target_variable].astype('int64')
    # Convert boolean values to binary (existing logic)
    elif original_type == bool or set(str(v).lower() for v in unique_values if pd.notna(v)) <= {
        'true',
        'false',
        '1',
        '0',
    }:
        print("Converting boolean/text values to binary (0/1)")
        df[target_variable] = df[target_variable].map(
            {
                True: 1,
                False: 0,
                'True': 1,
                'False': 0,
                'true': 1,
                'false': 0,
                'TRUE': 1,
                'FALSE': 0,
                1: 1,
                0: 0,
                '1': 1,
                '0': 0,
            }
        )
        # Convert to appropriate integer type
        if df[target_variable].isna().any():
            df[target_variable] = df[target_variable].astype('Int64')
        else:
            df[target_variable] = df[target_variable].astype('int64')
    else:
        # Handle arbitrary binary values - map first value to 0, second to 1
        first_value, second_value = unique_values[0], unique_values[1]
        print(f"Mapping arbitrary binary values: '{first_value}' -> 0, '{second_value}' -> 1")

        df[target_variable] = df[target_variable].map({first_value: 0, second_value: 1})
        # Convert to appropriate integer type
        if df[target_variable].isna().any():
            df[target_variable] = df[target_variable].astype('Int64')
        else:
            df[target_variable] = df[target_variable].astype('int64')

    # Ensure values are 0 and 1
    final_unique = sorted(df[target_variable].dropna().unique())
    if set(final_unique) == {0, 1}:
        print(f"[OK] Target variable successfully encoded as binary: {final_unique}")
        print(f"Data type: {df[target_variable].dtype}")

        # Show distribution
        value_counts = df[target_variable].value_counts().sort_index()
        print(f"Distribution: {dict(value_counts)}")

        # Show missing values if any
        missing_count = df[target_variable].isna().sum()
        if missing_count > 0:
            print(f"Missing values: {missing_count}")
    else:
        print(f"Warning: Target variable values are not binary: {final_unique}")
        print("Expected values: [0, 1]")

    return df


def convert_to_categorical(
    df, categorical_threshold=10, convert_numeric=True, inplace=False, exclude_columns=None
):
    """Convert appropriate columns to pandas Categorical dtype.

    This function:
    1. Preserves existing Categorical columns
    2. Converts string/object columns to Categorical
    3. Optionally converts numeric columns with few unique values to Categorical

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to process
    categorical_threshold : int
        Maximum unique values for numeric columns to be converted to categorical
    convert_numeric : bool
        If True, convert numeric columns with few unique values to categorical
    inplace : bool
        If True, modify the DataFrame in place and return None
    exclude_columns : list of str, optional
        Column names to exclude from categorical conversion. Useful for temporal
        columns used in splitting that should remain numeric for comparisons.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with categorical columns converted, or None if inplace=True
    """
    if not inplace:
        df = df.copy()

    if exclude_columns is None:
        exclude_columns = []

    for col in df.columns:
        # Skip excluded columns
        if col in exclude_columns:
            continue

        # Rule 1: Keep existing Categorical columns as-is
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            continue

        # Rule 2: Convert string/object columns to Categorical
        # Uses is_string_dtype to handle both 'object' (pandas <3) and 'str' (pandas >=3)
        elif pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype('category')

        # Rule 3: Optionally convert numeric columns with few unique values to Categorical
        # Note: Binary numeric columns (n=2) are excluded to keep them as numeric for modeling
        elif convert_numeric and pd.api.types.is_numeric_dtype(df[col]):
            unique_count = df[col].nunique()
            if 2 < unique_count < categorical_threshold:
                # If float values have no decimals, convert to int first
                if pd.api.types.is_float_dtype(df[col]):
                    if (df[col].dropna() % 1 == 0).all():
                        df[col] = df[col].astype('Int64')  # Nullable integer type
                df[col] = df[col].astype('category')

    if not inplace:
        return df


def analyze_categorical_columns(df):
    """Analyze and display information about Categorical columns in the DataFrame.

    Classifies all categorical dtype columns as either binary (2 unique values) or
    multi-category (>2 unique values). The categorization is based solely on the number
    of unique values, regardless of whether the values are numeric-looking or string-based.

    Note: The decision of what should be categorical dtype should have already been made
    by convert_to_categorical(). This function simply reports and classifies what's there.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to analyze

    Returns
    -------
    dict
        Dictionary containing lists of binary and multi-category column names
    """
    # Get categorical columns using select_dtypes
    categorical_cols = df.select_dtypes(include=['category']).columns.tolist()

    if not categorical_cols:
        print("No categorical columns found in the DataFrame.")
        return {'binary': [], 'multi_category': []}

    # Classify by number of unique categories
    binary_cols = []
    multi_category_cols = []

    for col in categorical_cols:
        n_categories = df[col].nunique()
        if n_categories == 2:
            binary_cols.append(col)
        else:
            multi_category_cols.append(col)

    print(f"Found {len(categorical_cols)} total categorical columns:")
    print(f"  {len(binary_cols)} binary columns")
    print(f"  {len(multi_category_cols)} multi-category columns")

    if binary_cols:
        print("\nBinary categorical columns:")
        for col in binary_cols:
            categories = df[col].cat.categories.tolist()
            value_counts = df[col].value_counts().to_dict()
            print(f"  {col}: {categories} - {value_counts}")

    if multi_category_cols:
        print("\nMulti-category columns:")
        for col in multi_category_cols:
            categories = df[col].cat.categories.tolist()
            n_cats = len(categories)
            value_counts = df[col].value_counts().to_dict()
            # Truncate long category lists
            if n_cats > 10:
                display_cats = categories[:10] + ['...']
            else:
                display_cats = categories
            print(f"  {col}: {n_cats} categories {display_cats} - {value_counts}")

    return {
        'binary': binary_cols,
        'multi_category': multi_category_cols,
    }


def order_binary_categories(categories):
    """Determine sensible ordering for binary categorical variables.

    This function applies heuristics to order binary categories such that:
    - "Negative" or "absence" values map to 0
    - "Positive" or "presence" values map to 1

    Parameters
    ----------
    categories : list
        List of exactly 2 category values

    Returns
    -------
    list
        Ordered list [negative_value, positive_value]

    Examples
    --------
    >>> order_binary_categories(['yes', 'no'])
    ['no', 'yes']
    >>> order_binary_categories(['Y', 'N'])
    ['N', 'Y']
    >>> order_binary_categories(['Male', 'Female'])
    ['Male', 'Female']  # No clear ordering, keeps original
    """
    if len(categories) != 2:
        raise ValueError(f"Expected exactly 2 categories, got {len(categories)}")

    ordered_categories = categories.copy()

    # Common patterns for ordering
    negative_indicators = ['no', 'none', 'false', '0', 'absent', 'negative', 'bad', 'low', 'n']
    positive_indicators = ['yes', 'true', '1', 'present', 'positive', 'good', 'high', 'y']

    # Check if we can determine a sensible ordering
    cat_lower = [str(cat).lower() for cat in categories]

    # Try to find negative indicator in first position
    for neg in negative_indicators:
        if any(neg in cat for cat in cat_lower):
            neg_idx = next(i for i, cat in enumerate(cat_lower) if neg in cat)
            pos_idx = 1 - neg_idx
            ordered_categories = [categories[neg_idx], categories[pos_idx]]
            return ordered_categories

    # Try to find positive indicator in second position
    for pos in positive_indicators:
        if any(pos in cat for cat in cat_lower):
            pos_idx = next(i for i, cat in enumerate(cat_lower) if pos in cat)
            neg_idx = 1 - pos_idx
            ordered_categories = [categories[neg_idx], categories[pos_idx]]
            return ordered_categories

    # No clear ordering found, return original order
    return ordered_categories


def create_binary_encoding(df, binary_cols):
    """Create integer encoding mappings for binary categorical variables.

    This function converts detected binary variables into integer encoding mappings
    suitable for use with preprocess_data()'s integer_encoding parameter.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the binary categorical variables
    binary_cols : list
        List of column names that are binary categorical

    Returns
    -------
    dict
        Dictionary mapping column names to ordered category lists for ordinal encoding

    Examples
    --------
    Binary variable 'own_telephone' with categories ['none', 'yes'] becomes:
    {'own_telephone': ['none', 'yes']} -> none=0, yes=1
    """
    ordinal_mappings = {}

    if not binary_cols:
        print("No binary categorical variables to process for ordinal mapping")
        return ordinal_mappings

    print(f"Creating ordinal mappings for {len(binary_cols)} binary variables:")

    for col in binary_cols:
        if col not in df.columns:
            print(f"  Warning: '{col}' not found in DataFrame")
            continue

        if not isinstance(df[col].dtype, pd.CategoricalDtype):
            print(f"  Warning: '{col}' is not categorical type")
            continue

        categories = df[col].cat.categories.tolist()

        if len(categories) != 2:
            print(f"  Warning: '{col}' has {len(categories)} categories, expected 2")
            continue

        # Use shared ordering logic
        ordered_categories = order_binary_categories(categories)

        # Store the mapping
        ordinal_mappings[col] = ordered_categories

        print(
            f"  {col}: {categories} -> {ordered_categories} (0={ordered_categories[0]}, 1={ordered_categories[1]})"
        )

    return ordinal_mappings


def create_ordinal_encoding(df, multi_cat_cols, ordering_method='alphabetical'):
    """Create ordinal encoding mappings for multi-category categorical variables.

    This function converts multi-category variables into ordinal encoding mappings
    suitable for use with preprocess_data()'s integer_encoding parameter.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the multi-category categorical variables
    multi_cat_cols : list
        List of column names that are multi-category categorical
    ordering_method : str
        Method for ordering categories ('alphabetical' or 'frequency')

    Returns
    -------
    dict
        Dictionary mapping column names to ordered category lists for ordinal encoding

    Examples
    --------
    Multi-category variable 'purpose' with categories ['business', 'car', 'education'] becomes:
    {'purpose': ['business', 'car', 'education']} -> business=0, car=1, education=2
    """
    ordinal_mappings = {}

    if not multi_cat_cols:
        print("No multi-category variables to process for ordinal mapping")
        return ordinal_mappings

    print(f"Creating ordinal mappings for {len(multi_cat_cols)} multi-category variables:")

    for col in multi_cat_cols:
        if col not in df.columns:
            print(f"  Warning: '{col}' not found in DataFrame")
            continue

        if not isinstance(df[col].dtype, pd.CategoricalDtype):
            print(f"  Warning: '{col}' is not categorical type")
            continue

        categories = df[col].cat.categories.tolist()

        if len(categories) <= 2:
            print(f"  Warning: '{col}' has {len(categories)} categories, expected >2")
            continue

        # Order categories based on the specified method
        if ordering_method == 'alphabetical':
            ordered_categories = sorted(categories)
        elif ordering_method == 'frequency':
            freq_counts = df[col].value_counts()
            ordered_categories = freq_counts.index.tolist()
        else:
            ordered_categories = sorted(categories)

        # Store the mapping
        ordinal_mappings[col] = ordered_categories

        # Display shortened version if too many categories
        if len(ordered_categories) > 3:
            display_cats = ordered_categories[:3] + ['...']
        else:
            display_cats = ordered_categories
        print(f"  {col}: {len(categories)} categories -> {display_cats}")

    return ordinal_mappings


def ensure_consistent_categories(df, all_categories, all_integer_mappings):
    """Ensure all categorical columns have all possible categories.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to process
    all_categories : dict
        Dictionary mapping column names to all possible categories for one-hot encoding
    all_integer_mappings : dict
        Dictionary mapping column names to ordered categories for ordinal encoding

    Returns
    -------
    pd.DataFrame
        DataFrame with consistent categorical representations
    """
    df_consistent = df.copy()

    # Handle categorical (one-hot) variables
    for col, categories in all_categories.items():
        if col in df_consistent.columns:
            df_consistent[col] = pd.Categorical(
                df_consistent[col], categories=categories, ordered=False
            )

    # Handle ordinal variables
    for col, ordered_categories in all_integer_mappings.items():
        if col in df_consistent.columns:
            df_consistent[col] = pd.Categorical(
                df_consistent[col], categories=ordered_categories, ordered=True
            )

    return df_consistent
