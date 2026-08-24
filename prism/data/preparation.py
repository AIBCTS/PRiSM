"""In-memory data preparation for PRiSM.

Composes the existing preprocessing utilities into a single call that takes a
raw DataFrame and returns everything the PRiSM pipeline needs downstream
(splits, fitted scaler, one-hot group manager, feature names) without touching
the config/.env machinery or writing intermediate files to disk.

The composition mirrors ``example_notebooks/preprocessing.ipynb``: binary
target encoding, ID removal, categorical detection, splitting, consistent
category sets across splits, per-split encoding/imputation with the training
split's reference columns propagated to test/val, and a scaler fitted on the
training split only.
"""

import contextlib
import io
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from prism.data_utilities import (
    analyze_categorical_columns,
    convert_to_categorical,
    create_binary_encoding,
    enforce_binary_target_encoding,
    ensure_consistent_categories,
    split_data_predefined,
    split_data_temporal_or_random,
)
from prism.preprocessing import (
    OneHotGroupManager,
    PRiSMScaler,
    collapse_onehot_features,
    preprocess_data,
)


def _stdout_sink(verbose: bool):
    """Return a context manager that swallows stdout unless *verbose*.

    The preprocessing utilities this module composes all print their progress
    unconditionally and none of them takes a flag, so silencing happens once
    here rather than being threaded through every one of them.
    """
    return contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())


@dataclass
class PreparedData:
    """Bundle of everything PRiSM needs after preprocessing.

    The ``scaler``, ``group_manager`` and ``feature_names`` fields are exactly
    the trio the partial-response calculator requires together when one-hot
    encoded features are present.

    Examples
    --------
    >>> from prism.data import load_htx_example_splits
    >>> data = load_htx_example_splits(verbose=False)
    >>> data.feature_names[:3]
    ['DAT_VENTILATOR_TRR', 'DAT_DIAL_AFTER_LIST', 'DAT_TRANSFUSIONS']
    >>> data.X_train.shape[1] == len(data.feature_names)
    True
    """

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    X_val: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    y_val: pd.Series
    scaler: PRiSMScaler
    group_manager: Optional[OneHotGroupManager]
    feature_names: List[str]
    collapsed_feature_names: List[str]
    target_column: str
    metadata: Dict


def prepare_dataset(
    df: pd.DataFrame,
    target_column: str,
    *,
    id_column: Optional[str] = None,
    split_column: Optional[str] = None,
    split_ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2),
    categorical_threshold: int = 8,
    convert_numeric: bool = True,
    invert_target: bool = False,
    reference_column_strategy: str = 'most_common',
    integer_encoding: Optional[Dict[str, List[str]]] = None,
    max_onehot_categories: Optional[int] = 20,
    random_seed: int = 257,
    verbose: bool = True,
) -> PreparedData:
    """Preprocess a raw DataFrame into PRiSM-ready train/test/val splits.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data including the target column (and optionally ID/split columns).
    target_column : str
        Name of the binary target column. Any two-valued column is accepted;
        it is encoded to 0/1 via ``enforce_binary_target_encoding``.
    id_column : str, optional
        Identifier column to drop before modelling.
    split_column : str, optional
        Column with predefined 'train'/'test'/'val' assignments. If None, a
        random stratified split with ``split_ratios`` is performed instead.
    split_ratios : tuple of float, optional
        (train, test, val) fractions for the random split. Ignored when
        ``split_column`` is given.
    categorical_threshold : int, optional
        Numeric columns with at most this many unique values are treated as
        categorical (when ``convert_numeric`` is True).
    convert_numeric : bool, optional
        Whether low-cardinality numeric columns are treated as categorical.
    invert_target : bool, optional
        Flip the 0/1 encoding of the target after binary encoding.
    reference_column_strategy : str, optional
        Reference-category selection for one-hot groups on the training split:
        'alphabetical', 'most_common' (alias: 'frequency'), or 'manual'.
        Test/val always reuse the training split's reference columns.
    integer_encoding : dict, optional
        Manual ordinal encodings, mapping column name to ordered category list.
        Listed columns are ordinal-encoded; remaining multi-category columns
        are one-hot encoded.
    max_onehot_categories : int or None, optional
        Refuse to one-hot encode a column with more distinct values than this.
        Guards against encoding an identifier-like column, which produces one
        near-empty dummy per row. Pass None to disable the check. See
        :func:`prism.preprocessing.encode_categorical_features`.
    random_seed : int, optional
        Seed for splitting and random imputation.
    verbose : bool, optional
        Whether to print preprocessing progress. The composed utilities print
        unconditionally, so False suppresses their output wholesale.

    Returns
    -------
    PreparedData
        Splits as X/y pairs plus the fitted scaler, one-hot group manager,
        feature names (one-hot expanded) and collapsed feature names.

    Examples
    --------
    >>> from prism.data import load_example_dataset, prepare_dataset
    >>> df = load_example_dataset('htx_example')
    >>> data = prepare_dataset(
    ...     df,
    ...     target_column='MORTALITY_365D',
    ...     id_column='TRR_ID_CODE',
    ...     split_column='split',
    ...     verbose=False,
    ... )
    >>> data.X_train.shape
    (3166, 33)
    >>> sorted(set(data.y_train))
    [0, 1]

    Leaving the identifier in is rejected rather than silently one-hot encoded
    into one column per row.

    >>> prepare_dataset(df, target_column='MORTALITY_365D', verbose=False)
    Traceback (most recent call last):
        ...
    ValueError: Refusing to one-hot encode column 'TRR_ID_CODE': ...

    One-hot groups move as a single variable downstream, which is why the
    collapsed name list is shorter than the one-hot expanded one.

    >>> len(data.feature_names), len(data.collapsed_feature_names)
    (33, 22)
    """
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )
    with _stdout_sink(verbose):
        df = df.copy()

        # Binary-encode the target (e.g. Yes/No -> 1/0)
        df = enforce_binary_target_encoding(df, target_column)
        if invert_target:
            df[target_column] = 1 - df[target_column]

        if id_column is not None and id_column in df.columns:
            df = df.drop(columns=[id_column])

        # Detect categoricals; keep the split column as-is for splitting
        exclude_from_categorical = [split_column] if split_column else []
        df = convert_to_categorical(
            df,
            categorical_threshold=categorical_threshold,
            convert_numeric=convert_numeric,
            exclude_columns=exclude_from_categorical,
        )
        categorical_info = analyze_categorical_columns(df)
        binary_vars = categorical_info['binary']
        multi_cat_vars = categorical_info['multi_category']

        # Split before encoding so all encoding choices derive from the raw splits
        if split_column is not None:
            if split_column not in df.columns:
                raise ValueError(f"Split column '{split_column}' not found in DataFrame")
            train_df, test_df, val_df = split_data_predefined(df, split_column=split_column)
            train_df = train_df.drop(columns=[split_column])
            test_df = test_df.drop(columns=[split_column])
            val_df = val_df.drop(columns=[split_column])
        else:
            train_df, test_df, val_df = split_data_temporal_or_random(
                df,
                temporal_column=None,
                train_size=split_ratios[0],
                test_size=split_ratios[1],
                val_size=split_ratios[2],
                stratify=df[target_column],
                random_state=random_seed,
            )

        # Encoding strategy: binary -> ordinal, multi-category -> one-hot,
        # with manual ordinal overrides taking precedence
        integer_encoding = integer_encoding or {}
        binary_mappings = create_binary_encoding(df, binary_vars)
        all_integer_mappings = {**binary_mappings, **integer_encoding}
        onehot_cols = [
            col
            for col in multi_cat_vars
            if col != target_column and col not in all_integer_mappings
        ]

        # Pool category sets across splits so one-hot columns are identical everywhere
        all_categories = {}
        for col in onehot_cols:
            values = set()
            for split_df in (train_df, test_df, val_df):
                values.update(split_df[col].dropna().unique())
            all_categories[col] = sorted(values)

        train_df = ensure_consistent_categories(train_df, all_categories, all_integer_mappings)
        test_df = ensure_consistent_categories(test_df, all_categories, all_integer_mappings)
        val_df = ensure_consistent_categories(val_df, all_categories, all_integer_mappings)

        # Encode + impute each split; test/val reuse the training reference columns
        # so 'most_common' cannot pick different references per split
        train_processed, train_metadata = preprocess_data(
            train_df,
            integer_encoding=all_integer_mappings,
            categorical_imputation_method='random',
            numerical_imputation_method='random',
            random_state=random_seed,
            reference_column_strategy=reference_column_strategy,
            drop_reference_columns=True,
            save_metadata_path=False,
            max_onehot_categories=max_onehot_categories,
        )
        train_references = train_metadata.get('reference_columns', {}).get('references', {})
        manual_kwargs = dict(
            integer_encoding=all_integer_mappings,
            categorical_imputation_method='random',
            numerical_imputation_method='random',
            random_state=random_seed,
            reference_column_strategy='manual',
            manual_reference_columns=train_references,
            drop_reference_columns=True,
            save_metadata_path=False,
            max_onehot_categories=max_onehot_categories,
        )
        test_processed, _ = preprocess_data(test_df, **manual_kwargs)
        val_processed, _ = preprocess_data(val_df, **manual_kwargs)

        # Align column order to the training split and verify consistency
        for name, split_df in (('test', test_processed), ('val', val_processed)):
            if set(split_df.columns) != set(train_processed.columns):
                missing = set(train_processed.columns) - set(split_df.columns)
                extra = set(split_df.columns) - set(train_processed.columns)
                raise RuntimeError(
                    f"Column mismatch between train and {name} splits after preprocessing. "
                    f"Missing: {sorted(missing)}; extra: {sorted(extra)}"
                )
        test_processed = test_processed[train_processed.columns]
        val_processed = val_processed[train_processed.columns]

        y_train = train_processed[target_column].astype(int)
        y_test = test_processed[target_column].astype(int)
        y_val = val_processed[target_column].astype(int)
        X_train = train_processed.drop(columns=[target_column])
        X_test = test_processed.drop(columns=[target_column])
        X_val = val_processed.drop(columns=[target_column])

        feature_names = X_train.columns.tolist()
        group_manager = train_metadata.get('onehot_group_manager')

        scaler = PRiSMScaler(scaler='median_std')
        scaler.fit(X_train)

        if group_manager is not None:
            _, collapsed_feature_names = collapse_onehot_features(
                X_train.values, group_manager, feature_names
            )
        else:
            collapsed_feature_names = list(feature_names)

        return PreparedData(
            X_train=X_train,
            X_test=X_test,
            X_val=X_val,
            y_train=y_train,
            y_test=y_test,
            y_val=y_val,
            scaler=scaler,
            group_manager=group_manager,
            feature_names=feature_names,
            collapsed_feature_names=collapsed_feature_names,
            target_column=target_column,
            metadata=train_metadata,
        )


def load_htx_example_splits(random_seed: int = 257, verbose: bool = True) -> PreparedData:
    """Load and preprocess the bundled htx_example dataset in one call.

    Applies the same choices as the ``htx_example`` pipeline config: target
    ``MORTALITY_365D``, ID column ``TRR_ID_CODE`` dropped, predefined split
    from the ``split`` column, and most-common reference categories.

    Returns
    -------
    PreparedData
        See :func:`prepare_dataset`.

    Examples
    --------
    This is the shortest path to PRiSM-ready data. Preprocessing reports its
    progress on stdout, so the call below silences it.

    >>> from prism.data import load_htx_example_splits
    >>> data = load_htx_example_splits(verbose=False)
    >>> data.X_train.shape, data.X_test.shape, data.X_val.shape
    ((3166, 33), (869, 33), (965, 33))
    >>> data.target_column
    'MORTALITY_365D'
    >>> sorted(data.group_manager.groups_dict)
    ['DAT_DIAB', 'DAT_DIAG_NEW', 'DAT_HCV_SEROSTATUS', 'DAT_MED_COND_TRR', 'DAT_VAD_DEVICE_TY_TRR']
    """
    from prism.data import load_example_dataset

    df = load_example_dataset('htx_example')
    return prepare_dataset(
        df,
        target_column='MORTALITY_365D',
        id_column='TRR_ID_CODE',
        split_column='split',
        categorical_threshold=8,
        reference_column_strategy='most_common',
        random_seed=random_seed,
        verbose=verbose,
    )
