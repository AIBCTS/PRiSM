"""
Feature Label Management for PRiSM

Provides clean separation between:
- column_names: Data identifiers (e.g., 'recbilirubin', 'diagn_CAD')
- user_labels: Display text for plots (e.g., 'Serum bilirubin\n(umol/L)')

This prevents confusion between technical column names and human-readable labels.

Also provides utilities for generating label file templates during preprocessing.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class FeatureLabelManager:
    """
    Manages mapping between column names and user-friendly labels.

    Purpose
    -------
    Keep clear separation between:
    - column_names: Used in data operations (e.g., 'recbilirubin')
    - user_labels: Used in plots (e.g., 'Serum bilirubin\n(umol/L)')

    This prevents mixing up technical names with display text and handles
    special formatting like newlines in labels.

    Examples
    --------
    >>> # Load from CSV
    >>> manager = FeatureLabelManager.from_csv('generic_variable_labels.csv')
    >>> manager.get_label('recbilirubin')
    'Serum bilirubin\\n(umol/L)'
    >>> manager.get_label('unknown_feature')
    'unknown_feature'  # Fallback to column name

    >>> # Create manually
    >>> manager = FeatureLabelManager({'age': 'Patient Age (years)'})
    >>> manager.get_label('age')
    'Patient Age (years)'
    """

    def __init__(self, column_to_label_map: Optional[Dict[str, str]] = None):
        """
        Initialize FeatureLabelManager.

        Parameters
        ----------
        column_to_label_map : Optional[Dict[str, str]]
            Dictionary mapping column names to user labels
            Example: {'recbilirubin': 'Serum bilirubin\\n(umol/L)'}
        """
        # Sanitize labels: normalize \r\n and \r to \n
        # This fixes matplotlib glyph warnings for carriage returns
        if column_to_label_map:
            self.column_to_label = {
                k: v.replace('\r\n', '\n').replace('\r', '\n') if isinstance(v, str) else v
                for k, v in column_to_label_map.items()
            }
        else:
            self.column_to_label = {}
        logger.debug(f"Initialized FeatureLabelManager with {len(self.column_to_label)} mappings")

    @classmethod
    def from_csv(
        cls, csv_path: Path, column_name_col: str = 'processed_name', label_col: str = 'user_label'
    ) -> 'FeatureLabelManager':
        """
        Load from generic_variable_labels.csv format.

        Parameters
        ----------
        csv_path : Path
            Path to CSV file containing label mappings
        column_name_col : str, default='processed_name'
            Name of column containing column names
        label_col : str, default='user_label'
            Name of column containing user labels

        Returns
        -------
        FeatureLabelManager
            Initialized manager with mappings from CSV

        Examples
        --------
        >>> manager = FeatureLabelManager.from_csv('generic_variable_labels.csv')
        >>> len(manager.column_to_label) > 0
        True
        """
        csv_path = Path(csv_path)

        if not csv_path.exists():
            logger.warning(f"CSV file not found: {csv_path}. Creating empty manager.")
            return cls({})

        try:
            # Read CSV with proper handling of newlines in quoted fields
            df = pd.read_csv(csv_path, skipinitialspace=True)

            # Check required columns exist
            if column_name_col not in df.columns or label_col not in df.columns:
                logger.error(
                    f"Required columns not found in CSV. "
                    f"Expected: '{column_name_col}', '{label_col}'. "
                    f"Found: {list(df.columns)}"
                )
                return cls({})

            # Create mapping from column name to label
            # Drop rows with missing column names
            df_clean = df[[column_name_col, label_col]].dropna(subset=[column_name_col])

            # Create dictionary mapping
            column_to_label = {}
            for _, row in df_clean.iterrows():
                col_name = row[column_name_col]
                label = row[label_col]

                # Use column name if label is NaN
                if pd.isna(label):
                    label = col_name
                else:
                    # Normalize newlines: replace \r\n and \r with \n
                    # This fixes display issues in matplotlib where \r shows as a box
                    label = label.replace('\r\n', '\n').replace('\r', '\n')

                column_to_label[col_name] = label

            logger.info(f"Loaded {len(column_to_label)} label mappings from {csv_path}")
            return cls(column_to_label)

        except Exception as e:
            logger.error(f"Failed to load labels from {csv_path}: {e}")
            return cls({})

    def get_label(self, column_name: str) -> str:
        """
        Get user label for column, fallback to column name if not found.

        Parameters
        ----------
        column_name : str
            Column name to look up

        Returns
        -------
        str
            User label if found, otherwise column name

        Examples
        --------
        >>> manager = FeatureLabelManager({'age': 'Patient Age'})
        >>> manager.get_label('age')
        'Patient Age'
        >>> manager.get_label('unknown')
        'unknown'
        """
        return self.column_to_label.get(column_name, column_name)

    def has_label(self, column_name: str) -> bool:
        """
        Check if a label mapping exists for a column.

        Parameters
        ----------
        column_name : str
            Column name to check

        Returns
        -------
        bool
            True if mapping exists, False otherwise

        Examples
        --------
        >>> manager = FeatureLabelManager({'age': 'Patient Age'})
        >>> manager.has_label('age')
        True
        >>> manager.has_label('unknown')
        False
        """
        return column_name in self.column_to_label

    def __len__(self) -> int:
        """Return number of mappings."""
        return len(self.column_to_label)

    def __repr__(self) -> str:
        """String representation."""
        return f"FeatureLabelManager({len(self)} mappings)"


def generate_label_file_template(
    original_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    target_variable: Optional[str],
    id_variable: Optional[str],
    output_path: Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Generate a label file template with variable names and placeholders for user labels.

    Creates a CSV file that users can edit to provide human-readable labels for
    all variables, including:
    - Original variables (unchanged during preprocessing)
    - Ordinal-encoded variables (with _ordinal suffix)
    - One-hot encoded category columns (grouped together with group name and reference)
    - One-hot group names (for collapsed features in plots)
    - Dropped reference columns (for complete category labeling in nomograms)

    Related rows are kept together: for each one-hot group, the group name row,
    reference column row, and all category column rows appear consecutively.

    Parameters
    ----------
    original_df : pd.DataFrame
        Original DataFrame before preprocessing
    processed_df : pd.DataFrame
        Processed DataFrame after preprocessing
    target_variable : str or None
        Name of target variable
    id_variable : str or None
        Name of ID variable (if detected)
    output_path : Path
        Path to save the label file
    metadata : dict or None
        Preprocessing metadata containing encoding information.
        Expected keys:
        - 'encoding': Dict with encoding info per variable
        - 'reference_columns': Dict with dropped reference column info

    Returns
    -------
    pd.DataFrame
        Label template DataFrame with columns:
        - original_name: Base variable name
        - processed_name: Column name after preprocessing
        - user_label: Placeholder for user-defined label
        - notes: Description of variable type

    Examples
    --------
    >>> label_df = generate_label_file_template(
    ...     df_raw, df_processed, 'target', 'id',
    ...     Path('labels.csv'), preprocessing_metadata
    ... )
    >>> print(label_df.head())
    """
    original_columns = list(original_df.columns)
    processed_columns = list(processed_df.columns)

    label_data = []

    # Track which processed columns we've handled
    handled_columns = set()

    # Build lookup structures from metadata
    onehot_groups = {}  # group_name -> list of category columns in processed_df
    reference_columns = {}  # group_name -> reference column name
    ordinal_groups = {}  # group_name -> ordinal column name

    if metadata:
        # Get reference column info
        ref_info = metadata.get('reference_columns', {})
        reference_columns = ref_info.get('references', {})

        # Get encoding info to identify one-hot and ordinal groups
        encoding_info = metadata.get('encoding', {})
        for orig_col, enc_info in encoding_info.items():
            if enc_info.get('encoding_type') == 'one-hot':
                # Get all created columns for this group
                created_cols = enc_info.get('created_columns', [])
                # Filter to only those in processed_df (excludes dropped reference)
                cols_in_df = [c for c in created_cols if c in processed_columns]
                onehot_groups[orig_col] = cols_in_df
            elif enc_info.get('encoding_type') == 'ordinal':
                created_cols = enc_info.get('created_columns', [])
                if created_cols:
                    ordinal_groups[orig_col] = created_cols[0]

    # 1. Add special rows for target and ID variables first
    if id_variable and id_variable in processed_columns:
        label_data.append(
            {
                'original_name': id_variable,
                'processed_name': id_variable,
                'user_label': f"[USER: Fill in label for {id_variable}]",
                'notes': 'ID VARIABLE',
            }
        )
        handled_columns.add(id_variable)

    if target_variable and target_variable in processed_columns:
        label_data.append(
            {
                'original_name': target_variable,
                'processed_name': target_variable,
                'user_label': f"[USER: Fill in label for {target_variable}]",
                'notes': 'TARGET VARIABLE',
            }
        )
        handled_columns.add(target_variable)

    # 2. Process columns, grouping one-hot encoded variables together
    for col in processed_columns:
        if col in handled_columns:
            continue

        # Check if this column is part of a one-hot group
        is_onehot_member = False
        for group_name, group_cols in onehot_groups.items():
            if col in group_cols:
                is_onehot_member = True
                # Only process this group once (when we encounter the first member)
                if group_name not in handled_columns:
                    # Add group name row first
                    label_data.append(
                        {
                            'original_name': group_name,
                            'processed_name': group_name,
                            'user_label': f"[USER: Fill in label for {group_name}]",
                            'notes': 'One-hot group name (for collapsed plots)',
                        }
                    )
                    handled_columns.add(group_name)

                    # Add reference column row (if exists)
                    if group_name in reference_columns:
                        ref_col = reference_columns[group_name]
                        label_data.append(
                            {
                                'original_name': group_name,
                                'processed_name': ref_col,
                                'user_label': f"[USER: Fill in label for {ref_col}]",
                                'notes': f'One-hot reference (dropped) from {group_name}',
                            }
                        )
                        handled_columns.add(ref_col)

                    # Add all category columns for this group
                    for cat_col in group_cols:
                        label_data.append(
                            {
                                'original_name': group_name,
                                'processed_name': cat_col,
                                'user_label': f"[USER: Fill in label for {cat_col}]",
                                'notes': f'One-hot encoded from {group_name}',
                            }
                        )
                        handled_columns.add(cat_col)
                break

        if is_onehot_member:
            continue

        # Check if this is an ordinal column
        is_ordinal = False
        for group_name, ordinal_col in ordinal_groups.items():
            if col == ordinal_col:
                is_ordinal = True
                label_data.append(
                    {
                        'original_name': group_name,
                        'processed_name': col,
                        'user_label': f"[USER: Fill in label for {col}]",
                        'notes': f'Ordinal encoded from {group_name}',
                    }
                )
                handled_columns.add(col)
                break

        if is_ordinal:
            continue

        # Regular column (original or unknown)
        if col in original_columns:
            label_data.append(
                {
                    'original_name': col,
                    'processed_name': col,
                    'user_label': f"[USER: Fill in label for {col}]",
                    'notes': 'Original variable',
                }
            )
        else:
            # Try to find base name for encoded variables without metadata
            # Note: Ordinal columns can no longer be detected without metadata
            # since they no longer have the _ordinal suffix
            base_name = None
            note_type = 'Unknown encoding (metadata required)'

            # Check if this might be a one-hot encoded column
            for orig_col in original_columns:
                if col.startswith(orig_col + '_'):
                    base_name = orig_col
                    note_type = 'Possibly one-hot encoded'
                    break

            if base_name:
                label_data.append(
                    {
                        'original_name': base_name,
                        'processed_name': col,
                        'user_label': f"[USER: Fill in label for {col}]",
                        'notes': f'{note_type} from {base_name}',
                    }
                )
            else:
                label_data.append(
                    {
                        'original_name': 'NONE',
                        'processed_name': col,
                        'user_label': f"[USER: Fill in label for {col}]",
                        'notes': 'Generated unique ID variable',
                    }
                )

        handled_columns.add(col)

    # Create DataFrame
    label_df = pd.DataFrame(label_data)

    # Remove duplicates (keep first occurrence)
    label_df = label_df.drop_duplicates(subset=['processed_name'], keep='first')

    # Save to CSV
    output_path = Path(output_path)
    label_df.to_csv(output_path, index=False)

    logger.info(f"Generated label file template: {output_path}")
    logger.info(f"Template contains {len(label_df)} variables")

    return label_df
