"""
Unit tests for FeatureLabelManager and generate_label_file_template.

Tests loading labels from CSV, mapping column names to user labels,
fallback behavior, label file generation, and various edge cases.
"""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from prism.feature_labels import FeatureLabelManager, generate_label_file_template


class TestFeatureLabelManagerInit:
    """Test initialization and basic functionality."""

    def test_init_empty(self):
        """Test creating empty manager."""
        manager = FeatureLabelManager()
        assert len(manager) == 0
        assert isinstance(manager.column_to_label, dict)

    def test_init_with_dict(self):
        """Test creating manager with mappings."""
        mappings = {'age': 'Patient Age', 'sex': 'Gender'}
        manager = FeatureLabelManager(mappings)
        assert len(manager) == 2
        assert manager.get_label('age') == 'Patient Age'
        assert manager.get_label('sex') == 'Gender'

    def test_repr(self):
        """Test string representation."""
        manager = FeatureLabelManager({'age': 'Patient Age'})
        repr_str = repr(manager)
        assert 'FeatureLabelManager' in repr_str
        assert '1' in repr_str


class TestFeatureLabelManagerGetLabel:
    """Test get_label method."""

    def test_get_label_exists(self):
        """Test getting label that exists."""
        manager = FeatureLabelManager({'age': 'Patient Age'})
        assert manager.get_label('age') == 'Patient Age'

    def test_get_label_missing_fallback(self):
        """Test fallback to column name when label missing."""
        manager = FeatureLabelManager({'age': 'Patient Age'})
        assert manager.get_label('unknown') == 'unknown'

    def test_get_label_with_newlines(self):
        """Test labels with newlines preserved."""
        label_with_newline = 'Serum bilirubin\n(umol/L)'
        manager = FeatureLabelManager({'recbilirubin': label_with_newline})
        assert manager.get_label('recbilirubin') == label_with_newline
        assert '\n' in manager.get_label('recbilirubin')

    def test_get_labels_batch(self):
        """Test getting multiple labels with list comprehension."""
        manager = FeatureLabelManager(
            {'age': 'Patient Age', 'sex': 'Gender', 'weight': 'Body Weight'}
        )
        labels = [manager.get_label(name) for name in ['age', 'sex', 'unknown']]
        assert labels == ['Patient Age', 'Gender', 'unknown']

    def test_get_labels_empty_list(self):
        """Test getting labels for empty list with list comprehension."""
        manager = FeatureLabelManager({'age': 'Patient Age'})
        labels = [manager.get_label(name) for name in []]
        assert labels == []


class TestFeatureLabelManagerAddMapping:
    """Test direct dictionary access for adding mappings."""

    def test_add_mapping_single(self):
        """Test adding single mapping with direct dict access."""
        manager = FeatureLabelManager()
        manager.column_to_label['age'] = 'Patient Age'
        assert manager.get_label('age') == 'Patient Age'
        assert len(manager) == 1

    def test_add_mapping_overwrite(self):
        """Test overwriting existing mapping with direct dict access."""
        manager = FeatureLabelManager({'age': 'Old Label'})
        manager.column_to_label['age'] = 'New Label'
        assert manager.get_label('age') == 'New Label'
        assert len(manager) == 1

    def test_add_mappings_batch(self):
        """Test adding multiple mappings with dict.update()."""
        manager = FeatureLabelManager()
        manager.column_to_label.update(
            {'age': 'Patient Age', 'sex': 'Gender', 'weight': 'Body Weight'}
        )
        assert len(manager) == 3
        assert manager.get_label('age') == 'Patient Age'
        assert manager.get_label('sex') == 'Gender'
        assert manager.get_label('weight') == 'Body Weight'

    def test_add_mappings_merge(self):
        """Test adding mappings to existing manager with dict.update()."""
        manager = FeatureLabelManager({'age': 'Patient Age'})
        manager.column_to_label.update({'sex': 'Gender', 'weight': 'Body Weight'})
        assert len(manager) == 3
        assert manager.get_label('age') == 'Patient Age'
        assert manager.get_label('sex') == 'Gender'


class TestFeatureLabelManagerHasLabel:
    """Test has_label method."""

    def test_has_label_true(self):
        """Test has_label returns True for existing mapping."""
        manager = FeatureLabelManager({'age': 'Patient Age'})
        assert manager.has_label('age') is True

    def test_has_label_false(self):
        """Test has_label returns False for missing mapping."""
        manager = FeatureLabelManager({'age': 'Patient Age'})
        assert manager.has_label('unknown') is False

    def test_has_label_empty_manager(self):
        """Test has_label on empty manager."""
        manager = FeatureLabelManager()
        assert manager.has_label('age') is False


class TestFeatureLabelManagerFromCSV:
    """Test loading from CSV file."""

    def test_from_csv_valid_file(self):
        """Test loading from valid CSV file."""
        # Create temporary CSV file
        csv_content = '''processed_name,user_label,notes
recageyear,Age (year),Original variable
recweightkg,Weight (kg),Original variable
diagn_CAD,Ischemic cardiomyopathy,One-hot encoded
'''
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write(csv_content)
            temp_path = f.name

        try:
            manager = FeatureLabelManager.from_csv(temp_path)
            assert len(manager) == 3
            assert manager.get_label('recageyear') == 'Age (year)'
            assert manager.get_label('recweightkg') == 'Weight (kg)'
            assert manager.get_label('diagn_CAD') == 'Ischemic cardiomyopathy'
        finally:
            os.unlink(temp_path)

    def test_from_csv_with_newlines_in_labels(self):
        """Test CSV with newlines in quoted labels."""
        csv_content = '''processed_name,user_label
recbilirubin,"Serum bilirubin
(umol/L)"
recvad,"Ventricular
assist device"
'''
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write(csv_content)
            temp_path = f.name

        try:
            manager = FeatureLabelManager.from_csv(temp_path)
            assert len(manager) == 2
            # Check for newline (Unix or Windows)
            label = manager.get_label('recbilirubin')
            assert 'Serum bilirubin' in label and '(umol/L)' in label
            assert '\n' in label or '\r\n' in label
            # Check second label also has newline
            label2 = manager.get_label('recvad')
            assert '\n' in label2 or '\r\n' in label2
        finally:
            os.unlink(temp_path)

    def test_from_csv_missing_label_uses_column_name(self):
        """Test that missing labels fall back to column name."""
        csv_content = '''processed_name,user_label
recageyear,Age (year)
recweightkg,
diagn_CAD,
'''
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write(csv_content)
            temp_path = f.name

        try:
            manager = FeatureLabelManager.from_csv(temp_path)
            assert manager.get_label('recageyear') == 'Age (year)'
            # Missing labels should use column name
            assert manager.get_label('recweightkg') == 'recweightkg'
            assert manager.get_label('diagn_CAD') == 'diagn_CAD'
        finally:
            os.unlink(temp_path)

    def test_from_csv_missing_column_name_skipped(self):
        """Test that rows with missing column names are skipped."""
        csv_content = '''processed_name,user_label
recageyear,Age (year)
,Missing Column Name
diagn_CAD,Ischemic cardiomyopathy
'''
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write(csv_content)
            temp_path = f.name

        try:
            manager = FeatureLabelManager.from_csv(temp_path)
            assert len(manager) == 2
            assert manager.get_label('recageyear') == 'Age (year)'
            assert manager.get_label('diagn_CAD') == 'Ischemic cardiomyopathy'
        finally:
            os.unlink(temp_path)

    def test_from_csv_file_not_found(self):
        """Test loading from non-existent file returns empty manager."""
        manager = FeatureLabelManager.from_csv('/nonexistent/path/file.csv')
        assert len(manager) == 0
        assert isinstance(manager, FeatureLabelManager)

    def test_from_csv_wrong_columns(self):
        """Test CSV with wrong column names returns empty manager."""
        csv_content = '''wrong_column,another_wrong_column
value1,value2
'''
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write(csv_content)
            temp_path = f.name

        try:
            manager = FeatureLabelManager.from_csv(temp_path)
            assert len(manager) == 0
        finally:
            os.unlink(temp_path)

    def test_from_csv_custom_column_names(self):
        """Test loading with custom column name parameters."""
        csv_content = '''feature,display_name
recageyear,Age (year)
recweightkg,Weight (kg)
'''
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write(csv_content)
            temp_path = f.name

        try:
            manager = FeatureLabelManager.from_csv(
                temp_path, column_name_col='feature', label_col='display_name'
            )
            assert len(manager) == 2
            assert manager.get_label('recageyear') == 'Age (year)'
            assert manager.get_label('recweightkg') == 'Weight (kg)'
        finally:
            os.unlink(temp_path)

    def test_from_csv_empty_file(self):
        """Test loading from empty CSV returns empty manager."""
        csv_content = '''processed_name,user_label
'''
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write(csv_content)
            temp_path = f.name

        try:
            manager = FeatureLabelManager.from_csv(temp_path)
            assert len(manager) == 0
        finally:
            os.unlink(temp_path)

    def test_from_csv_duplicate_column_names(self):
        """Test CSV with duplicate column names (last one wins)."""
        csv_content = '''processed_name,user_label
recageyear,Age (year)
recageyear,Patient Age
diagn_CAD,Ischemic cardiomyopathy
'''
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write(csv_content)
            temp_path = f.name

        try:
            manager = FeatureLabelManager.from_csv(temp_path)
            # Last mapping should win
            assert manager.get_label('recageyear') == 'Patient Age'
            assert manager.get_label('diagn_CAD') == 'Ischemic cardiomyopathy'
        finally:
            os.unlink(temp_path)


class TestFeatureLabelManagerIntegration:
    """Integration tests with realistic scenarios."""

    def test_typical_usage_workflow(self):
        """Test typical usage workflow."""
        # Create manager from CSV
        csv_content = '''processed_name,user_label
recageyear,Age (year)
recweightkg,Weight (kg)
diagn_CAD,"Ischemic
cardiomyopathy"
'''
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write(csv_content)
            temp_path = f.name

        try:
            manager = FeatureLabelManager.from_csv(temp_path)

            # Add additional mappings programmatically with direct dict access
            manager.column_to_label['custom_feature'] = 'Custom Feature Label'

            # Get labels for plotting with list comprehension
            feature_names = ['recageyear', 'recweightkg', 'diagn_CAD', 'custom_feature', 'unknown']
            labels = [manager.get_label(name) for name in feature_names]

            assert len(labels) == 5
            assert labels[0] == 'Age (year)'
            assert labels[1] == 'Weight (kg)'
            assert '\n' in labels[2]  # Has newline
            assert labels[3] == 'Custom Feature Label'
            assert labels[4] == 'unknown'  # Fallback
        finally:
            os.unlink(temp_path)

    def test_one_hot_encoded_features(self):
        """Test handling one-hot encoded feature labels."""
        mappings = {
            'diagn_CAD': 'Ischemic cardiomyopathy',
            'diagn_Cardiomyopathy': 'Idiopathic cardiomyopathy',
            'diagn_Congenital': 'Congenital heart disease',
            'recethcat_African American': 'African american ethnicity',
            'recethcat_Caucasian': 'Caucasian ethnicity',
        }
        manager = FeatureLabelManager(mappings)

        # Get labels for one-hot group with list comprehension
        diagn_features = ['diagn_CAD', 'diagn_Cardiomyopathy', 'diagn_Congenital']
        diagn_labels = [manager.get_label(name) for name in diagn_features]

        assert len(diagn_labels) == 3
        assert diagn_labels[0] == 'Ischemic cardiomyopathy'
        assert diagn_labels[1] == 'Idiopathic cardiomyopathy'
        assert diagn_labels[2] == 'Congenital heart disease'


class TestGenerateLabelFileTemplate:
    """Tests for generate_label_file_template function."""

    @pytest.fixture
    def original_df(self):
        """Sample original DataFrame before preprocessing."""
        return pd.DataFrame(
            {
                'id': [1, 2, 3],
                'age': [25, 35, 45],
                'weight': [70, 80, 90],
                'diagn': ['CAD', 'Cardiomyopathy', 'Congenital'],
                'status': ['Active', 'Inactive', 'Unknown'],
                'severity': ['Low', 'Medium', 'High'],
                'target': [0, 1, 0],
            }
        )

    @pytest.fixture
    def processed_df(self):
        """Sample processed DataFrame after preprocessing (one-hot, ordinal encoding)."""
        return pd.DataFrame(
            {
                'age': [25, 35, 45],
                'weight': [70, 80, 90],
                'diagn_CAD': [1, 0, 0],
                'diagn_Congenital': [0, 0, 1],
                'status_Active': [1, 0, 0],
                'status_Inactive': [0, 1, 0],
                'severity': [0, 1, 2],
                'target': [0, 1, 0],
            }
        )

    @pytest.fixture
    def sample_metadata(self):
        """Sample preprocessing metadata with encoding information."""
        return {
            'encoding': {
                'diagn': {
                    'encoding_type': 'one-hot',
                    'created_columns': ['diagn_CAD', 'diagn_Cardiomyopathy', 'diagn_Congenital'],
                },
                'status': {
                    'encoding_type': 'one-hot',
                    'created_columns': ['status_Active', 'status_Inactive', 'status_Unknown'],
                },
                'severity': {'encoding_type': 'ordinal', 'created_columns': ['severity']},
            },
            'reference_columns': {
                'references': {'diagn': 'diagn_Cardiomyopathy', 'status': 'status_Unknown'},
                'dropped_columns': ['diagn_Cardiomyopathy', 'status_Unknown'],
            },
        }

    def test_basic_output_file_creation(self, original_df, processed_df, sample_metadata):
        """Test that function creates a CSV file with correct structure."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            output_path = f.name

        try:
            result = generate_label_file_template(
                original_df=original_df,
                processed_df=processed_df,
                target_variable='target',
                id_variable='id',
                output_path=output_path,
                metadata=sample_metadata,
            )

            # Check file exists and can be read
            df = pd.read_csv(output_path)
            assert list(df.columns) == ['original_name', 'processed_name', 'user_label', 'notes']
            assert len(df) > 0
            # Also check return value
            assert isinstance(result, pd.DataFrame)
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_includes_reference_columns(self, original_df, processed_df, sample_metadata):
        """Test that dropped reference columns are included in output."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            output_path = f.name

        try:
            result = generate_label_file_template(
                original_df=original_df,
                processed_df=processed_df,
                target_variable='target',
                id_variable='id',
                output_path=output_path,
                metadata=sample_metadata,
            )

            processed_names = result['processed_name'].tolist()

            # Should include the dropped reference columns
            assert 'diagn_Cardiomyopathy' in processed_names
            assert 'status_Unknown' in processed_names
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_reference_columns_marked_in_notes(self, original_df, processed_df, sample_metadata):
        """Test that reference columns have appropriate note."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            output_path = f.name

        try:
            result = generate_label_file_template(
                original_df=original_df,
                processed_df=processed_df,
                target_variable='target',
                id_variable='id',
                output_path=output_path,
                metadata=sample_metadata,
            )

            ref_row = result[result['processed_name'] == 'diagn_Cardiomyopathy']
            assert len(ref_row) == 1
            assert 'reference' in ref_row['notes'].values[0].lower()
            assert 'dropped' in ref_row['notes'].values[0].lower()
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_one_hot_group_rows_together(self, original_df, processed_df, sample_metadata):
        """Test that one-hot group rows are kept together."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            output_path = f.name

        try:
            result = generate_label_file_template(
                original_df=original_df,
                processed_df=processed_df,
                target_variable='target',
                id_variable='id',
                output_path=output_path,
                metadata=sample_metadata,
            )

            processed_names = result['processed_name'].tolist()

            # Find indices of diagn-related rows (including group name)
            diagn_indices = [
                i
                for i, name in enumerate(processed_names)
                if name.startswith('diagn') or name == 'diagn'
            ]

            # They should be consecutive
            if len(diagn_indices) > 1:
                for i in range(len(diagn_indices) - 1):
                    assert (
                        diagn_indices[i + 1] - diagn_indices[i] == 1
                    ), f"One-hot group rows should be consecutive: {diagn_indices}"
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_group_name_row_exists(self, original_df, processed_df, sample_metadata):
        """Test that group name row exists for one-hot encoded variables."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            output_path = f.name

        try:
            result = generate_label_file_template(
                original_df=original_df,
                processed_df=processed_df,
                target_variable='target',
                id_variable='id',
                output_path=output_path,
                metadata=sample_metadata,
            )

            # Find group name row (original_name == processed_name for group)
            diagn_group_rows = result[
                (result['original_name'] == 'diagn') & (result['processed_name'] == 'diagn')
            ]
            assert (
                len(diagn_group_rows) == 1
            ), f"Should have group name row for diagn. Result:\n{result[result['original_name'] == 'diagn']}"

            # Check note indicates it's a group
            notes = diagn_group_rows['notes'].values[0].lower()
            assert 'one-hot' in notes and 'group' in notes
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_reference_row_before_other_categories(
        self, original_df, processed_df, sample_metadata
    ):
        """Test that reference row comes before other category rows (after group)."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            output_path = f.name

        try:
            result = generate_label_file_template(
                original_df=original_df,
                processed_df=processed_df,
                target_variable='target',
                id_variable='id',
                output_path=output_path,
                metadata=sample_metadata,
            )

            processed_names = result['processed_name'].tolist()

            # Find group name, reference, and first category indices
            diagn_group_idx = None
            diagn_ref_idx = None
            diagn_cad_idx = None

            for i, name in enumerate(processed_names):
                if name == 'diagn':
                    diagn_group_idx = i
                elif name == 'diagn_Cardiomyopathy':
                    diagn_ref_idx = i
                elif name == 'diagn_CAD':
                    diagn_cad_idx = i

            # Order should be: group name -> reference -> other categories
            if diagn_group_idx is not None and diagn_ref_idx is not None:
                assert diagn_group_idx < diagn_ref_idx, "Group name should come before reference"
            if diagn_ref_idx is not None and diagn_cad_idx is not None:
                assert (
                    diagn_ref_idx < diagn_cad_idx
                ), "Reference should come before other categories"
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_variables(self):
        """Test handling of ordinal variables."""
        original_df = pd.DataFrame(
            {'severity': ['Low', 'Medium', 'High'], 'age': [25, 35, 45], 'target': [0, 1, 0]}
        )
        processed_df = pd.DataFrame(
            {'severity': [0, 1, 2], 'age': [25, 35, 45], 'target': [0, 1, 0]}
        )
        metadata = {
            'encoding': {
                'severity': {
                    'encoding_type': 'ordinal',
                    'created_columns': ['severity'],
                    'categories': ['Low', 'Medium', 'High'],
                }
            },
            'reference_columns': {'references': {}, 'dropped_columns': []},
        }

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            output_path = f.name

        try:
            result = generate_label_file_template(
                original_df=original_df,
                processed_df=processed_df,
                target_variable='target',
                id_variable=None,
                output_path=output_path,
                metadata=metadata,
            )

            severity_row = result[result['processed_name'] == 'severity']
            assert len(severity_row) == 1
            assert 'ordinal' in severity_row['notes'].values[0].lower()
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_original_variables(self):
        """Test handling of original (unencoded) variables."""
        original_df = pd.DataFrame(
            {
                'age': [25, 35, 45],
                'weight': [70, 80, 90],
                'height': [170, 175, 180],
                'target': [0, 1, 0],
            }
        )
        processed_df = pd.DataFrame(
            {
                'age': [25, 35, 45],
                'weight': [70, 80, 90],
                'height': [170, 175, 180],
                'target': [0, 1, 0],
            }
        )
        metadata = {'encoding': {}, 'reference_columns': {'references': {}, 'dropped_columns': []}}

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            output_path = f.name

        try:
            result = generate_label_file_template(
                original_df=original_df,
                processed_df=processed_df,
                target_variable='target',
                id_variable=None,
                output_path=output_path,
                metadata=metadata,
            )

            # Should have age, weight, height (not target)
            original_vars = result[result['notes'].str.contains('Original', case=False, na=False)]
            assert len(original_vars) == 3
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_without_metadata(self):
        """Test function works without metadata (minimal mode)."""
        original_df = pd.DataFrame(
            {
                'age': [25, 35, 45],
                'weight': [70, 80, 90],
                'diagn_CAD': [1, 0, 0],
                'target': [0, 1, 0],
            }
        )
        processed_df = original_df.copy()

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            output_path = f.name

        try:
            result = generate_label_file_template(
                original_df=original_df,
                processed_df=processed_df,
                target_variable='target',
                id_variable=None,
                output_path=output_path,
                metadata=None,
            )

            # Should have target, age, weight, diagn_CAD
            assert len(result) == 4
            processed_names = result['processed_name'].tolist()
            assert 'target' in processed_names
            assert 'age' in processed_names
            assert 'weight' in processed_names
            assert 'diagn_CAD' in processed_names
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_excludes_target_and_id(self, original_df, processed_df, sample_metadata):
        """Test that target and id variables are present but marked appropriately."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            output_path = f.name

        try:
            result = generate_label_file_template(
                original_df=original_df,
                processed_df=processed_df,
                target_variable='target',
                id_variable='id',
                output_path=output_path,
                metadata=sample_metadata,
            )

            # Target should be present with TARGET VARIABLE note
            target_row = result[result['processed_name'] == 'target']
            assert len(target_row) == 1
            assert 'TARGET' in target_row['notes'].values[0].upper()

            # ID should NOT be in output if not in processed_df
            # (In this fixture, id is not in processed_df)
            id_rows = result[result['processed_name'] == 'id']
            assert len(id_rows) == 0  # id was dropped in preprocessing
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_empty_dataframes(self):
        """Test with empty DataFrames."""
        original_df = pd.DataFrame()
        processed_df = pd.DataFrame()

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            output_path = f.name

        try:
            result = generate_label_file_template(
                original_df=original_df,
                processed_df=processed_df,
                target_variable=None,
                id_variable=None,
                output_path=output_path,
                metadata=None,
            )

            assert len(result) == 0
            # Verify file was written (may be empty)
            assert os.path.exists(output_path)
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_path_as_string_or_pathlib(self, original_df, processed_df, sample_metadata):
        """Test that output_path accepts both string and Path."""
        # Test with string
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            output_path_str = f.name

        try:
            generate_label_file_template(
                original_df=original_df,
                processed_df=processed_df,
                target_variable='target',
                id_variable='id',
                output_path=output_path_str,
                metadata=sample_metadata,
            )
            assert os.path.exists(output_path_str)
        finally:
            if os.path.exists(output_path_str):
                os.unlink(output_path_str)

        # Test with Path
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            output_path_pathlib = Path(f.name)

        try:
            generate_label_file_template(
                original_df=original_df,
                processed_df=processed_df,
                target_variable='target',
                id_variable='id',
                output_path=output_path_pathlib,
                metadata=sample_metadata,
            )
            assert output_path_pathlib.exists()
        finally:
            if output_path_pathlib.exists():
                output_path_pathlib.unlink()
