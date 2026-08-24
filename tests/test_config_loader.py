"""Tests for prism.config_loader module."""

import logging
import tempfile
from pathlib import Path

import pandas as pd
import pytest
import yaml

from prism.config_loader import (
    DEFAULT_ID_CANDIDATES,
    DEFAULT_LABEL_FILE_CANDIDATES,
    DEFAULT_TARGET_CANDIDATES,
    LassoConfigurationError,
    LassoLambdaConfig,
    apply_lasso_lambda_selection,
    detect_target_and_id_columns,
    get_lasso_lambda_config,
    load_config_if_exists,
    load_label_file,
    parse_lasso_lambda_config,
)


class TestDefaultCandidates:
    """Test that default candidate lists are properly defined."""

    def test_default_target_candidates_not_empty(self):
        """Default target candidates should have entries."""
        assert len(DEFAULT_TARGET_CANDIDATES) > 0

    def test_default_id_candidates_not_empty(self):
        """Default ID candidates should have entries."""
        assert len(DEFAULT_ID_CANDIDATES) > 0

    def test_default_label_file_candidates_not_empty(self):
        """Default label file candidates should have entries."""
        assert len(DEFAULT_LABEL_FILE_CANDIDATES) > 0

    def test_common_targets_in_defaults(self):
        """Common target names should be in defaults."""
        assert 'target' in DEFAULT_TARGET_CANDIDATES
        assert 'y' in DEFAULT_TARGET_CANDIDATES
        assert 'event_oneyear' in DEFAULT_TARGET_CANDIDATES

    def test_common_ids_in_defaults(self):
        """Common ID names should be in defaults."""
        assert 'id' in DEFAULT_ID_CANDIDATES
        assert 'patient_id' in DEFAULT_ID_CANDIDATES


class TestLoadConfigIfExists:
    """Tests for load_config_if_exists function."""

    def test_returns_empty_dict_when_no_config_dir(self):
        """Should return empty dict when config directory doesn't exist."""
        result = load_config_if_exists('nonexistent', config_dir='/nonexistent/path')
        assert result == {}

    def test_returns_empty_dict_when_no_config_file(self):
        """Should return empty dict when config file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = load_config_if_exists('nonexistent_dataset', config_dir=tmpdir)
            assert result == {}

    def test_loads_valid_yaml_config(self):
        """Should load valid YAML config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_data = {
                'random_seed': 42,
                'target_candidates': ['my_target'],
                'splitting_method': 'temporal',
            }
            config_path = Path(tmpdir) / 'test_dataset.yaml'
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)

            result = load_config_if_exists('test_dataset', config_dir=tmpdir)
            assert result == config_data

    def test_returns_empty_dict_for_empty_yaml(self):
        """Should return empty dict for empty YAML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'empty_dataset.yaml'
            config_path.touch()  # Create empty file

            result = load_config_if_exists('empty_dataset', config_dir=tmpdir)
            assert result == {}

    def test_returns_empty_dict_for_invalid_yaml(self):
        """Should return empty dict for invalid YAML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'invalid_dataset.yaml'
            with open(config_path, 'w') as f:
                f.write("invalid: yaml: content: [")  # Invalid YAML

            result = load_config_if_exists('invalid_dataset', config_dir=tmpdir)
            assert result == {}

    def test_loads_complex_config(self):
        """Should load config with complex nested structures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_data = {
                'integer_encoding': {
                    'education': ['low', 'medium', 'high'],
                    'income': ['poor', 'middle', 'rich'],
                },
                'split_ratios': [0.6, 0.2, 0.2],
                'reference_columns': {
                    'category_a': 'category_a_ref',
                    'category_b': 'category_b_ref',
                },
            }
            config_path = Path(tmpdir) / 'complex_dataset.yaml'
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)

            result = load_config_if_exists('complex_dataset', config_dir=tmpdir)
            assert result == config_data
            assert result['integer_encoding']['education'] == ['low', 'medium', 'high']


class TestDetectTargetAndIdColumns:
    """Tests for detect_target_and_id_columns function."""

    def test_detects_target_from_candidates(self):
        """Should detect target column from candidate list."""
        df = pd.DataFrame({'target': [0, 1, 0], 'feature1': [1, 2, 3]})
        target_col, id_col = detect_target_and_id_columns(df, verbose=False)
        assert target_col == 'target'

    def test_detects_id_from_candidates(self):
        """Should detect ID column from candidate list."""
        df = pd.DataFrame({'patient_id': ['A', 'B', 'C'], 'feature1': [1, 2, 3]})
        target_col, id_col = detect_target_and_id_columns(df, verbose=False)
        assert id_col == 'patient_id'

    def test_detects_both_target_and_id(self):
        """Should detect both target and ID columns."""
        df = pd.DataFrame(
            {
                'target': [0, 1, 0],
                'id': ['A', 'B', 'C'],
                'feature1': [1, 2, 3],
            }
        )
        target_col, id_col = detect_target_and_id_columns(df, verbose=False)
        assert target_col == 'target'
        assert id_col == 'id'

    def test_respects_candidate_order(self):
        """Should respect order of candidates (first match wins)."""
        df = pd.DataFrame(
            {
                'y': [0, 1, 0],
                'target': [1, 0, 1],
                'feature1': [1, 2, 3],
            }
        )
        # 'target' should win over 'y' because it comes earlier in default candidates
        target_col, id_col = detect_target_and_id_columns(df, verbose=False)
        assert target_col == 'target'

    def test_uses_custom_candidates(self):
        """Should use custom candidate lists when provided."""
        df = pd.DataFrame(
            {
                'my_custom_target': [0, 1, 0],
                'my_custom_id': ['A', 'B', 'C'],
                'feature1': [1, 2, 3],
            }
        )
        target_col, id_col = detect_target_and_id_columns(
            df,
            target_candidates=['my_custom_target'],
            id_candidates=['my_custom_id'],
            verbose=False,
        )
        assert target_col == 'my_custom_target'
        assert id_col == 'my_custom_id'

    def test_fallback_to_binary_column(self):
        """Should fall back to binary column when no candidate matches."""
        df = pd.DataFrame(
            {
                'binary_outcome': [0, 1, 0, 1],
                'feature1': [1, 2, 3, 4],
                'feature2': [5, 6, 7, 8],
            }
        )
        target_col, id_col = detect_target_and_id_columns(
            df,
            target_candidates=['nonexistent_target'],
            verbose=False,
        )
        assert target_col == 'binary_outcome'

    def test_returns_none_when_no_match(self):
        """Should return None when no columns match."""
        df = pd.DataFrame(
            {
                'feature1': [1, 2, 3],
                'feature2': [4, 5, 6],
                'feature3': [7, 8, 9],  # No binary column
            }
        )
        target_col, id_col = detect_target_and_id_columns(
            df,
            target_candidates=['nonexistent'],
            id_candidates=['also_nonexistent'],
            verbose=False,
        )
        # No binary fallback either since no 0/1 column
        assert target_col is None
        assert id_col is None

    def test_verbose_mode(self, capsys):
        """Should print messages when verbose=True."""
        df = pd.DataFrame({'target': [0, 1], 'id': ['A', 'B']})
        detect_target_and_id_columns(df, verbose=True)
        captured = capsys.readouterr()
        assert 'Detecting target and ID columns' in captured.out
        assert 'Found target column' in captured.out

    def test_handles_missing_values_in_binary_detection(self):
        """Should handle NaN values when detecting binary columns."""
        df = pd.DataFrame(
            {
                'binary_with_nan': [0, 1, None, 0, 1],
                'feature1': [1, 2, 3, 4, 5],
            }
        )
        target_col, id_col = detect_target_and_id_columns(
            df,
            target_candidates=['nonexistent'],
            verbose=False,
        )
        assert target_col == 'binary_with_nan'


class TestLoadLabelFile:
    """Tests for load_label_file function."""

    def test_returns_none_when_no_file_found(self):
        """Should return None when no label file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = load_label_file(
                label_file_candidates=['nonexistent.csv'],
                data_dir=tmpdir,
            )
            assert result is None

    def test_loads_template_format(self):
        """Should load template format with processed_name and user_label columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            label_data = pd.DataFrame(
                {
                    'original_name': ['var1', 'var2'],
                    'processed_name': ['var1_ordinal', 'var2_ordinal'],
                    'user_label': ['Age (years)', 'Weight (kg)'],
                    'notes': ['Original', 'Original'],
                }
            )
            label_path = Path(tmpdir) / 'generic_variable_labels.csv'
            label_data.to_csv(label_path, index=False)

            result = load_label_file(
                label_file_candidates=['generic_variable_labels.csv'],
                data_dir=tmpdir,
            )
            assert result is not None
            assert result['var1_ordinal'] == 'Age (years)'
            assert result['var2_ordinal'] == 'Weight (kg)'

    def test_skips_placeholder_labels(self):
        """Should skip labels that start with [USER:."""
        with tempfile.TemporaryDirectory() as tmpdir:
            label_data = pd.DataFrame(
                {
                    'original_name': ['var1', 'var2'],
                    'processed_name': ['var1_ordinal', 'var2_ordinal'],
                    'user_label': ['Age (years)', '[USER: provide label]'],
                    'notes': ['Original', 'Original'],
                }
            )
            label_path = Path(tmpdir) / 'generic_variable_labels.csv'
            label_data.to_csv(label_path, index=False)

            result = load_label_file(
                label_file_candidates=['generic_variable_labels.csv'],
                data_dir=tmpdir,
            )
            assert result is not None
            assert result['var1_ordinal'] == 'Age (years)'
            assert result['var2_ordinal'] == 'var2_ordinal'  # Falls back to name

    def test_loads_simple_two_column_format(self):
        """Should load simple two-column format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            label_path = Path(tmpdir) / 'labels__.csv'
            with open(label_path, 'w') as f:
                f.write('var1,Age\n')
                f.write('var2,Weight\n')

            result = load_label_file(
                label_file_candidates=['labels__.csv'],
                data_dir=tmpdir,
            )
            assert result is not None
            assert result['var1'] == 'Age'
            assert result['var2'] == 'Weight'

    def test_handles_escaped_newlines(self):
        """Should convert escaped newlines in labels."""
        with tempfile.TemporaryDirectory() as tmpdir:
            label_path = Path(tmpdir) / 'htx_variable_labels.csv'
            with open(label_path, 'w') as f:
                f.write('var1,Age\\n(years)\n')

            result = load_label_file(
                label_file_candidates=['htx_variable_labels.csv'],
                data_dir=tmpdir,
            )
            assert result is not None
            assert result['var1'] == 'Age\n(years)'

    def test_dataset_prefix_substitution(self):
        """Should substitute {dataset_prefix} in file paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            label_path = Path(tmpdir) / 'mydata_label.csv'
            with open(label_path, 'w') as f:
                f.write('var1,Age\n')

            result = load_label_file(
                label_file_candidates=['{dataset_prefix}_label.csv'],
                data_dir=tmpdir,
                dataset_prefix='mydata',
            )
            assert result is not None
            assert result['var1'] == 'Age'

    def test_searches_candidates_in_order(self):
        """Should search candidate files in order and return first found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create second file only
            label_path = Path(tmpdir) / 'second.csv'
            with open(label_path, 'w') as f:
                f.write('var1,From Second\n')

            result = load_label_file(
                label_file_candidates=['first.csv', 'second.csv'],
                data_dir=tmpdir,
            )
            assert result is not None
            assert result['var1'] == 'From Second'


class TestLoadLabelFilePrecedence:
    """Tests for label file precedence and shadowing behavior."""

    def test_dataset_specific_takes_precedence_over_generic(self):
        """Dataset-specific file should be selected over generic template."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create generic file
            generic_path = Path(tmpdir) / 'generic_variable_labels.csv'
            pd.DataFrame(
                {
                    'processed_name': ['var1'],
                    'user_label': ['From Generic'],
                }
            ).to_csv(generic_path, index=False)

            # Create dataset-specific file
            specific_path = Path(tmpdir) / 'credit-g_labels.csv'
            with open(specific_path, 'w') as f:
                f.write('var1,From Specific\n')

            result = load_label_file(
                label_file_candidates=[
                    'credit-g_labels.csv',
                    'generic_variable_labels.csv',
                ],
                data_dir=tmpdir,
                dataset_prefix='credit-g',
            )

            assert result is not None
            assert result['var1'] == 'From Specific'

    def test_warns_on_multiple_dataset_specific_files(self, caplog):
        """Should warn when multiple dataset-specific files exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two dataset-specific files
            for filename in ['credit-g_variable_labels.csv', 'credit-g_labels.csv']:
                path = Path(tmpdir) / filename
                with open(path, 'w') as f:
                    f.write('var1,Label\n')

            with caplog.at_level(logging.WARNING):
                result = load_label_file(
                    label_file_candidates=[
                        'credit-g_variable_labels.csv',
                        'credit-g_labels.csv',
                    ],
                    data_dir=tmpdir,
                    dataset_prefix='credit-g',
                )

            assert result is not None
            assert any('Multiple dataset-specific' in rec.message for rec in caplog.records)

    def test_info_on_dataset_specific_shadowing_generic(self, caplog):
        """Should log info when dataset-specific shadows generic (expected behavior)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create both types
            for filename in ['credit-g_labels.csv', 'generic_variable_labels.csv']:
                path = Path(tmpdir) / filename
                if filename == 'generic_variable_labels.csv':
                    pd.DataFrame(
                        {
                            'processed_name': ['var1'],
                            'user_label': ['Label'],
                        }
                    ).to_csv(path, index=False)
                else:
                    with open(path, 'w') as f:
                        f.write('var1,Label\n')

            with caplog.at_level(logging.INFO):
                result = load_label_file(
                    label_file_candidates=[
                        'credit-g_labels.csv',
                        'generic_variable_labels.csv',
                    ],
                    data_dir=tmpdir,
                    dataset_prefix='credit-g',
                )

            assert result is not None
            # Should be INFO level, not WARNING (this is expected behavior)
            assert any('dataset-specific' in rec.message.lower() for rec in caplog.records)

    def test_verbose_mode_prints_scan_details(self, capsys):
        """Verbose mode should print scan details to stdout."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'test_labels.csv'
            with open(path, 'w') as f:
                f.write('var1,Label\n')

            result = load_label_file(
                label_file_candidates=['test_labels.csv'],
                data_dir=tmpdir,
                verbose=True,
            )

            captured = capsys.readouterr()
            assert 'Scanning for label files' in captured.out
            assert 'Candidates to check' in captured.out
            assert 'Found' in captured.out

    def test_verbose_shows_selected_marker(self, capsys):
        """Verbose mode should mark which file was selected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two files
            for filename in ['first.csv', 'second.csv']:
                with open(Path(tmpdir) / filename, 'w') as f:
                    f.write('var1,Label\n')

            result = load_label_file(
                label_file_candidates=['first.csv', 'second.csv'],
                data_dir=tmpdir,
                verbose=True,
            )

            captured = capsys.readouterr()
            assert 'SELECTED' in captured.out
            assert 'shadowed' in captured.out

    def test_no_verbose_output_by_default(self, capsys):
        """Verbose should default to False (no print output)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'test.csv'
            with open(path, 'w') as f:
                f.write('var1,Label\n')

            result = load_label_file(
                label_file_candidates=['test.csv'],
                data_dir=tmpdir,
            )

            captured = capsys.readouterr()
            # Should have NO print output (only logger output)
            assert captured.out == ''

    def test_handles_none_dataset_prefix(self):
        """Should not crash when dataset_prefix is None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'test.csv'
            with open(path, 'w') as f:
                f.write('var1,Label\n')

            result = load_label_file(
                label_file_candidates=['test.csv'],
                data_dir=tmpdir,
                dataset_prefix=None,
            )

            assert result is not None
            assert result['var1'] == 'Label'

    def test_respects_new_precedence_order(self):
        """Should respect new precedence order (dataset-specific before generic)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files in reverse order to ensure precedence works
            generic_path = Path(tmpdir) / 'data_label.csv'
            with open(generic_path, 'w') as f:
                f.write('var1,Generic Label\n')

            specific_path = Path(tmpdir) / 'mydata_variable_labels.csv'
            with open(specific_path, 'w') as f:
                f.write('var1,Specific Label\n')

            # Use DEFAULT_LABEL_FILE_CANDIDATES to test the actual precedence
            result = load_label_file(
                data_dir=tmpdir,
                dataset_prefix='mydata',
            )

            assert result is not None
            # Should get specific label, not generic
            assert result['var1'] == 'Specific Label'

    def test_warns_on_multiple_generic_files(self, caplog):
        """Should warn when multiple generic files exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple generic files
            for filename in ['data_label.csv', 'generic_variable_labels.csv']:
                path = Path(tmpdir) / filename
                if filename == 'generic_variable_labels.csv':
                    pd.DataFrame(
                        {
                            'processed_name': ['var1'],
                            'user_label': ['Label'],
                        }
                    ).to_csv(path, index=False)
                else:
                    with open(path, 'w') as f:
                        f.write('var1,Label\n')

            with caplog.at_level(logging.WARNING):
                result = load_label_file(
                    label_file_candidates=['data_label.csv', 'generic_variable_labels.csv'],
                    data_dir=tmpdir,
                    dataset_prefix=None,
                )

            assert result is not None
            assert any('Multiple generic' in rec.message for rec in caplog.records)

    def test_verbose_with_no_files_found(self, capsys):
        """Verbose mode should print when no files are found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = load_label_file(
                label_file_candidates=['nonexistent.csv'],
                data_dir=tmpdir,
                verbose=True,
            )

            captured = capsys.readouterr()
            assert 'No label files found' in captured.out
            assert result is None

    def test_dataset_specific_precedence_with_multiple_formats(self):
        """Dataset-specific files should take precedence across different formats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create generic template format
            generic_path = Path(tmpdir) / 'generic_variable_labels.csv'
            pd.DataFrame(
                {
                    'processed_name': ['var1', 'var2'],
                    'user_label': ['Generic Var1', 'Generic Var2'],
                }
            ).to_csv(generic_path, index=False)

            # Create dataset-specific simple format (different format than generic)
            specific_path = Path(tmpdir) / 'htx_variable_labels.csv'
            with open(specific_path, 'w') as f:
                f.write('var1,Specific Var1\n')
                f.write('var2,Specific Var2\n')

            result = load_label_file(
                label_file_candidates=[
                    'htx_variable_labels.csv',
                    'generic_variable_labels.csv',
                ],
                data_dir=tmpdir,
                dataset_prefix='htx',
            )

            assert result is not None
            assert result['var1'] == 'Specific Var1'
            assert result['var2'] == 'Specific Var2'

    def test_detects_template_format_by_column_headers(self):
        """Should detect template format by column headers, not just filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dataset-specific file with template format
            # (Not named 'template' or 'generic_variable_labels')
            specific_path = Path(tmpdir) / 'mydata_variable_labels.csv'
            pd.DataFrame(
                {
                    'original_name': ['var1', 'var2'],
                    'processed_name': ['var1_processed', 'var2_processed'],
                    'user_label': ['Variable One', 'Variable Two'],
                    'notes': ['Note 1', 'Note 2'],
                }
            ).to_csv(specific_path, index=False)

            result = load_label_file(
                label_file_candidates=['mydata_variable_labels.csv'],
                data_dir=tmpdir,
                dataset_prefix='mydata',
            )

            assert result is not None
            assert result['var1_processed'] == 'Variable One'
            assert result['var2_processed'] == 'Variable Two'

    def test_skips_placeholder_labels_in_template_format(self):
        """Should skip placeholder labels starting with [USER: in template format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            label_path = Path(tmpdir) / 'test_labels.csv'
            pd.DataFrame(
                {
                    'original_name': ['var1', 'var2', 'var3'],
                    'processed_name': ['var1', 'var2', 'var3'],
                    'user_label': ['Real Label', '[USER: provide label]', 'Another Label'],
                    'notes': ['', '', ''],
                }
            ).to_csv(label_path, index=False)

            result = load_label_file(
                label_file_candidates=['test_labels.csv'],
                data_dir=tmpdir,
            )

            assert result is not None
            assert result['var1'] == 'Real Label'
            assert result['var2'] == 'var2'  # Falls back to processed_name
            assert result['var3'] == 'Another Label'


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_config_provides_custom_candidates(self):
        """Config-loaded candidates should work with detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create config
            config_data = {
                'target_candidates': ['custom_outcome'],
                'id_candidates': ['custom_id'],
            }
            config_path = Path(tmpdir) / 'test.yaml'
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)

            # Load config
            config = load_config_if_exists('test', config_dir=tmpdir)

            # Use config candidates for detection
            df = pd.DataFrame(
                {
                    'custom_outcome': [0, 1, 0],
                    'custom_id': ['A', 'B', 'C'],
                    'feature': [1, 2, 3],
                }
            )
            target_col, id_col = detect_target_and_id_columns(
                df,
                target_candidates=config.get('target_candidates', DEFAULT_TARGET_CANDIDATES),
                id_candidates=config.get('id_candidates', DEFAULT_ID_CANDIDATES),
                verbose=False,
            )

            assert target_col == 'custom_outcome'
            assert id_col == 'custom_id'


class TestLassoLambdaConfig:
    """Tests for LASSO lambda selection configuration parsing."""

    def test_parse_max_test_auc_with_defaults(self):
        """Should parse max_test_auc with default values."""
        config = {
            'lasso_lambda_selection': {
                'blackbox': {'method': 'max_test_auc'},
                'prn': {'method': 'max_test_auc'},
            }
        }
        result = parse_lasso_lambda_config(config, 'blackbox')
        assert result.method == 'max_test_auc'
        assert result.beta_threshold == 0.1
        assert result.target_ratio == 0.99

    def test_parse_max_test_auc_with_custom_values(self):
        """Should parse max_test_auc with custom values."""
        config = {
            'lasso_lambda_selection': {
                'blackbox': {
                    'method': 'max_test_auc',
                    'target_ratio': 0.998,
                    'beta_threshold': 0.05,
                },
                'prn': {'method': 'max_test_auc'},
            }
        }
        result = parse_lasso_lambda_config(config, 'blackbox')
        assert result.method == 'max_test_auc'
        assert result.beta_threshold == 0.05
        assert result.target_ratio == 0.998

    def test_parse_min_test_auc(self):
        """Should parse min_test_auc with required min_auc."""
        config = {
            'lasso_lambda_selection': {
                'blackbox': {'method': 'min_test_auc', 'min_auc': 0.75},
                'prn': {'method': 'min_test_auc', 'min_auc': 0.70},
            }
        }
        result = parse_lasso_lambda_config(config, 'blackbox')
        assert result.method == 'min_test_auc'
        assert result.min_auc == 0.75
        assert result.beta_threshold == 0.1

    def test_parse_by_features(self):
        """Should parse by_features with required target_features."""
        config = {
            'lasso_lambda_selection': {
                'blackbox': {'method': 'by_features', 'target_features': 10},
                'prn': {'method': 'by_features', 'target_features': 15},
            }
        }
        result = parse_lasso_lambda_config(config, 'blackbox')
        assert result.method == 'by_features'
        assert result.target_features == 10

    def test_parse_by_index(self):
        """Should parse by_index with required lambda_index."""
        config = {
            'lasso_lambda_selection': {
                'blackbox': {'method': 'by_index', 'lambda_index': 42},
                'prn': {'method': 'by_index', 'lambda_index': 12},
            }
        }
        result = parse_lasso_lambda_config(config, 'blackbox')
        assert result.method == 'by_index'
        assert result.lambda_index == 42

    def test_missing_config_raises_error(self):
        """Should raise LassoConfigurationError when config is missing."""
        with pytest.raises(LassoConfigurationError, match="not configured"):
            parse_lasso_lambda_config({}, 'blackbox')

    def test_missing_config_none_raises_error(self):
        """Should raise LassoConfigurationError when config is None."""
        with pytest.raises(LassoConfigurationError, match="not configured"):
            parse_lasso_lambda_config(None, 'blackbox')

    def test_missing_stage_raises_error(self):
        """Should raise LassoConfigurationError when stage is missing."""
        config = {'lasso_lambda_selection': {'blackbox': {'method': 'max_test_auc'}}}
        with pytest.raises(LassoConfigurationError, match="'prn' not configured"):
            parse_lasso_lambda_config(config, 'prn')

    def test_missing_method_raises_error(self):
        """Should raise LassoConfigurationError when method is missing."""
        config = {
            'lasso_lambda_selection': {
                'blackbox': {'threshold': 0.1},  # No method
                'prn': {'method': 'max_test_auc'},
            }
        }
        with pytest.raises(LassoConfigurationError, match="Missing 'method'"):
            parse_lasso_lambda_config(config, 'blackbox')

    def test_invalid_method_raises_error(self):
        """Should raise LassoConfigurationError for invalid method."""
        config = {
            'lasso_lambda_selection': {
                'blackbox': {'method': 'invalid_method'},
                'prn': {'method': 'max_test_auc'},
            }
        }
        with pytest.raises(LassoConfigurationError, match="Invalid LASSO selection method"):
            parse_lasso_lambda_config(config, 'blackbox')

    def test_by_features_missing_target_raises_error(self):
        """Should raise LassoConfigurationError when target_features is missing."""
        config = {
            'lasso_lambda_selection': {
                'blackbox': {'method': 'by_features'},  # Missing target_features
                'prn': {'method': 'max_test_auc'},
            }
        }
        with pytest.raises(LassoConfigurationError, match="requires 'target_features'"):
            parse_lasso_lambda_config(config, 'blackbox')

    def test_by_features_invalid_target_raises_error(self):
        """Should raise LassoConfigurationError for invalid target_features."""
        config = {
            'lasso_lambda_selection': {
                'blackbox': {'method': 'by_features', 'target_features': -5},
                'prn': {'method': 'max_test_auc'},
            }
        }
        with pytest.raises(LassoConfigurationError, match="must be a positive integer"):
            parse_lasso_lambda_config(config, 'blackbox')

    def test_by_index_missing_lambda_index_raises_error(self):
        """Should raise LassoConfigurationError when lambda_index is missing."""
        config = {
            'lasso_lambda_selection': {
                'blackbox': {'method': 'by_index'},  # Missing lambda_index
                'prn': {'method': 'max_test_auc'},
            }
        }
        with pytest.raises(LassoConfigurationError, match="requires 'lambda_index'"):
            parse_lasso_lambda_config(config, 'blackbox')

    def test_by_index_invalid_lambda_index_raises_error(self):
        """Should raise LassoConfigurationError for negative lambda_index."""
        config = {
            'lasso_lambda_selection': {
                'blackbox': {'method': 'by_index', 'lambda_index': -1},
                'prn': {'method': 'max_test_auc'},
            }
        }
        with pytest.raises(LassoConfigurationError, match="must be a non-negative integer"):
            parse_lasso_lambda_config(config, 'blackbox')

    def test_min_test_auc_missing_min_auc_raises_error(self):
        """Should raise LassoConfigurationError when min_auc is missing."""
        config = {
            'lasso_lambda_selection': {
                'blackbox': {'method': 'min_test_auc'},  # Missing min_auc
                'prn': {'method': 'max_test_auc'},
            }
        }
        with pytest.raises(LassoConfigurationError, match="requires 'min_auc'"):
            parse_lasso_lambda_config(config, 'blackbox')

    def test_min_test_auc_invalid_range_raises_error(self):
        """Should raise LassoConfigurationError for invalid min_auc range."""
        config = {
            'lasso_lambda_selection': {
                'blackbox': {'method': 'min_test_auc', 'min_auc': 1.5},
                'prn': {'method': 'max_test_auc'},
            }
        }
        with pytest.raises(LassoConfigurationError, match="must be a float between 0 and 1"):
            parse_lasso_lambda_config(config, 'blackbox')

    def test_max_test_auc_invalid_target_ratio_raises_error(self):
        """Should raise LassoConfigurationError for invalid target_ratio."""
        config = {
            'lasso_lambda_selection': {
                'blackbox': {'method': 'max_test_auc', 'target_ratio': 1.5},
                'prn': {'method': 'max_test_auc'},
            }
        }
        with pytest.raises(LassoConfigurationError, match="must be a float between 0 and 1"):
            parse_lasso_lambda_config(config, 'blackbox')

    def test_parses_prn_stage(self):
        """Should correctly parse PRN stage configuration."""
        config = {
            'lasso_lambda_selection': {
                'blackbox': {'method': 'max_test_auc'},
                'prn': {'method': 'by_features', 'target_features': 5, 'beta_threshold': 0.2},
            }
        }
        result = parse_lasso_lambda_config(config, 'prn')
        assert result.method == 'by_features'
        assert result.target_features == 5
        assert result.beta_threshold == 0.2

    def test_model_specific_config(self):
        """Should parse model-specific configuration when model is specified."""
        config = {
            'lasso_lambda_selection': {
                'blackbox': {'method': 'max_test_auc', 'target_ratio': 0.99},
                'prn': {'method': 'max_test_auc'},
                'mlp': {
                    'blackbox': {'method': 'by_features', 'target_features': 10},
                    'prn': {'method': 'by_features', 'target_features': 8},
                },
            }
        }
        # With model='mlp', should get model-specific config
        result = get_lasso_lambda_config(config, 'blackbox', model='mlp')
        assert result.method == 'by_features'
        assert result.target_features == 10

        result = get_lasso_lambda_config(config, 'prn', model='mlp')
        assert result.method == 'by_features'
        assert result.target_features == 8

    def test_model_specific_falls_back_to_generic(self):
        """Should fall back to generic config when model-specific not found."""
        config = {
            'lasso_lambda_selection': {
                'blackbox': {'method': 'max_test_auc', 'target_ratio': 0.998},
                'prn': {'method': 'max_test_auc', 'target_ratio': 0.95},
            }
        }
        # Model 'xgb' not in config, should fall back to generic
        result = get_lasso_lambda_config(config, 'blackbox', model='xgb')
        assert result.method == 'max_test_auc'
        assert result.target_ratio == 0.998

    def test_model_specific_with_partial_stages(self):
        """Should fall back to generic for stages not in model-specific config."""
        config = {
            'lasso_lambda_selection': {
                'blackbox': {'method': 'max_test_auc', 'target_ratio': 0.99},
                'prn': {'method': 'max_test_auc', 'target_ratio': 0.95},
                'mlp': {
                    'blackbox': {'method': 'by_features', 'target_features': 10},
                    # prn not specified for mlp
                },
            }
        }
        # mlp.blackbox exists, should use model-specific
        result = get_lasso_lambda_config(config, 'blackbox', model='mlp')
        assert result.method == 'by_features'
        assert result.target_features == 10

        # mlp.prn not in config, should fall back to generic
        result = get_lasso_lambda_config(config, 'prn', model='mlp')
        assert result.method == 'max_test_auc'
        assert result.target_ratio == 0.95

    def test_parse_model_specific_directly(self):
        """Should parse model-specific config when called directly with model param."""
        config = {
            'lasso_lambda_selection': {
                'xgb': {
                    'blackbox': {'method': 'min_test_auc', 'min_auc': 0.80},
                    'prn': {'method': 'min_test_auc', 'min_auc': 0.75},
                },
            }
        }
        result = parse_lasso_lambda_config(config, 'blackbox', model='xgb')
        assert result.method == 'min_test_auc'
        assert result.min_auc == 0.80


class TestNonInferiorityLassoConfig:
    """Tests for non_inferiority LASSO lambda selection configuration."""

    def test_parse_non_inferiority_with_default_ni_level(self):
        """Should parse non_inferiority with default ni_level=0.1."""
        config = {
            'lasso_lambda_selection': {
                'blackbox': {'method': 'non_inferiority'},
                'prn': {'method': 'non_inferiority'},
            }
        }
        result = parse_lasso_lambda_config(config, 'blackbox')
        assert result.method == 'non_inferiority'
        assert result.ni_level == 0.1
        assert result.beta_threshold == 0.1

    def test_parse_non_inferiority_with_custom_ni_level(self):
        """Should parse non_inferiority with custom ni_level."""
        config = {
            'lasso_lambda_selection': {
                'blackbox': {'method': 'non_inferiority', 'ni_level': 0.2},
                'prn': {'method': 'non_inferiority', 'ni_level': 0.15},
            }
        }
        result = parse_lasso_lambda_config(config, 'blackbox')
        assert result.method == 'non_inferiority'
        assert result.ni_level == 0.2

        result_prn = parse_lasso_lambda_config(config, 'prn')
        assert result_prn.ni_level == 0.15

    def test_parse_non_inferiority_with_custom_beta_threshold(self):
        """Should parse non_inferiority with custom beta_threshold."""
        config = {
            'lasso_lambda_selection': {
                'blackbox': {'method': 'non_inferiority', 'ni_level': 0.1, 'beta_threshold': 0.05},
                'prn': {'method': 'non_inferiority'},
            }
        }
        result = parse_lasso_lambda_config(config, 'blackbox')
        assert result.method == 'non_inferiority'
        assert result.ni_level == 0.1
        assert result.beta_threshold == 0.05

    def test_parse_non_inferiority_ni_level_one_valid(self):
        """ni_level=1.0 should be valid (accepts down to AUC=0.5)."""
        config = {
            'lasso_lambda_selection': {
                'blackbox': {'method': 'non_inferiority', 'ni_level': 1.0},
                'prn': {'method': 'non_inferiority'},
            }
        }
        result = parse_lasso_lambda_config(config, 'blackbox')
        assert result.method == 'non_inferiority'
        assert result.ni_level == 1.0

    def test_parse_non_inferiority_ni_level_zero_raises_error(self):
        """ni_level=0 should raise error (identical to max_test_auc)."""
        config = {
            'lasso_lambda_selection': {
                'blackbox': {'method': 'non_inferiority', 'ni_level': 0},
                'prn': {'method': 'non_inferiority'},
            }
        }
        with pytest.raises(LassoConfigurationError, match="must be a float in range"):
            parse_lasso_lambda_config(config, 'blackbox')

    def test_parse_non_inferiority_negative_ni_level_raises_error(self):
        """Negative ni_level should raise error."""
        config = {
            'lasso_lambda_selection': {
                'blackbox': {'method': 'non_inferiority', 'ni_level': -0.1},
                'prn': {'method': 'non_inferiority'},
            }
        }
        with pytest.raises(LassoConfigurationError, match="must be a float in range"):
            parse_lasso_lambda_config(config, 'blackbox')

    def test_parse_non_inferiority_ni_level_greater_than_one_raises_error(self):
        """ni_level > 1 should raise error."""
        config = {
            'lasso_lambda_selection': {
                'blackbox': {'method': 'non_inferiority', 'ni_level': 1.5},
                'prn': {'method': 'non_inferiority'},
            }
        }
        with pytest.raises(LassoConfigurationError, match="must be a float in range"):
            parse_lasso_lambda_config(config, 'blackbox')

    def test_get_lasso_lambda_config_returns_default_for_missing_config(self):
        """Should return default config when non_inferiority not configured."""
        # Test with None config
        result = get_lasso_lambda_config(None, 'blackbox')
        assert result.method == 'max_test_auc'  # Default method

    def test_non_inferiority_in_model_specific_config(self):
        """Should parse non_inferiority in model-specific config."""
        config = {
            'lasso_lambda_selection': {
                'blackbox': {'method': 'max_test_auc'},
                'prn': {'method': 'max_test_auc'},
                'mlp': {
                    'blackbox': {'method': 'non_inferiority', 'ni_level': 0.15},
                    'prn': {'method': 'non_inferiority', 'ni_level': 0.2},
                },
            }
        }
        result = get_lasso_lambda_config(config, 'blackbox', model='mlp')
        assert result.method == 'non_inferiority'
        assert result.ni_level == 0.15

        result_prn = get_lasso_lambda_config(config, 'prn', model='mlp')
        assert result_prn.method == 'non_inferiority'
        assert result_prn.ni_level == 0.2


class TestApplyLassoLambdaSelection:
    """Tests for apply_lasso_lambda_selection with reference_auc."""

    def test_reference_auc_forwarded_to_non_inferiority(self):
        """reference_auc should be passed through to select_lambda_non_inferiority."""
        from unittest.mock import MagicMock

        mock_results = MagicMock()
        mock_results.select_lambda_non_inferiority.return_value = 5

        config = LassoLambdaConfig(method='non_inferiority', ni_level=0.15)
        result = apply_lasso_lambda_selection(mock_results, config, reference_auc=0.85)

        mock_results.select_lambda_non_inferiority.assert_called_once_with(
            threshold=config.beta_threshold,
            ni_level=0.15,
            reference_auc=0.85,
        )
        assert result == 5

    def test_reference_auc_none_forwarded_to_non_inferiority(self):
        """reference_auc=None should be forwarded (backward compat)."""
        from unittest.mock import MagicMock

        mock_results = MagicMock()
        mock_results.select_lambda_non_inferiority.return_value = 3

        config = LassoLambdaConfig(method='non_inferiority', ni_level=0.1)
        apply_lasso_lambda_selection(mock_results, config)

        mock_results.select_lambda_non_inferiority.assert_called_once_with(
            threshold=config.beta_threshold,
            ni_level=0.1,
            reference_auc=None,
        )

    def test_reference_auc_ignored_for_max_test_auc(self):
        """reference_auc should not affect max_test_auc method."""
        from unittest.mock import MagicMock

        mock_results = MagicMock()
        mock_results.select_lambda_max_test_auc.return_value = 9

        config = LassoLambdaConfig(method='max_test_auc')
        result = apply_lasso_lambda_selection(mock_results, config, reference_auc=0.85)

        mock_results.select_lambda_max_test_auc.assert_called_once()
        mock_results.select_lambda_non_inferiority.assert_not_called()
        assert result == 9
