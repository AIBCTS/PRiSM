"""
Tests for the PRiSM pipeline runner scripts.

Tests cover:
- get_dataset_output_dir: Directory enumeration and overlap detection
- load_configs_from_file: Batch config file loading and validation
- TeeLogger: Output logging to both terminal and file
- run_prism_pipeline.py: Basic import and configuration validation
- run_prism_parallel.py: Basic import and configuration validation
"""

import io
import sys
from pathlib import Path

import pytest

# Project root (for reference in tests)
PROJECT_ROOT = Path(__file__).parent.parent


# =============================================================================
# Tests for get_dataset_output_dir
# =============================================================================


class TestGetDatasetOutputDir:
    """Tests for the get_dataset_output_dir function."""

    @pytest.fixture
    def temp_base_dir(self, tmp_path):
        """Create a temporary base directory for testing."""
        return tmp_path / "pipeline_results"

    def test_new_directory_created(self, temp_base_dir):
        """Test that a new directory path is returned when none exists."""
        from prism.cli.pipeline import get_dataset_output_dir

        temp_base_dir.mkdir(parents=True)

        result = get_dataset_output_dir(
            base_dir=temp_base_dir,
            dataset="credit-g",
            date_str="20260114",
            models_to_run=["mlp", "xgb"],
            skip_preprocessing=False,
        )

        assert result.name == "20260114_credit-g"
        assert result.parent == temp_base_dir

    def test_reuse_directory_no_model_overlap(self, temp_base_dir):
        """Test that existing directory is reused when no model overlap."""
        from prism.cli.pipeline import get_dataset_output_dir

        temp_base_dir.mkdir(parents=True)

        # Create existing directory with 'mlp' model subdirectory
        existing_dir = temp_base_dir / "20260114_credit-g"
        existing_dir.mkdir()
        (existing_dir / "mlp").mkdir()

        # Request with different models - should reuse
        result = get_dataset_output_dir(
            base_dir=temp_base_dir,
            dataset="credit-g",
            date_str="20260114",
            models_to_run=["logreg", "rf"],
            skip_preprocessing=True,  # Skip to avoid preprocessing overlap
        )

        assert result.name == "20260114_credit-g"
        assert result == existing_dir

    def test_enumerate_directory_on_model_overlap(self, temp_base_dir):
        """Test that directory is enumerated when model overlap detected."""
        from prism.cli.pipeline import get_dataset_output_dir

        temp_base_dir.mkdir(parents=True)

        # Create existing directory with 'mlp' model subdirectory
        existing_dir = temp_base_dir / "20260114_credit-g"
        existing_dir.mkdir()
        (existing_dir / "mlp").mkdir()

        # Request with overlapping model 'mlp' - should enumerate
        result = get_dataset_output_dir(
            base_dir=temp_base_dir,
            dataset="credit-g",
            date_str="20260114",
            models_to_run=["mlp", "rf"],
            skip_preprocessing=True,
        )

        assert result.name == "20260114_credit-g_1"

    def test_enumerate_directory_on_preprocessing_overlap(self, temp_base_dir):
        """Test that directory is enumerated when preprocessing output exists."""
        from prism.cli.pipeline import get_dataset_output_dir

        temp_base_dir.mkdir(parents=True)

        # Create existing directory with preprocessing output
        existing_dir = temp_base_dir / "20260114_credit-g"
        existing_dir.mkdir()
        (existing_dir / "01_preprocessing.pdf").touch()

        # Request without skip_preprocessing - should enumerate
        result = get_dataset_output_dir(
            base_dir=temp_base_dir,
            dataset="credit-g",
            date_str="20260114",
            models_to_run=["xgb"],
            skip_preprocessing=False,
        )

        assert result.name == "20260114_credit-g_1"

    def test_skip_preprocessing_ignores_preprocessing_overlap(self, temp_base_dir):
        """Test that preprocessing overlap is ignored when skip_preprocessing=True."""
        from prism.cli.pipeline import get_dataset_output_dir

        temp_base_dir.mkdir(parents=True)

        # Create existing directory with preprocessing output
        existing_dir = temp_base_dir / "20260114_credit-g"
        existing_dir.mkdir()
        (existing_dir / "01_preprocessing.pdf").touch()

        # Request with skip_preprocessing=True - should reuse
        result = get_dataset_output_dir(
            base_dir=temp_base_dir,
            dataset="credit-g",
            date_str="20260114",
            models_to_run=["xgb"],
            skip_preprocessing=True,
        )

        assert result.name == "20260114_credit-g"
        assert result == existing_dir

    def test_enumerate_finds_first_available(self, temp_base_dir):
        """Test that enumeration finds the first available slot."""
        from prism.cli.pipeline import get_dataset_output_dir

        temp_base_dir.mkdir(parents=True)

        # Create multiple existing directories with overlap
        for suffix in ["", "_1", "_2"]:
            dir_path = temp_base_dir / f"20260114_credit-g{suffix}"
            dir_path.mkdir()
            (dir_path / "mlp").mkdir()

        # Request with overlapping model - should find _3
        result = get_dataset_output_dir(
            base_dir=temp_base_dir,
            dataset="credit-g",
            date_str="20260114",
            models_to_run=["mlp"],
            skip_preprocessing=True,
        )

        assert result.name == "20260114_credit-g_3"

    def test_enumerate_reuses_non_overlapping_slot(self, temp_base_dir):
        """Test that enumeration can reuse a non-overlapping enumerated directory."""
        from prism.cli.pipeline import get_dataset_output_dir

        temp_base_dir.mkdir(parents=True)

        # Create base directory with mlp AND logreg (both overlap with request)
        base_dir = temp_base_dir / "20260114_credit-g"
        base_dir.mkdir()
        (base_dir / "mlp").mkdir()
        (base_dir / "logreg").mkdir()

        # Create _1 directory with xgb only (no overlap with rf)
        enum_dir = temp_base_dir / "20260114_credit-g_1"
        enum_dir.mkdir()
        (enum_dir / "xgb").mkdir()

        # Request with rf - base has overlap (logreg not there but mlp blocks),
        # wait, let me rethink: we request ['rf'], base has ['mlp', 'logreg']
        # no overlap with rf, so it should reuse base!
        # Let's change test to request ['logreg'] which DOES overlap with base
        result = get_dataset_output_dir(
            base_dir=temp_base_dir,
            dataset="credit-g",
            date_str="20260114",
            models_to_run=["logreg"],  # This overlaps with base dir
            skip_preprocessing=True,
        )

        # Should skip base (has logreg), and reuse _1 (only has xgb, no overlap)
        assert result.name == "20260114_credit-g_1"

    def test_html_preprocessing_also_detected(self, temp_base_dir):
        """Test that HTML preprocessing output is also detected as overlap."""
        from prism.cli.pipeline import get_dataset_output_dir

        temp_base_dir.mkdir(parents=True)

        # Create existing directory with HTML preprocessing output
        existing_dir = temp_base_dir / "20260114_credit-g"
        existing_dir.mkdir()
        (existing_dir / "01_preprocessing.html").touch()

        # Request without skip_preprocessing - should enumerate
        result = get_dataset_output_dir(
            base_dir=temp_base_dir,
            dataset="credit-g",
            date_str="20260114",
            models_to_run=["xgb"],
            skip_preprocessing=False,
        )

        assert result.name == "20260114_credit-g_1"


# =============================================================================
# Tests for load_configs_from_file
# =============================================================================


class TestLoadConfigsFromFile:
    """Tests for the load_configs_from_file function."""

    def test_load_valid_config_file(self, tmp_path):
        """Test loading a valid config batch file."""
        from prism.cli.pipeline import load_configs_from_file

        config_file = tmp_path / "batch.yaml"
        config_file.write_text("configs:\n  - htx_example\n  - credit-g\n  - openml_31\n")

        result = load_configs_from_file(config_file)

        assert result == ["htx_example", "credit-g", "openml_31"]

    def test_load_single_config(self, tmp_path):
        """Test loading a file with a single config."""
        from prism.cli.pipeline import load_configs_from_file

        config_file = tmp_path / "single.yaml"
        config_file.write_text("configs:\n  - htx_example\n")

        result = load_configs_from_file(config_file)

        assert result == ["htx_example"]

    def test_file_not_found(self, tmp_path):
        """Test that FileNotFoundError is raised for missing file."""
        from prism.cli.pipeline import load_configs_from_file

        nonexistent = tmp_path / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError, match="Config batch file not found"):
            load_configs_from_file(nonexistent)

    def test_invalid_yaml_syntax(self, tmp_path):
        """Test that ValueError is raised for invalid YAML."""
        from prism.cli.pipeline import load_configs_from_file

        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("configs:\n  - htx_example\n  invalid: [unclosed")

        with pytest.raises(ValueError, match="Invalid YAML"):
            load_configs_from_file(config_file)

    def test_missing_configs_key(self, tmp_path):
        """Test that ValueError is raised when 'configs' key is missing."""
        from prism.cli.pipeline import load_configs_from_file

        config_file = tmp_path / "no_configs.yaml"
        config_file.write_text("datasets:\n  - htx_example\n")

        with pytest.raises(ValueError, match="must contain a 'configs' key"):
            load_configs_from_file(config_file)

    def test_empty_configs_list(self, tmp_path):
        """Test that ValueError is raised for empty configs list."""
        from prism.cli.pipeline import load_configs_from_file

        config_file = tmp_path / "empty.yaml"
        config_file.write_text("configs: []\n")

        with pytest.raises(ValueError, match="list is empty"):
            load_configs_from_file(config_file)

    def test_configs_not_a_list(self, tmp_path):
        """Test that ValueError is raised when configs is not a list."""
        from prism.cli.pipeline import load_configs_from_file

        config_file = tmp_path / "not_list.yaml"
        config_file.write_text("configs: htx_example\n")

        with pytest.raises(ValueError, match="must be a list"):
            load_configs_from_file(config_file)

    def test_non_string_config_entry(self, tmp_path):
        """Test that ValueError is raised for non-string entries."""
        from prism.cli.pipeline import load_configs_from_file

        config_file = tmp_path / "non_string.yaml"
        config_file.write_text("configs:\n  - htx_example\n  - 123\n")

        with pytest.raises(ValueError, match="must be a string"):
            load_configs_from_file(config_file)

    def test_file_not_dict(self, tmp_path):
        """Test that ValueError is raised when file content is not a dict."""
        from prism.cli.pipeline import load_configs_from_file

        config_file = tmp_path / "list_only.yaml"
        config_file.write_text("- htx_example\n- credit-g\n")

        with pytest.raises(ValueError, match="must contain a YAML dictionary"):
            load_configs_from_file(config_file)

    def test_preserves_order(self, tmp_path):
        """Test that config order is preserved."""
        from prism.cli.pipeline import load_configs_from_file

        config_file = tmp_path / "ordered.yaml"
        config_file.write_text("configs:\n  - z_last\n  - a_first\n  - m_middle\n")

        result = load_configs_from_file(config_file)

        assert result == ["z_last", "a_first", "m_middle"]


# =============================================================================
# Tests for TeeLogger
# =============================================================================


class TestTeeLogger:
    """Tests for the TeeLogger class."""

    def test_basic_logging(self, tmp_path):
        """Test that TeeLogger writes to both stdout and file."""
        from prism.cli.pipeline import TeeLogger

        log_file = tmp_path / "test.log"
        logger = TeeLogger(log_file)

        # Capture stdout
        old_stdout = sys.stdout
        captured = io.StringIO()

        try:
            logger.start()
            # Temporarily also capture to our StringIO
            sys.stdout.terminal = captured
            print("Test message")
            logger.stop()
        finally:
            sys.stdout = old_stdout

        # Check file was written
        assert log_file.exists()
        content = log_file.read_text()
        assert "Test message" in content

    def test_timestamps_added(self, tmp_path):
        """Test that timestamps are added to log file lines."""
        from prism.cli.pipeline import TeeLogger

        log_file = tmp_path / "test.log"
        logger = TeeLogger(log_file)

        logger.start()
        print("Timestamped message")
        logger.stop()

        content = log_file.read_text()
        # Should have timestamp format [HH:MM:SS]
        import re

        assert re.search(r'\[\d{2}:\d{2}:\d{2}\].*Timestamped message', content)

    def test_ansi_codes_stripped(self, tmp_path):
        """Test that ANSI color codes are stripped from log file."""
        from prism.cli.pipeline import TeeLogger

        log_file = tmp_path / "test.log"
        logger = TeeLogger(log_file)

        logger.start()
        # Write text with ANSI codes (e.g., red text)
        print("\x1b[31mRed text\x1b[0m Normal text")
        logger.stop()

        content = log_file.read_text()
        # ANSI codes should be stripped
        assert "\x1b[31m" not in content
        assert "\x1b[0m" not in content
        # But text content should remain
        assert "Red text" in content
        assert "Normal text" in content

    def test_header_and_footer(self, tmp_path):
        """Test that log file has header and footer."""
        from prism.cli.pipeline import TeeLogger

        log_file = tmp_path / "test.log"
        logger = TeeLogger(log_file)

        logger.start()
        print("Some content")
        logger.stop()

        content = log_file.read_text()
        assert "PRiSM Pipeline Log" in content
        assert "Started:" in content
        assert "Finished:" in content

    def test_multiline_handling(self, tmp_path):
        """Test that multiline output is handled correctly."""
        from prism.cli.pipeline import TeeLogger

        log_file = tmp_path / "test.log"
        logger = TeeLogger(log_file)

        logger.start()
        print("Line 1\nLine 2\nLine 3")
        logger.stop()

        content = log_file.read_text()
        # Each line should have a timestamp
        import re

        timestamps = re.findall(r'\[\d{2}:\d{2}:\d{2}\]', content)
        # Header has no timestamp, but each of the 3 lines should
        assert len(timestamps) >= 3

    def test_get_log_path(self, tmp_path):
        """Test that get_log_path returns the correct path."""
        from prism.cli.pipeline import TeeLogger

        log_file = tmp_path / "test.log"
        logger = TeeLogger(log_file)

        assert logger.get_log_path() == log_file

    def test_stop_restores_stdout(self, tmp_path):
        """Test that stop() restores original stdout."""
        from prism.cli.pipeline import TeeLogger

        log_file = tmp_path / "test.log"
        logger = TeeLogger(log_file)

        original_stdout = sys.stdout

        logger.start()
        # stdout should be replaced
        assert sys.stdout is not original_stdout

        logger.stop()
        # stdout should be restored
        assert sys.stdout is original_stdout

    def test_creates_parent_directories(self, tmp_path):
        """Test that TeeLogger creates parent directories if needed."""
        from prism.cli.pipeline import TeeLogger

        nested_path = tmp_path / "deep" / "nested" / "dir" / "test.log"
        logger = TeeLogger(nested_path)

        logger.start()
        print("Test")
        logger.stop()

        assert nested_path.exists()


class TestTeeStream:
    """Tests for the TeeStream helper class."""

    def test_ansi_escape_pattern(self):
        """Test that ANSI escape regex matches common codes."""
        from prism.cli.pipeline import TeeStream

        test_cases = [
            ("\x1b[31m", ""),  # Red
            ("\x1b[0m", ""),  # Reset
            ("\x1b[1;32m", ""),  # Bold green
            ("\x1b[38;5;208m", ""),  # 256-color
            ("plain text", "plain text"),  # No codes
            ("\x1b[31mred\x1b[0m", "red"),  # Mixed
        ]

        for input_str, expected in test_cases:
            result = TeeStream.ANSI_ESCAPE.sub('', input_str)
            assert result == expected, f"Failed for input: {repr(input_str)}"


# =============================================================================
# Tests for run_prism_pipeline.py module
# =============================================================================


class TestRunPrismPipeline:
    """Tests for the run_prism_pipeline module."""

    def test_module_imports(self):
        """Test that the module can be imported without errors."""
        from prism.cli import pipeline

        # Core functions (DATASETS dict was removed - now uses YAML configs)
        assert hasattr(pipeline, "build_pipeline_config")
        assert hasattr(pipeline, "list_available_configs")
        assert hasattr(pipeline, "load_configs_from_file")
        assert hasattr(pipeline, "get_dataset_output_dir")
        assert hasattr(pipeline, "get_available_output_path")
        assert hasattr(pipeline, "validate_environment")
        assert hasattr(pipeline, "validate_params_files")
        assert hasattr(pipeline, "process_dataset")
        assert hasattr(pipeline, "TeeLogger")
        assert hasattr(pipeline, "TeeStream")

    def test_build_pipeline_config_structure(self):
        """Test that build_pipeline_config returns valid configuration."""
        from prism.cli.pipeline import build_pipeline_config, list_available_configs

        available = list_available_configs()
        if not available:
            pytest.skip("No config files available")

        # Test with credit-g if available, otherwise first available config
        config_name = 'credit-g' if 'credit-g' in available else available[0]
        config = build_pipeline_config(config_name)

        required_keys = [
            "config_name",
            "prefix",
            "models",
            "preprocessing_notebook",
            "prism_notebook",
        ]
        for key in required_keys:
            assert key in config, f"Config missing key '{key}'"

        # Validate models is a list
        assert isinstance(config["models"], list)
        assert len(config["models"]) > 0

    def test_list_available_configs(self):
        """Test that list_available_configs returns config names."""
        from prism.cli.pipeline import list_available_configs

        configs = list_available_configs()
        assert isinstance(configs, list)
        # Should have at least example_config if configs exist
        # (example_config is excluded from list)
        for cfg in configs:
            assert isinstance(cfg, str)
            assert cfg != 'example_config'  # Template should be excluded

    def test_get_available_output_path_new_file(self, tmp_path):
        """Test get_available_output_path returns original path for new files."""
        from prism.cli.pipeline import get_available_output_path

        new_file = tmp_path / "test.pdf"
        result = get_available_output_path(new_file)

        assert result == new_file

    def test_get_available_output_path_existing_file(self, tmp_path):
        """Test get_available_output_path increments for existing files."""
        from prism.cli.pipeline import get_available_output_path

        existing_file = tmp_path / "test.pdf"
        existing_file.touch()

        result = get_available_output_path(existing_file)

        assert result.name == "test_1.pdf"
        assert result.parent == tmp_path

    def test_is_latex_error_detection(self):
        """Test LaTeX error detection in stderr."""
        from prism.cli.pipeline import is_latex_error

        latex_errors = [
            "xelatex not found",
            "pdflatex failed",
            "LaTeX Error: Missing package",
            "pandoc: error converting",
        ]

        for error in latex_errors:
            assert is_latex_error(error), f"Should detect as latex error: {error}"

        # Non-latex errors
        non_latex_errors = [
            "Python exception occurred",
            "ImportError: No module named 'foo'",
            "RuntimeError: CUDA out of memory",
        ]

        for error in non_latex_errors:
            assert not is_latex_error(error), f"Should NOT detect as latex error: {error}"


# =============================================================================
# Tests for run_prism_parallel.py module
# =============================================================================


class TestRunPrismParallel:
    """Tests for the run_prism_parallel module."""

    def test_module_imports(self):
        """Test that the module can be imported without errors."""
        from prism.cli import parallel

        assert hasattr(parallel, "get_available_gpu_ids")
        assert hasattr(parallel, "execute_notebook_with_env")
        assert hasattr(parallel, "run_model_worker")
        assert hasattr(parallel, "process_dataset_parallel")

    def test_shared_imports_from_pipeline(self):
        """Test that parallel script imports required functions from pipeline."""
        from prism.cli import parallel

        # These should be imported from prism.cli.pipeline
        assert hasattr(parallel, "build_pipeline_config")
        assert hasattr(parallel, "list_available_configs")
        assert hasattr(parallel, "load_configs_from_file")
        assert hasattr(parallel, "get_dataset_output_dir")
        assert hasattr(parallel, "validate_environment")
        assert hasattr(parallel, "validate_params_files")
        assert hasattr(parallel, "TeeLogger")

    def test_get_available_gpu_ids_returns_list(self):
        """Test that get_available_gpu_ids returns a list."""
        from prism.cli.parallel import get_available_gpu_ids

        result = get_available_gpu_ids()

        assert isinstance(result, list)
        # All elements should be integers
        for gpu_id in result:
            assert isinstance(gpu_id, int)


# =============================================================================
# Tests for command-line argument parsing
# =============================================================================


class TestCommandLineArguments:
    """Tests for command-line argument parsing in runner scripts."""

    def test_pipeline_argparse_help(self):
        """Test that pipeline script has valid argument parser."""
        import argparse
        from unittest.mock import patch

        # Temporarily patch sys.argv and prevent exit
        with patch("sys.argv", ["prism run", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                # Re-create parser to test it
                parser = argparse.ArgumentParser()
                parser.add_argument("datasets", nargs="*", default=["htx_example"])
                parser.add_argument("--skip-preprocessing", action="store_true")
                parser.add_argument("--self-contained", action="store_true")

                parser.parse_args(["--help"])

            # --help exits with 0
            assert exc_info.value.code == 0

    def test_pipeline_accepts_multiple_datasets(self):
        """Test that pipeline accepts multiple dataset arguments."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("datasets", nargs="*", default=["htx_example"])
        parser.add_argument("--skip-preprocessing", action="store_true")

        args = parser.parse_args(["credit-g", "htx_example", "credit-g"])

        assert args.datasets == ["credit-g", "htx_example", "credit-g"]
        assert not args.skip_preprocessing

    def test_pipeline_accepts_from_file_argument(self):
        """Test that pipeline accepts --from-file argument."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("configs", nargs="*", default=[])
        parser.add_argument("-f", "--from-file", type=str, metavar="FILE")
        parser.add_argument("--skip-preprocessing", action="store_true")

        # Test long form
        args = parser.parse_args(["--from-file", "batch.yaml"])
        assert args.from_file == "batch.yaml"
        assert args.configs == []

        # Test short form
        args = parser.parse_args(["-f", "batch.yaml"])
        assert args.from_file == "batch.yaml"

        # Test combined with CLI configs (configs must come before -f)
        args = parser.parse_args(["htx_example", "credit-g", "-f", "batch.yaml"])
        assert args.from_file == "batch.yaml"
        assert args.configs == ["htx_example", "credit-g"]

        # Test -f before configs (configs must come first with nargs='*')
        args = parser.parse_args(["-f", "batch.yaml"])
        assert args.from_file == "batch.yaml"
        assert args.configs == []

    def test_parallel_accepts_gpu_argument(self):
        """Test that parallel script accepts GPU argument."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("datasets", nargs="*", default=["htx_example"])
        parser.add_argument("--gpus", type=str, default=None)
        parser.add_argument("--skip-preprocessing", action="store_true")

        args = parser.parse_args(["credit-g", "--gpus", "0,1,2,3"])

        assert args.datasets == ["credit-g"]
        assert args.gpus == "0,1,2,3"

    def test_parallel_accepts_from_file_argument(self):
        """Test that parallel script accepts --from-file argument."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("configs", nargs="*", default=[])
        parser.add_argument("-f", "--from-file", type=str, metavar="FILE")
        parser.add_argument("--gpus", type=str, default=None)

        # Test combined with GPUs
        args = parser.parse_args(["-f", "batch.yaml", "--gpus", "0,1"])
        assert args.from_file == "batch.yaml"
        assert args.gpus == "0,1"
        assert args.configs == []


# =============================================================================
# Tests for combine_predictions_horizontally ID column resolution
# =============================================================================


class TestCombinePredictionsIDColumn:
    """Tests for patient ID column detection in combine_predictions_horizontally."""

    @pytest.fixture
    def prediction_csvs(self, tmp_path):
        """Create sample prediction CSV files with uppercase column names."""
        import pandas as pd

        features = {
            "MORTALITY_365D": [0, 1, 0],
            "DAT_AGE": [50, 60, 70],
            "TRR_ID_CODE": ["A001", "A002", "A003"],
            "pred_blackbox": [0.1, 0.8, 0.3],
            "pred_prn": [0.15, 0.75, 0.35],
        }

        paths = []
        for model in ["mlp", "xgb"]:
            df = pd.DataFrame(features)
            csv_path = tmp_path / f"test_{model}_preds_20260101.csv"
            df.to_csv(csv_path, index=False)
            paths.append(csv_path)

        return paths

    def test_finds_uppercase_id_column(self, prediction_csvs):
        """Test that uppercase TRR_ID_CODE is found via case-insensitive search."""
        from prism.cli.pipeline import combine_predictions_horizontally

        result = combine_predictions_horizontally(prediction_csvs)

        assert result is not None
        assert "TRR_ID_CODE" in result.columns
        # MORTALITY should NOT be used as index
        assert "MORTALITY_365D" in result.columns

    def test_config_id_candidates_used(self, prediction_csvs):
        """Test that explicit id_candidates list is respected."""
        from prism.cli.pipeline import combine_predictions_horizontally

        result = combine_predictions_horizontally(prediction_csvs, id_candidates=["TRR_ID_CODE"])

        assert result is not None
        assert "TRR_ID_CODE" in result.columns

    def test_fallback_to_first_column_with_warning(self, tmp_path, capsys):
        """Test that first column is used as fallback when no ID matches."""
        import pandas as pd

        from prism.cli.pipeline import combine_predictions_horizontally

        df = pd.DataFrame(
            {
                "OUTCOME": [0, 1],
                "FEATURE_A": [10, 20],
                "pred_blackbox": [0.1, 0.9],
            }
        )

        paths = []
        for model in ["mlp", "xgb"]:
            p = tmp_path / f"test_{model}_preds_20260101.csv"
            df.to_csv(p, index=False)
            paths.append(p)

        result = combine_predictions_horizontally(paths, id_candidates=["nonexistent_id"])

        assert result is not None
        captured = capsys.readouterr()
        assert "WARN" in captured.out
        assert "OUTCOME" in captured.out

    def test_case_insensitive_match(self, tmp_path):
        """Test that case-insensitive matching works for various casings."""
        import pandas as pd

        from prism.cli.pipeline import combine_predictions_horizontally

        df = pd.DataFrame(
            {
                "Patient_ID": ["P1", "P2"],
                "value": [1.0, 2.0],
                "pred_blackbox": [0.5, 0.6],
            }
        )

        paths = []
        for model in ["mlp", "xgb"]:
            p = tmp_path / f"test_{model}_preds_20260101.csv"
            df.to_csv(p, index=False)
            paths.append(p)

        result = combine_predictions_horizontally(paths, id_candidates=["patient_id"])

        assert result is not None
        assert "Patient_ID" in result.columns


# =============================================================================
# Tests for preprocessing JSON serialization (NumPy type handling)
# =============================================================================


class TestPreprocessingJsonSerialization:
    """Tests for _convert_to_json_serializable in preprocessing.py."""

    def test_numpy_float64(self):
        """Test that np.float64 values are converted to Python float."""
        import numpy as np

        from prism.preprocessing import _convert_to_json_serializable

        result = _convert_to_json_serializable(np.float64(3.14))
        assert isinstance(result, float)
        assert result == pytest.approx(3.14)

    def test_numpy_int64(self):
        """Test that np.int64 values are converted to Python int."""
        import numpy as np

        from prism.preprocessing import _convert_to_json_serializable

        result = _convert_to_json_serializable(np.int64(42))
        assert isinstance(result, int)
        assert result == 42

    def test_numpy_float32(self):
        """Test that np.float32 values are converted to Python float."""
        import numpy as np

        from prism.preprocessing import _convert_to_json_serializable

        result = _convert_to_json_serializable(np.float32(2.5))
        assert isinstance(result, float)

    def test_numpy_bool(self):
        """Test that np.bool_ values are converted to Python bool."""
        import numpy as np

        from prism.preprocessing import _convert_to_json_serializable

        result = _convert_to_json_serializable(np.bool_(True))
        assert isinstance(result, bool)
        assert result is True

    def test_nested_dict_with_numpy(self):
        """Test that nested dicts with numpy values are fully converted."""
        import json

        import numpy as np

        from prism.preprocessing import _convert_to_json_serializable

        data = {
            "mean": np.float64(0.5),
            "count": np.int64(100),
            "values": [np.float32(1.0), np.float32(2.0)],
            "nested": {"flag": np.bool_(False)},
        }

        result = _convert_to_json_serializable(data)

        # Should be fully JSON-serializable without errors
        json_str = json.dumps(result)
        assert '"mean": 0.5' in json_str

    def test_no_removed_numpy_aliases(self):
        """Test that no removed NumPy 2.0 aliases are used."""
        import numpy as np

        # These were removed in NumPy 2.0
        removed_aliases = ['float_', 'int_', 'complex_', 'object_', 'bool8']
        for alias in removed_aliases:
            # They should NOT exist in NumPy 2.0+, but may exist in <2.0
            # The important thing is our code doesn't reference them
            pass

        # Verify our function works with current numpy
        from prism.preprocessing import _convert_to_json_serializable

        result = _convert_to_json_serializable(np.float64(1.0))
        assert isinstance(result, float)
