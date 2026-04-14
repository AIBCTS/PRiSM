#!/usr/bin/env python3
"""
PRiSM Notebook Pipeline Runner

Executes preprocessing, modeling, and PRiSM analysis notebooks using YAML configs.

Config files are located in example_notebooks/config/ and specify:
- dataset: The raw data file to use (maps to data/raw/{dataset}.csv)
- models: List of models to train and analyze
- Preprocessing settings (splitting, encoding, etc.)

Outputs are saved to example_notebooks/pipeline_results/{YYYYMMDD}_{config}/:
- Numbered notebook exports (PDF with HTML fallback)
- Trained models, predictions, performance summaries (in models/ subfolder)
- Nomogram JSON data (in models/nomogram/ subfolder)
- Log files for each pipeline stage
- Pipeline run log with timestamps (pipeline_run_YYYYMMDD_HHMMSS.log)

Output Directory Structure:
    pipeline_results/
    +-- pipeline_run_20251218_143022.log  # Timestamped run log
    +-- 20251218_htx_example/
        +-- preprocessing_metadata_htx_example_*.json
        +-- 01_preprocessing.pdf
        +-- mlp/
        |   +-- 02_train_mlp.pdf
        |   +-- 03_prism_analysis_mlp.pdf
        |   +-- models/
        |       +-- htx_example_mlp/
        |       |   +-- htx_example_mlp_model_*.pt
        |       +-- predictions/
        |       +-- performance_summaries/
        |       +-- nomogram/
        +-- xgb/
            +-- 04_train_xgb.pdf
            +-- 05_prism_analysis_xgb.pdf
            +-- models/

Examples
--------
    # Run with a config (loads example_notebooks/config/htx_example.yaml)
    python run_prism_pipeline.py htx_example

    # Run multiple configs
    python run_prism_pipeline.py htx_example openml_31

    # Run configs from a batch file
    python run_prism_pipeline.py --from-file example_notebooks/config/my_batch.yaml
    python run_prism_pipeline.py -f my_batch.yaml  # short form

    # Combine CLI configs with batch file (CLI runs first)
    python run_prism_pipeline.py extra_config -f batch.yaml

    # Skip preprocessing (assumes preprocessed files exist)
    python run_prism_pipeline.py htx_example --skip-preprocessing

    # Create fully self-contained run (copies data to output folder)
    python run_prism_pipeline.py htx_example --self-contained

    # Skip hyperparameter tuning even if enabled in YAML config
    python run_prism_pipeline.py htx_example --no-tune

    # List available configs
    python run_prism_pipeline.py --list-configs

Batch Config File Format
------------------------
Create a YAML file with a 'configs' list:

    # my_batch.yaml
    configs:
      - htx_example
      - openml_31

Hyperparameter Tuning
---------------------
Tuning is controlled via YAML config (hyperparameter_tuning.{model}.enabled: true).
When enabled, tuning runs inline during each model's training notebook execution.

To skip tuning (use defaults or previously saved params): --no-tune

For pre-tuning multiple models or sharing params across runs, use the dedicated
tuning script and specify params_file in your YAML config:
    python run_hyperparameter_tuning.py htx_example --models mlp xgb
    # Then in YAML: hyperparameter_tuning.mlp.params_file: 'models/..._best_params.json'
"""

import argparse
import atexit
import json
import os
import re
import shutil
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml

# Project paths -- CLI operates on user's working directory
PROJECT_ROOT = Path.cwd()
NOTEBOOKS_DIR = PROJECT_ROOT / "example_notebooks"

from prism.config import INTERIM_DATA_DIR, MODELS_DIR, PROCESSED_DATA_DIR  # noqa: E402
from prism.config_loader import get_models_from_config, load_config  # noqa: E402

# Generate today's date for output directory
today_date = datetime.now().strftime("%Y%m%d")
# OUTPUT_DIR is set per-dataset in process_dataset() as: {date}_{dataset}/

# Global state for cleanup
temp_files_to_cleanup = []
current_subprocess = None
current_output_dir = None  # Set per-dataset
run_start_time = None  # Timestamp when run starts, used to filter output summary
gpu_monitor_stop_event = None
gpu_monitor_thread = None
tee_logger = None  # Global logger for tee output


# =============================================================================
# Logging Utilities (Tee to Terminal + File)
# =============================================================================


class TeeLogger:
    """Tee stdout/stderr to both terminal and a log file with timestamps.

    Features:
    - Duplicates all output to terminal and log file
    - Strips ANSI color codes from log file for cleaner logs
    - Adds timestamps to each line in log file
    - Thread-safe for basic operations
    """

    # Regex to match ANSI escape codes
    ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    def __init__(self, log_file_path: Path):
        self.log_file_path = log_file_path
        self.log_file = None
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self._line_buffer = ""

    def start(self):
        """Start tee logging - redirect stdout/stderr."""
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(self.log_file_path, 'w', encoding='utf-8')

        # Write header
        self.log_file.write(f"{'='*70}\n")
        self.log_file.write("PRiSM Pipeline Log\n")
        self.log_file.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log_file.write(f"{'='*70}\n\n")
        self.log_file.flush()

        # Replace stdout/stderr with tee wrappers
        sys.stdout = TeeStream(self.original_stdout, self.log_file, add_timestamps=True)
        sys.stderr = TeeStream(
            self.original_stderr, self.log_file, add_timestamps=True, prefix="[STDERR] "
        )

    def stop(self):
        """Stop tee logging - restore stdout/stderr and close log file."""
        if self.log_file:
            # Restore original streams
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr

            # Write footer
            self.log_file.write(f"\n{'='*70}\n")
            self.log_file.write(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.log_file.write(f"{'='*70}\n")
            self.log_file.close()
            self.log_file = None

    def get_log_path(self) -> Path:
        """Return the path to the log file."""
        return self.log_file_path


class TeeStream:
    """A stream that writes to both terminal and log file."""

    ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    def __init__(self, terminal_stream, log_file, add_timestamps=True, prefix=""):
        self.terminal = terminal_stream
        self.log_file = log_file
        self.add_timestamps = add_timestamps
        self.prefix = prefix
        self._at_line_start = True

    def write(self, message):
        # Write to terminal with original formatting (including ANSI codes)
        self.terminal.write(message)

        # Strip ANSI codes for log file
        clean_message = self.ANSI_ESCAPE.sub('', message)

        # Add timestamps at the start of each line
        if self.add_timestamps and clean_message:
            timestamped_lines = []
            lines = clean_message.split('\n')

            for i, line in enumerate(lines):
                is_last = i == len(lines) - 1

                if self._at_line_start and line:
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    timestamped_lines.append(f"[{timestamp}] {self.prefix}{line}")
                else:
                    timestamped_lines.append(line)

                # Track if we're at line start for next write
                if not is_last:
                    self._at_line_start = True
                else:
                    # Last segment: at line start only if it's empty (ended with \n)
                    self._at_line_start = line == ''

            clean_message = '\n'.join(timestamped_lines)

        self.log_file.write(clean_message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def fileno(self):
        return self.terminal.fileno()

    def isatty(self):
        return self.terminal.isatty()


# =============================================================================
# PRiSM-Only Mode Validation
# =============================================================================


def validate_prism_only_source(
    source_dir: Path,
    dataset_prefix: str,
    models: List[str],
) -> Dict:
    """Validate source directory for prism-only mode and return validated paths.

    Checks that all required files exist in the source directory:
    - Preprocessing metadata JSON
    - Model files for each model
    - Processed data files (in self-contained dir or standard location)

    Parameters
    ----------
    source_dir : Path
        Path to previous pipeline run directory
    dataset_prefix : str
        Dataset prefix (from 'dataset' field in YAML config)
    models : List[str]
        List of models to process

    Returns
    -------
    Dict
        Dictionary with validated paths:
        - metadata_file: Path to preprocessing metadata
        - processed_data_dir: Path to processed data directory
        - model_dirs: {model: Path} mapping to model directories

    Raises
    ------
    FileNotFoundError
        If source directory or required files are missing
    ValueError
        If dataset prefix in filenames doesn't match config
    """
    errors = []

    print(f"\n[PRiSM-Only] Validating source: {source_dir}")

    # 1. Check source directory exists
    if not source_dir.exists():
        raise FileNotFoundError(
            f"ERROR: PRiSM-only source directory not found:\n"
            f"  {source_dir}\n\n"
            f"Ensure prism_only_source_dir in your config points to a valid pipeline run."
        )

    if not source_dir.is_dir():
        raise FileNotFoundError(
            f"ERROR: prism_only_source_dir is not a directory:\n" f"  {source_dir}"
        )

    # 2. Find preprocessing metadata
    metadata_pattern = f"preprocessing_metadata_{dataset_prefix}_*.json"
    metadata_files = list(source_dir.glob(metadata_pattern))

    if not metadata_files:
        errors.append(
            f"  Preprocessing metadata not found:\n"
            f"    Expected: {source_dir}/{metadata_pattern}\n"
            f"    Searched: {source_dir}"
        )
        metadata_file = None
    else:
        # Use latest by mtime if multiple
        metadata_file = max(metadata_files, key=lambda p: p.stat().st_mtime)
        print(f"  [OK] Metadata: {metadata_file.name}")

    # 3. Locate processed data directory
    # Check self-contained location first, then standard
    self_contained_processed = source_dir / "data" / "processed"

    if self_contained_processed.exists() and self_contained_processed.is_dir():
        processed_data_dir = self_contained_processed
        print(f"  [OK] Processed data: {processed_data_dir} (self-contained)")
    else:
        # Fall back to standard location
        processed_data_dir = PROCESSED_DATA_DIR
        print(f"  [OK] Processed data: {processed_data_dir} (standard location)")

    # Verify processed data files exist (check for at least train file)
    # Pattern: {dataset}_{model}_{timestamp}_{split}.csv OR {dataset}_{split}.csv
    train_files = list(processed_data_dir.glob(f"{dataset_prefix}_*_train.csv"))
    if not train_files:
        # Try simpler pattern without model
        train_files = list(processed_data_dir.glob(f"{dataset_prefix}_train.csv"))

    if not train_files:
        errors.append(
            f"  Processed data files not found:\n"
            f"    Expected: {dataset_prefix}_*_train.csv or {dataset_prefix}_train.csv\n"
            f"    Searched: {processed_data_dir}"
        )

    # 4. Validate model files for each model
    model_dirs = {}

    for model in models:
        # mlp10 uses mlp model
        model_to_find = 'mlp' if model == 'mlp10' else model

        # Expected path: {source}/{model}/models/{dataset}_{model}/
        model_base_dir = source_dir / model_to_find / "models"

        if not model_base_dir.exists():
            errors.append(
                f"  Model '{model}' directory not found:\n"
                f"    Expected: {model_base_dir}\n"
                f"    Searched: {source_dir / model_to_find}/"
            )
            continue

        # Find model file using pattern
        model_files = find_model_files(model_base_dir, dataset_prefix, model_to_find)

        if not model_files:
            errors.append(
                f"  Model '{model}' file not found:\n"
                f"    Expected: {model_base_dir}/{dataset_prefix}_{model_to_find}/{dataset_prefix}_{model_to_find}_model_*.pt\n"
                f"    Searched: {model_base_dir}"
            )
            continue

        # Check prefix matches
        latest_model = model_files[0]
        filename = latest_model.name
        expected_prefix = f"{dataset_prefix}_{model_to_find}_"

        if not filename.startswith(expected_prefix):
            # Extract actual prefix from filename for error message
            # Pattern: {prefix}_{model}_model_{timestamp}.pt
            parts = filename.split('_model_')
            if parts:
                actual_prefix = parts[0].rsplit('_', 1)[0] if '_' in parts[0] else parts[0]
            else:
                actual_prefix = "unknown"

            errors.append(
                f"  Prefix mismatch for '{model}':\n"
                f"    Config dataset: '{dataset_prefix}'\n"
                f"    File prefix: '{actual_prefix}' (from {filename})"
            )
            continue

        model_dirs[model] = model_base_dir
        print(f"  [OK] {model} model: {latest_model.name}")

    # 5. Check for partial responses files (optional, used for caching)
    # These are optional - if not found, partial responses will be recalculated
    partial_responses_files = {}
    for model in models:
        model_to_find = 'mlp' if model == 'mlp10' else model
        pr_dir = source_dir / model_to_find / "models" / "partial_responses"

        if pr_dir.exists():
            pr_pattern = f"blackbox_{dataset_prefix}_{model_to_find}_*_partial_responses.pt"
            pr_files = list(pr_dir.glob(pr_pattern))
            if pr_files:
                latest_pr = max(pr_files, key=lambda p: p.stat().st_mtime)
                partial_responses_files[model] = latest_pr
                print(f"  [OK] {model} partial responses: {latest_pr.name}")
            else:
                print(f"  [INFO] {model} partial responses: not found (will recalculate)")
        else:
            print(f"  [INFO] {model} partial responses dir not found (will recalculate)")

    # 6. Check for LASSO results files (optional, used for caching)
    # These are optional - if not found, LASSO sweep will be re-run
    lasso_results_files = {}
    for model in models:
        model_to_find = 'mlp' if model == 'mlp10' else model
        lasso_dir = source_dir / model_to_find / "models" / "lasso_results"

        if lasso_dir.exists():
            lasso_pattern = f"blackbox_{dataset_prefix}_{model_to_find}_*_lasso.pt"
            lasso_files = list(lasso_dir.glob(lasso_pattern))
            if lasso_files:
                latest_lasso = max(lasso_files, key=lambda p: p.stat().st_mtime)
                lasso_results_files[model] = latest_lasso
                print(f"  [OK] {model} LASSO results: {latest_lasso.name}")
            else:
                print(f"  [INFO] {model} LASSO results: not found (will recalculate)")
        else:
            print(f"  [INFO] {model} LASSO results dir not found (will recalculate)")

    # 7. Check for PRN model files (optional, used for PRN caching)
    # These are optional - if not found and load_cached_prn=true, will fail at runtime
    prn_model_files = {}
    prn_partial_responses_files = {}
    prn_lasso_results_files = {}

    for model in models:
        model_to_find = 'mlp' if model == 'mlp10' else model
        prn_model_identifier = f"{dataset_prefix}_{model_to_find}_prn"

        # Check PRN model file
        prn_model_dir = source_dir / model_to_find / "models" / prn_model_identifier
        if prn_model_dir.exists():
            prn_pattern = f"{prn_model_identifier}_model_tuned.pt"
            prn_files = list(prn_model_dir.glob(prn_pattern))
            if prn_files:
                latest_prn = max(prn_files, key=lambda p: p.stat().st_mtime)
                prn_model_files[model] = latest_prn
                print(f"  [OK] {model} PRN model: {latest_prn.name}")
            else:
                print(f"  [INFO] {model} PRN model: not found")
        else:
            print(f"  [INFO] {model} PRN model dir not found")

        # Check PRN partial responses (optional)
        pr_dir = source_dir / model_to_find / "models" / "partial_responses"
        if pr_dir.exists():
            prn_pr_pattern = f"prn_{dataset_prefix}_{model_to_find}_*_partial_responses.pt"
            prn_pr_files = list(pr_dir.glob(prn_pr_pattern))
            if prn_pr_files:
                latest_prn_pr = max(prn_pr_files, key=lambda p: p.stat().st_mtime)
                prn_partial_responses_files[model] = latest_prn_pr
                print(f"  [OK] {model} PRN partial responses: {latest_prn_pr.name}")
            else:
                print(f"  [INFO] {model} PRN partial responses: not found (will recalculate)")

        # Check PRN LASSO results (optional)
        lasso_dir = source_dir / model_to_find / "models" / "lasso_results"
        if lasso_dir.exists():
            prn_lasso_pattern = f"prn_{dataset_prefix}_{model_to_find}_*_lasso.pt"
            prn_lasso_files = list(lasso_dir.glob(prn_lasso_pattern))
            if prn_lasso_files:
                latest_prn_lasso = max(prn_lasso_files, key=lambda p: p.stat().st_mtime)
                prn_lasso_results_files[model] = latest_prn_lasso
                print(f"  [OK] {model} PRN LASSO results: {latest_prn_lasso.name}")
            else:
                print(f"  [INFO] {model} PRN LASSO results: not found (will recalculate)")

    # 8. Report results
    if errors:
        error_msg = (
            "\nERROR: PRiSM-only validation failed:\n\n"
            + "\n\n".join(errors)
            + "\n\nEnsure prism_only_source_dir points to a complete pipeline run\n"
            f"matching the 'dataset' field ('{dataset_prefix}') in your config."
        )
        raise FileNotFoundError(error_msg)

    print("[PRiSM-Only] Validation complete - all files found\n")

    return {
        "metadata_file": metadata_file,
        "processed_data_dir": processed_data_dir,
        "model_dirs": model_dirs,
        "partial_responses_files": partial_responses_files,
        "lasso_results_files": lasso_results_files,
        "prn_model_files": prn_model_files,
        "prn_partial_responses_files": prn_partial_responses_files,
        "prn_lasso_results_files": prn_lasso_results_files,
    }


def validate_load_cached_prn_requirements(
    config: Dict,
    validated_paths: Dict,
    models: List[str],
) -> None:
    """Validate that PRN model files exist when load_cached_prn=true.

    This is a fail-fast validation to catch missing PRN files early,
    before the pipeline starts running notebooks.

    Parameters
    ----------
    config : Dict
        Configuration dictionary from YAML
    validated_paths : Dict
        Dictionary returned by validate_prism_only_source()
    models : List[str]
        List of model names to check

    Raises
    ------
    FileNotFoundError
        If load_cached_prn is enabled but PRN model files are missing for any model
    """
    if not config.get('load_cached_prn', False):
        return  # Not using PRN caching, skip validation

    missing_prn_models = []
    for model in models:
        if model not in validated_paths.get('prn_model_files', {}):
            missing_prn_models.append(model)

    if missing_prn_models:
        raise FileNotFoundError(
            f"ERROR: load_cached_prn=true but PRN model files not found for:\n"
            f"  {missing_prn_models}\n\n"
            f"Either:\n"
            f"  1. Set load_cached_prn=false to train PRN from scratch\n"
            f"  2. Ensure PRN was trained in the source run before enabling caching\n"
            f"  3. Check that the source directory contains PRN model files"
        )

    print("[OK] PRN cache validation passed - all required PRN models found")


def validate_prism_only_source_config_ordering(
    configs_to_process: List[str],
) -> Dict[str, str]:
    """Validate ordering of configs with prism_only_source_config references.

    Ensures that any config referencing another config via prism_only_source_config
    appears AFTER the referenced config in the processing order. This is required
    because the referenced config's output directory is needed as the source.

    Parameters
    ----------
    configs_to_process : List[str]
        Ordered list of config names to process

    Returns
    -------
    Dict[str, str]
        Mapping of config name -> source config name for configs using
        prism_only_source_config. Empty dict if none use this feature.

    Raises
    ------
    ValueError
        If a config references a config not in the list, or if a config
        references a config that appears after it in the processing order.
    """
    # Build position index for ordering validation
    config_order = {name: i for i, name in enumerate(configs_to_process)}

    # Track which configs use prism_only_source_config
    source_config_refs = {}

    for config_name in configs_to_process:
        yaml_config, _ = load_config(config_name)
        source_config = yaml_config.get('prism_only_source_config')

        if source_config:
            # Validate referenced config exists in the list
            if source_config not in config_order:
                raise ValueError(
                    f"Config '{config_name}' references '{source_config}' via "
                    f"prism_only_source_config, but '{source_config}' is not in the "
                    f"configs list: {configs_to_process}\n"
                    f"Either add '{source_config}' to the list or use "
                    f"prism_only_source_dir with an explicit path instead."
                )

            # Validate ordering: referenced config must come BEFORE referencing config
            if config_order[source_config] >= config_order[config_name]:
                raise ValueError(
                    f"Config ordering error: '{config_name}' references '{source_config}' "
                    f"via prism_only_source_config, but '{source_config}' appears at "
                    f"position {config_order[source_config]} while '{config_name}' is at "
                    f"position {config_order[config_name]}.\n"
                    f"The referenced config must appear BEFORE the referencing config.\n"
                    f"Reorder to: [..., '{source_config}', ..., '{config_name}', ...]"
                )

            source_config_refs[config_name] = source_config

    return source_config_refs


def save_prism_only_reproducibility(
    output_dir: Path,
    config_name: str,
    dataset_prefix: str,
    source_dir: Path,
    source_config_name: str = None,
    models: List[str] = None,
) -> None:
    """Save reproducibility artifacts for PRiSM-only runs.

    For prism-only runs, we save different artifacts than full runs:
    - Config YAML copy (the prism-only config with different LASSO/PRN settings)
    - Source reference file (JSON with source path/config name, timestamp)
    - Copy of source's preprocessing metadata (for reference without needing source)
    - PRN tuning results (if PRN hyperparameter tuning ran)

    We do NOT duplicate:
    - Raw data (already in source or standard location)
    - Interim/processed data (using existing from source)

    Parameters
    ----------
    output_dir : Path
        Output directory for this prism-only run
    config_name : str
        Name of the prism-only config file (without .yaml extension)
    dataset_prefix : str
        Dataset prefix (for finding metadata and param files)
    source_dir : Path
        Path to the source directory (previous pipeline run)
    source_config_name : str, optional
        If using prism_only_source_config, the name of the source config
    models : List[str], optional
        List of models processed (for finding PRN param files)
    """
    print("\nSaving PRiSM-only reproducibility artifacts...")

    repro_dir = output_dir / "reproducibility"
    repro_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy config YAML file
    config_dir = PROJECT_ROOT / "example_notebooks" / "config"
    config_file = config_dir / f"{config_name}.yaml"

    if config_file.exists():
        dest_config = repro_dir / f"{config_name}.yaml"
        try:
            shutil.copy(config_file, dest_config)
            print(f"  [OK] Copied config: {config_file.name}")
        except Exception as e:
            print(f"  [WARN] Could not copy config: {e}")
    else:
        print(f"  [WARN] Config file not found: {config_file}")

    # 2. Save source reference file
    source_ref = {
        "prism_only_run": True,
        "generated_at": datetime.now().isoformat(),
        "config_name": config_name,
        "dataset_prefix": dataset_prefix,
        "source_directory": str(source_dir),
        "source_config_name": source_config_name,  # None if using prism_only_source_dir
        "models_processed": models or [],
    }

    source_ref_file = repro_dir / "prism_only_source_reference.json"
    try:
        with open(source_ref_file, 'w', encoding='utf-8') as f:
            json.dump(source_ref, f, indent=2)
        print(f"  [OK] Saved source reference: {source_ref_file.name}")
    except Exception as e:
        print(f"  [WARN] Could not save source reference: {e}")

    # 3. Copy source's preprocessing metadata
    metadata_pattern = f"preprocessing_metadata_{dataset_prefix}_*.json"
    source_metadata_files = list(source_dir.glob(metadata_pattern))

    if source_metadata_files:
        latest_metadata = max(source_metadata_files, key=lambda p: p.stat().st_mtime)
        dest_metadata = repro_dir / f"source_{latest_metadata.name}"
        try:
            shutil.copy(latest_metadata, dest_metadata)
            print(f"  [OK] Copied source metadata: {latest_metadata.name}")
        except Exception as e:
            print(f"  [WARN] Could not copy source metadata: {e}")
    else:
        print(f"  [WARN] No preprocessing metadata found in source: {source_dir}")

    # 4. Collect PRN tuning results from this run's model directories
    if models:
        prn_params_collected = 0
        for model in models:
            model_dir = output_dir / model / "models"
            if not model_dir.exists():
                continue

            # PRN params are saved as {dataset}_{blackbox}_prn_best_params.json
            prn_pattern = f"{dataset_prefix}_{model}_prn_best_params.json"
            prn_files = list(model_dir.glob(prn_pattern))

            for prn_file in prn_files:
                dest_prn = repro_dir / prn_file.name
                try:
                    shutil.copy(prn_file, dest_prn)
                    prn_params_collected += 1
                except Exception as e:
                    print(f"  [WARN] Could not copy PRN params {prn_file.name}: {e}")

        if prn_params_collected > 0:
            print(f"  [OK] Collected {prn_params_collected} PRN tuning result(s)")

    print(f"  Reproducibility artifacts saved to: {repro_dir}")


# =============================================================================
# Config Discovery and Loading
# =============================================================================


def load_configs_from_file(file_path: Path) -> List[str]:
    """Load a list of config names from a YAML batch file.

    Expected format:
        configs:
          - htx_example
          - openml_31

    Parameters
    ----------
    file_path : Path
        Path to the YAML file containing config list

    Returns
    -------
    List[str]
        List of config names

    Raises
    ------
    FileNotFoundError
        If the file doesn't exist
    ValueError
        If the file format is invalid
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Config batch file not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {file_path}: {e}")

    if not isinstance(data, dict):
        raise ValueError("Config batch file must contain a YAML dictionary with 'configs' key")

    if 'configs' not in data:
        raise ValueError(
            "Config batch file must contain a 'configs' key with a list of config names"
        )

    configs = data['configs']
    if not isinstance(configs, list):
        raise ValueError("'configs' must be a list of config names")

    if not configs:
        raise ValueError("'configs' list is empty")

    # Validate each entry is a string
    for i, cfg in enumerate(configs):
        if not isinstance(cfg, str):
            raise ValueError(f"Config at index {i} must be a string, got {type(cfg).__name__}")

    return configs


def list_available_configs() -> List[str]:
    """List all available config files in example_notebooks/config/."""
    config_dir = NOTEBOOKS_DIR / "config"
    if not config_dir.exists():
        return []

    configs = []
    for yaml_file in config_dir.glob("*.yaml"):
        # Skip example_config.yaml template
        if yaml_file.stem != "example_config":
            configs.append(yaml_file.stem)
    return sorted(configs)


def build_pipeline_config(config_name: str) -> Dict:
    """Load YAML config and build pipeline configuration dict.

    Parameters
    ----------
    config_name : str
        Name of the config file (without .yaml extension)

    Returns
    -------
    dict
        Pipeline configuration with keys:
        - config_name: The config name (for output directory naming)
        - prefix: Dataset prefix (from 'dataset' field in YAML)
        - models: List of models to train
        - preprocessing_notebook: Always "preprocessing.ipynb"
        - prism_notebook: Always "prism_analysis.ipynb"
        - model_notebook_template: Always "train_{model}.ipynb"
    """
    yaml_config, dataset_name = load_config(config_name)
    models = get_models_from_config(yaml_config, default_models=['mlp'])

    return {
        "config_name": config_name,
        "prefix": dataset_name,
        "models": models,
        "preprocessing_notebook": "preprocessing.ipynb",
        "prism_notebook": "prism_analysis.ipynb",
        "model_notebook_template": "train_{model}.ipynb",
    }


# =============================================================================
# Model File Utilities
# =============================================================================


def find_model_files(model_dir: Path, prefix: str, model_type: str) -> List[Path]:
    """Find model files matching the training notebook naming convention.

    Training notebooks save models as:
    - Directory: MODELS_DIR / {prefix}_{model_type}/
    - Filename: {prefix}_{model_type}_model_{timestamp}.pt

    Parameters
    ----------
    model_dir : Path
        Base directory to search in (e.g., .../mlp/models/)
    prefix : str
        Dataset prefix (e.g., "htx_example")
    model_type : str
        Model type (e.g., "mlp", "xgb")

    Returns
    -------
    List[Path]
        List of matching model file paths, sorted by modification time (newest first)
    """
    model_identifier = f"{prefix}_{model_type}"
    model_files = []

    # Pattern 1: In subdirectory (how training notebooks save)
    # e.g., .../models/htx_example_mlp/htx_example_mlp_model_*.pt
    model_files = list(model_dir.glob(f"{model_identifier}/{model_identifier}_model_*.pt"))

    # Pattern 2: Directly in directory (fallback)
    # e.g., .../models/htx_example_mlp_model_*.pt
    if not model_files:
        model_files = list(model_dir.glob(f"{model_identifier}_model_*.pt"))

    # Pattern 3: Legacy .pth format (for backwards compatibility)
    # e.g., .../models/htx_example_mlp.pth
    if not model_files:
        legacy_path = model_dir / f"{model_identifier}.pth"
        if legacy_path.exists():
            model_files = [legacy_path]

    # Sort by modification time, newest first
    return sorted(model_files, key=lambda p: p.stat().st_mtime, reverse=True)


# =============================================================================
# Prediction Concatenation Utilities
# =============================================================================


def extract_model_prefix_from_filename(filename: str) -> str | None:
    """Extract model prefix from filename pattern: {prefix}_{model_prefix}_preds_{timestamp}.csv"""
    # Pattern: {prefix}_{model_prefix}_preds_{timestamp}.csv
    pattern = r'^[^_]+_(.+?)_preds_\d{8}_\d{6}\.csv$'
    match = re.search(pattern, filename)
    if match:
        return match.group(1)
    # Fallback: more flexible pattern
    pattern_flexible = r'^[^_]+_(.+?)_preds_.+\.csv$'
    match_flexible = re.search(pattern_flexible, filename)
    if match_flexible:
        return match_flexible.group(1)
    return None


def combine_predictions_horizontally(
    prediction_files: List[Path],
    id_candidates: List[str] = None,
) -> pd.DataFrame | None:
    """Combine prediction CSV files horizontally by patient ID.

    Parameters
    ----------
    prediction_files : List[Path]
        List of prediction CSV file paths
    id_candidates : list of str, optional
        Ordered list of candidate ID column names to search for (case-insensitive).
        If None, uses DEFAULT_ID_CANDIDATES from config_loader.

    Returns
    -------
    pd.DataFrame or None
        Combined dataframe with predictions from all models, or None if no valid files
    """
    from prism.config_loader import DEFAULT_ID_CANDIDATES

    if id_candidates is None:
        id_candidates = DEFAULT_ID_CANDIDATES

    prediction_cols = ['pred_blackbox', 'pred_blackbox_nomogram', 'pred_prn', 'pred_prn_nomogram']
    dfs_with_models = []

    for csv_file in prediction_files:
        model_prefix = extract_model_prefix_from_filename(csv_file.name)
        if model_prefix is None:
            print(f"  [WARN] {csv_file.name}: Could not extract model prefix, skipping")
            continue
        try:
            df = pd.read_csv(csv_file)
            dfs_with_models.append((df, model_prefix, csv_file.name))
        except Exception as e:
            print(f"  [ERROR] {csv_file.name}: Error reading - {e}")
            continue

    if not dfs_with_models:
        return None

    # Find patient ID column (case-insensitive search through candidates)
    base_df, base_model, _ = dfs_with_models[0]
    columns_lower = {col.lower(): col for col in base_df.columns}
    patient_id = None
    for candidate in id_candidates:
        if candidate in base_df.columns:
            patient_id = candidate
            break
        if candidate.lower() in columns_lower:
            patient_id = columns_lower[candidate.lower()]
            break
    if patient_id is None:
        # Last resort: use first column
        patient_id = base_df.columns[0]
        print(
            f"  [WARN] No ID column found from candidates {id_candidates}, "
            f"using first column '{patient_id}'"
        )

    # Get non-prediction columns from base dataframe
    non_pred_cols = [col for col in base_df.columns if col not in prediction_cols]
    result_df = base_df[non_pred_cols].copy()
    result_df = result_df.set_index(patient_id)

    # Add prediction columns from each model
    for df, model_prefix, filename in dfs_with_models:
        if patient_id not in df.columns:
            print(f"  [WARN] {filename}: Missing ID column '{patient_id}', skipping")
            continue

        model_data = df.set_index(patient_id)
        for pred_col in prediction_cols:
            if pred_col in df.columns:
                new_col_name = f"{pred_col}_{model_prefix}"
                result_df[new_col_name] = model_data[pred_col]

    return result_df.reset_index()


def concatenate_all_predictions(
    output_dir: Path, dataset_prefix: str, config_name: str = None
) -> None:
    """Find and concatenate all prediction files from a pipeline run.

    Parameters
    ----------
    output_dir : Path
        The output directory for this pipeline run
    dataset_prefix : str
        Dataset prefix to match prediction files
    config_name : str, optional
        Config name to load id_candidates from YAML config
    """
    # Load id_candidates from config if available
    id_candidates = None
    if config_name:
        try:
            yaml_config, _ = load_config(config_name)
            id_candidates = yaml_config.get('id_candidates')
        except Exception:
            pass

    # Find all prediction files across model subdirectories
    prediction_files = list(
        output_dir.glob(f"*/models/predictions/{dataset_prefix}_*_preds_*.csv")
    )

    if not prediction_files:
        print("\nNo prediction files found to concatenate.")
        return

    print(f"\nConcatenating predictions from {len(prediction_files)} files...")
    for pf in sorted(prediction_files):
        model = extract_model_prefix_from_filename(pf.name)
        print(f"  - {pf.name} (model: {model})")

    combined_df = combine_predictions_horizontally(prediction_files, id_candidates=id_candidates)

    if combined_df is None or combined_df.empty:
        print("  [WARN] No valid predictions to combine")
        return

    # Save combined predictions to output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{dataset_prefix}_combined_predictions_{timestamp}.csv"

    try:
        combined_df.to_csv(output_file, index=False)
        print(f"  [OK] Combined predictions saved: {output_file.name}")
        print(f"       Shape: {combined_df.shape[0]:,} rows x {combined_df.shape[1]} columns")
    except Exception as e:
        print(f"  [ERROR] Error saving combined predictions: {e}")


# =============================================================================
# Hyperparameter Tuning Integration
# =============================================================================


def validate_params_files(config_name: str, models: List[str]) -> None:
    """Validate that all params_file paths in the config exist.

    This function checks upfront that any params_file specified in the
    hyperparameter_tuning config actually exists, failing fast before
    any expensive computation starts.

    Parameters
    ----------
    config_name : str
        Config name to load and validate
    models : list of str
        List of models to check (e.g., ['mlp', 'xgb'])

    Raises
    ------
    FileNotFoundError
        If any specified params_file doesn't exist
    """
    from prism.config_loader import get_tuning_config

    yaml_config, _ = load_config(config_name)

    # Check each model's tuning config
    # Also check 'prn' since PRiSM analysis uses it
    models_to_check = list(models) + ['prn']

    errors = []
    for model_type in models_to_check:
        tuning_config = get_tuning_config(yaml_config, model_type)

        # Check if params_file is specified
        if hasattr(tuning_config, 'params_file') and tuning_config.params_file:
            params_file = Path(tuning_config.params_file)

            # Resolve relative paths from project root
            if not params_file.is_absolute():
                params_file = PROJECT_ROOT / params_file

            if not params_file.exists():
                errors.append(
                    f"  - {model_type}: params_file not found: {tuning_config.params_file}\n"
                    f"    Resolved path: {params_file}"
                )

    if errors:
        error_msg = (
            f"ERROR: Invalid params_file path(s) in config '{config_name}':\n"
            + "\n".join(errors)
            + "\n\nPlease check that the specified params_file paths exist, or remove them to use defaults."
        )
        raise FileNotFoundError(error_msg)


def run_hyperparameter_tuning_for_config(
    config_name: str,
    config: Dict,
    dataset: str,
) -> None:
    """Run hyperparameter tuning for all models in a config.

    Uses run_hyperparameter_tuning.py's logic to tune models before training.
    Saves best parameters to models/{dataset}_{model}_best_params.json.

    Parameters
    ----------
    config_name : str
        Config file name (without .yaml)
    config : Dict
        Loaded pipeline config dictionary
    dataset : str
        Dataset prefix (e.g., "htx_example")
    """
    import numpy as np
    import torch

    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING")
    print("=" * 70)

    try:
        from prism.config_loader import get_tuning_config
        from prism.device_tools import get_device
        from prism.hyperparameter_tuning import (
            print_tuning_summary,
            run_hyperparameter_tuning,
            save_best_params,
        )
    except ImportError as e:
        print(f"WARNING: Hyperparameter tuning not available: {e}")
        print("Install optuna to enable hyperparameter tuning.")
        return

    # Load preprocessed data
    try:
        train_file = INTERIM_DATA_DIR / f"{dataset}_train.csv"
        test_file = INTERIM_DATA_DIR / f"{dataset}_test.csv"

        if not train_file.exists() or not test_file.exists():
            print(f"ERROR: Preprocessed data not found for {dataset}")
            print("Run preprocessing first before tuning.")
            return

        data_train = pd.read_csv(train_file, comment='#')
        data_test = pd.read_csv(test_file, comment='#')

        # Detect target and ID columns
        target_candidates = ['var1', 'event_oneyear', 'target', 'outcome', 'y', 'label']
        target_column = None
        for candidate in target_candidates:
            if candidate in data_train.columns:
                target_column = candidate
                break

        if target_column is None:
            print("ERROR: Could not detect target column")
            return

        id_candidates = ['trr_id_code', 'id', 'patient_id', 'subject_id']
        id_column = None
        for candidate in id_candidates:
            if candidate in data_train.columns:
                id_column = candidate
                break

        drop_cols = [target_column]
        if id_column:
            drop_cols.append(id_column)

        X_train = data_train.drop(drop_cols, axis=1)
        y_train = data_train[target_column]
        X_test = data_test.drop(drop_cols, axis=1)
        y_test = data_test[target_column]

        print(f"Loaded data: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
        print(f"Target: {target_column}, Features: {X_train.shape[1]}")

    except Exception as e:
        print(f"ERROR: Failed to load data for tuning: {e}")
        return

    # Get device and seed
    device = get_device()
    yaml_config, _ = load_config(config_name)
    random_seed = yaml_config.get('random_seed', 257)

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    print(f"Device: {device}")
    print(f"Random seed: {random_seed}")

    # Tune each model
    models = config.get('models', ['mlp'])
    for model_type in models:
        print(f"\n--- Tuning {model_type.upper()} ---")

        tuning_config = get_tuning_config(yaml_config, model_type)
        if not tuning_config.enabled:
            print(f"Tuning disabled for {model_type} in config. Skipping.")
            continue

        try:
            best_params, study, _best_model = run_hyperparameter_tuning(
                model_type=model_type,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                tuning_config=tuning_config,
                random_seed=random_seed,
                device=device,
            )
            print_tuning_summary(study, model_type)
            save_best_params(
                best_params,
                model_type,
                dataset,
                MODELS_DIR,
                study,
                random_seed=random_seed,
                tuning_config=tuning_config,
            )
        except Exception as e:
            print(f"WARNING: Tuning failed for {model_type}: {e}")
            continue

    print("\n" + "=" * 70)
    print("TUNING COMPLETE - Proceeding to training")
    print("=" * 70 + "\n")


# =============================================================================
# Reproducibility Utilities
# =============================================================================


def save_reproducibility_artifacts(
    output_dir: Path,
    config_name: str,
    dataset_prefix: str,
    interim_dir: Path = None,
) -> None:
    """Save reproducibility artifacts to output directory.

    Creates a 'reproducibility' folder containing:
    - Raw input CSV (zipped to save space)
    - Split data CSVs (zipped to save space)
    - SHA256 hashes of all data files (from preprocessing metadata)
    - Config YAML file (copied)

    Parameters
    ----------
    output_dir : Path
        Output directory for this pipeline run
    config_name : str
        Name of the config file (without .yaml extension)
    dataset_prefix : str
        Dataset prefix (used for finding data files)
    interim_dir : Path, optional
        Override for interim data directory (for self-contained mode)
    """
    import zipfile

    from prism.config import PROCESSED_DATA_DIR

    RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

    print("\nSaving reproducibility artifacts...")

    repro_dir = output_dir / "reproducibility"
    repro_dir.mkdir(parents=True, exist_ok=True)

    # Load preprocessing metadata - hashes are always present (computed by preprocessing notebook)
    metadata_pattern = f"preprocessing_metadata_{dataset_prefix}_*.json"
    # Check output_dir first (where pipeline copies it), then PROCESSED_DATA_DIR
    metadata_files = list(output_dir.glob(metadata_pattern))
    if not metadata_files:
        metadata_files = list(PROCESSED_DATA_DIR.glob(metadata_pattern))

    if not metadata_files:
        print(f"  [WARN] No preprocessing metadata found for {dataset_prefix}")
        print(f"         Searched in: {output_dir} and {PROCESSED_DATA_DIR}")
        return

    latest_metadata = max(metadata_files, key=lambda p: p.stat().st_mtime)
    with open(latest_metadata, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    provenance = metadata.get('data_provenance', {})

    # 1. Zip raw input CSV (hash already in metadata)
    raw_csv = RAW_DATA_DIR / f"{dataset_prefix}.csv"

    if raw_csv.exists():
        raw_zip_path = repro_dir / f"{dataset_prefix}_raw.zip"
        try:
            with zipfile.ZipFile(raw_zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.write(raw_csv, raw_csv.name)
            print(f"  [OK] Zipped raw data: {raw_zip_path.name}")
        except Exception as e:
            print(f"  [WARN] Could not zip raw data: {e}")
    else:
        print(f"  [WARN] Raw data file not found: {raw_csv}")

    # 2. Zip and copy split data CSVs (hashes already in metadata)
    if interim_dir is None:
        interim_dir = INTERIM_DATA_DIR

    split_files = [
        interim_dir / f"{dataset_prefix}_train.csv",
        interim_dir / f"{dataset_prefix}_test.csv",
        interim_dir / f"{dataset_prefix}_val.csv",
    ]

    zip_path = repro_dir / f"{dataset_prefix}_splits.zip"
    files_zipped = 0

    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for split_file in split_files:
                if split_file.exists():
                    zf.write(split_file, split_file.name)
                    files_zipped += 1
                else:
                    print(f"  [WARN] Split file not found: {split_file.name}")

        if files_zipped > 0:
            print(f"  [OK] Zipped {files_zipped} split files: {zip_path.name}")
        else:
            zip_path.unlink()
            print("  [WARN] No split files found to zip")
    except Exception as e:
        print(f"  [WARN] Could not zip split files: {e}")

    # 3. Copy config YAML file
    config_dir = PROJECT_ROOT / "example_notebooks" / "config"
    config_file = config_dir / f"{config_name}.yaml"

    if config_file.exists():
        dest_config = repro_dir / f"{config_name}.yaml"
        try:
            shutil.copy(config_file, dest_config)
            print(f"  [OK] Copied config: {config_file.name}")
        except Exception as e:
            print(f"  [WARN] Could not copy config: {e}")
    else:
        print(f"  [WARN] Config file not found: {config_file}")

    # 4. Save data provenance (hashes from preprocessing metadata)
    hash_file = repro_dir / "data_hashes.json"
    hash_data = {
        "algorithm": provenance.get("hash_algorithm", "sha256"),
        "generated_at": datetime.now().isoformat(),
        "config_name": config_name,
        "dataset_prefix": dataset_prefix,
        "source_metadata": latest_metadata.name,
        "input": provenance.get("input"),
        "output": provenance.get("output"),
    }

    try:
        with open(hash_file, 'w', encoding='utf-8') as f:
            json.dump(hash_data, f, indent=2)
        print(f"  [OK] Saved data provenance: {hash_file.name}")
    except Exception as e:
        print(f"  [WARN] Could not save hashes: {e}")

    print(f"  Reproducibility artifacts saved to: {repro_dir}")


def save_tuning_artifacts(
    output_dir: Path,
    dataset_prefix: str,
    models: List[str],
) -> None:
    """Save hyperparameter tuning artifacts to reproducibility folder AFTER training.

    This function should be called after all model training is complete,
    collecting best_params.json files from each model's output directory.

    Parameters
    ----------
    output_dir : Path
        Output directory for this pipeline run
    dataset_prefix : str
        Dataset prefix (used for finding param files)
    models : List[str]
        List of model types that were trained
    """
    repro_dir = output_dir / "reproducibility"
    repro_dir.mkdir(parents=True, exist_ok=True)

    print("\nSaving tuning artifacts...")

    collected_count = 0

    # Collect best_params from each model's directory
    for model in models:
        model_dir = output_dir / model / "models"
        if not model_dir.exists():
            continue

        # Look for any best_params.json files in this model's directory
        for params_file in model_dir.glob("*_best_params.json"):
            dest_params = repro_dir / params_file.name
            try:
                shutil.copy(params_file, dest_params)
                print(f"  [OK] Copied tuning params: {params_file.name}")
                collected_count += 1
            except Exception as e:
                print(f"  [WARN] Could not copy tuning params {params_file.name}: {e}")

    if collected_count == 0:
        print("  [INFO] No hyperparameter tuning files generated during this run")
    else:
        print(f"  Tuning artifacts saved to: {repro_dir}")


def copy_existing_best_params(
    dataset_prefix: str,
    model_type: str,
    target_dir: Path,
) -> bool:
    """Copy existing best_params files from global models/ to run-specific directory.

    This allows previously tuned hyperparameters to be loaded when running via
    the pipeline runner, which redirects MODELS_DIR to a run-specific location.

    Parameters
    ----------
    dataset_prefix : str
        Dataset prefix (e.g., 'htx_example')
    model_type : str
        Model type (e.g., 'mlp', 'xgb', 'logreg', 'rf')
    target_dir : Path
        Target directory (run-specific models folder)

    Returns
    -------
    bool
        True if any params file was copied
    """
    global_models_dir = PROJECT_ROOT / "models"
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    copied = False

    # Look for best_params file in global models directory
    params_filename = f"{dataset_prefix}_{model_type}_best_params.json"
    source_file = global_models_dir / params_filename

    if source_file.exists():
        dest_file = target_dir / params_filename
        if not dest_file.exists():  # Don't overwrite if already exists
            try:
                shutil.copy(source_file, dest_file)
                print(f"  -> Copied existing tuned params: {params_filename}")
                copied = True
            except Exception as e:
                print(f"  [WARN] Could not copy {params_filename}: {e}")

    return copied


# =============================================================================
# GPU Health Monitor (prevents CUDA connection loss during long runs)
# =============================================================================


def gpu_health_monitor(stop_event, interval_seconds=300):
    """Background thread that periodically pings GPUs to keep driver connection alive.

    This helps prevent NVML/CUDA connection staleness during long runs (12+ hours)
    by ensuring regular GPU driver communication from the Python process.

    Performs actual tensor operations on each GPU individually to detect per-GPU failures.
    """
    import torch

    # Track which GPUs were available at start
    initial_gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    while not stop_event.is_set():
        try:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                failed_gpus = []

                # Check each GPU individually with actual tensor operations
                for i in range(gpu_count):
                    try:
                        # Perform actual computation on each GPU to keep connection active
                        # This is more robust than just get_device_properties
                        test_tensor = torch.tensor([1.0], device=f'cuda:{i}')
                        _ = test_tensor.sum()
                        torch.cuda.synchronize(i)
                        del test_tensor
                    except Exception as gpu_error:
                        failed_gpus.append((i, str(gpu_error)))

                # Report any per-GPU failures
                if failed_gpus:
                    for gpu_id, error in failed_gpus:
                        print(f"[GPU Monitor] WARNING: GPU {gpu_id} health check failed: {error}")

                # Warn if GPU count changed
                if gpu_count != initial_gpu_count:
                    print(
                        f"[GPU Monitor] WARNING: GPU count changed from {initial_gpu_count} to {gpu_count}"
                    )

                # Clear CUDA cache to prevent memory buildup from health checks
                torch.cuda.empty_cache()
            else:
                # CUDA became unavailable
                print("[GPU Monitor] WARNING: CUDA is no longer available!")

        except Exception as e:
            print(f"[GPU Monitor] WARNING: GPU health check failed: {e}")

        # Wait for interval or until stop is signaled
        stop_event.wait(interval_seconds)


def start_gpu_monitor(interval_seconds=300):
    """Start the GPU health monitor thread.

    Parameters
    ----------
    interval_seconds : int
        How often to ping GPUs (default: 300 = 5 minutes)
    """
    global gpu_monitor_stop_event, gpu_monitor_thread
    import threading

    # Check if GPUs are available before starting
    try:
        import torch

        if not torch.cuda.is_available():
            return  # No GPUs, no need for monitor
        gpu_count = torch.cuda.device_count()
    except ImportError:
        return  # No torch, no need for monitor

    gpu_monitor_stop_event = threading.Event()
    gpu_monitor_thread = threading.Thread(
        target=gpu_health_monitor,
        args=(gpu_monitor_stop_event, interval_seconds),
        daemon=True,
        name="gpu-health-monitor",
    )
    gpu_monitor_thread.start()
    print(
        f"[GPU Monitor] Started background GPU health monitor ({gpu_count} GPUs, {interval_seconds//60} min interval)"
    )


def stop_gpu_monitor():
    """Stop the GPU health monitor thread."""
    if gpu_monitor_stop_event:
        gpu_monitor_stop_event.set()
    if gpu_monitor_thread and gpu_monitor_thread.is_alive():
        gpu_monitor_thread.join(timeout=5)
        print("[GPU Monitor] Stopped")


def validate_environment():
    """Validate that script is running in correct environment with prism installed"""
    # Check if prism is importable
    try:
        import prism

        print(f"[OK] Found prism package (version {prism.__version__})")
    except ImportError:
        print("ERROR: 'prism' package not found in current Python environment")
        print(f"Current Python: {sys.executable}")
        print("\nPlease ensure you:")
        print("  1. Created a virtual environment")
        print("  2. Installed dependencies: pip install -r requirements.txt")
        print("  3. Activated the virtual environment before running this script")
        sys.exit(1)

    # Check if in virtual environment (warning, not error)
    if not hasattr(sys, 'real_prefix') and not (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    ):
        print("WARNING: Not running in a virtual environment")
        print(f"Current Python: {sys.executable}")
        print("Continuing anyway, but this may cause issues...\n")
    else:
        print(f"[OK] Running in virtual environment: {sys.prefix}")

    # Check required packages for nbconvert
    try:
        import jupyter  # noqa: F401
        import nbconvert  # noqa: F401

        print("[OK] Jupyter and nbconvert are available")
    except ImportError as e:
        print(f"ERROR: Required package not found: {e}")
        print("Please install all requirements: pip install -r requirements.txt")
        sys.exit(1)

    print()  # Blank line for readability


def cleanup_on_exit():
    """Clean up temporary files on exit"""
    # Stop GPU monitor
    stop_gpu_monitor()

    # Stop tee logger
    if tee_logger:
        tee_logger.stop()

    # Kill any running subprocess
    if current_subprocess and current_subprocess.poll() is None:
        print("\nTerminating running notebook execution...")
        try:
            current_subprocess.terminate()
            current_subprocess.wait(timeout=5)
        except Exception:
            try:
                current_subprocess.kill()
            except Exception:
                pass

    # Clean up temporary files
    if temp_files_to_cleanup:
        print(f"Cleaning up {len(temp_files_to_cleanup)} temporary files...")
        for temp_file in temp_files_to_cleanup:
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except Exception:
                pass


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print(f"\nReceived signal {signum}. Shutting down...")
    cleanup_on_exit()
    sys.exit(0)


def update_config(
    dataset_prefix: str,
    models_dir: Path = None,
    interim_dir: Path = None,
    processed_dir: Path = None,
    config_name: str = None,
) -> None:
    """Update environment variables to override config settings.

    Parameters
    ----------
    dataset_prefix : str
        The dataset prefix to set
    models_dir : Path, optional
        If provided, redirect MODELS_DIR to this path
    interim_dir : Path, optional
        If provided, redirect INTERIM_DATA_DIR to this path
    processed_dir : Path, optional
        If provided, redirect PROCESSED_DATA_DIR to this path
    config_name : str, optional
        The config name (for PRISM_CONFIG). If provided, takes precedence over .env
    """
    print(f"Updating configuration environment for {dataset_prefix}")

    # Set config name if provided - this takes precedence over .env file
    # prism/config.py skips load_dotenv() when PRISM_CONFIG is already set
    if config_name:
        os.environ['PRISM_CONFIG'] = config_name

    # Also set dataset prefix for backwards compatibility
    os.environ['PRISM_DATASET_PREFIX'] = dataset_prefix

    # Handle MODELS_DIR override
    if models_dir:
        # Use forward slashes for consistency
        path_str = str(models_dir).replace('\\', '/')
        os.environ['PRISM_MODELS_DIR'] = path_str
        print(f"  -> Redirecting MODELS_DIR to {models_dir}")
    elif 'PRISM_MODELS_DIR' in os.environ:
        # Clean up if not specified (revert to default in config.py)
        del os.environ['PRISM_MODELS_DIR']

    # Handle INTERIM_DATA_DIR override
    if interim_dir:
        path_str = str(interim_dir).replace('\\', '/')
        os.environ['PRISM_INTERIM_DATA_DIR'] = path_str
        print(f"  -> Redirecting INTERIM_DATA_DIR to {interim_dir}")
    elif 'PRISM_INTERIM_DATA_DIR' in os.environ:
        del os.environ['PRISM_INTERIM_DATA_DIR']

    # Handle PROCESSED_DATA_DIR override
    if processed_dir:
        path_str = str(processed_dir).replace('\\', '/')
        os.environ['PRISM_PROCESSED_DATA_DIR'] = path_str
        print(f"  -> Redirecting PROCESSED_DATA_DIR to {processed_dir}")
    elif 'PRISM_PROCESSED_DATA_DIR' in os.environ:
        del os.environ['PRISM_PROCESSED_DATA_DIR']


def modify_notebook_model_prefix(
    notebook_path: Path, model_prefix: str, dataset: str = ""
) -> Path:
    """Create a temporary copy of notebook with modified model_prefix"""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook_data = json.load(f)

    # Fix kernel to use python3 (venv's kernel) instead of venv_prism (system kernel)
    if 'metadata' not in notebook_data:
        notebook_data['metadata'] = {}
    if 'kernelspec' not in notebook_data['metadata']:
        notebook_data['metadata']['kernelspec'] = {}
    notebook_data['metadata']['kernelspec']['name'] = 'python3'
    notebook_data['metadata']['kernelspec']['display_name'] = 'venv_prism'

    # Find and modify the model_prefix cell
    for cell in notebook_data.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = ''.join(cell.get('source', []))
            if 'model_prefix =' in source:
                # Update the source to set the correct model_prefix
                new_source = []
                for line in cell['source']:
                    if line.strip().startswith('model_prefix =') and not line.strip().startswith(
                        '#'
                    ):
                        new_source.append(f'model_prefix = "{model_prefix}"\n')
                    elif line.strip().startswith('# model_prefix ='):
                        new_source.append(line)  # Keep comments
                    elif 'model_prefix =' in line and line.strip().startswith('#'):
                        new_source.append(line)  # Keep other commented lines
                    else:
                        new_source.append(line)
                cell['source'] = new_source
                break

    # Create temporary file with clean descriptive name
    # Extract base name from original notebook (not from already-modified temp files)
    base_name = notebook_path.stem
    # Remove version suffix and any _temp_ prefix if present
    base_name = base_name.replace('_v1.0.0', '').replace('_temp_', '')

    # Create simple, clear naming based on notebook type
    if 'prism' in base_name.lower():
        stage = "prism"
    elif 'model' in base_name.lower():
        stage = "model"
    elif 'preprocessing' in base_name.lower():
        stage = "preprocessing"
    else:
        stage = base_name

    # Include dataset name in temp file: dataset_stage_model format
    if dataset:
        temp_filename = f"_temp_{dataset}_{stage}_{model_prefix}.ipynb"
    else:
        temp_filename = f"_temp_{stage}_{model_prefix}.ipynb"

    # Keep temp files in NOTEBOOKS_DIR to ensure kernel has access to installed packages
    temp_dir = NOTEBOOKS_DIR
    temp_path = temp_dir / temp_filename

    # Ensure unique filename
    counter = 1
    while temp_path.exists():
        if dataset:
            temp_path = temp_dir / f"_temp_{dataset}_{stage}_{model_prefix}_{counter}.ipynb"
        else:
            temp_path = temp_dir / f"_temp_{stage}_{model_prefix}_{counter}.ipynb"
        counter += 1

    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(notebook_data, f, indent=2)

    # Track for cleanup
    temp_files_to_cleanup.append(temp_path)
    return temp_path


def modify_htx_notebook_paths(
    notebook_path: Path, dataset: str = "", model_prefix: str = ""
) -> Path:
    """Create a temporary copy of htx notebook with corrected file paths"""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook_data = json.load(f)

    # Fix kernel to use python3 (venv's kernel) instead of venv_prism (system kernel)
    if 'metadata' not in notebook_data:
        notebook_data['metadata'] = {}
    if 'kernelspec' not in notebook_data['metadata']:
        notebook_data['metadata']['kernelspec'] = {}
    notebook_data['metadata']['kernelspec']['name'] = 'python3'
    notebook_data['metadata']['kernelspec']['display_name'] = 'venv_prism'

    # Find and modify cells that reference htx_variable_labels.csv
    for cell in notebook_data.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = ''.join(cell.get('source', []))
            if 'htx_variable_labels.csv' in source:
                # Fix the path to htx_variable_labels.csv
                new_source = []
                for line in cell['source']:
                    if 'htx_variable_labels.csv' in line:
                        # Replace with absolute path using regex to handle all variations
                        htx_labels_path = (
                            PROJECT_ROOT / "example_notebooks" / "htx_variable_labels.csv"
                        )
                        # Convert to string and use forward slashes to avoid JSON escape issues
                        htx_labels_path_str = str(htx_labels_path).replace('\\', '/')

                        # Pattern to match any path that ends with htx_variable_labels.csv
                        # This handles: 'htx_variable_labels.csv', 'example_notebooks/htx_variable_labels.csv', etc.
                        pattern = r"['\"](?:.*[/\\])?htx_variable_labels\.csv['\"]"
                        new_line = re.sub(pattern, f"'{htx_labels_path_str}'", line)

                        # If no quoted version found, try unquoted
                        if new_line == line:
                            pattern = r"(?:.*[/\\])?htx_variable_labels\.csv"
                            new_line = re.sub(pattern, htx_labels_path_str, line)

                        new_source.append(new_line)
                    else:
                        new_source.append(line)
                cell['source'] = new_source

    # Create temporary file with clean descriptive name - NEVER modify original files!
    # If we received a temp file, use it as-is (htx fixes don't change the name)
    if notebook_path.stem.startswith('_temp_'):
        # Already a temp file with proper naming, keep the same name
        temp_filename = f"{notebook_path.stem}.ipynb"
    else:
        # Original notebook, build name from scratch
        base_name = notebook_path.stem.replace('_v1.0.0', '')

        # Determine stage from notebook name
        if 'prism' in base_name.lower():
            stage = "prism"
        elif 'model' in base_name.lower():
            stage = "model"
        elif 'preprocessing' in base_name.lower():
            stage = "preprocessing"
        else:
            stage = base_name

        # Build filename with dataset and model_prefix if provided
        if dataset and model_prefix:
            temp_filename = f"_temp_{dataset}_{stage}_{model_prefix}.ipynb"
        elif dataset:
            temp_filename = f"_temp_{dataset}_{stage}.ipynb"
        elif model_prefix:
            temp_filename = f"_temp_{stage}_{model_prefix}.ipynb"
        else:
            temp_filename = f"_temp_{stage}.ipynb"

    # Keep temp files in NOTEBOOKS_DIR to ensure kernel has access to installed packages
    temp_dir = NOTEBOOKS_DIR
    temp_path = temp_dir / temp_filename

    # Ensure unique filename
    counter = 1
    while temp_path.exists():
        if notebook_path.stem.startswith('_temp_'):
            # If it was a temp file, just increment
            base_temp_name = notebook_path.stem
            temp_path = temp_dir / f"{base_temp_name}_{counter}.ipynb"
        else:
            # Original notebook, increment with full structure
            if dataset and model_prefix:
                temp_path = temp_dir / f"_temp_{dataset}_{stage}_{model_prefix}_{counter}.ipynb"
            elif dataset:
                temp_path = temp_dir / f"_temp_{dataset}_{stage}_{counter}.ipynb"
            elif model_prefix:
                temp_path = temp_dir / f"_temp_{stage}_{model_prefix}_{counter}.ipynb"
            else:
                temp_path = temp_dir / f"_temp_{stage}_{counter}.ipynb"
        counter += 1

    # Write the modified notebook data to the NEW temporary file
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(notebook_data, f, indent=2)

    # Track for cleanup
    temp_files_to_cleanup.append(temp_path)
    return temp_path


def gather_log_files(
    output_dir: Path,
    dataset_prefix: str,
    model_prefix: str = None,
    stage: str = "preprocessing",
) -> None:
    """Gather log files into the output directory.

    Since MODELS_DIR is redirected to the output folder, model outputs (predictions,
    performance summaries, nomograms) are already written directly there. This function
    only needs to gather log files which are created in PROJECT_ROOT.

    Parameters
    ----------
    output_dir : Path
        Directory to copy log files to
    dataset_prefix : str
        Dataset prefix for finding log files
    model_prefix : str, optional
        Model prefix for finding model-specific logs
    stage : str
        Pipeline stage: "preprocessing", "modelling", or "prism"
    """
    if stage == "preprocessing":
        log_pattern = f"preprocessing_{dataset_prefix}_*.log"
    elif stage == "modelling" and model_prefix:
        # Actual log files use "model_" prefix (not "modelling_")
        # Pattern: model_{prefix}_{prefix}_{model}_*.log (with dataset prefix duplicated)
        log_pattern = f"model_{dataset_prefix}_{dataset_prefix}_{model_prefix}_*.log"
    elif stage == "prism" and model_prefix:
        # Pattern: prism_{prefix}_{prefix}_{model}_*.log (with dataset prefix duplicated)
        log_pattern = f"prism_{dataset_prefix}_{dataset_prefix}_{model_prefix}_*.log"
    else:
        return

    # Log files are in the logs/ directory
    logs_dir = PROJECT_ROOT / "logs"

    # Search in logs directory
    log_files = list(logs_dir.glob(log_pattern))

    if log_files:
        latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
        dest = output_dir / latest_log.name
        if not dest.exists():
            try:
                # Use copy() instead of copy2() to give the copied file a new timestamp
                shutil.copy(latest_log, dest)
                print(f"  Copied log: {latest_log.name}")
            except Exception as e:
                print(f"  Warning: Could not copy {latest_log.name}: {e}")


def cleanup_png_files(output_dir: Path) -> None:
    """Remove PNG files generated by nbconvert in the output directory"""
    png_files = list(output_dir.glob("*.png"))
    if png_files:
        print(f"Cleaning up {len(png_files)} PNG files...")
        for png_file in png_files:
            try:
                png_file.unlink()
            except Exception as e:
                print(f"Warning: Could not delete {png_file.name}: {e}")


def get_dataset_output_dir(
    base_dir: Path,
    dataset: str,
    date_str: str,
    models_to_run: list,
    skip_preprocessing: bool = False,
) -> Path:
    """Get the appropriate output directory for a dataset run.

    Handles directory enumeration to avoid overwriting existing results:
    - If directory doesn't exist, use base name (e.g., 20260114_htx_example)
    - If directory exists but no overlap with models/preprocessing, reuse it
    - If any overlap detected, enumerate (e.g., 20260114_htx_example_1)

    Parameters
    ----------
    base_dir : Path
        Parent directory (e.g., pipeline_results/)
    dataset : str
        Dataset name (e.g., "htx_example")
    date_str : str
        Date string for directory naming (e.g., "20260114")
    models_to_run : list
        List of model names that will be processed
    skip_preprocessing : bool
        Whether preprocessing is being skipped

    Returns
    -------
    Path
        The output directory to use
    """
    base_name = f"{date_str}_{dataset}"
    candidate = base_dir / base_name

    # If directory doesn't exist, use it
    if not candidate.exists():
        return candidate

    # Directory exists - check for overlaps
    existing_model_dirs = {d.name for d in candidate.iterdir() if d.is_dir()}
    models_set = set(models_to_run)

    # Check model directory overlap
    model_overlap = existing_model_dirs & models_set

    # Check preprocessing overlap (if not skipping)
    preprocessing_overlap = False
    if not skip_preprocessing:
        preprocessing_file = candidate / "01_preprocessing.pdf"
        preprocessing_html = candidate / "01_preprocessing.html"
        if preprocessing_file.exists() or preprocessing_html.exists():
            preprocessing_overlap = True

    # If no overlap, reuse the existing directory
    if not model_overlap and not preprocessing_overlap:
        print(f"Reusing existing directory (no overlap with models: {models_set})")
        return candidate

    # Overlap detected - find next available enumerated directory
    if model_overlap:
        print(f"Model overlap detected: {model_overlap}")
    if preprocessing_overlap:
        print("Preprocessing output already exists")

    counter = 1
    while True:
        enumerated = base_dir / f"{base_name}_{counter}"
        if not enumerated.exists():
            print(f"Creating new enumerated directory: {enumerated.name}")
            return enumerated

        # Check this enumerated directory for overlaps too
        existing_model_dirs = {d.name for d in enumerated.iterdir() if d.is_dir()}
        model_overlap = existing_model_dirs & models_set

        preprocessing_overlap = False
        if not skip_preprocessing:
            preprocessing_file = enumerated / "01_preprocessing.pdf"
            preprocessing_html = enumerated / "01_preprocessing.html"
            if preprocessing_file.exists() or preprocessing_html.exists():
                preprocessing_overlap = True

        if not model_overlap and not preprocessing_overlap:
            print(f"Reusing existing directory (no overlap): {enumerated.name}")
            return enumerated

        counter += 1
        if counter > 100:  # Safety limit
            raise RuntimeError(
                f"Could not find available directory after 100 attempts: {base_name}"
            )


def get_available_output_path(output_path: Path) -> Path:
    """Get an available output path, incrementing filename if file exists or is locked.

    If the target file exists or cannot be written (e.g., open in PDF reader),
    returns an incremented filename like 'name_1.pdf', 'name_2.pdf', etc.
    """
    if not output_path.exists():
        return output_path

    # Try to check if file is writable (not locked by another process)
    try:
        # Attempt to open file for writing to check if it's locked
        with open(output_path, 'a'):
            pass
        # File exists but is writable - still increment to avoid overwriting
    except (IOError, OSError, PermissionError):
        # File is locked (e.g., open in PDF reader)
        print(f"File is locked or inaccessible: {output_path.name}")

    # Find an available incremented filename
    counter = 1
    stem = output_path.stem
    suffix = output_path.suffix
    parent = output_path.parent

    while True:
        new_path = parent / f"{stem}_{counter}{suffix}"
        if not new_path.exists():
            # Also check if the new path is writable
            try:
                # Try creating/touching the file briefly
                new_path.touch()
                new_path.unlink()
                print(f"Using incremented filename: {new_path.name}")
                return new_path
            except (IOError, OSError, PermissionError):
                pass
        counter += 1
        if counter > 100:  # Safety limit
            raise RuntimeError(
                f"Could not find available filename after 100 attempts: {output_path}"
            )


def is_latex_error(stderr: str) -> bool:
    """Check if the error is related to LaTeX/PDF conversion rather than notebook execution."""
    latex_indicators = [
        'xelatex',
        'pdflatex',
        'latex',
        'texlive',
        'miktex',
        'pandoc',
        'nbconvert failed: PDF creating failed',
        'LaTeX Error',
        'TeX capacity exceeded',
        'Missing $ inserted',
        'Undefined control sequence',
        '.tex',
        'texconv',
    ]
    stderr_lower = stderr.lower()
    return any(indicator.lower() in stderr_lower for indicator in latex_indicators)


def execute_and_export_notebook(
    notebook_path: Path,
    output_path: Path,
    model_prefix: str = None,
    is_htx_notebook: bool = False,
) -> bool:
    """Execute notebook and export to PDF, falling back to HTML if PDF export fails.

    Parameters
    ----------
    notebook_path : Path
        Path to the source notebook
    output_path : Path
        Desired output path (with .pdf extension, will be changed to .html if fallback needed)
    model_prefix : str, optional
        Model prefix to set in the notebook
    is_htx_notebook : bool
        Whether this is an HTX notebook requiring path fixes

    Returns
    -------
    bool
        True if export succeeded (PDF or HTML), False otherwise
    """
    try:
        temp_notebook = None

        if model_prefix:
            print(f"Executing {notebook_path.name} with model_prefix={model_prefix}")
            # First apply model prefix modification
            dataset_name = output_path.parent.name  # Extract dataset from output path
            temp_notebook = modify_notebook_model_prefix(notebook_path, model_prefix, dataset_name)
            # Then apply htx path fixes if needed
            if is_htx_notebook:
                htx_fixed_notebook = modify_htx_notebook_paths(
                    temp_notebook, dataset_name, model_prefix
                )
                temp_notebook.unlink()  # Clean up intermediate file
                if temp_notebook in temp_files_to_cleanup:
                    temp_files_to_cleanup.remove(temp_notebook)
                temp_notebook = htx_fixed_notebook
            notebook_to_run = temp_notebook
        elif is_htx_notebook:
            print(f"Executing {notebook_path.name}")
            dataset_name = output_path.parent.name  # Extract dataset from output path
            temp_notebook = modify_htx_notebook_paths(notebook_path, dataset_name)
            notebook_to_run = temp_notebook
        else:
            print(f"Executing {notebook_path.name}")
            notebook_to_run = notebook_path

        # Use the current Python executable (which should be the activated venv's Python)
        python_exe = sys.executable
        venv_dir = Path(sys.executable).parent  # Directory containing python executable

        # Ensure kernel uses same Python by prioritizing venv in PATH
        env = os.environ.copy()
        env['PATH'] = f"{venv_dir}{os.pathsep}{env.get('PATH', '')}"
        # Force UTF-8 encoding on Windows to avoid UnicodeDecodeError from
        # IPython's deduperreload extension reading stdlib files as cp1252
        env['PYTHONUTF8'] = '1'

        # Convert paths for cross-platform compatibility
        notebook_path_str = str(notebook_to_run).replace('\\', '/')

        # =================================================================
        # Stage 1: Execute notebook once and save in-place
        # =================================================================
        # This separates execution from format conversion, so if PDF export
        # fails (e.g., LaTeX not installed), we can fall back to HTML without
        # re-executing the notebook (which can take 10-40+ minutes).

        cmd_execute = f'"{python_exe}" -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=python3 "{notebook_path_str}"'

        global current_subprocess
        current_subprocess = subprocess.Popen(
            cmd_execute,
            shell=True,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = current_subprocess.communicate()
        exec_returncode = current_subprocess.returncode
        current_subprocess = None

        if exec_returncode != 0:
            print(f"Failed to execute {notebook_path.name}")
            print(f"Error: {stderr}")
            # Clean up temp notebook and return early
            if temp_notebook and temp_notebook.exists():
                temp_notebook.unlink()
                if temp_notebook in temp_files_to_cleanup:
                    temp_files_to_cleanup.remove(temp_notebook)
            return False

        # =================================================================
        # Stage 2: Convert executed notebook to output format
        # =================================================================
        # The notebook has been executed and saved with outputs. Now we just
        # need to convert it to PDF or HTML (fast, no re-execution needed).

        pdf_output_path = output_path.with_suffix('.pdf')
        pdf_output_path = get_available_output_path(pdf_output_path)
        output_path_str = str(pdf_output_path).replace('\\', '/')

        # Try PDF export first (no --execute flag - notebook already executed)
        cmd_pdf = f'"{python_exe}" -m jupyter nbconvert --to pdf --output "{output_path_str}" "{notebook_path_str}"'

        current_subprocess = subprocess.Popen(
            cmd_pdf,
            shell=True,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = current_subprocess.communicate()
        pdf_returncode = current_subprocess.returncode
        current_subprocess = None

        export_succeeded = False
        final_output_path = pdf_output_path

        if pdf_returncode == 0:
            print(f"Successfully exported: {pdf_output_path.name}")
            export_succeeded = True
        elif is_latex_error(stderr):
            # PDF export failed due to LaTeX issues - try HTML fallback
            # This is fast since the notebook is already executed!
            print("PDF export failed (LaTeX error), falling back to HTML...")

            html_output_path = output_path.with_suffix('.html')
            html_output_path = get_available_output_path(html_output_path)
            html_output_path_str = str(html_output_path).replace('\\', '/')

            # No --execute flag - notebook already executed
            cmd_html = f'"{python_exe}" -m jupyter nbconvert --to html --output "{html_output_path_str}" "{notebook_path_str}"'

            current_subprocess = subprocess.Popen(
                cmd_html,
                shell=True,
                cwd=PROJECT_ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout_html, stderr_html = current_subprocess.communicate()
            html_returncode = current_subprocess.returncode
            current_subprocess = None

            if html_returncode == 0:
                print(f"Successfully exported (HTML fallback): {html_output_path.name}")
                export_succeeded = True
                final_output_path = html_output_path
            else:
                print(f"HTML fallback also failed for {notebook_path.name}")
                print(f"Error: {stderr_html}")
        else:
            # Non-LaTeX PDF conversion error
            print(f"Failed to convert {notebook_path.name} to PDF")
            print(f"Error: {stderr}")

        # Clean up temporary file if created
        if temp_notebook and temp_notebook.exists():
            temp_notebook.unlink()
            # Remove from cleanup list since we handled it
            if temp_notebook in temp_files_to_cleanup:
                temp_files_to_cleanup.remove(temp_notebook)

        if export_succeeded:
            # Clean up PNG files generated during nbconvert
            cleanup_png_files(final_output_path.parent)
            return True

        return False

    except Exception as e:
        print(f"Exception executing {notebook_path.name}: {e}")
        if temp_notebook and temp_notebook.exists():
            temp_notebook.unlink()
            # Remove from cleanup list since we handled it
            if temp_notebook in temp_files_to_cleanup:
                temp_files_to_cleanup.remove(temp_notebook)
        return False


def check_preprocessed_files_exist(dataset: str, config: Dict) -> bool:
    """Check if preprocessed files exist for a dataset"""
    try:
        # Expected preprocessed files based on dataset prefix
        prefix = config["prefix"]
        expected_files = [f"{prefix}_train.csv", f"{prefix}_test.csv", f"{prefix}_val.csv"]

        # Check if files exist in INTERIM_DATA_DIR (where preprocessing notebooks save files)
        missing_files = []
        for filename in expected_files:
            file_path = INTERIM_DATA_DIR / filename
            if not file_path.exists():
                missing_files.append(filename)

        if missing_files:
            print(f"Missing preprocessed files: {', '.join(missing_files)}")
            print(f"Searched in: {INTERIM_DATA_DIR}")
            return False

        print(f"Found all required preprocessed files for {dataset}")
        return True

    except Exception as e:
        print(f"Error checking preprocessed files: {e}")
        return False


def execute_preprocessing_notebook(dataset: str, config: Dict, output_dir: Path) -> bool:
    """Execute preprocessing notebook and export to PDF (or HTML as fallback)"""
    notebook_path = NOTEBOOKS_DIR / config["preprocessing_notebook"]

    # Numbered output: 01_preprocessing.pdf
    output_path = output_dir / "01_preprocessing.pdf"

    print(f"Executing preprocessing notebook for {dataset}")

    # Check if this is an htx notebook that needs path fixes
    is_htx = dataset.startswith("htx")

    success = execute_and_export_notebook(notebook_path, output_path, is_htx_notebook=is_htx)

    # Gather preprocessing log files if successful
    if success:
        gather_log_files(output_dir, config["prefix"], stage="preprocessing")

    return success


def process_dataset(
    dataset: str,
    config: Dict,
    output_dir: Path,
    skip_preprocessing: bool = False,
    self_contained: bool = False,
    reproducibility: bool = True,
    prism_only: bool = False,
    prism_only_paths: Dict = None,
    prism_only_source_dir: Path = None,
    prism_only_source_config: str = None,
) -> Tuple[int, int]:
    """Process a complete dataset pipeline.

    Parameters
    ----------
    dataset : str
        Dataset name (e.g., "htx_example")
    config : Dict
        Dataset configuration from DATASETS
    output_dir : Path
        Base output directory for this dataset
    skip_preprocessing : bool
        Skip preprocessing step
    self_contained : bool
        If True, also redirect data directories to output folder
    reproducibility : bool
        If True, save reproducibility artifacts (data, hashes, config) to output folder
    prism_only : bool
        If True, skip preprocessing and training, run only PRiSM analysis
    prism_only_paths : Dict, optional
        Validated paths from validate_prism_only_source() when prism_only=True
        Contains: metadata_file, processed_data_dir, model_dirs
    prism_only_source_dir : Path, optional
        Source directory for prism-only mode (for reproducibility tracking)
    prism_only_source_config : str, optional
        Source config name if using prism_only_source_config (for reproducibility tracking)
    """
    global current_output_dir
    current_output_dir = output_dir

    print(f"\nProcessing dataset: {dataset}")
    print(f"Models to process: {config['models']}")
    if prism_only:
        print("Mode: PRiSM-only (using pre-trained models)")

    successes = 0
    total = 0

    # Data directories setup
    interim_dir = None
    processed_dir = None

    # PRiSM-only mode: use source paths
    if prism_only and prism_only_paths:
        processed_dir = prism_only_paths.get("processed_data_dir")
        # Skip preprocessing and training entirely
        print("Skipping preprocessing (--prism-only mode)")
        print("Skipping model training (--prism-only mode)")
    elif self_contained:
        # Full isolation: redirect data directories too
        interim_dir = output_dir / "data" / "interim"
        processed_dir = output_dir / "data" / "processed"
        interim_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

    # 1. Execute preprocessing notebook (if not skipped and configured)
    # Skip entirely in prism-only mode
    if not prism_only:
        should_skip_preprocessing = (
            skip_preprocessing
            or not config.get("preprocessing_notebook")
            or config.get("preprocessing_notebook") == ""
        )

        if should_skip_preprocessing:
            if skip_preprocessing:
                print("Skipping preprocessing (--skip-preprocessing flag)")
            else:
                print("Skipping preprocessing (no preprocessing notebook configured)")

            # Check if required preprocessed files exist - exit if not
            if not check_preprocessed_files_exist(dataset, config):
                print(
                    "ERROR: Required preprocessed files not found. Cannot proceed with model training."
                )
                print("Run preprocessing first or ensure the required files exist.")
                return 0, 1  # 0 successes, 1 failure
        else:
            # Update config for preprocessing only (no models_dir yet)
            update_config(
                config["prefix"],
                models_dir=None,
                interim_dir=interim_dir,
                processed_dir=processed_dir,
                config_name=config["config_name"],
            )

            total += 1
            if execute_preprocessing_notebook(dataset, config, output_dir):
                successes += 1
                # Copy preprocessing metadata to top level if it exists
                if processed_dir:
                    metadata_pattern = f"preprocessing_metadata_{config['prefix']}_*.json"
                    metadata_files = list(processed_dir.glob(metadata_pattern))
                else:
                    from prism.config import PROCESSED_DATA_DIR

                    metadata_pattern = f"preprocessing_metadata_{config['prefix']}_*.json"
                    metadata_files = list(PROCESSED_DATA_DIR.glob(metadata_pattern))

                if metadata_files:
                    latest_metadata = max(metadata_files, key=lambda p: p.stat().st_mtime)
                    dest = output_dir / latest_metadata.name
                    if not dest.exists():
                        try:
                            # Use copy() instead of copy2() to give the copied file a new timestamp
                            shutil.copy(latest_metadata, dest)
                            print(f"  Copied metadata: {latest_metadata.name}")
                        except Exception as e:
                            print(f"  Warning: Could not copy metadata: {e}")
            else:
                print("Preprocessing failed, but continuing with model training...")

        # Save reproducibility artifacts after preprocessing (or after verifying preprocessed files exist)
        if reproducibility:
            save_reproducibility_artifacts(
                output_dir=output_dir,
                config_name=config["config_name"],
                dataset_prefix=config["prefix"],
                interim_dir=interim_dir,
            )

    # 2. Process each model
    # In prism-only mode, numbering starts at 01 (no preprocessing/training outputs)
    model_counter = 1 if prism_only else 2

    for model in config["models"]:
        # Create model-specific output directory for notebooks and models
        model_output_dir = output_dir / model
        model_output_dir.mkdir(parents=True, exist_ok=True)

        # Create models subdirectory inside model folder
        models_dir = model_output_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        # In prism-only mode, copy model file from source to new output directory
        # This ensures outputs (nomograms, LASSO results, etc.) go to the new directory
        if prism_only and prism_only_paths:
            # Load YAML config to check caching flags and avoid copying files
            # that will be recalculated (prevents stale files in output directory)
            yaml_config, _ = load_config(config["config_name"])
            _load_cached_prn = yaml_config.get('load_cached_prn', False)
            _force_recalc_pr = yaml_config.get('force_recalculate_partial_responses', False)
            _force_recalc_prn_pr = yaml_config.get(
                'force_recalculate_prn_partial_responses', False
            )
            _force_recalc_lasso = yaml_config.get('force_recalculate_lasso', False)
            _force_recalc_prn_lasso = yaml_config.get('force_recalculate_prn_lasso', False)
            # mlp10 uses mlp model
            model_to_use = 'mlp' if model == 'mlp10' else model
            source_model_dir = prism_only_paths["model_dirs"].get(model)
            if source_model_dir:
                # Find the model file in source directory
                source_model_files = find_model_files(
                    source_model_dir, config["prefix"], model_to_use
                )
                if source_model_files:
                    source_model_file = source_model_files[0]  # Latest by mtime

                    # Create destination directory structure matching source
                    # Pattern: models/{dataset}_{model}/{dataset}_{model}_model_*.pt
                    dest_model_subdir = models_dir / f"{config['prefix']}_{model_to_use}"
                    dest_model_subdir.mkdir(parents=True, exist_ok=True)
                    dest_model_file = dest_model_subdir / source_model_file.name

                    # Copy model file to new output directory
                    if not dest_model_file.exists():
                        shutil.copy2(source_model_file, dest_model_file)
                        print(
                            f"  Copied pre-trained {model_to_use} model: {source_model_file.name}"
                        )
                    else:
                        print(f"  Using existing model copy: {dest_model_file.name}")

            # Copy PRN model file if available and load_cached_prn is enabled
            # Skip copying when PRN will be retrained (avoids stale files in output)
            if _load_cached_prn:
                source_prn_model_file = prism_only_paths.get("prn_model_files", {}).get(model)
                if source_prn_model_file and source_prn_model_file.exists():
                    # PRN model is stored in: models/{dataset}_{model}_prn/{dataset}_{model}_prn_model_tuned.pt
                    prn_model_identifier = f"{config['prefix']}_{model_to_use}_prn"
                    dest_prn_model_subdir = models_dir / prn_model_identifier
                    dest_prn_model_subdir.mkdir(parents=True, exist_ok=True)
                    dest_prn_model_file = dest_prn_model_subdir / source_prn_model_file.name
                    if not dest_prn_model_file.exists():
                        shutil.copy2(source_prn_model_file, dest_prn_model_file)
                        print(f"  Copied cached PRN model: {source_prn_model_file.name}")
                    else:
                        print(f"  Using existing PRN model: {dest_prn_model_file.name}")
            else:
                print("  Skipping PRN model copy (load_cached_prn=false, PRN will be retrained)")

            # Copy partial responses file if available (saves expensive GPU recalculation)
            # Skip when force_recalculate_partial_responses is set
            if _force_recalc_pr:
                print(
                    "  Skipping blackbox partial responses copy (force_recalculate_partial_responses=true)"
                )
            else:
                source_pr_file = prism_only_paths.get("partial_responses_files", {}).get(model)
                if source_pr_file and source_pr_file.exists():
                    dest_pr_dir = models_dir / "partial_responses"
                    dest_pr_dir.mkdir(exist_ok=True)
                    dest_pr_file = dest_pr_dir / source_pr_file.name
                    if not dest_pr_file.exists():
                        shutil.copy2(source_pr_file, dest_pr_file)
                        print(f"  Copied cached partial responses: {source_pr_file.name}")
                    else:
                        print(f"  Using existing partial responses: {dest_pr_file.name}")

            # Copy LASSO results file if available (allows different lambda selection)
            # Skip when force_recalculate_lasso is set (LASSO will be recomputed with new seed)
            if _force_recalc_lasso:
                print("  Skipping blackbox LASSO results copy (force_recalculate_lasso=true)")
            else:
                source_lasso_file = prism_only_paths.get("lasso_results_files", {}).get(model)
                if source_lasso_file and source_lasso_file.exists():
                    dest_lasso_dir = models_dir / "lasso_results"
                    dest_lasso_dir.mkdir(exist_ok=True)
                    dest_lasso_file = dest_lasso_dir / source_lasso_file.name
                    if not dest_lasso_file.exists():
                        shutil.copy2(source_lasso_file, dest_lasso_file)
                        print(f"  Copied cached LASSO results: {source_lasso_file.name}")
                    else:
                        print(f"  Using existing LASSO results: {dest_lasso_file.name}")

            # Copy PRN partial responses file if available (for PRN caching)
            # Only copy when load_cached_prn=true AND not force-recalculating PRN partial responses
            if _load_cached_prn and not _force_recalc_prn_pr:
                source_prn_pr_file = prism_only_paths.get("prn_partial_responses_files", {}).get(
                    model
                )
                if source_prn_pr_file and source_prn_pr_file.exists():
                    dest_pr_dir = models_dir / "partial_responses"
                    dest_pr_dir.mkdir(exist_ok=True)
                    dest_prn_pr_file = dest_pr_dir / source_prn_pr_file.name
                    if not dest_prn_pr_file.exists():
                        shutil.copy2(source_prn_pr_file, dest_prn_pr_file)
                        print(f"  Copied cached PRN partial responses: {source_prn_pr_file.name}")
                    else:
                        print(f"  Using existing PRN partial responses: {dest_prn_pr_file.name}")
            elif not _load_cached_prn:
                print("  Skipping PRN partial responses copy (load_cached_prn=false)")
            else:
                print(
                    "  Skipping PRN partial responses copy (force_recalculate_prn_partial_responses=true)"
                )

            # Copy PRN LASSO results file if available (for PRN caching)
            # Only copy when load_cached_prn=true AND not force-recalculating PRN LASSO
            if _load_cached_prn and not _force_recalc_prn_lasso:
                source_prn_lasso_file = prism_only_paths.get("prn_lasso_results_files", {}).get(
                    model
                )
                if source_prn_lasso_file and source_prn_lasso_file.exists():
                    dest_lasso_dir = models_dir / "lasso_results"
                    dest_lasso_dir.mkdir(exist_ok=True)
                    dest_prn_lasso_file = dest_lasso_dir / source_prn_lasso_file.name
                    if not dest_prn_lasso_file.exists():
                        shutil.copy2(source_prn_lasso_file, dest_prn_lasso_file)
                        print(f"  Copied cached PRN LASSO results: {source_prn_lasso_file.name}")
                    else:
                        print(f"  Using existing PRN LASSO results: {dest_prn_lasso_file.name}")
            elif not _load_cached_prn:
                print("  Skipping PRN LASSO results copy (load_cached_prn=false)")
            else:
                print("  Skipping PRN LASSO results copy (force_recalculate_prn_lasso=true)")

            # Point MODELS_DIR to new output directory (not source)
            # This ensures all outputs go to the new directory
            update_config(
                config["prefix"],
                models_dir=models_dir,
                interim_dir=interim_dir,
                processed_dir=processed_dir,
                config_name=config["config_name"],
            )
        else:
            # Note: Hyperparameters are loaded based on config settings:
            # - If params_file is specified in config -> load from that file
            # - Otherwise -> use defaults
            # No implicit copying; use explicit params_file in config for reproducibility

            # Update config to point models to this specific model's directory
            update_config(
                config["prefix"],
                models_dir=models_dir,
                interim_dir=interim_dir,
                processed_dir=processed_dir,
                config_name=config["config_name"],
            )

        # Skip mlp10 dependency training in prism-only mode
        # Check if mlp10 requires mlp model first
        if model == "mlp10" and not prism_only:
            # Check if mlp model has been run for this dataset
            mlp_model_dir = output_dir / "mlp" / "models"
            mlp_already_in_list = "mlp" in config["models"]

            # Use helper function to find model files with proper pattern
            mlp_model_files = find_model_files(mlp_model_dir, config["prefix"], "mlp")
            mlp_model_exists = len(mlp_model_files) > 0

            if mlp_model_exists:
                print(f"Found existing MLP model: {mlp_model_files[0].name}")

            if not mlp_model_exists and not mlp_already_in_list:
                print("Model mlp10 requires mlp model, but mlp not found. Training mlp first...")
                total += 1

                # Create mlp output directory and models subdirectory
                mlp_output_dir = output_dir / "mlp"
                mlp_output_dir.mkdir(parents=True, exist_ok=True)
                mlp_models_dir = mlp_output_dir / "models"
                mlp_models_dir.mkdir(parents=True, exist_ok=True)

                # Update config to point to mlp models directory
                update_config(
                    config["prefix"],
                    models_dir=mlp_models_dir,
                    interim_dir=interim_dir,
                    processed_dir=processed_dir,
                    config_name=config["config_name"],
                )

                # Get mlp notebook from template
                mlp_notebook = config["model_notebook_template"].format(model="mlp")
                mlp_notebook_path = NOTEBOOKS_DIR / "modelling" / mlp_notebook
                mlp_output_path = mlp_output_dir / f"{model_counter:02d}_train_mlp.pdf"

                # Check if this is an htx notebook that needs path fixes
                is_htx = dataset.startswith("htx")

                if execute_and_export_notebook(
                    mlp_notebook_path, mlp_output_path, is_htx_notebook=is_htx
                ):
                    successes += 1
                    gather_log_files(
                        mlp_output_dir, config["prefix"], model_prefix="mlp", stage="modelling"
                    )
                    model_counter += 1

                    # Restore config to current model's directory
                    update_config(
                        config["prefix"],
                        models_dir=models_dir,
                        interim_dir=interim_dir,
                        processed_dir=processed_dir,
                        config_name=config["config_name"],
                    )
                else:
                    print("Failed to train mlp model required for mlp10")

        # Skip training in prism-only mode
        # Skip training for mlp10 (uses mlp model)
        if model != "mlp10" and not prism_only:
            # Train model
            total += 1
            model_notebook = config["model_notebook_template"].format(model=model)
            notebook_path = NOTEBOOKS_DIR / "modelling" / model_notebook
            output_path = model_output_dir / f"{model_counter:02d}_train_{model}.pdf"

            # Check if this is an htx notebook that needs path fixes
            is_htx = dataset.startswith("htx")

            if execute_and_export_notebook(notebook_path, output_path, is_htx_notebook=is_htx):
                successes += 1
                gather_log_files(
                    model_output_dir, config["prefix"], model_prefix=model, stage="modelling"
                )

            model_counter += 1

        # PRiSM analysis for this model
        total += 1
        prism_notebook_path = NOTEBOOKS_DIR / config["prism_notebook"]
        prism_output_path = model_output_dir / f"{model_counter:02d}_prism_analysis_{model}.pdf"

        # Check if this is an htx notebook that needs path fixes
        is_htx = dataset.startswith("htx")

        if execute_and_export_notebook(
            prism_notebook_path, prism_output_path, model_prefix=model, is_htx_notebook=is_htx
        ):
            successes += 1
            gather_log_files(model_output_dir, config["prefix"], model_prefix=model, stage="prism")

        model_counter += 1

    # Concatenate all model predictions into a single file
    if len(config["models"]) > 1:
        concatenate_all_predictions(output_dir, config["prefix"], config.get("config_name"))

    # Save tuning artifacts (best_params) AFTER all training completes
    # Skip in prism-only mode (no new models trained)
    if reproducibility and not prism_only:
        save_tuning_artifacts(
            output_dir=output_dir,
            dataset_prefix=config["prefix"],
            models=config["models"],
        )

    # Save prism-only specific reproducibility artifacts
    if reproducibility and prism_only and prism_only_source_dir:
        save_prism_only_reproducibility(
            output_dir=output_dir,
            config_name=config["config_name"],
            dataset_prefix=config["prefix"],
            source_dir=prism_only_source_dir,
            source_config_name=prism_only_source_config,
            models=config["models"],
        )

    print(f"{dataset} completed: {successes}/{total} successful")
    return successes, total


def print_output_summary(output_dir: Path, dataset: str) -> None:
    """Print a summary of output files produced during this run only."""
    print(f"\n{'-'*60}")
    print(f"Output Summary for {dataset}")
    print(f"{'-'*60}")
    print(f" {output_dir}")

    if not output_dir.exists():
        print("  (no outputs)")
        return

    def has_new_files(path: Path) -> bool:
        """Check if directory contains any files/dirs created during this run."""
        if not path.is_dir():
            return False

        for item in path.iterdir():
            if run_start_time and item.stat().st_mtime >= run_start_time:
                return True
            if item.is_dir() and has_new_files(item):
                return True
        return False

    def print_tree(path: Path, prefix: str = "  "):
        """Recursively print directory tree, only showing files/dirs created or containing new files."""
        items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name))

        # Filter items: include if new OR if directory containing new files
        filtered_items = []
        for item in items:
            if run_start_time:
                is_new = item.stat().st_mtime >= run_start_time
                if item.is_dir():
                    # Include directory if it's new OR contains new files
                    if is_new or has_new_files(item):
                        filtered_items.append(item)
                elif is_new:
                    # Include file if it's new
                    filtered_items.append(item)
            else:
                filtered_items.append(item)

        if not filtered_items:
            return

        for i, item in enumerate(filtered_items):
            is_last = i == len(filtered_items) - 1
            connector = "+-- " if is_last else "|-- "
            if item.is_dir():
                print(f"{prefix}{connector}[{item.name}]/")
                extension = "    " if is_last else "|   "
                print_tree(item, prefix + extension)
            else:
                # Show file with size
                size = item.stat().st_size
                if size < 1024:
                    size_str = f"{size} B"
                elif size < 1024 * 1024:
                    size_str = f"{size/1024:.1f} KB"
                else:
                    size_str = f"{size/(1024*1024):.1f} MB"
                print(f"{prefix}{connector}{item.name} ({size_str})")

    print_tree(output_dir)


def main():
    """Main pipeline execution"""
    global run_start_time

    # Check that we're running from a cloned repo with notebooks
    if not NOTEBOOKS_DIR.exists():
        print(
            "ERROR: example_notebooks/ directory not found in current directory.\n"
            "This command requires a cloned PRiSM repository.\n"
            "Run from the repo root directory, e.g.:\n"
            "  cd PRiSM && prism run htx_example"
        )
        return 1

    # Record start time to filter output summary later
    run_start_time = datetime.now().timestamp()

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run PRiSM Notebook Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_prism_pipeline.py htx_example
    python run_prism_pipeline.py htx_example --skip-preprocessing
    python run_prism_pipeline.py htx_example --self-contained
    python run_prism_pipeline.py htx_example my_config
    python run_prism_pipeline.py --list-configs

    # PRiSM-only mode (requires prism_only_source_dir in config)
    python run_prism_pipeline.py my_config --prism-only
        """,
    )

    parser.add_argument(
        'configs',
        nargs='*',
        default=[],
        help='Config name(s) to process. Each config corresponds to a YAML file in example_notebooks/config/. Use --list-configs to see available configs.',
    )
    parser.add_argument(
        '-f',
        '--from-file',
        type=str,
        metavar='FILE',
        help='Load config names from a YAML batch file. Format: configs: [config1, config2, ...]. CLI configs run first, then file configs.',
    )
    parser.add_argument(
        '--skip-preprocessing',
        action='store_true',
        help='Skip preprocessing step (assumes preprocessed files exist)',
    )
    parser.add_argument(
        '--self-contained',
        action='store_true',
        help='Create fully isolated run with data copied to output folder (uses more disk space)',
    )
    parser.add_argument(
        '--list-configs',
        action='store_true',
        help='List available config files and exit',
    )
    parser.add_argument(
        '--reproducibility',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Save reproducibility artifacts (raw data, split data zipped, hashes, config) to output folder (default: True). Use --no-reproducibility to disable.',
    )
    parser.add_argument(
        '--no-tune',
        action='store_true',
        help='Skip hyperparameter tuning even if enabled in YAML config. '
        'Models will use previously saved params (if available) or defaults.',
    )
    parser.add_argument(
        '--prism-only',
        action='store_true',
        help='Run only PRiSM analysis using models from prism_only_source_dir in config. '
        'Skips preprocessing and model training.',
    )
    args = parser.parse_args()

    # Handle --list-configs
    if args.list_configs:
        available = list_available_configs()
        if available:
            print("Available configs:")
            for cfg in available:
                try:
                    pipeline_cfg = build_pipeline_config(cfg)
                    dataset = pipeline_cfg['prefix']
                    models = ', '.join(pipeline_cfg['models'])
                    print(f"  {cfg}: dataset={dataset}, models=[{models}]")
                except Exception as e:
                    print(f"  {cfg}: (error loading: {e})")
        else:
            print("No config files found in example_notebooks/config/")
        sys.exit(0)

    # Build list of configs to process (CLI args first, then file configs)
    configs_to_process = list(args.configs)  # CLI configs first

    if args.from_file:
        try:
            file_path = Path(args.from_file)
            file_configs = load_configs_from_file(file_path)
            print(f"Loaded {len(file_configs)} configs from {file_path}")
            # Add file configs that aren't already in CLI list
            for cfg in file_configs:
                if cfg not in configs_to_process:
                    configs_to_process.append(cfg)
                else:
                    print(f"  (skipping duplicate: {cfg})")
        except (FileNotFoundError, ValueError) as e:
            print(f"Error loading config file: {e}")
            sys.exit(1)

    # Default to htx_example if no configs specified
    if not configs_to_process:
        configs_to_process = ['htx_example']
        print("No configs specified, using default: htx_example")

    # Validate all configs upfront (fail fast)
    available_configs = list_available_configs()
    invalid_configs = [cfg for cfg in configs_to_process if cfg not in available_configs]

    if invalid_configs:
        print(f"Error: Config file(s) not found: {', '.join(invalid_configs)}")
        print(f"Available configs: {', '.join(available_configs)}")
        print("Use --list-configs to see available config files with their settings.")
        sys.exit(1)

    # Validate prism_only_source_config ordering upfront (fail fast)
    # This ensures configs referenced by prism_only_source_config appear before the referencing config
    try:
        source_config_refs = validate_prism_only_source_config_ordering(configs_to_process)
        if source_config_refs:
            print(f"Validated prism_only_source_config references: {source_config_refs}")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Set up tee logger (logs to pipeline_results/)
    global tee_logger
    pipeline_results_dir = PROJECT_ROOT / "example_notebooks" / "pipeline_results"
    pipeline_results_dir.mkdir(parents=True, exist_ok=True)

    log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"pipeline_run_{log_timestamp}.log"
    log_path = pipeline_results_dir / log_filename
    tee_logger = TeeLogger(log_path)
    tee_logger.start()
    print(f"Logging to: {log_path}")

    # Set PRISM_NO_TUNE if --no-tune flag is passed
    # This tells notebooks to skip inline tuning even if YAML has enabled: true
    if args.no_tune:
        os.environ['PRISM_NO_TUNE'] = '1'
        print("Hyperparameter tuning disabled via --no-tune flag")

    # Validate environment before starting
    validate_environment()

    # Start GPU health monitor for long runs
    start_gpu_monitor()

    # Register signal handlers and cleanup
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(cleanup_on_exit)

    total_successes = 0
    total_attempts = 0
    output_dirs = []

    # Track completed configs' output directories for prism_only_source_config resolution
    config_output_dirs = {}  # config_name -> output_dir

    # Process each config sequentially
    print(
        f"Starting PRiSM Pipeline for {len(configs_to_process)} config(s): {', '.join(configs_to_process)}"
    )

    for i, config_name in enumerate(configs_to_process, 1):
        print(f"\n{'='*80}")
        print(f"Processing config {i}/{len(configs_to_process)}: {config_name}")
        print(f"{'='*80}")

        # Load config from YAML
        try:
            config = build_pipeline_config(config_name)
        except Exception as e:
            print(f"Error loading config '{config_name}': {e}")
            total_attempts += 1
            continue

        dataset = config["prefix"]
        print(f"Dataset: {dataset}, Models: {config['models']}")

        # Load raw YAML config for per-config mode settings
        yaml_config, _ = load_config(config_name)

        # Determine per-config mode settings (CLI flag OR config setting)
        # self_contained: CLI flag or config setting
        config_self_contained = args.self_contained or yaml_config.get('self_contained', False)

        # prism_only: CLI flag, config setting, OR presence of prism_only_source_dir/source_config
        source_dir_str = yaml_config.get('prism_only_source_dir')
        source_config_name = yaml_config.get('prism_only_source_config')
        config_prism_only = (
            getattr(args, 'prism_only', False)
            or yaml_config.get('prism_only', False)
            or bool(source_dir_str)  # Auto-enable when source dir is specified
            or bool(source_config_name)  # Auto-enable when source config is specified
        )

        # Print per-config mode
        mode_parts = []
        if config_self_contained:
            mode_parts.append("self-contained")
        if config_prism_only:
            mode_parts.append("prism-only")
        if mode_parts:
            print(f"Mode: {', '.join(mode_parts)}")

        # PRiSM-only mode: validate source directory or resolve source config
        prism_only_paths = None
        source_dir = None  # Track for reproducibility
        if config_prism_only:
            # Resolve source directory from prism_only_source_config or prism_only_source_dir
            if source_config_name:
                # Look up the referenced config's output directory
                if source_config_name not in config_output_dirs:
                    # This shouldn't happen if validation passed, but handle gracefully
                    print(f"ERROR: Referenced config '{source_config_name}' has not completed yet")
                    print("This may indicate the config failed earlier in the run.")
                    total_attempts += 1
                    continue
                source_dir = config_output_dirs[source_config_name]
                print(
                    f"[PRiSM-Only] Using output from config '{source_config_name}': {source_dir}"
                )
            elif source_dir_str:
                # Resolve path relative to project root
                source_dir = Path(source_dir_str)
                if not source_dir.is_absolute():
                    source_dir = PROJECT_ROOT / source_dir
            else:
                print(
                    f"ERROR: prism-only mode requires 'prism_only_source_dir' or 'prism_only_source_config' in config '{config_name}'"
                )
                print("Add one of the following to your YAML config:")
                print(
                    "  prism_only_source_dir: 'example_notebooks/pipeline_results/YYYYMMDD_config_name'"
                )
                print(
                    "  prism_only_source_config: 'other_config_name'  # for multi-config batch runs"
                )
                total_attempts += 1
                continue

            # Validate source directory (fail-fast)
            try:
                prism_only_paths = validate_prism_only_source(
                    source_dir=source_dir,
                    dataset_prefix=dataset,
                    models=config['models'],
                )
                # Validate PRN caching requirements if load_cached_prn is enabled
                validate_load_cached_prn_requirements(
                    config=config,
                    validated_paths=prism_only_paths,
                    models=config['models'],
                )
            except (FileNotFoundError, ValueError) as e:
                print(e)
                total_attempts += 1
                continue

        # Validate params_file paths upfront (fail fast before expensive computation)
        # Skip in prism-only mode (using pre-trained models)
        if not config_prism_only:
            try:
                validate_params_files(config_name, config['models'])
                print("[OK] All params_file paths validated")
            except FileNotFoundError as e:
                print(e)
                total_attempts += 1
                continue

        # Get appropriate output directory (handles overlap detection and enumeration)
        # Use config_name for directory naming (not dataset)
        pipeline_results_dir = PROJECT_ROOT / "example_notebooks" / "pipeline_results"
        pipeline_results_dir.mkdir(parents=True, exist_ok=True)

        # In prism-only mode, skip_preprocessing is implied
        skip_preprocessing_for_dir = args.skip_preprocessing or config_prism_only

        config_output_dir = get_dataset_output_dir(
            base_dir=pipeline_results_dir,
            dataset=config_name,  # Use config name for directory naming
            date_str=today_date,
            models_to_run=config["models"],
            skip_preprocessing=skip_preprocessing_for_dir,
        )
        config_output_dir.mkdir(parents=True, exist_ok=True)
        output_dirs.append((config_name, config_output_dir))

        print(f"Output directory: {config_output_dir}")

        successes, attempts = process_dataset(
            dataset,
            config,
            config_output_dir,
            skip_preprocessing=args.skip_preprocessing,
            self_contained=config_self_contained,
            reproducibility=args.reproducibility,
            prism_only=config_prism_only,
            prism_only_paths=prism_only_paths,
            prism_only_source_dir=source_dir,
            prism_only_source_config=source_config_name,
        )
        total_successes += successes
        total_attempts += attempts

        # Track completed config's output directory for prism_only_source_config resolution
        config_output_dirs[config_name] = config_output_dir

        print(f"\n{config_name} Pipeline Complete!")
        print(f"{config_name}: {successes}/{attempts} notebooks successful")
        if attempts > 0:
            print(f"Success rate: {100*successes/attempts:.1f}%")

        # Print output summary for this config
        print_output_summary(config_output_dir, config_name)

    # Final summary
    print(f"\n{'='*80}")
    print("ALL CONFIGS COMPLETE!")
    print(f"{'='*80}")
    print(f"Total: {total_successes}/{total_attempts} notebooks successful")

    if total_successes == total_attempts:
        print("SUCCESS: All notebooks across all configs completed successfully!")
    else:
        print(f"FAILED: {total_attempts - total_successes} notebooks failed")

    print("\nOutput directories:")
    for cfg_name, output_dir in output_dirs:
        print(f"  {cfg_name}: {output_dir}")

    # Print log file location
    if tee_logger:
        print(f"\nLog file: {tee_logger.get_log_path()}")

        # Stop logger before exit (also done in cleanup_on_exit, but explicit is clearer)
        tee_logger.stop()


if __name__ == "__main__":
    main()
