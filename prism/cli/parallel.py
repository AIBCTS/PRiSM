#!/usr/bin/env python3
"""
PRiSM Parallel Pipeline Runner (Multi-GPU)

Executes model training and PRiSM analysis in parallel across multiple GPUs.
Use this on multi-GPU systems like DGX Station A100.

For single-GPU or MPS systems, use run_prism_pipeline.py instead.

Pipeline Flow:
    1. Preprocessing (sequential, shared across all models)
    2. Model training + PRiSM analysis (parallel, one model per GPU)
    3. Prediction concatenation (sequential, after all models complete)

Environment Variables (set per worker):
    PRISM_DATASET_PREFIX - Dataset name for this worker
    PRISM_MODELS_DIR - Output directory for this worker's models
    CUDA_VISIBLE_DEVICES - GPU assignment for this worker

Output:
    All output is logged to pipeline_results/pipeline_parallel_run_YYYYMMDD_HHMMSS.log
    with timestamps for debugging long-running jobs.

Examples
--------
    # Auto-detect GPUs and run in parallel with a config
    python run_prism_parallel.py htx_example

    # Specify GPUs explicitly (DGX with 4 GPUs)
    python run_prism_parallel.py htx_example --gpus 0,1,2,3

    # Run configs from a batch file
    python run_prism_parallel.py --from-file example_notebooks/config/my_batch.yaml
    python run_prism_parallel.py -f my_batch.yaml  # short form

    # Combine CLI configs with batch file (CLI runs first)
    python run_prism_parallel.py extra_config -f batch.yaml --gpus 0,1,2,3

    # Skip preprocessing (assumes already done)
    python run_prism_parallel.py htx_example --skip-preprocessing

    # Skip hyperparameter tuning even if enabled in YAML config
    python run_prism_parallel.py htx_example --no-tune

    # Create fully self-contained run (copies data to output folder)
    python run_prism_parallel.py htx_example --self-contained

    # Multiple configs
    python run_prism_parallel.py htx_example my_config --gpus 0,1,2,3

    # List available configs
    python run_prism_parallel.py --list-configs

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
Each model tunes independently in parallel on its assigned GPU.

To skip tuning (use defaults or previously saved params): --no-tune

For sharing tuned params across models or runs, pre-tune with the dedicated script
and specify params_file in your YAML config:
    python run_hyperparameter_tuning.py htx_example --models mlp xgb
    # Then in YAML: hyperparameter_tuning.mlp.params_file: 'models/..._best_params.json'
"""

import argparse
import atexit
import os
import shutil
import signal
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Project paths -- CLI operates on user's working directory
PROJECT_ROOT = Path.cwd()
NOTEBOOKS_DIR = PROJECT_ROOT / "example_notebooks"

from prism.cli.pipeline import (  # noqa: E402
    TeeLogger,
    build_pipeline_config,
    check_preprocessed_files_exist,
    cleanup_png_files,
    concatenate_all_predictions,
    find_model_files,
    gather_log_files,
    get_available_output_path,
    get_dataset_output_dir,
    is_latex_error,
    list_available_configs,
    load_configs_from_file,
    modify_htx_notebook_paths,
    modify_notebook_model_prefix,
    print_output_summary,
    save_prism_only_reproducibility,
    save_reproducibility_artifacts,
    save_tuning_artifacts,
    start_gpu_monitor,
    stop_gpu_monitor,
    validate_environment,
    validate_load_cached_prn_requirements,
    validate_params_files,
    validate_prism_only_source,
    validate_prism_only_source_config_ordering,
)
from prism.config_loader import load_config  # noqa: E402

# Global state
temp_files_to_cleanup = []
run_start_time = None
shutdown_requested = False  # Flag to stop submitting new tasks
main_process_pid = os.getpid()  # Track main process for signal handling
active_executor = None  # Reference to executor for forceful shutdown
worker_pids = set()  # Track worker process PIDs for termination
tee_logger = None  # Global logger for tee output


def get_available_gpu_ids() -> List[int]:
    """Return list of available CUDA GPU IDs."""
    try:
        import torch

        if not torch.cuda.is_available():
            return []
        return list(range(torch.cuda.device_count()))
    except ImportError:
        # Fallback: try nvidia-smi
        try:
            result = subprocess.run(
                ['nvidia-smi', '-L'], capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                lines = [
                    line for line in result.stdout.strip().split('\n') if line.startswith('GPU')
                ]
                return list(range(len(lines)))
        except Exception:
            pass
        return []


def execute_notebook_with_env(
    notebook_path: Path,
    output_path: Path,
    env: Dict[str, str],
    model_prefix: str = None,
    is_htx_notebook: bool = False,
) -> bool:
    """Execute notebook with custom environment variables.

    This is similar to execute_and_export_notebook from run_prism_pipeline.py
    but accepts a custom environment dict for parallel execution.
    """
    try:
        temp_notebook = None
        notebook_to_run = notebook_path

        # Extract dataset name for temp file naming
        dataset_name = env.get('PRISM_DATASET_PREFIX', 'unknown')

        if model_prefix:
            print(
                f"  [{dataset_name}] Executing {notebook_path.name} with model_prefix={model_prefix}"
            )
            temp_notebook = modify_notebook_model_prefix(notebook_path, model_prefix, dataset_name)
            if is_htx_notebook:
                htx_fixed = modify_htx_notebook_paths(temp_notebook, dataset_name, model_prefix)
                temp_notebook.unlink()
                temp_notebook = htx_fixed
            notebook_to_run = temp_notebook
        elif is_htx_notebook:
            print(f"  [{dataset_name}] Executing {notebook_path.name}")
            temp_notebook = modify_htx_notebook_paths(notebook_path, dataset_name)
            notebook_to_run = temp_notebook
        else:
            print(f"  [{dataset_name}] Executing {notebook_path.name}")

        # Use the current Python executable
        python_exe = sys.executable
        venv_dir = Path(sys.executable).parent

        # Build environment with venv priority
        exec_env = env.copy()
        exec_env['PATH'] = f"{venv_dir}{os.pathsep}{exec_env.get('PATH', '')}"
        # Force UTF-8 encoding on Windows to avoid UnicodeDecodeError from
        # IPython's deduperreload extension reading stdlib files as cp1252
        exec_env['PYTHONUTF8'] = '1'

        notebook_path_str = str(notebook_to_run).replace('\\', '/')

        # Stage 1: Execute notebook and save in-place
        # Use Popen with process group so we can terminate the entire tree on shutdown
        cmd_execute = f'"{python_exe}" -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=python3 "{notebook_path_str}"'

        try:
            # Start process in its own process group for clean termination
            process = subprocess.Popen(
                cmd_execute,
                shell=True,
                cwd=PROJECT_ROOT,
                env=exec_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,  # Creates new process group
            )

            # Wait for completion (this can take hours)
            stdout, stderr = process.communicate()
            returncode = process.returncode

        except Exception as e:
            print(f"  [{dataset_name}] Process error: {e}")
            if temp_notebook and temp_notebook.exists():
                temp_notebook.unlink()
            return False

        if returncode != 0:
            print(f"  [{dataset_name}] FAILED to execute {notebook_path.name}")
            print(f"  Error: {stderr if stderr else 'Unknown error'}")
            if temp_notebook and temp_notebook.exists():
                temp_notebook.unlink()
            return False

        # Stage 2: Convert to PDF or HTML
        pdf_output_path = output_path.with_suffix('.pdf')
        pdf_output_path = get_available_output_path(pdf_output_path)
        output_path_str = str(pdf_output_path).replace('\\', '/')

        cmd_pdf = f'"{python_exe}" -m jupyter nbconvert --to pdf --output "{output_path_str}" "{notebook_path_str}"'

        pdf_result = subprocess.run(
            cmd_pdf,
            shell=True,
            cwd=PROJECT_ROOT,
            env=exec_env,
            capture_output=True,
            text=True,
        )

        export_succeeded = False
        final_output_path = pdf_output_path

        if pdf_result.returncode == 0:
            print(f"  [{dataset_name}] Exported: {pdf_output_path.name}")
            export_succeeded = True
        elif is_latex_error(pdf_result.stderr):
            # Fallback to HTML
            html_output_path = output_path.with_suffix('.html')
            html_output_path = get_available_output_path(html_output_path)
            html_output_path_str = str(html_output_path).replace('\\', '/')

            cmd_html = f'"{python_exe}" -m jupyter nbconvert --to html --output "{html_output_path_str}" "{notebook_path_str}"'

            html_result = subprocess.run(
                cmd_html,
                shell=True,
                cwd=PROJECT_ROOT,
                env=exec_env,
                capture_output=True,
                text=True,
            )

            if html_result.returncode == 0:
                print(f"  [{dataset_name}] Exported (HTML): {html_output_path.name}")
                export_succeeded = True
                final_output_path = html_output_path
            else:
                print(f"  [{dataset_name}] HTML export also failed")
        else:
            print(f"  [{dataset_name}] PDF export failed: {pdf_result.stderr[:200]}")

        # Cleanup
        if temp_notebook and temp_notebook.exists():
            temp_notebook.unlink()

        if export_succeeded:
            cleanup_png_files(final_output_path.parent)

        return export_succeeded

    except Exception as e:
        print(f"  Exception: {e}")
        return False


def cleanup_gpu_resources():
    """Explicitly clean up GPU resources and garbage collect."""
    try:
        from prism.device_tools import cleanup_gpu_memory

        cleanup_gpu_memory()  # device=None cleans all backends with correct order
    except Exception:
        pass


def cleanup_on_exit():
    """Clean up temporary files and resources on exit."""
    # Stop GPU monitor (imported from run_prism_pipeline)
    stop_gpu_monitor()

    # Stop tee logger
    if tee_logger:
        tee_logger.stop()

    # 1. Clean up local temp files
    if temp_files_to_cleanup:
        print(f"\nCleaning up {len(temp_files_to_cleanup)} temporary files...")
        for p in temp_files_to_cleanup:
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass

    # 2. Clean up prism.cli.pipeline temp files (since we populate that list via imports)
    try:
        from prism.cli import pipeline as _pipeline_mod

        if _pipeline_mod.temp_files_to_cleanup:
            for p in _pipeline_mod.temp_files_to_cleanup:
                try:
                    if p.exists():
                        p.unlink()
                except Exception:
                    pass
    except Exception:
        pass

    # 3. GPU cleanup
    cleanup_gpu_resources()


def terminate_worker_processes():
    """Forcefully terminate all worker processes and their children."""
    # Shutdown executor without waiting
    if active_executor:
        try:
            # Python 3.9+: cancel_futures=True
            active_executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            # Python 3.8: no cancel_futures parameter
            active_executor.shutdown(wait=False)

    # Kill any remaining child processes
    import psutil

    try:
        current_process = psutil.Process(os.getpid())
        children = current_process.children(recursive=True)

        for child in children:
            try:
                print(f"  Terminating PID {child.pid} ({child.name()})...")
                child.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # Wait briefly for graceful termination
        gone, alive = psutil.wait_procs(children, timeout=3)

        # Force kill any remaining
        for child in alive:
            try:
                print(f"  Force killing PID {child.pid}...")
                child.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    except ImportError:
        # psutil not available, use os.kill on known PIDs
        print("  (psutil not available, using basic process termination)")
    except Exception as e:
        print(f"  Warning: Error during process cleanup: {e}")


def signal_handler(signum, frame):
    """Handle signals gracefully in the main process."""
    global shutdown_requested

    # Only handle in main process (workers should ignore)
    if os.getpid() != main_process_pid:
        return

    if shutdown_requested:
        # Second signal - force exit
        print("\nForced shutdown - killing all processes...")
        terminate_worker_processes()
        cleanup_on_exit()
        os._exit(1)

    print(f"\nReceived signal {signum}. Terminating workers...")
    shutdown_requested = True

    # Immediately terminate worker processes (don't wait for 4-hour tasks)
    terminate_worker_processes()

    print("Workers terminated. Cleaning up...")
    cleanup_on_exit()
    sys.exit(0)


def run_model_worker(
    model: str,
    dataset_prefix: str,
    output_dir: Path,
    gpu_id: int,
    config: Dict,
    model_counter_start: int = 2,
    prism_only: bool = False,
    prism_only_model_dir: Path = None,
    prism_only_pr_file: Path = None,
    prism_only_lasso_file: Path = None,
    prism_only_prn_model_file: Path = None,
    prism_only_prn_pr_file: Path = None,
    prism_only_prn_lasso_file: Path = None,
) -> Tuple[str, int, int, int]:
    """Run training + PRiSM analysis for a single model.

    Called in a subprocess with isolated environment.

    Parameters
    ----------
    model : str
        Model name (e.g., 'mlp', 'xgb')
    dataset_prefix : str
        Dataset prefix
    output_dir : Path
        Output directory for this run
    gpu_id : int
        GPU ID to use
    config : Dict
        Pipeline configuration
    model_counter_start : int
        Starting number for notebook output files
    prism_only : bool
        If True, skip training and only run PRiSM analysis
    prism_only_model_dir : Path, optional
        Source model directory for prism-only mode
    prism_only_pr_file : Path, optional
        Source partial responses file for prism-only mode (avoids expensive recalculation)
    prism_only_lasso_file : Path, optional
        Source LASSO results file for prism-only mode (allows different lambda selection)
    prism_only_prn_model_file : Path, optional
        Source PRN model file for prism-only mode (for PRN caching)
    prism_only_prn_pr_file : Path, optional
        Source PRN partial responses file for prism-only mode (for PRN caching)
    prism_only_prn_lasso_file : Path, optional
        Source PRN LASSO results file for prism-only mode (for PRN caching)

    Returns
    -------
    Tuple of (model_name, successes, total_attempts, next_counter)
    """
    try:
        successes = 0
        total = 0
        counter = model_counter_start

        # Create model-specific output directory
        model_output_dir = output_dir / model
        model_output_dir.mkdir(parents=True, exist_ok=True)

        models_dir = model_output_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        # Note: Hyperparameters are loaded based on config settings:
        # - If params_file is specified in config -> load from that file
        # - Otherwise -> use defaults
        # No implicit copying; use explicit params_file in config for reproducibility

        # Build environment for this worker
        # Set PRISM_CONFIG so notebooks pick up the correct config, not .env file
        env = os.environ.copy()
        env['PRISM_CONFIG'] = config['config_name']
        env['PRISM_DATASET_PREFIX'] = dataset_prefix
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        # In prism-only mode, copy model file to new output directory
        # This ensures outputs (nomograms, LASSO results, etc.) go to the new directory
        if prism_only and prism_only_model_dir:
            # Load YAML config to check caching flags and avoid copying files
            # that will be recalculated (prevents stale files in output directory)
            yaml_config, _ = load_config(config['config_name'])
            _load_cached_prn = yaml_config.get('load_cached_prn', False)
            _force_recalc_pr = yaml_config.get('force_recalculate_partial_responses', False)
            _force_recalc_prn_pr = yaml_config.get(
                'force_recalculate_prn_partial_responses', False
            )
            _force_recalc_lasso = yaml_config.get('force_recalculate_lasso', False)
            _force_recalc_prn_lasso = yaml_config.get('force_recalculate_prn_lasso', False)

            # mlp10 uses mlp model
            model_to_use = 'mlp' if model == 'mlp10' else model

            # Find the model file in source directory
            source_model_files = find_model_files(
                prism_only_model_dir, dataset_prefix, model_to_use
            )
            if source_model_files:
                source_model_file = source_model_files[0]  # Latest by mtime

                # Create destination directory structure matching source
                # Pattern: models/{dataset}_{model}/{dataset}_{model}_model_*.pt
                dest_model_subdir = models_dir / f"{dataset_prefix}_{model_to_use}"
                dest_model_subdir.mkdir(parents=True, exist_ok=True)
                dest_model_file = dest_model_subdir / source_model_file.name

                # Copy model file to new output directory
                if not dest_model_file.exists():
                    shutil.copy2(source_model_file, dest_model_file)
                    print(
                        f"[GPU {gpu_id}] Copied pre-trained {model_to_use} model: {source_model_file.name}"
                    )
                else:
                    print(f"[GPU {gpu_id}] Using existing model copy: {dest_model_file.name}")

            # Copy partial responses file if available (saves expensive GPU recalculation)
            # Skip when force_recalculate_partial_responses is set
            if _force_recalc_pr:
                print(
                    f"[GPU {gpu_id}] Skipping blackbox partial responses copy (force_recalculate_partial_responses=true)"
                )
            elif prism_only_pr_file and prism_only_pr_file.exists():
                dest_pr_dir = models_dir / "partial_responses"
                dest_pr_dir.mkdir(exist_ok=True)
                dest_pr_file = dest_pr_dir / prism_only_pr_file.name
                if not dest_pr_file.exists():
                    shutil.copy2(prism_only_pr_file, dest_pr_file)
                    print(
                        f"[GPU {gpu_id}] Copied cached partial responses: {prism_only_pr_file.name}"
                    )
                else:
                    print(f"[GPU {gpu_id}] Using existing partial responses: {dest_pr_file.name}")

            # Copy LASSO results file if available (allows different lambda selection)
            # Skip when force_recalculate_lasso is set (LASSO will be recomputed with new seed)
            if _force_recalc_lasso:
                print(
                    f"[GPU {gpu_id}] Skipping blackbox LASSO results copy (force_recalculate_lasso=true)"
                )
            elif prism_only_lasso_file and prism_only_lasso_file.exists():
                dest_lasso_dir = models_dir / "lasso_results"
                dest_lasso_dir.mkdir(exist_ok=True)
                dest_lasso_file = dest_lasso_dir / prism_only_lasso_file.name
                if not dest_lasso_file.exists():
                    shutil.copy2(prism_only_lasso_file, dest_lasso_file)
                    print(
                        f"[GPU {gpu_id}] Copied cached LASSO results: {prism_only_lasso_file.name}"
                    )
                else:
                    print(f"[GPU {gpu_id}] Using existing LASSO results: {dest_lasso_file.name}")

            # Copy PRN model file if available and load_cached_prn is enabled
            # Skip copying when PRN will be retrained (avoids stale files in output)
            if _load_cached_prn:
                if prism_only_prn_model_file and prism_only_prn_model_file.exists():
                    # PRN model directory structure: models/{dataset}_{model}_prn/{dataset}_{model}_prn_model_tuned.pt
                    model_to_use = 'mlp' if model == 'mlp10' else model
                    dest_prn_model_subdir = models_dir / f"{dataset_prefix}_{model_to_use}_prn"
                    dest_prn_model_subdir.mkdir(parents=True, exist_ok=True)
                    dest_prn_model_file = dest_prn_model_subdir / prism_only_prn_model_file.name
                    if not dest_prn_model_file.exists():
                        shutil.copy2(prism_only_prn_model_file, dest_prn_model_file)
                        print(
                            f"[GPU {gpu_id}] Copied cached PRN model: {prism_only_prn_model_file.name}"
                        )
                    else:
                        print(
                            f"[GPU {gpu_id}] Using existing PRN model: {dest_prn_model_file.name}"
                        )
            else:
                print(
                    f"[GPU {gpu_id}] Skipping PRN model copy (load_cached_prn=false, PRN will be retrained)"
                )

            # Copy PRN partial responses file if available (for PRN caching)
            # Only copy when load_cached_prn=true AND not force-recalculating PRN partial responses
            if _load_cached_prn and not _force_recalc_prn_pr:
                if prism_only_prn_pr_file and prism_only_prn_pr_file.exists():
                    dest_pr_dir = models_dir / "partial_responses"
                    dest_pr_dir.mkdir(exist_ok=True)
                    dest_prn_pr_file = dest_pr_dir / prism_only_prn_pr_file.name
                    if not dest_prn_pr_file.exists():
                        shutil.copy2(prism_only_prn_pr_file, dest_prn_pr_file)
                        print(
                            f"[GPU {gpu_id}] Copied cached PRN partial responses: {prism_only_prn_pr_file.name}"
                        )
                    else:
                        print(
                            f"[GPU {gpu_id}] Using existing PRN partial responses: {dest_prn_pr_file.name}"
                        )
            elif not _load_cached_prn:
                print(
                    f"[GPU {gpu_id}] Skipping PRN partial responses copy (load_cached_prn=false)"
                )
            else:
                print(
                    f"[GPU {gpu_id}] Skipping PRN partial responses copy (force_recalculate_prn_partial_responses=true)"
                )

            # Copy PRN LASSO results file if available (for PRN caching)
            # Only copy when load_cached_prn=true AND not force-recalculating PRN LASSO
            if _load_cached_prn and not _force_recalc_prn_lasso:
                if prism_only_prn_lasso_file and prism_only_prn_lasso_file.exists():
                    dest_lasso_dir = models_dir / "lasso_results"
                    dest_lasso_dir.mkdir(exist_ok=True)
                    dest_prn_lasso_file = dest_lasso_dir / prism_only_prn_lasso_file.name
                    if not dest_prn_lasso_file.exists():
                        shutil.copy2(prism_only_prn_lasso_file, dest_prn_lasso_file)
                        print(
                            f"[GPU {gpu_id}] Copied cached PRN LASSO results: {prism_only_prn_lasso_file.name}"
                        )
                    else:
                        print(
                            f"[GPU {gpu_id}] Using existing PRN LASSO results: {dest_prn_lasso_file.name}"
                        )
            elif not _load_cached_prn:
                print(f"[GPU {gpu_id}] Skipping PRN LASSO results copy (load_cached_prn=false)")
            else:
                print(
                    f"[GPU {gpu_id}] Skipping PRN LASSO results copy (force_recalculate_prn_lasso=true)"
                )

            # Point MODELS_DIR to new output directory (not source)
            env['PRISM_MODELS_DIR'] = str(models_dir)
            print(f"[GPU {gpu_id}] Starting {model} for {dataset_prefix} (PRiSM-only)")
        else:
            env['PRISM_MODELS_DIR'] = str(models_dir)
            print(f"[GPU {gpu_id}] Starting {model} for {dataset_prefix}")

        is_htx = dataset_prefix.startswith("htx")

        # 1. Train model (skip in prism-only mode)
        if not prism_only:
            # For mlp10: train mlp as a dependency (mlp10 PRiSM analysis loads the mlp model)
            # Training is fast compared to PRiSM, so this doesn't significantly delay mlp10
            # and allows mlp10 to run fully in parallel without waiting for mlp's PRiSM to finish
            train_model = 'mlp' if model == 'mlp10' else model

            total += 1
            model_notebook = config["model_notebook_template"].format(model=train_model)
            notebook_path = NOTEBOOKS_DIR / "modelling" / model_notebook
            output_path = model_output_dir / f"{counter:02d}_train_{train_model}.pdf"

            if execute_notebook_with_env(notebook_path, output_path, env, is_htx_notebook=is_htx):
                successes += 1
                gather_log_files(
                    model_output_dir, dataset_prefix, model_prefix=train_model, stage="modelling"
                )

            counter += 1

        # 2. PRiSM analysis
        total += 1
        prism_notebook_path = NOTEBOOKS_DIR / config["prism_notebook"]
        prism_output_path = model_output_dir / f"{counter:02d}_prism_analysis_{model}.pdf"

        if execute_notebook_with_env(
            prism_notebook_path, prism_output_path, env, model_prefix=model, is_htx_notebook=is_htx
        ):
            successes += 1
            gather_log_files(model_output_dir, dataset_prefix, model_prefix=model, stage="prism")

        counter += 1

        print(f"[GPU {gpu_id}] Completed {model}: {successes}/{total}")
        return (model, successes, total, counter)
    finally:
        cleanup_gpu_resources()


def run_preprocessing(
    dataset: str,
    config: Dict,
    output_dir: Path,
    processed_dir: Path = None,
) -> bool:
    """Run preprocessing notebook (sequential, before parallel model execution).

    Parameters
    ----------
    dataset : str
        Dataset name
    config : Dict
        Pipeline configuration
    output_dir : Path
        Output directory for this run
    processed_dir : Path, optional
        If provided (self-contained mode), search here for metadata instead of
        the global PROCESSED_DATA_DIR.
    """
    print(f"\nRunning preprocessing for {dataset}...")

    # Set environment for preprocessing
    # Set PRISM_CONFIG so notebooks pick up the correct config, not .env file
    env = os.environ.copy()
    env['PRISM_CONFIG'] = config['config_name']
    env['PRISM_DATASET_PREFIX'] = config['prefix']
    # Don't override MODELS_DIR for preprocessing

    notebook_path = NOTEBOOKS_DIR / config["preprocessing_notebook"]
    output_path = output_dir / "01_preprocessing.pdf"

    is_htx = dataset.startswith("htx")

    success = execute_notebook_with_env(notebook_path, output_path, env, is_htx_notebook=is_htx)

    if success:
        gather_log_files(output_dir, config["prefix"], stage="preprocessing")

        # Copy preprocessing metadata to top level if it exists
        if processed_dir:
            metadata_pattern = f"preprocessing_metadata_{config['prefix']}_*.json"
            metadata_files = list(processed_dir.glob(metadata_pattern))
        else:
            from prism.config import PROCESSED_DATA_DIR

            metadata_pattern = f"preprocessing_metadata_{config['prefix']}_*.json"
            metadata_files = list(PROCESSED_DATA_DIR.glob(metadata_pattern))

        if metadata_files:
            import shutil

            latest = max(metadata_files, key=lambda p: p.stat().st_mtime)
            dest = output_dir / latest.name
            if not dest.exists():
                shutil.copy(latest, dest)
                print(f"  Copied metadata: {latest.name}")

    return success


def process_dataset_parallel(
    dataset: str,
    config: Dict,
    output_dir: Path,
    gpu_ids: List[int],
    skip_preprocessing: bool = False,
    reproducibility: bool = True,
    self_contained: bool = False,
    prism_only: bool = False,
    prism_only_paths: Dict = None,
    prism_only_source_dir: Path = None,
    prism_only_source_config: str = None,
) -> Tuple[int, int]:
    """Process a dataset with parallel model execution across GPUs.

    Parameters
    ----------
    dataset : str
        Dataset name
    config : Dict
        Pipeline configuration
    output_dir : Path
        Output directory for this run
    gpu_ids : List[int]
        List of GPU IDs to use
    skip_preprocessing : bool
        Skip preprocessing step
    reproducibility : bool
        Save reproducibility artifacts (data, hashes, config) to output folder
    self_contained : bool
        If True, redirect data directories to output folder for full isolation
    prism_only : bool
        If True, skip preprocessing and training, run only PRiSM analysis
    prism_only_paths : Dict, optional
        Validated paths from validate_prism_only_source() when prism_only=True
    prism_only_source_dir : Path, optional
        Source directory for prism-only mode (for reproducibility tracking)
    prism_only_source_config : str, optional
        Source config name if using prism_only_source_config (for reproducibility tracking)

    Returns
    -------
    Tuple[int, int]
        (total_successes, total_attempts)
    """
    print(f"\n{'='*60}")
    print(f"Processing {dataset} in parallel on GPUs: {gpu_ids}")
    print(f"Models: {config['models']}")
    if prism_only:
        print("Mode: PRiSM-only (using pre-trained models)")
    print(f"{'='*60}")

    total_successes = 0
    total_attempts = 0

    # Data directories setup for self-contained mode
    interim_dir = None
    processed_dir = None

    # PRiSM-only mode: use source paths for processed data
    if prism_only and prism_only_paths:
        processed_dir = prism_only_paths.get("processed_data_dir")
        if processed_dir:
            os.environ['PRISM_PROCESSED_DATA_DIR'] = str(processed_dir).replace('\\', '/')
        print("Skipping preprocessing (--prism-only mode)")
        print("Skipping model training (--prism-only mode)")
    elif self_contained:
        # Full isolation: redirect data directories to output folder
        interim_dir = output_dir / "data" / "interim"
        processed_dir = output_dir / "data" / "processed"
        interim_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        # Set environment variables so all notebooks (preprocessing and workers) use these dirs
        os.environ['PRISM_INTERIM_DATA_DIR'] = str(interim_dir).replace('\\', '/')
        os.environ['PRISM_PROCESSED_DATA_DIR'] = str(processed_dir).replace('\\', '/')
        print(f"Self-contained mode: data dirs -> {output_dir / 'data'}")
    else:
        # Clean up data dir overrides (revert to defaults) for non-self-contained runs
        if 'PRISM_INTERIM_DATA_DIR' in os.environ:
            del os.environ['PRISM_INTERIM_DATA_DIR']
        if 'PRISM_PROCESSED_DATA_DIR' in os.environ:
            del os.environ['PRISM_PROCESSED_DATA_DIR']

    # 1. Preprocessing (sequential) - skip in prism-only mode
    if not prism_only:
        if skip_preprocessing:
            print("Skipping preprocessing (--skip-preprocessing)")
            if not check_preprocessed_files_exist(dataset, config):
                print("ERROR: Preprocessed files not found. Cannot continue.")
                return 0, 1
        elif config.get("preprocessing_notebook"):
            total_attempts += 1
            if run_preprocessing(dataset, config, output_dir, processed_dir=processed_dir):
                total_successes += 1
            else:
                print("WARNING: Preprocessing failed, but continuing with model training...")

        # Save reproducibility artifacts after preprocessing
        if reproducibility:
            save_reproducibility_artifacts(
                output_dir=output_dir,
                config_name=config["config_name"],
                dataset_prefix=config["prefix"],
                interim_dir=interim_dir,
            )

    # 2. All models run in parallel
    # mlp10 trains its own mlp model as a dependency (training is fast),
    # so it doesn't need to wait for the mlp worker's PRiSM analysis to complete
    models = config['models'].copy()

    # 3. Run all models in parallel
    max_workers = min(len(models), len(gpu_ids))
    print(f"\nStarting parallel execution: {len(models)} models on {max_workers} GPUs")

    model_results = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Track executor for signal handler to terminate
        global active_executor
        active_executor = executor

        futures = {}

        for i, model in enumerate(models):
            # Check for shutdown before submitting new tasks
            if shutdown_requested:
                print(f"Shutdown requested, skipping remaining models: {models[i:]}")
                break

            gpu_id = gpu_ids[i % len(gpu_ids)]

            # Get source model directory and cached files for prism-only mode
            prism_only_model_dir = None
            prism_only_pr_file = None
            prism_only_lasso_file = None
            prism_only_prn_model_file = None
            prism_only_prn_pr_file = None
            prism_only_prn_lasso_file = None
            if prism_only and prism_only_paths:
                prism_only_model_dir = prism_only_paths["model_dirs"].get(model)
                prism_only_pr_file = prism_only_paths.get("partial_responses_files", {}).get(model)
                prism_only_lasso_file = prism_only_paths.get("lasso_results_files", {}).get(model)
                prism_only_prn_model_file = prism_only_paths.get("prn_model_files", {}).get(model)
                prism_only_prn_pr_file = prism_only_paths.get(
                    "prn_partial_responses_files", {}
                ).get(model)
                prism_only_prn_lasso_file = prism_only_paths.get(
                    "prn_lasso_results_files", {}
                ).get(model)

            # In prism-only mode, counter starts at 1 (no preprocessing/training)
            model_counter_start = 1 if prism_only else 2

            future = executor.submit(
                run_model_worker,
                model,
                config['prefix'],
                output_dir,
                gpu_id,
                config,
                model_counter_start,
                prism_only,
                prism_only_model_dir,
                prism_only_pr_file,
                prism_only_lasso_file,
                prism_only_prn_model_file,
                prism_only_prn_pr_file,
                prism_only_prn_lasso_file,
            )
            futures[future] = model

        # Collect results as they complete
        for future in as_completed(futures):
            model = futures[future]
            try:
                name, successes, attempts, _ = future.result()
                model_results[name] = (successes, attempts)
                total_successes += successes
                total_attempts += attempts
                print(f"  {name} done: {successes}/{attempts}")
            except Exception as e:
                print(f"  {model} ERROR: {e}")
                # In prism-only mode, only 1 attempt (prism); otherwise 2 (train + prism)
                expected_attempts = 1 if prism_only else 2
                model_results[model] = (0, expected_attempts)
                total_attempts += expected_attempts

            # Check for shutdown after each completion
            if shutdown_requested:
                print("Shutdown requested, cancelling pending tasks...")
                for f in futures:
                    f.cancel()
                break

    # Clear executor reference
    active_executor = None

    # 4. Concatenate predictions
    if len(models) > 1:
        concatenate_all_predictions(output_dir, config['prefix'], config.get('config_name'))

    # 5. Save tuning artifacts (best_params) AFTER all training completes
    # Skip in prism-only mode (no new models trained)
    if reproducibility and not prism_only:
        save_tuning_artifacts(
            output_dir=output_dir,
            dataset_prefix=config['prefix'],
            models=config['models'],
        )

    # 6. Save prism-only specific reproducibility artifacts
    if reproducibility and prism_only and prism_only_source_dir:
        save_prism_only_reproducibility(
            output_dir=output_dir,
            config_name=config["config_name"],
            dataset_prefix=config["prefix"],
            source_dir=prism_only_source_dir,
            source_config_name=prism_only_source_config,
            models=config["models"],
        )

    return total_successes, total_attempts


def main():
    """Main entry point for parallel pipeline."""
    global run_start_time

    if not NOTEBOOKS_DIR.exists():
        print(
            "ERROR: example_notebooks/ directory not found in current directory.\n"
            "This command requires a cloned PRiSM repository.\n"
            "Run from the repo root directory, e.g.:\n"
            "  cd PRiSM && prism run-parallel htx_example"
        )
        return 1

    run_start_time = datetime.now().timestamp()

    parser = argparse.ArgumentParser(
        description='PRiSM Parallel Pipeline Runner (Multi-GPU)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_prism_parallel.py htx_example
    python run_prism_parallel.py htx_example --gpus 0,1,2,3
    python run_prism_parallel.py htx_example --skip-preprocessing
    python run_prism_parallel.py htx_example my_config
    python run_prism_parallel.py --list-configs

    # PRiSM-only mode (requires prism_only_source_dir in config)
    python run_prism_parallel.py my_config --prism-only --gpus 0,1
        """,
    )

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Register atexit
    atexit.register(cleanup_on_exit)

    parser.add_argument(
        'configs',
        nargs='*',
        default=[],
        help='Config name(s) to process. Use --list-configs to see available configs.',
    )
    parser.add_argument(
        '-f',
        '--from-file',
        type=str,
        metavar='FILE',
        help='Load config names from a YAML batch file. Format: configs: [config1, config2, ...]. CLI configs run first, then file configs.',
    )
    parser.add_argument(
        '--gpus',
        type=str,
        default=None,
        help='Comma-separated GPU IDs (e.g., "0,1,2,3"). Auto-detected if not specified.',
    )
    parser.add_argument(
        '--skip-preprocessing',
        action='store_true',
        help='Skip preprocessing (assumes preprocessed files exist)',
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
        '--self-contained',
        action='store_true',
        help='Create fully isolated run with data copied to output folder (uses more disk space)',
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
    invalid_configs = [c for c in configs_to_process if c not in available_configs]
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
    log_filename = f"pipeline_parallel_run_{log_timestamp}.log"
    log_path = pipeline_results_dir / log_filename
    tee_logger = TeeLogger(log_path)
    tee_logger.start()
    print(f"Logging to: {log_path}")

    # Set PRISM_NO_TUNE if --no-tune flag is passed
    # This tells notebooks to skip inline tuning even if YAML has enabled: true
    if args.no_tune:
        os.environ['PRISM_NO_TUNE'] = '1'
        print("Hyperparameter tuning disabled via --no-tune flag")

    # Validate environment
    validate_environment()

    # Detect GPUs
    if args.gpus:
        gpu_ids = [int(g.strip()) for g in args.gpus.split(',')]
    else:
        gpu_ids = get_available_gpu_ids()

    if len(gpu_ids) < 2:
        print("\n" + "=" * 60)
        print("WARNING: Less than 2 GPUs detected.")
        print(f"Detected GPUs: {gpu_ids if gpu_ids else 'None'}")
        print("\nFor single-GPU or CPU systems, use run_prism_pipeline.py instead:")
        print("    python run_prism_pipeline.py htx_example")
        print("=" * 60 + "\n")
        sys.exit(1)

    print(f"Detected {len(gpu_ids)} GPUs: {gpu_ids}")

    # Start GPU health monitor to prevent driver connection loss
    if gpu_ids:
        start_gpu_monitor()

    # Process each config
    today_date = datetime.now().strftime("%Y%m%d")
    total_successes = 0
    total_attempts = 0
    output_dirs = []

    # Track completed configs' output directories for prism_only_source_config resolution
    config_output_dirs = {}  # config_name -> output_dir

    for config_name in configs_to_process:
        # Check for shutdown before starting next config
        if shutdown_requested:
            print("\nShutdown requested, skipping remaining configs")
            break

        # Load config from YAML
        try:
            config = build_pipeline_config(config_name)
        except Exception as e:
            print(f"Error loading config '{config_name}': {e}")
            total_attempts += 1
            continue

        dataset = config["prefix"]
        print(f"\nConfig: {config_name}, Dataset: {dataset}, Models: {config['models']}")

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
        pipeline_results_dir = PROJECT_ROOT / "example_notebooks" / "pipeline_results"
        pipeline_results_dir.mkdir(parents=True, exist_ok=True)

        # In prism-only mode, skip_preprocessing is implied
        skip_preprocessing_for_dir = args.skip_preprocessing or config_prism_only

        output_dir = get_dataset_output_dir(
            base_dir=pipeline_results_dir,
            dataset=config_name,  # Use config name for directory naming
            date_str=today_date,
            models_to_run=config["models"],
            skip_preprocessing=skip_preprocessing_for_dir,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dirs.append((config_name, output_dir))

        print(f"Output directory: {output_dir}")

        successes, attempts = process_dataset_parallel(
            dataset,
            config,
            output_dir,
            gpu_ids,
            skip_preprocessing=args.skip_preprocessing,
            reproducibility=args.reproducibility,
            self_contained=config_self_contained,
            prism_only=config_prism_only,
            prism_only_paths=prism_only_paths,
            prism_only_source_dir=source_dir,
            prism_only_source_config=source_config_name,
        )

        total_successes += successes
        total_attempts += attempts

        # Track completed config's output directory for prism_only_source_config resolution
        config_output_dirs[config_name] = output_dir

        print(f"\n{config_name} completed: {successes}/{attempts}")
        print_output_summary(output_dir, config_name)

    # Final summary
    print(f"\n{'='*60}")
    print("PARALLEL PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Total: {total_successes}/{total_attempts} notebooks successful")

    if total_successes == total_attempts:
        print("SUCCESS: All notebooks completed!")
    else:
        print(f"FAILED: {total_attempts - total_successes} notebooks failed")

    print("\nOutput directories:")
    for cfg_name, out_dir in output_dirs:
        print(f"  {cfg_name}: {out_dir}")

    # Print log file location
    if tee_logger:
        print(f"\nLog file: {tee_logger.get_log_path()}")
        tee_logger.stop()

    stop_gpu_monitor()
    cleanup_gpu_resources()


if __name__ == "__main__":
    main()
