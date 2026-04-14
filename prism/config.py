import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
# BUT only if not already set by pipeline runner (environment vars take precedence)
# This allows run_prism_pipeline.py to override .env settings
if not (os.environ.get('PRISM_CONFIG') or os.environ.get('PRISM_DATASET')):
    load_dotenv()

# Paths
# PRISM_PROJECT_DIR allows pip-installed users to point PROJ_ROOT at their working directory
_proj_root_override = os.environ.get('PRISM_PROJECT_DIR')
PROJ_ROOT = (
    Path(_proj_root_override).resolve()
    if _proj_root_override
    else Path(__file__).resolve().parents[1]
)
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

# Support environment variable overrides for parallel execution
# These allow run_prism_parallel.py to set isolated paths per worker
_interim_override = os.environ.get('PRISM_INTERIM_DATA_DIR')
INTERIM_DATA_DIR = Path(_interim_override) if _interim_override else DATA_DIR / "interim"

_processed_override = os.environ.get('PRISM_PROCESSED_DATA_DIR')
PROCESSED_DATA_DIR = Path(_processed_override) if _processed_override else DATA_DIR / "processed"

# Models directory - can be overridden via PRISM_MODELS_DIR for parallel execution
_models_dir_override = os.environ.get('PRISM_MODELS_DIR')
MODELS_DIR = Path(_models_dir_override) if _models_dir_override else PROJ_ROOT / "models"


# =============================================================================
# CONFIG/DATASET RESOLUTION
# =============================================================================
# Priority: PRISM_CONFIG > PRISM_DATASET > PRISM_DATASET_PREFIX (legacy)
#
# PRISM_CONFIG: Name of YAML config file (e.g., "htx_example" loads htx_example.yaml)
#               The config file must contain a "dataset:" field specifying the dataset
# PRISM_DATASET: Quick mode - just the dataset name, uses notebook defaults
# PRISM_DATASET_PREFIX: Legacy variable, works like PRISM_DATASET


def _load_config_for_dataset_resolution(config_name: str) -> dict:
    """Load config file to extract dataset name. Minimal loader to avoid circular imports."""
    import yaml

    config_dir = PROJ_ROOT / "example_notebooks" / "config"
    config_file = config_dir / f"{config_name}.yaml"

    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception:
            pass
    return {}


# Read environment variables
_config_name = os.environ.get('PRISM_CONFIG')
_dataset_override = os.environ.get('PRISM_DATASET') or os.environ.get('PRISM_DATASET_PREFIX')

# Resolve CONFIG_NAME and DATASET_PREFIX
if _config_name:
    # Full config mode: load YAML to get dataset name
    CONFIG_NAME = _config_name
    _config = _load_config_for_dataset_resolution(_config_name)
    # Dataset can be explicit in YAML, or fallback to config name
    DATASET_PREFIX = _config.get('dataset', _config_name)
    logger.info(f"Config mode: CONFIG_NAME={CONFIG_NAME}, DATASET_PREFIX={DATASET_PREFIX}")
elif _dataset_override:
    # Quick mode: just dataset name, no config file required
    CONFIG_NAME = None
    DATASET_PREFIX = _dataset_override
    logger.info(f"Quick mode: DATASET_PREFIX={DATASET_PREFIX}")
else:
    # No .env settings - notebooks will prompt user to set up .env
    # Provide a sensible default for backwards compatibility during transition
    CONFIG_NAME = None
    DATASET_PREFIX = None
    logger.warning(
        "No PRISM_CONFIG or PRISM_DATASET set in .env file. "
        "Copy .env.example to .env and configure your dataset."
    )

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
