"""PRiSM: Partial Responses in Structured Models.

Converts a trained probabilistic binary classifier for tabular data into a
standalone nomogram that can be audited term by term. PRiSM takes the shape of
each nonlinear effect and interaction from the source model, not merely which
variables mattered, and lets the observed outcome select and weight them. The
result replaces the source classifier as the deployed predictor; a single
regularization parameter sets where you sit on the sparsity/performance
trade-off.

The method runs in two phases. Phase I decomposes the source model's log-odds
into univariate main effects and bivariate interactions (a functional ANOVA
decomposition), then a LASSO pass against the observed outcome selects and
weights the terms worth keeping. Phase II trains a Partial Response Network
whose architecture is fixed by the selected effects, re-decomposes it, and runs
a second LASSO pass.

Binary classification only. Effects are additive on the log-odds scale.

Start with `example_notebooks/prism_demonstration.ipynb` in the repository, or
`prism.data.load_htx_example_splits` for the bundled example dataset.
"""

# Must run before anything imports torch or xgboost (see configure_openmp), so it
# is the first thing prism does on import.
import os
import platform

from prism._env_setup import configure_openmp

configure_openmp(platform.system(), os.environ)

# os and platform exist only for the call above. Without this they show up in
# dir(prism) as public names, which is misleading in an API listing.
del os, platform

from prism import config  # noqa: F401,E402
from prism import data as data  # noqa: F401
from prism.config_loader import (  # noqa: F401
    DEFAULT_ID_CANDIDATES,
    DEFAULT_LABEL_FILE_CANDIDATES,
    DEFAULT_TARGET_CANDIDATES,
    detect_target_and_id_columns,
    load_config_if_exists,
    load_label_file,
)
from prism.feature_labels import FeatureLabelManager, generate_label_file_template  # noqa: F401
from prism.notebook_utils import (  # noqa: F401
    get_analysis_params,
    load_model_checkpoint,
    load_preprocessing_metadata,
    load_train_test_val_data,
    validate_dataset_configured,
)
from prism.plotting_data import PlottingDataBundle  # noqa: F401

from .__version__ import __version__  # noqa: F401
