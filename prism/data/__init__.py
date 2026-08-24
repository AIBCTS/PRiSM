"""Bundled example datasets for PRiSM.

Provides convenience functions to load the example datasets shipped with the
package, so users can get started without downloading external files.

Example
-------
    >>> from prism.data import load_example_dataset, list_example_datasets
    >>> print(list_example_datasets())
    ['credit-g', 'htx_example']
    >>> df = load_example_dataset('htx_example')
    >>> print(df.shape)
    (5000, 25)

Datasets
--------
- ``htx_example``: synthetic heart-transplant example data generated for PRiSM.
- ``credit-g``: the Statlog German Credit dataset (H. Hofmann, 1994), obtained
  via OpenML dataset 31 "credit-g" (https://www.openml.org/d/31); original
  source UCI Machine Learning Repository.

Also provides in-memory data preparation helpers (``prepare_dataset``,
``load_htx_example_splits``) that turn a raw DataFrame into PRiSM-ready
splits without the config/.env pipeline machinery.
"""

from importlib.resources import files
from pathlib import Path

import pandas as pd

from prism.data.preparation import (  # noqa: F401
    PreparedData,
    load_htx_example_splits,
    prepare_dataset,
)

_DATASETS = {
    "htx_example": "htx_example.csv",
    # Statlog German Credit (H. Hofmann, 1994), via OpenML dataset 31.
    "credit-g": "credit-g.csv",
}


def list_example_datasets():
    """Return sorted list of available bundled dataset names.

    Examples
    --------
    >>> from prism.data import list_example_datasets
    >>> list_example_datasets()
    ['credit-g', 'htx_example']
    """
    return sorted(_DATASETS.keys())


def get_dataset_path(name):
    """Return the filesystem Path to a bundled dataset CSV.

    Parameters
    ----------
    name : str
        Dataset name (e.g. ``'htx_example'``).

    Returns
    -------
    pathlib.Path

    Raises
    ------
    ValueError
        If *name* is not a known bundled dataset.

    Examples
    --------
    >>> from prism.data import get_dataset_path
    >>> get_dataset_path('htx_example').name
    'htx_example.csv'
    >>> get_dataset_path('nope')
    Traceback (most recent call last):
        ...
    ValueError: Unknown dataset 'nope'. Available datasets: credit-g, htx_example
    """
    if name not in _DATASETS:
        available = ", ".join(sorted(_DATASETS.keys()))
        raise ValueError(f"Unknown dataset '{name}'. Available datasets: {available}")
    return Path(str(files("prism.data").joinpath(_DATASETS[name])))


def load_example_dataset(name, **kwargs):
    """Load a bundled example dataset as a pandas DataFrame.

    Parameters
    ----------
    name : str
        Dataset name (e.g. ``'htx_example'``).
    **kwargs
        Additional keyword arguments passed to :func:`pandas.read_csv`.

    Returns
    -------
    pandas.DataFrame

    Raises
    ------
    ValueError
        If *name* is not a known bundled dataset.

    Examples
    --------
    >>> from prism.data import load_example_dataset
    >>> df = load_example_dataset('htx_example')
    >>> df.shape
    (5000, 25)
    >>> 'MORTALITY_365D' in df.columns
    True

    Keyword arguments are forwarded to :func:`pandas.read_csv`.

    >>> load_example_dataset('htx_example', nrows=10).shape
    (10, 25)
    """
    path = get_dataset_path(name)
    return pd.read_csv(path, **kwargs)
