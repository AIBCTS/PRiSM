"""Bundled example datasets for PRiSM.

Provides convenience functions to load the example datasets shipped with the
package, so users can get started without downloading external files.

Example
-------
    >>> from prism.data import load_example_dataset, list_example_datasets
    >>> print(list_example_datasets())
    ['htx_example']
    >>> df = load_example_dataset('htx_example')
    >>> print(df.shape)
"""

from importlib.resources import files
from pathlib import Path

import pandas as pd

_DATASETS = {
    "htx_example": "htx_example.csv",
}


def list_example_datasets():
    """Return sorted list of available bundled dataset names."""
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
    """
    path = get_dataset_path(name)
    return pd.read_csv(path, **kwargs)
