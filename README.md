# PRiSM - Partial Responses in Structured Models

PRiSM (Partial Responses in Structured Models) is a method designed to transform **black-box classifiers into inherently interpretable models without compromising predictive performance**. This repository provides user-friendly code to implement PRiSM models, making it accessible for non-experts in machine learning.

## Getting Started

The PRiSM tool runs using `python`. To help with setup, we use `make`. It is assumed you have the following installed: `python 3.11`, `pip`, and [`make`](https://cookiecutter-data-science.drivendata.org/using-the-template/#make-as-a-task-runner).

Open a terminal and navigate to the project folder (where this `README.md` is). Start by creating the python virtual environment `venv_prism` by running:

```bash
make create_environment
```

Creating the environment may take a few minutes. To activate the virtual environment run:

On Windows: `.\venv_prism\Scripts\activate`

On Unix-like system: `.\venv_prism\bin\activate`

When running the Jupyter notebooks, select `venv_prism` as the kernel. See <\notebooks\4.01-hpi-unos-modelling-full.ipynb> for an example runthrough of the PRiSM method (_to be replaced with a proper tutorial notebook_). The notebook can be opened directly in VS Code, or in Jupyter directly by running `jupyter notebook \notebooks\4.01-hpi-unos-modelling-full.ipynb`.

Project documentation will eventually be hosted via github pages, but for now it can be viewed locally. See <docs/README.md>.

## Project Organization

```txt
├── Makefile           <- Makefile with convenience commands like `make create_environment` or `make train`
├── README.md          <- The top-level README for developers using this project
├── data               <- NB: the data directory is not included in source control.
│   ├── interim        <- Intermediate data that has been transformed
│   ├── processed      <- The final, canonical data sets for modeling
│   └── raw            <- The original, immutable data dump
│
├── docs               <- A default mkdocs project; see mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for prism
│                         and configuration for tools like black
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── prism              <- Source code for use in this project
│   ├── __init__.py    <- Makes prism a Python module
│   ├── config.py      <- Contains definitions like DATA_DIR
│   ├── maskedmlp.py   <- Masked MLP model definition and related functions
│   ├── PRiSM_functions.py  <- Other functions related to the PRiSM method
│   ├── prlasso.py     <- Partial response LASSO
│   ├── prnomogram.py  <- Partial response nomogram plotting functions
│   └── save_models.py <- Helper functions for saving model parameters, models, and metrics
```

For notebook numbering guidance, see the [conventions here](https://cookiecutter-data-science.drivendata.org/using-the-template/#open-a-notebook).

To export notebooks to pdf and .py, use `nbautoexport export notebooks` from the root project directory.

<p><small>Structure based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>.</small></p>