#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = prism

# Allow specification of a custom Python interpreter
# e.g. for DGX2 use CUSTOM_PYTHON=/opt/python/3.11.7/bin/python3 make create_environment
CUSTOM_PYTHON ?=

# Determine the appropriate Python interpreter
ifdef CUSTOM_PYTHON
    PYTHON_INTERPRETER = $(CUSTOM_PYTHON)
else
    ifeq ($(OS),Windows_NT)
        # Windows: prefer the launcher with an explicitly supported version (3.12, then 3.11),
        # since a bare `python`/`py` can resolve to 3.10 or 3.14 (no numpy<2.0 wheels there).
        PY312_CHECK := $(shell py -3.12 --version 2>nul)
        ifeq ($(PY312_CHECK),)
            PY311_CHECK := $(shell py -3.11 --version 2>nul)
            ifeq ($(PY311_CHECK),)
                # No supported launcher version found; fall back to whatever is on PATH
                # (the version check below will reject it if it is outside 3.11-3.12).
                PYTHON_CHECK := $(shell where python 2>nul)
                ifeq ($(PYTHON_CHECK),)
                    PYTHON_CHECK := $(shell where py 2>nul)
                    ifeq ($(PYTHON_CHECK),)
                        PYTHON_CHECK := $(shell where python3 2>nul)
                        ifeq ($(PYTHON_CHECK),)
                            PYTHON_INTERPRETER = python
                            $(warning Python not found in PATH. Install Python 3.11/3.12 or specify full path: make CUSTOM_PYTHON=/path/to/python.exe create_environment)
                        else
                            PYTHON_INTERPRETER = python3
                        endif
                    else
                        PYTHON_INTERPRETER = py
                    endif
                else
                    PYTHON_INTERPRETER = python
                endif
            else
                PYTHON_INTERPRETER = py -3.11
            endif
        else
            PYTHON_INTERPRETER = py -3.12
        endif
    else
        # Unix/Linux: Try python3, then python, then py
        PYTHON_CHECK := $(shell which python3 2>/dev/null)
        ifeq ($(PYTHON_CHECK),)
            PYTHON_CHECK := $(shell which python 2>/dev/null)
            ifeq ($(PYTHON_CHECK),)
                PYTHON_CHECK := $(shell which py 2>/dev/null)
                ifeq ($(PYTHON_CHECK),)
                    PYTHON_INTERPRETER = python3
                    $(warning Python not found in PATH. Install Python or specify full path: make CUSTOM_PYTHON=/path/to/python3 create_environment)
                else
                    PYTHON_INTERPRETER = py
                endif
            else
                PYTHON_INTERPRETER = python
            endif
        else
            PYTHON_INTERPRETER = python3
        endif
    endif
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Set up python interpreter environment
.PHONY: create_environment
create_environment:
	@echo "Checking Python version..."
	@$(PYTHON_INTERPRETER) -c "import sys; print('Python {}.{} detected.'.format(sys.version_info.major, sys.version_info.minor))"
	@$(PYTHON_INTERPRETER) -c "import sys; v = sys.version_info; sys.exit('ERROR: Python {}.{} is not supported. PRiSM requires Python 3.11 or 3.12 (numpy<2.0 has no wheels for 3.13+). On Windows try: py -3.12 -m venv venv_$(PROJECT_NAME); or specify one: make CUSTOM_PYTHON=/path/to/python3.12 create_environment'.format(v.major, v.minor)) if not ((3, 11) <= (v.major, v.minor) <= (3, 12)) else None"
	@echo "Creating virtual environment using Python interpreter: $(PYTHON_INTERPRETER)"
	@$(PYTHON_INTERPRETER) -m venv venv_$(PROJECT_NAME) --clear --copies
	@echo ">>> New venv created. Activating and installing requirements..."
ifeq ($(OS),Windows_NT)
	@venv_$(PROJECT_NAME)\Scripts\python -m pip install --upgrade pip setuptools
	@venv_$(PROJECT_NAME)\Scripts\pip install -r requirements.txt
else
	@. venv_$(PROJECT_NAME)/bin/activate && \
		pip install --upgrade pip setuptools && \
		pip install -r requirements.txt
endif
	@echo ">>> Environment setup complete. Activate with:"
ifeq ($(OS),Windows_NT)
	@echo ">>> venv_$(PROJECT_NAME)\Scripts\activate"
else
	@echo ">>> source venv_$(PROJECT_NAME)/bin/activate"
endif

## Set up python interpreter environment with dev dependencies (pytest, jupytext, etc.)
.PHONY: create_environment_dev
create_environment_dev:
	@echo "Checking Python version..."
	@$(PYTHON_INTERPRETER) -c "import sys; print('Python {}.{} detected.'.format(sys.version_info.major, sys.version_info.minor))"
	@$(PYTHON_INTERPRETER) -c "import sys; v = sys.version_info; sys.exit('ERROR: Python {}.{} is not supported. PRiSM requires Python 3.11 or 3.12 (numpy<2.0 has no wheels for 3.13+). On Windows try: py -3.12 -m venv venv_$(PROJECT_NAME); or specify one: make CUSTOM_PYTHON=/path/to/python3.12 create_environment'.format(v.major, v.minor)) if not ((3, 11) <= (v.major, v.minor) <= (3, 12)) else None"
	@echo "Creating virtual environment using Python interpreter: $(PYTHON_INTERPRETER)"
	@$(PYTHON_INTERPRETER) -m venv venv_$(PROJECT_NAME) --clear --copies
	@echo ">>> New venv created. Activating and installing requirements with dev dependencies..."
ifeq ($(OS),Windows_NT)
	@venv_$(PROJECT_NAME)\Scripts\python -m pip install --upgrade pip setuptools
	@venv_$(PROJECT_NAME)\Scripts\pip install -r requirements.txt
	@venv_$(PROJECT_NAME)\Scripts\pip install -e ".[dev,test]"
else
	@. venv_$(PROJECT_NAME)/bin/activate && \
		pip install --upgrade pip setuptools && \
		pip install -r requirements.txt && \
		pip install -e ".[dev,test]"
endif
	@echo ">>> Development environment setup complete. Activate with:"
ifeq ($(OS),Windows_NT)
	@echo ">>> venv_$(PROJECT_NAME)\Scripts\activate"
else
	@echo ">>> source venv_$(PROJECT_NAME)/bin/activate"
endif

## (Re)install Python Dependencies (run inside the activated venv; uses the active `python`,
## not the pinned bootstrap interpreter, so packages land in the venv)
.PHONY: requirements
requirements:
	python -m pip install -U pip
	python -m pip install -r requirements.txt

## Install GPU acceleration extras (cupy for CUDA 12.x) -- run inside the activated venv
.PHONY: requirements-gpu
requirements-gpu:
	python -m pip install -e ".[gpu]"
	@echo ">>> GPU extras installed. Verify with:"
	@echo ">>> python -c 'import cupy; print(cupy.cuda.runtime.getDeviceCount(), \"GPU(s) available\")'"

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 prism tests
	isort --check --diff --settings-path setup.cfg prism tests
	black --check --config pyproject.toml prism tests

## Format source code with black
.PHONY: format
format:
	black --config pyproject.toml prism tests
	isort --settings-path setup.cfg prism tests

## Run unit tests
.PHONY: test
test:
	pytest tests/ -v

## Run tests with coverage report
.PHONY: test-coverage
test-coverage:
	pytest tests/ -v --cov=prism --cov-report=term-missing

## Generate HTML coverage report
.PHONY: test-html
test-html:
	pytest tests/ -v --cov=prism --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

## Run only fast tests (exclude slow)
.PHONY: test-fast
test-fast:
	pytest tests/ -v -m "not slow"

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Make Dataset
# .PHONY: data
# data: requirements
# 	$(PYTHON_INTERPRETER) prism/data/make_dataset.py

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "$(PRINT_HELP_PYSCRIPT)" < $(MAKEFILE_LIST)
