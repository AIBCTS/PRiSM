#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = prism
PYTHON_VERSION = 3.11

# Allow specification of a custom Python interpreter
# e.g. for DGX2 use CUSTOM_PYTHON=/opt/python/3.11.7/bin/python3 make create_environment
CUSTOM_PYTHON ?=

# Determine the appropriate Python interpreter
ifdef CUSTOM_PYTHON
    PYTHON_INTERPRETER = $(CUSTOM_PYTHON)
else
    ifeq ($(OS),Windows_NT)
        PYTHON_INTERPRETER = python
    else
        ifeq ($(shell which python3),)
            PYTHON_INTERPRETER = python
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
	@$(PYTHON_INTERPRETER) -c "import sys; \
		current_version = '{}.{}'.format(sys.version_info.major, sys.version_info.minor); \
		print(f'Python {current_version} detected.'); \
		if current_version != '$(PYTHON_VERSION)': \
			print('\033[93mWarning: Python $(PYTHON_VERSION) is recommended for this project.\033[0m')"
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

## (Re)install Python Dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	
## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 prism
	isort --check --diff --profile black prism
	black --check --config pyproject.toml prism

## Format source code with black
.PHONY: format
format:
	black --config pyproject.toml prism

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
