#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = prism
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Set up python interpreter environment
.PHONY: create_environment
create_environment:
ifeq ($(OS),Windows_NT)
	@echo "Creating virtual environment on Windows (this may take a few minutes)..."
	@python -m venv venv_$(PROJECT_NAME) --clear --copies
	@echo ">>> New venv created. Activating and installing requirements..."
	@venv_$(PROJECT_NAME)\Scripts\python -m pip install --upgrade pip setuptools
	@venv_$(PROJECT_NAME)\Scripts\pip install -r requirements.txt
else
	@echo "Creating virtual environment on Unix-like system (this may take a few minutes)..."
	@python3 -m venv venv_$(PROJECT_NAME) --clear --copies
	@echo ">>> New venv created. Activating and installing requirements (this may take a few minutes)..."
	@venv_$(PROJECT_NAME)/bin/python -m pip install --upgrade pip setuptools
	@venv_$(PROJECT_NAME)/bin/pip install -r requirements.txt
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
	@python -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
