#!/usr/bin/env python3
"""Backward-compatible entry point. Delegates to prism.cli.tune."""
# Re-export all public names so that `from run_hyperparameter_tuning import X` works
from prism.cli.tune import *  # noqa: F401,F403
from prism.cli.tune import main

import sys

if __name__ == '__main__':
    sys.exit(main())
