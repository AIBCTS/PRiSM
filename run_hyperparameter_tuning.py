#!/usr/bin/env python3
"""Deprecated entry point. Delegates to prism.cli.tune.

Use `prism tune` instead. This wrapper is kept only so that existing invocations
of `python run_hyperparameter_tuning.py` keep working, and will be removed in a
future release.
"""

import sys

# Re-export all public names so that `from run_hyperparameter_tuning import X` works
from prism.cli.tune import *  # noqa: F401,F403
from prism.cli.tune import main

if __name__ == '__main__':
    sys.exit(main())
