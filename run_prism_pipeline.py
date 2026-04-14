#!/usr/bin/env python3
"""Backward-compatible entry point. Delegates to prism.cli.pipeline."""
# Re-export all public names so that `from run_prism_pipeline import X` works
from prism.cli.pipeline import *  # noqa: F401,F403
from prism.cli.pipeline import main

import sys

if __name__ == '__main__':
    sys.exit(main())
