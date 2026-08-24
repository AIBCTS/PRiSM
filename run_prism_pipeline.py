#!/usr/bin/env python3
"""Deprecated entry point. Delegates to prism.cli.pipeline.

Use `prism run` instead. This wrapper is kept only so that existing invocations
of `python run_prism_pipeline.py` keep working, and will be removed in a future
release.
"""

import sys

# Re-export all public names so that `from run_prism_pipeline import X` works
from prism.cli.pipeline import *  # noqa: F401,F403
from prism.cli.pipeline import main

if __name__ == '__main__':
    sys.exit(main())
