#!/usr/bin/env python3
"""Deprecated entry point. Delegates to prism.cli.parallel.

Use `prism run-parallel` instead. This wrapper is kept only so that existing
invocations of `python run_prism_parallel.py` keep working, and will be removed
in a future release.
"""

import sys

# Re-export all public names so that `from run_prism_parallel import X` works
from prism.cli.parallel import *  # noqa: F401,F403
from prism.cli.parallel import main

if __name__ == '__main__':
    sys.exit(main())
