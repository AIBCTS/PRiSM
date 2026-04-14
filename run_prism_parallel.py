#!/usr/bin/env python3
"""Backward-compatible entry point. Delegates to prism.cli.parallel."""
# Re-export all public names so that `from run_prism_parallel import X` works
from prism.cli.parallel import *  # noqa: F401,F403
from prism.cli.parallel import main

import sys

if __name__ == '__main__':
    sys.exit(main())
