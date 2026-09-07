"""Single source of truth for the package version.

The build backend reads the version from here, and prism/__init__.py re-exports
it so that prism.__version__ is the version string.
"""

__version__ = "0.1.1"
