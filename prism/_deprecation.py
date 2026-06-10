"""Deprecation utilities for the PRiSM public API."""

import warnings


class PrismDeprecationWarning(FutureWarning):
    """Warning category for deprecated PRiSM APIs.

    Derives from FutureWarning so it is shown by default and is not silenced
    by the blanket ``ignore::DeprecationWarning`` filter in the test suite.
    """


def warn_deprecated(old: str, new: str) -> None:
    """Emit a PrismDeprecationWarning pointing callers from `old` to `new`."""
    warnings.warn(
        f"{old} is deprecated and may change behavior in a future release; " f"use {new} instead.",
        PrismDeprecationWarning,
        stacklevel=3,
    )
