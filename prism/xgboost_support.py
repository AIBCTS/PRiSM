"""Import XGBoost with an actionable error when its OpenMP runtime is missing."""

import platform


def import_xgboost():
    """Import and return the ``xgboost`` module.

    On macOS, XGBoost loads an external OpenMP runtime (``libomp``) when the
    module is imported; if it is absent the raw loader error is cryptic. Convert
    that specific failure into an actionable message. Any other import error is
    re-raised unchanged.

    Returns
    -------
    module
        The imported ``xgboost`` module.
    """
    try:
        import xgboost as xgb
    except Exception as exc:
        message = str(exc)
        if platform.system() == "Darwin" and ("libomp" in message or "libxgboost" in message):
            raise ImportError(
                "XGBoost could not load its OpenMP runtime (libomp) on macOS. "
                "Install it with `brew install libomp` (see docs/SETUP_MACOS.md), then retry."
            ) from exc
        raise
    return xgb
