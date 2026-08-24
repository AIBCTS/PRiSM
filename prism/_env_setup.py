"""Process-environment configuration applied at package import time.

Kept dependency-free (standard library only) so it can run as the very first
thing in ``prism/__init__`` (before anything imports torch or xgboost) and be
unit-tested in isolation.
"""


def configure_openmp(system, environ):
    """Serialize OpenMP on macOS to avoid a torch/xgboost libomp conflict.

    On macOS, PyTorch and XGBoost each load their own copy of the OpenMP
    runtime (``libomp``). With both present in one process, XGBoost's parallel
    initialization collides with the second runtime: a deadlock on older
    toolchains, a hard crash (``OMP: Error #179`` -> segfault) on current ones.
    Forcing OpenMP to a single thread avoids the collision.

    Must be called before torch or xgboost is imported. ``setdefault`` is used
    so an explicit user-provided ``OMP_NUM_THREADS`` is left untouched.

    Parameters
    ----------
    system : str
        Platform identifier, e.g. the result of ``platform.system()``.
    environ : MutableMapping
        The process environment to mutate, e.g. ``os.environ``.
    """
    if system == "Darwin":
        environ.setdefault("OMP_NUM_THREADS", "1")
