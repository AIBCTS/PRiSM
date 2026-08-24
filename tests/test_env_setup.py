"""Unit tests for prism._env_setup.configure_openmp."""

import pytest

from prism._env_setup import configure_openmp


@pytest.mark.unit
def test_sets_single_thread_default_on_macos():
    env = {}
    configure_openmp("Darwin", env)
    assert env["OMP_NUM_THREADS"] == "1"


@pytest.mark.unit
def test_preserves_explicit_value_on_macos():
    env = {"OMP_NUM_THREADS": "8"}
    configure_openmp("Darwin", env)
    assert env["OMP_NUM_THREADS"] == "8"


@pytest.mark.unit
@pytest.mark.parametrize("system", ["Linux", "Windows"])
def test_noop_off_macos(system):
    env = {}
    configure_openmp(system, env)
    assert "OMP_NUM_THREADS" not in env
