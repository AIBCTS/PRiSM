"""Unit tests for prism.xgboost_support.import_xgboost."""

import builtins
import platform

import pytest

from prism.xgboost_support import import_xgboost


@pytest.mark.unit
def test_returns_module_when_import_succeeds():
    xgb = import_xgboost()
    assert hasattr(xgb, "XGBClassifier")


@pytest.mark.unit
def test_macos_libomp_failure_becomes_actionable(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "xgboost":
            raise ImportError("Library not loaded: @rpath/libomp.dylib")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="brew install libomp"):
        import_xgboost()


@pytest.mark.unit
def test_unrelated_import_error_is_reraised(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "xgboost":
            raise ImportError("some other problem")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="some other problem"):
        import_xgboost()
