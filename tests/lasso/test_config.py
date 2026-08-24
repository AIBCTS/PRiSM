"""Tests for LASSO configuration module."""

import pytest

from prism.lasso.config import LassoConfig


class TestLassoConfig:
    """Tests for LassoConfig dataclass."""

    def test_default_initialization(self):
        """Test that LassoConfig initializes with default values."""
        config = LassoConfig()

        assert config.nlambda == 100
        assert config.max_iter == 10000
        assert config.min_lambda == 0.001
        assert config.max_lambda == 1000
        assert config.regression_tol == 1e-4
        assert config.batch_size == 10
        assert config.max_workers == -1
        assert config.real_time_plot is True
        assert config.seed == 257
        assert config.max_features is None
        assert config.feature_threshold == 0.1
        assert config.base_model_name is None

    def test_custom_initialization(self):
        """Test initialization with custom values."""
        config = LassoConfig(
            nlambda=50,
            max_iter=5000,
            min_lambda=0.01,
            max_lambda=100,
            regression_tol=1e-5,
            batch_size=20,
            max_workers=4,
            real_time_plot=False,
            seed=42,
            max_features=10,
            feature_threshold=0.05,
            base_model_name="test_model",
        )

        assert config.nlambda == 50
        assert config.max_iter == 5000
        assert config.min_lambda == 0.01
        assert config.max_lambda == 100
        assert config.regression_tol == 1e-5
        assert config.batch_size == 20
        assert config.max_workers == 4
        assert config.real_time_plot is False
        assert config.seed == 42
        assert config.max_features == 10
        assert config.feature_threshold == 0.05
        assert config.base_model_name == "test_model"

    def test_validate_default_config(self):
        """Test that default configuration passes validation."""
        config = LassoConfig()
        # Should not raise any errors
        config.validate()

    def test_validate_custom_valid_config(self):
        """Test validation with valid custom configuration."""
        config = LassoConfig(nlambda=50, min_lambda=0.01, max_lambda=10.0, max_features=15)
        # Should not raise any errors
        config.validate()

    # ========================================================================
    # Validation Error Tests
    # ========================================================================

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"nlambda": -1}, "nlambda must be positive"),
            ({"nlambda": 0}, "nlambda must be positive"),
            ({"max_iter": -100}, "max_iter must be positive"),
            ({"max_iter": 0}, "max_iter must be positive"),
            ({"min_lambda": -0.01}, "min_lambda must be positive"),
            ({"min_lambda": 0.0}, "min_lambda must be positive"),
            (
                {"min_lambda": 10.0, "max_lambda": 5.0},
                "max_lambda must be greater than min_lambda",
            ),
            ({"min_lambda": 1.0, "max_lambda": 1.0}, "max_lambda must be greater than min_lambda"),
            ({"batch_size": -5}, "batch_size must be positive"),
            ({"batch_size": 0}, "batch_size must be positive"),
            ({"regression_tol": -1e-4}, "regression_tol must be positive"),
            ({"regression_tol": 0.0}, "regression_tol must be positive"),
            ({"feature_threshold": -0.1}, "feature_threshold must be positive"),
            ({"feature_threshold": 0.0}, "feature_threshold must be positive"),
            ({"max_features": -10}, "max_features must be positive if specified"),
            ({"max_features": 0}, "max_features must be positive if specified"),
        ],
    )
    def test_validate_rejects_out_of_range_values(self, kwargs, message):
        """Zero, negative, and inverted-range values are rejected with a clear message."""
        config = LassoConfig(**kwargs)
        with pytest.raises(ValueError, match=message):
            config.validate()

    def test_validate_none_max_features(self):
        """Test that None max_features is valid (no limit)."""
        config = LassoConfig(max_features=None)
        # Should not raise errors
        config.validate()

    # ========================================================================
    # Edge Case Tests
    # ========================================================================

    def test_very_small_positive_values(self):
        """Test configuration with very small but valid positive values."""
        config = LassoConfig(
            min_lambda=1e-10,
            max_lambda=1e-9,
            regression_tol=1e-12,
            feature_threshold=1e-6,
        )
        # Should not raise errors
        config.validate()

    def test_very_large_values(self):
        """Test configuration with very large values."""
        config = LassoConfig(
            nlambda=10000,
            max_iter=1000000,
            max_lambda=1e6,
            batch_size=1000,
            max_features=10000,
        )
        # Should not raise errors
        config.validate()

    def test_minimal_valid_configuration(self):
        """Test configuration with minimal valid values."""
        config = LassoConfig(
            nlambda=1,
            max_iter=1,
            min_lambda=0.000001,
            max_lambda=0.000002,
            batch_size=1,
            regression_tol=1e-10,
            feature_threshold=1e-10,
            max_features=1,
        )
        # Should not raise errors
        config.validate()
