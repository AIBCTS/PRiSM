"""Shared fixtures for PRiSM tests."""

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.datasets import make_classification

# ============================================================================
# Random Seed Management
# ============================================================================


@pytest.fixture(scope="session", autouse=True)
def set_random_seeds():
    """Set random seeds for reproducibility across all tests."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


# ============================================================================
# Mock Models
# ============================================================================


class MockBinaryClassifier:
    """Simple mock classifier for testing.

    Provides deterministic predictions based on a linear combination of features.
    """

    def __init__(self, n_features=5, bias=0.5):
        """Initialize mock classifier.

        Parameters
        ----------
        n_features : int
            Number of input features
        bias : float
            Bias term added to predictions
        """
        self.n_features = n_features
        self.bias = bias
        # Create simple linear weights
        self.weights = np.linspace(0.1, 1.0, n_features)

    def predict_proba(self, x, device=None):
        """Predict probabilities.

        Parameters
        ----------
        x : np.ndarray or torch.Tensor
            Input features, shape (n_samples, n_features)
        device : torch.device, optional
            Device to use for computation

        Returns
        -------
        torch.Tensor
            Predicted probabilities, shape (n_samples,)
        """
        # Convert to tensor if numpy
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        # Move to device if specified
        if device is not None:
            x = x.to(device)

        # Convert weights to tensor
        weights = torch.from_numpy(self.weights).float()
        if device is not None:
            weights = weights.to(device)

        # Linear combination + sigmoid
        logits = torch.matmul(x, weights) + self.bias
        probs = torch.sigmoid(logits)
        return probs

    def __call__(self, x):
        """Make model callable."""
        return self.predict_proba(x)


@pytest.fixture
def mock_model():
    """Fixture providing a simple mock classifier."""
    return MockBinaryClassifier(n_features=5)


@pytest.fixture
def mock_model_10d():
    """Fixture providing a mock classifier with 10 features."""
    return MockBinaryClassifier(n_features=10)


# ============================================================================
# Test Data - Tensors
# ============================================================================


@pytest.fixture
def small_tensor_2d():
    """Small 2D tensor for quick tests."""
    return torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.5, 0.4, 0.3, 0.2, 0.1],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5, 0.5, 0.5],
        ],
        dtype=torch.float32,
    )


@pytest.fixture
def medium_tensor_2d():
    """Medium-sized 2D tensor (100 samples, 10 features)."""
    np.random.seed(42)
    data = np.random.randn(100, 10).astype(np.float32)
    return torch.from_numpy(data)


@pytest.fixture
def binary_target():
    """Binary classification target."""
    return torch.tensor([0, 1, 0, 1, 0], dtype=torch.long)


@pytest.fixture
def binary_target_100():
    """Binary target for 100 samples."""
    np.random.seed(42)
    return torch.from_numpy(np.random.randint(0, 2, size=100).astype(np.int64))


# ============================================================================
# Test Data - DataFrames
# ============================================================================


@pytest.fixture
def sample_dataframe():
    """Sample DataFrame with mixed feature types."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "age": [25, 35, 45, 55, 65, 30, 40, 50, 60, 70],
            "income": [30000, 45000, 60000, 75000, 90000, 35000, 50000, 65000, 80000, 95000],
            "education": ["HS", "BS", "MS", "PhD", "BS", "HS", "MS", "PhD", "BS", "HS"],
            "gender": ["M", "F", "M", "F", "M", "F", "M", "F", "M", "F"],
            "target": [0, 0, 1, 1, 1, 0, 1, 1, 0, 1],
        }
    )


@pytest.fixture
def temporal_dataframe():
    """DataFrame with temporal column for testing temporal splits."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "year": [2010, 2010, 2011, 2011, 2012, 2012, 2013, 2013, 2014, 2014] * 10,
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "target": np.random.randint(0, 2, 100),
        }
    )


@pytest.fixture
def predefined_split_dataframe():
    """DataFrame with predefined split column for testing predefined splits."""
    np.random.seed(42)
    n_train, n_test, n_val = 60, 20, 20
    return pd.DataFrame(
        {
            "feature1": np.random.randn(n_train + n_test + n_val),
            "feature2": np.random.randn(n_train + n_test + n_val),
            "target": np.random.randint(0, 2, n_train + n_test + n_val),
            "split": ["train"] * n_train + ["test"] * n_test + ["val"] * n_val,
        }
    )


@pytest.fixture
def classification_dataset():
    """Larger synthetic classification dataset."""
    np.random.seed(42)
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=7,
        n_redundant=2,
        n_classes=2,
        class_sep=1.0,
        random_state=42,
    )

    # Create DataFrame
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    return df


# ============================================================================
# One-Hot Encoding Groups
# ============================================================================


@pytest.fixture
def valid_onehot_groups():
    """Valid one-hot encoding group configurations."""
    return [
        [0, 1, 2],  # First group: features 0, 1, 2
        [3, 4],  # Second group: features 3, 4
    ]


@pytest.fixture
def invalid_onehot_groups_overlap():
    """Invalid one-hot groups with overlap."""
    return [
        [0, 1, 2],
        [2, 3, 4],  # Feature 2 appears in both groups
    ]


# ============================================================================
# Mock Scalers
# ============================================================================


class MockScaler:
    """Simple mock scaler for testing."""

    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def transform(self, x):
        """Standardize input."""
        return (x - self.mean) / self.std

    def inverse_transform(self, x):
        """Reverse standardization."""
        return x * self.std + self.mean

    def fit(self, x):
        """Fit scaler (no-op for mock)."""
        return self

    def fit_transform(self, x):
        """Fit and transform."""
        self.fit(x)
        return self.transform(x)


@pytest.fixture
def mock_scaler():
    """Fixture providing a simple mock scaler."""
    return MockScaler(mean=0.5, std=0.2)


# ============================================================================
# Mock LASSO Results
# ============================================================================


@pytest.fixture
def mock_partial_responses():
    """Mock partial response dictionary for LASSO testing."""
    np.random.seed(42)

    # Create mock partial responses
    # Shape: (n_samples, n_points_per_feature)
    n_samples = 100
    n_features = 10

    partial_responses = {}

    # Univariate responses
    for i in range(n_features):
        # Each feature has varying number of evaluation points
        n_points = np.random.randint(10, 20)
        partial_responses[f"f{i}"] = {
            "x_values": np.linspace(0, 1, n_points),
            "responses": np.random.randn(n_samples, n_points),
        }

    # Add a few bivariate interactions
    partial_responses["f0_f1"] = {
        "x_values": (np.linspace(0, 1, 15), np.linspace(0, 1, 15)),
        "responses": np.random.randn(n_samples, 15, 15),
    }

    return partial_responses


# ============================================================================
# Pytest Markers
# ============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests (fast, isolated)")
    config.addinivalue_line("markers", "integration: Integration tests (slower)")
    config.addinivalue_line("markers", "slow: Slow tests (can be skipped)")


# ============================================================================
# Parametrized Fixtures
# ============================================================================


@pytest.fixture(params=[0.6, 0.7, 0.8])
def train_ratio(request):
    """Different training set ratios."""
    return request.param


@pytest.fixture(params=[5, 10, 12])
def categorical_threshold(request):
    """Different categorical detection thresholds."""
    return request.param


@pytest.fixture(params=[0.001, 0.01, 0.1, 1.0, 10.0])
def lambda_value(request):
    """Different LASSO regularization values."""
    return request.param


# ============================================================================
# Models for Mathematical Validation (Added for Audit)
# ============================================================================


class LinearTestModel(torch.nn.Module):
    """Purely linear model with known coefficients for ground truth testing.

    This model has NO interactions and can be used to verify that bivariate
    partial responses are zero and univariate responses match the coefficients.
    """

    def __init__(self, coefficients, bias=0.0):
        """Initialize linear model.

        Parameters
        ----------
        coefficients : list of float
            Linear coefficients for each feature
        bias : float
            Intercept term
        """
        super().__init__()
        self.coefficients = torch.nn.Parameter(
            torch.tensor(coefficients, dtype=torch.float32), requires_grad=False
        )
        self.bias = torch.nn.Parameter(
            torch.tensor(bias, dtype=torch.float32), requires_grad=False
        )

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (n_samples, n_features)

        Returns
        -------
        torch.Tensor
            Probabilities, shape (n_samples,)
        """
        # Logits = x @ coefficients + bias
        logits = torch.matmul(x, self.coefficients) + self.bias
        # Return probabilities
        return torch.sigmoid(logits)

    def predict_proba(self, x, device=None):
        """Predict probabilities (wrapper for forward).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        device : torch.device, optional
            Device to use

        Returns
        -------
        torch.Tensor
            Probabilities
        """
        if device is not None:
            x = x.to(device)
            self.to(device)
        return self.forward(x)


class InteractionTestModel(torch.nn.Module):
    """Model with known interaction terms for ground truth testing.

    This model has explicit x1*x2 interaction to verify that bivariate
    partial responses correctly capture interactions.
    """

    def __init__(self, linear_coeffs, interaction_pairs, interaction_coeffs, bias=0.0):
        """Initialize interaction model.

        Parameters
        ----------
        linear_coeffs : list of float
            Linear coefficients for each feature
        interaction_pairs : list of tuple
            List of (i, j) pairs for interactions
        interaction_coeffs : list of float
            Coefficients for each interaction pair
        bias : float
            Intercept term
        """
        super().__init__()
        self.linear_coeffs = torch.nn.Parameter(
            torch.tensor(linear_coeffs, dtype=torch.float32), requires_grad=False
        )
        self.interaction_pairs = interaction_pairs
        self.interaction_coeffs = torch.nn.Parameter(
            torch.tensor(interaction_coeffs, dtype=torch.float32), requires_grad=False
        )
        self.bias = torch.nn.Parameter(
            torch.tensor(bias, dtype=torch.float32), requires_grad=False
        )

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (n_samples, n_features)

        Returns
        -------
        torch.Tensor
            Probabilities, shape (n_samples,)
        """
        # Linear terms
        logits = torch.matmul(x, self.linear_coeffs) + self.bias

        # Add interaction terms
        for (i, j), coeff in zip(self.interaction_pairs, self.interaction_coeffs):
            logits += coeff * x[:, i] * x[:, j]

        return torch.sigmoid(logits)

    def predict_proba(self, x, device=None):
        """Predict probabilities (wrapper for forward).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        device : torch.device, optional
            Device to use

        Returns
        -------
        torch.Tensor
            Probabilities
        """
        if device is not None:
            x = x.to(device)
            self.to(device)
        return self.forward(x)


class TestMLP(torch.nn.Module):
    """Small MLP for general testing.

    Used for reconstruction tests and other non-ground-truth validation.
    """

    def __init__(self, n_features=5, hidden_size=10):
        """Initialize MLP.

        Parameters
        ----------
        n_features : int
            Number of input features
        hidden_size : int
            Number of hidden units
        """
        super().__init__()
        self.fc1 = torch.nn.Linear(n_features, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (n_samples, n_features)

        Returns
        -------
        torch.Tensor
            Probabilities, shape (n_samples,)
        """
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x).squeeze()

    def predict_proba(self, x, device=None):
        """Predict probabilities (wrapper for forward).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        device : torch.device, optional
            Device to use

        Returns
        -------
        torch.Tensor
            Probabilities
        """
        if device is not None:
            x = x.to(device)
            self.to(device)
        return self.forward(x)


@pytest.fixture
def linear_test_model():
    """Linear model with known coefficients: [0.3, 0.5, -0.2, 0.1, 0.4]."""
    return LinearTestModel([0.3, 0.5, -0.2, 0.1, 0.4], bias=0.0)


@pytest.fixture
def interaction_test_model():
    """Model with linear terms and x0*x1 interaction.

    Linear: [1.0, 1.0, 0.0, 0.0, 0.0]
    Interaction: 0.5 * x0 * x1
    """
    return InteractionTestModel(
        linear_coeffs=[1.0, 1.0, 0.0, 0.0, 0.0],
        interaction_pairs=[(0, 1)],
        interaction_coeffs=[0.5],
        bias=0.0,
    )


@pytest.fixture
def test_mlp():
    """Small MLP for general testing."""
    torch.manual_seed(42)  # For reproducibility
    return TestMLP(n_features=5, hidden_size=10)


@pytest.fixture
def test_mlp_10d():
    """Small MLP with 10 features."""
    torch.manual_seed(42)
    return TestMLP(n_features=10, hidden_size=20)
