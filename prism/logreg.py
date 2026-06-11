import torch
import torch.nn as nn


# Logistic Regression class using PyTorch
class LogisticRegression(nn.Module):
    def __init__(self, input_features, random_seed=None):
        super(LogisticRegression, self).__init__()
        # Set random seed for reproducibility if provided
        if random_seed is not None:
            torch.manual_seed(random_seed)

        # Linear layer represents logistic regression
        self.linear = nn.Linear(input_features, 1)

        # Initialize weights similar to sklearn's default
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        # Apply linear layer (returns logits)
        logits = self.linear(x)
        return logits.squeeze()  # No sigmoid here for training with BCEWithLogitsLoss

    def predict_proba(self, X, device=None):
        """Probability of the positive class P(y=1) as a torch tensor."""
        # Handle device parameter
        if device is None:
            device = next(self.parameters()).device

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32, device=device)
        elif X.device != device:
            X = X.to(device)

        # Set to evaluation mode
        self.eval()
        with torch.no_grad():
            # Return probabilities (0-1) by applying sigmoid to the logits
            return torch.sigmoid(self.forward(X))

    def predict(self, X, device=None, threshold=0.5):
        """
        Binary class labels: (predict_proba(X, device) >= threshold) as a long tensor.

        Use predict_proba for the underlying probabilities.
        """
        return (self.predict_proba(X, device) >= threshold).long()

    def get_logits(self, X, device=None):
        """Helper method to get raw logits when needed"""
        if device is None:
            device = next(self.parameters()).device

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32, device=device)
        elif X.device != device:
            X = X.to(device)

        self.eval()
        with torch.no_grad():
            return self.forward(X)

    # Note: We intentionally do NOT override __call__ here.
    # PyTorch's nn.Module.__call__ handles forward() with proper hooks.
    # Overriding __call__ to call predict() would break gradient computation
    # during training because predict() uses torch.no_grad().
