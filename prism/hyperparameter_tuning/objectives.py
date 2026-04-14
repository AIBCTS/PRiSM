"""Optuna objective functions for hyperparameter tuning.

Each objective function trains a model with trial-suggested parameters
and returns the evaluation metric for optimization.

PyTorch-based objectives (MLP, LogReg, PRN) support GPU acceleration:
- When device=None, auto-detects and uses CUDA > MPS > CPU
- When device is specified explicitly, uses that device
- For safe parallel GPU trials with n_jobs > 1, use multiprocessing mode
  (auto-enabled in tuning.py when GPU is detected)
"""

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics as sklearn_metrics

from prism.device_tools import _free_all_gpu_caches
from prism.hyperparameter_tuning.search_spaces import get_search_space
from prism.logreg import LogisticRegression

# Import model classes and training functions
from prism.maskedmlp import MaskedMLP, apply_mask, train_model


def _convert_to_tensor(data, device):
    """Convert numpy array, pandas DataFrame/Series, or tensor to torch tensor on specified device."""
    import pandas as pd

    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        return torch.from_numpy(data.values).float().to(device)
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data).float().to(device)
    elif isinstance(data, torch.Tensor):
        return data.float().to(device)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


def _evaluate_predictions(y_true, y_pred, metric='test_auc'):
    """Evaluate predictions and return the specified metric.

    Args:
        y_true: True labels (numpy array or tensor)
        y_pred: Predicted probabilities (numpy array or tensor)
        metric: Metric to compute (default: 'test_auc')

    Returns:
        float: Computed metric value
    """
    # Convert to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    if metric in ['test_auc', 'val_auc', 'auc']:
        return sklearn_metrics.roc_auc_score(y_true, y_pred)
    elif metric == 'accuracy':
        y_pred_binary = (y_pred > 0.5).astype(int)
        return sklearn_metrics.accuracy_score(y_true, y_pred_binary)
    elif metric == 'brier':
        return sklearn_metrics.brier_score_loss(y_true, y_pred)
    else:
        raise ValueError(f"Unsupported metric: {metric}")


class MLPObjective:
    """Objective function for MLP hyperparameter tuning.

    Supports GPU acceleration when available. For safe parallel trials with
    n_jobs > 1, use multiprocessing mode (auto-enabled in tuning.py).
    """

    def __init__(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        tuning_config,
        random_seed,
        device=None,
        n_hidden=None,
        mask=None,
    ):
        """
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            tuning_config: TuningConfig instance
            random_seed: Random seed for reproducibility
            device: Device to use for training ('cuda', 'mps', 'cpu').
                    If None, auto-detects best available device.
            n_hidden: Fixed number of hidden units (if provided, won't tune)
            mask: Optional mask for MaskedMLP
        """
        self.tuning_config = tuning_config
        self.random_seed = random_seed
        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        self.device = device
        self.fixed_n_hidden = n_hidden
        self.mask = mask
        # Cache data tensors on device once (avoid per-trial conversion)
        self._X_train_t = _convert_to_tensor(X_train, self.device)
        self._y_train_t = _convert_to_tensor(y_train, self.device).squeeze()
        self._X_test_t = _convert_to_tensor(X_test, self.device)
        self._y_test_t = _convert_to_tensor(y_test, self.device).squeeze()
        # Track best model during tuning
        self.best_model = None
        self.best_value = float('-inf') if tuning_config.direction == 'maximize' else float('inf')

    def __call__(self, trial: optuna.Trial) -> float:
        """Run a single trial."""
        model = None
        optimizer = None
        criterion = None
        try:
            # Get hyperparameters from search space
            params = get_search_space('mlp', trial)

            # Override n_hidden if fixed
            if self.fixed_n_hidden is not None:
                params['n_hidden'] = self.fixed_n_hidden

            # Set random seed BEFORE model creation for reproducible weight initialization
            if self.random_seed is not None:
                np.random.seed(self.random_seed)
                torch.manual_seed(self.random_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(self.random_seed)

            # Create model (now with deterministic weight initialization)
            input_dim = self._X_train_t.shape[1]
            model = MaskedMLP(
                input_dim=input_dim, hidden_units=params['n_hidden'], output_dim=1, mask=self.mask
            ).to(self.device)

            # Setup training
            criterion = nn.BCELoss()
            optimizer = optim.Adam(
                model.parameters(), lr=params['lr'], weight_decay=params['weight_decay']
            )

            # Train model
            # max_epochs matches train_mlp.py (max_iter: 4000)
            max_epochs = 4000
            tolerance = 1e-4

            model = train_model(
                self._X_train_t,
                self._y_train_t,
                self._X_test_t,
                self._y_test_t,
                model,
                criterion,
                optimizer,
                epochs=max_epochs,
                patience=params['patience'],
                tolerance=tolerance,
                device=self.device,
                plot_loss=False,
                batch_size=params['batch_size'],
                verbose=False,  # Suppress console output during tuning
            )

            # Apply mask if present
            if self.mask is not None:
                apply_mask(model)

            # Evaluate on test set
            model.eval()
            with torch.no_grad():
                y_pred = model.predict(self._X_test_t, device=self.device)

            # Calculate metric
            score = _evaluate_predictions(self._y_test_t, y_pred, self.tuning_config.metric)

            # Track best model
            is_better = (
                (score > self.best_value)
                if self.tuning_config.direction == 'maximize'
                else (score < self.best_value)
            )
            if is_better:
                self.best_value = score
                # Deep copy the model to preserve its state
                import copy

                self.best_model = copy.deepcopy(model)

            return score
        finally:
            del model, optimizer, criterion
            if self.device != 'cpu':
                _free_all_gpu_caches()


class LogRegObjective:
    """Objective function for Logistic Regression hyperparameter tuning.

    Supports GPU acceleration when available. For safe parallel trials with
    n_jobs > 1, use multiprocessing mode (auto-enabled in tuning.py).
    """

    def __init__(self, X_train, y_train, X_test, y_test, tuning_config, random_seed, device=None):
        """
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            tuning_config: TuningConfig instance
            random_seed: Random seed for reproducibility
            device: Device to use for training ('cuda', 'mps', 'cpu').
                    If None, auto-detects best available device.
        """
        self.tuning_config = tuning_config
        self.random_seed = random_seed
        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        self.device = device
        # Cache data tensors on device once (avoid per-trial conversion)
        self._X_train_t = _convert_to_tensor(X_train, self.device)
        y_train_t = _convert_to_tensor(y_train, self.device)
        self._y_train_t = y_train_t.squeeze() if y_train_t.dim() > 1 else y_train_t
        self._X_test_t = _convert_to_tensor(X_test, self.device)
        y_test_t = _convert_to_tensor(y_test, self.device)
        self._y_test_t = y_test_t.squeeze() if y_test_t.dim() > 1 else y_test_t
        # Track best model during tuning
        self.best_model = None
        self.best_value = float('-inf') if tuning_config.direction == 'maximize' else float('inf')

    def __call__(self, trial: optuna.Trial) -> float:
        """Run a single trial."""
        model = None
        optimizer = None
        criterion = None
        try:
            # Get hyperparameters
            params = get_search_space('logreg', trial)

            # Create model
            input_dim = self._X_train_t.shape[1]
            model = LogisticRegression(input_features=input_dim, random_seed=self.random_seed).to(
                self.device
            )

            # Custom criterion that handles squeezed logits
            # LogReg.forward() returns squeezed output, but y_batch in train_model is (n, 1)
            class LogRegCriterion(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.bce_logits = nn.BCEWithLogitsLoss()

                def forward(self, output, target):
                    # output is already squeezed by LogReg.forward()
                    # target has shape (batch, 1) from train_model
                    return self.bce_logits(output, target.squeeze())

            criterion = LogRegCriterion()
            optimizer = optim.Adam(
                model.parameters(), lr=params['lr'], weight_decay=params['weight_decay']
            )

            # Train model
            # max_epochs matches train_logreg.py (epochs: 2000)
            # batch_size=None for full-batch training to match notebook behavior
            max_epochs = 2000
            tolerance = 1e-4

            model = train_model(
                self._X_train_t,
                self._y_train_t,
                self._X_test_t,
                self._y_test_t,
                model,
                criterion,
                optimizer,
                epochs=max_epochs,
                patience=params['patience'],
                tolerance=tolerance,
                device=self.device,
                plot_loss=False,
                batch_size=None,  # Full-batch training to match notebook
                verbose=False,  # Suppress console output during tuning
            )

            # Evaluate on test set
            model.eval()
            with torch.no_grad():
                y_pred = model.predict(self._X_test_t, device=self.device)

            # Calculate metric
            score = _evaluate_predictions(self._y_test_t, y_pred, self.tuning_config.metric)

            # Track best model
            is_better = (
                (score > self.best_value)
                if self.tuning_config.direction == 'maximize'
                else (score < self.best_value)
            )
            if is_better:
                self.best_value = score
                import copy

                self.best_model = copy.deepcopy(model)

            return score
        finally:
            del model, optimizer, criterion
            if self.device != 'cpu':
                _free_all_gpu_caches()


class XGBObjective:
    """Objective function for XGBoost hyperparameter tuning."""

    def __init__(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        tuning_config,
        random_seed,
        device='cpu',
        nthread=None,
    ):
        # Convert to numpy to avoid XGBoost feature name validation issues
        # (XGBoost doesn't accept feature names with [, ], or < characters)
        self.X_train = self._to_numpy(X_train)
        self.y_train = self._to_numpy(y_train)
        self.X_test = self._to_numpy(X_test)
        self.y_test = self._to_numpy(y_test)
        self.tuning_config = tuning_config
        self.random_seed = random_seed
        self.device = device  # PyTorch device for XGBoost GPU acceleration
        self.nthread = nthread  # Cap threads per model to avoid oversubscription
        # Track best model during tuning
        self.best_model = None
        self.best_value = float('-inf') if tuning_config.direction == 'maximize' else float('inf')

    def _to_numpy(self, data):
        """Convert input data to numpy array.

        Handles pandas DataFrames/Series, numpy arrays, and PyTorch tensors
        (including GPU tensors).
        """
        import pandas as pd

        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            return data.values
        elif isinstance(data, np.ndarray):
            return data
        elif isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        else:
            return np.array(data)

    def __call__(self, trial: optuna.Trial) -> float:
        """Run a single trial."""
        import xgboost as xgb

        # Get hyperparameters
        params = get_search_space('xgb', trial)

        # Create XGBoost model
        # Settings match train_xgb.py including early_stopping_rounds
        from prism.device_tools import to_xgb_device

        xgb_device = to_xgb_device(self.device)
        xgb_kwargs = dict(
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            n_estimators=params['n_estimators'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            min_child_weight=params['min_child_weight'],
            gamma=params['gamma'],
            reg_alpha=params['reg_alpha'],
            reg_lambda=params['reg_lambda'],
            random_state=self.random_seed,
            use_label_encoder=False,
            eval_metric='logloss',
            early_stopping_rounds=params['early_stopping_rounds'],
            device=xgb_device,  # GPU acceleration when CUDA available
            verbosity=0,  # Suppress device mismatch warnings with numpy input
        )
        if self.nthread is not None:
            xgb_kwargs['nthread'] = self.nthread
        model = xgb.XGBClassifier(**xgb_kwargs)

        # Train model with eval_set for early stopping
        # Two sets: train (validation_0) + test (validation_1) for loss tracking
        eval_set = [(self.X_train, self.y_train), (self.X_test, self.y_test)]
        model.fit(self.X_train, self.y_train, eval_set=eval_set, verbose=False)

        # Evaluate on test set
        y_pred = model.predict_proba(self.X_test)[:, 1]

        # Calculate metric
        score = _evaluate_predictions(self.y_test, y_pred, self.tuning_config.metric)

        # Track best model
        is_better = (
            (score > self.best_value)
            if self.tuning_config.direction == 'maximize'
            else (score < self.best_value)
        )
        if is_better:
            self.best_value = score
            import copy

            self.best_model = copy.deepcopy(model)
            # Attach training history from evals_result
            evals = model.evals_result()
            best_iter = getattr(model, 'best_iteration', None)
            self.best_model.training_history_ = {
                'train_loss': evals.get('validation_0', {}).get('logloss', []),
                'test_loss': evals.get('validation_1', {}).get('logloss', []),
                'best_epoch': best_iter,
            }

        return score


class RFObjective:
    """Objective function for Random Forest hyperparameter tuning."""

    def __init__(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        tuning_config,
        random_seed,
        device='cpu',
        nthread=None,
    ):
        # Convert to numpy to avoid XGBoost feature name validation issues
        self.X_train = self._to_numpy(X_train)
        self.y_train = self._to_numpy(y_train)
        self.X_test = self._to_numpy(X_test)
        self.y_test = self._to_numpy(y_test)
        self.tuning_config = tuning_config
        self.random_seed = random_seed
        self.device = device  # PyTorch device for XGBoost GPU acceleration
        self.nthread = nthread  # Cap threads per model to avoid oversubscription
        # Track best model during tuning
        self.best_model = None
        self.best_value = float('-inf') if tuning_config.direction == 'maximize' else float('inf')

    def _to_numpy(self, data):
        """Convert input data to numpy array.

        Handles pandas DataFrames/Series, numpy arrays, and PyTorch tensors
        (including GPU tensors).
        """
        import pandas as pd

        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            return data.values
        elif isinstance(data, np.ndarray):
            return data
        elif isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        else:
            return np.array(data)

    def __call__(self, trial: optuna.Trial) -> float:
        """Run a single trial."""
        import xgboost as xgb

        # Get hyperparameters
        params = get_search_space('rf', trial)

        # Create XGBoost Random Forest model
        from prism.device_tools import to_xgb_device

        xgb_device = to_xgb_device(self.device)
        rf_kwargs = dict(
            max_depth=params['max_depth'],
            n_estimators=params['n_estimators'],
            subsample=params['subsample'],
            colsample_bynode=params['colsample_bynode'],
            min_child_weight=params['min_child_weight'],
            random_state=self.random_seed,
            use_label_encoder=False,
            eval_metric='logloss',
            device=xgb_device,  # GPU acceleration when CUDA available
            verbosity=0,  # Suppress device mismatch warnings with numpy input
        )
        if self.nthread is not None:
            rf_kwargs['nthread'] = self.nthread
        model = xgb.XGBRFClassifier(**rf_kwargs)

        # Train model
        model.fit(self.X_train, self.y_train)

        # Evaluate on test set
        y_pred = model.predict_proba(self.X_test)[:, 1]

        # Calculate metric
        score = _evaluate_predictions(self.y_test, y_pred, self.tuning_config.metric)

        # Track best model
        is_better = (
            (score > self.best_value)
            if self.tuning_config.direction == 'maximize'
            else (score < self.best_value)
        )
        if is_better:
            self.best_value = score
            import copy

            self.best_model = copy.deepcopy(model)

        return score


class PRNObjective:
    """Objective function for PRN (Partial Response Network) hyperparameter tuning.

    Supports GPU acceleration when available. For safe parallel trials with
    n_jobs > 1, use multiprocessing mode (auto-enabled in tuning.py).
    """

    def __init__(
        self, X_train, y_train, X_test, y_test, tuning_config, random_seed, device=None, mask=None
    ):
        """
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            tuning_config: TuningConfig instance
            random_seed: Random seed for reproducibility
            device: Device to use for training ('cuda', 'mps', 'cpu').
                    If None, auto-detects best available device.
            mask: Mask for PRN (required)
        """
        self.tuning_config = tuning_config
        self.random_seed = random_seed
        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        self.device = device
        self.mask = mask
        # Cache data tensors on device once (avoid per-trial conversion)
        self._X_train_t = _convert_to_tensor(X_train, self.device)
        self._y_train_t = _convert_to_tensor(y_train, self.device).squeeze()
        self._X_test_t = _convert_to_tensor(X_test, self.device)
        self._y_test_t = _convert_to_tensor(y_test, self.device).squeeze()
        # Track best model during tuning
        self.best_model = None
        self.best_value = float('-inf') if tuning_config.direction == 'maximize' else float('inf')

        if mask is None:
            raise ValueError("PRN tuning requires a mask from LASSO feature selection")

    def __call__(self, trial: optuna.Trial) -> float:
        """Run a single trial."""
        model = None
        optimizer = None
        criterion = None
        try:
            # Get hyperparameters
            params = get_search_space('prn', trial)

            # Create PRN model (MaskedMLP with LASSO mask)
            # The mask was created by LASSO's get_mask() with shape (input_dim, hidden_units)
            # where hidden_units = n_selected_features * subnet_nodes
            # We use the mask shape directly to determine hidden_units
            input_dim = self._X_train_t.shape[1]
            mask_tensor = (
                torch.tensor(self.mask, dtype=torch.float32)
                if not isinstance(self.mask, torch.Tensor)
                else self.mask
            )

            # Determine hidden_units from mask shape
            # Mask has shape (input_dim, hidden_units) or (hidden_units, input_dim)
            if mask_tensor.shape[0] == input_dim:
                hidden_units = mask_tensor.shape[1]
            elif mask_tensor.shape[1] == input_dim:
                hidden_units = mask_tensor.shape[0]
            else:
                raise ValueError(
                    f"Mask shape {mask_tensor.shape} incompatible with input_dim {input_dim}"
                )

            # Set random seed BEFORE model creation for reproducible weight initialization
            if self.random_seed is not None:
                np.random.seed(self.random_seed)
                torch.manual_seed(self.random_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(self.random_seed)

            # Create PRN model (now with deterministic weight initialization)
            model = MaskedMLP(
                input_dim=input_dim, hidden_units=hidden_units, output_dim=1, mask=self.mask
            ).to(self.device)

            # Setup training
            criterion = nn.BCELoss()
            optimizer = optim.Adam(
                model.parameters(), lr=params['lr'], weight_decay=params['weight_decay']
            )

            # Train model
            # max_epochs matches prism_analysis.py (max_iter: 4000)
            max_epochs = 4000
            tolerance = 1e-4

            model = train_model(
                self._X_train_t,
                self._y_train_t,
                self._X_test_t,
                self._y_test_t,
                model,
                criterion,
                optimizer,
                epochs=max_epochs,
                patience=params['patience'],
                tolerance=tolerance,
                device=self.device,
                plot_loss=False,
                batch_size=params['batch_size'],
                verbose=False,  # Suppress console output during tuning
            )

            # Apply mask
            apply_mask(model)

            # Evaluate on test set
            model.eval()
            with torch.no_grad():
                y_pred = model.predict(self._X_test_t, device=self.device)

            # Calculate metric
            score = _evaluate_predictions(self._y_test_t, y_pred, self.tuning_config.metric)

            # Track best model
            is_better = (
                (score > self.best_value)
                if self.tuning_config.direction == 'maximize'
                else (score < self.best_value)
            )
            if is_better:
                self.best_value = score
                import copy

                self.best_model = copy.deepcopy(model)

            return score
        finally:
            del model, optimizer, criterion
            if self.device != 'cpu':
                _free_all_gpu_caches()


# Registry mapping model types to objective classes
OBJECTIVE_REGISTRY = {
    'mlp': MLPObjective,
    'logreg': LogRegObjective,
    'xgb': XGBObjective,
    'rf': RFObjective,
    'prn': PRNObjective,
}


def get_objective(model_type: str, **kwargs):
    """Get objective function for a given model type.

    Args:
        model_type: Model type ('mlp', 'logreg', 'xgb', 'rf', 'prn')
        **kwargs: Arguments to pass to objective constructor

    Returns:
        Objective function instance

    Raises:
        ValueError: If model_type is not supported
    """
    model_type_lower = model_type.lower()
    if model_type_lower not in OBJECTIVE_REGISTRY:
        raise ValueError(
            f"Unsupported model type: {model_type}. "
            f"Supported types: {list(OBJECTIVE_REGISTRY.keys())}"
        )
    return OBJECTIVE_REGISTRY[model_type_lower](**kwargs)
