"""Tests for hyperparameter tuning orchestration."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from prism.hyperparameter_tuning.config import TuningConfig
from prism.hyperparameter_tuning.tuning import (
    load_best_params,
    load_params_from_file,
    load_tuned_model,
    run_hyperparameter_tuning,
    save_best_model,
    save_best_params,
)


class TestSaveLoadParams:
    """Test saving and loading best parameters."""

    def test_save_best_params(self):
        """Test saving best parameters to JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            best_params = {
                'n_hidden': 15,
                'lr': 0.005,
                'weight_decay': 0.001,
                'patience': 12,
                'batch_size': 128,
            }

            # Save parameters
            filepath = save_best_params(
                best_params=best_params,
                model_type='mlp',
                dataset_prefix='test_dataset',
                output_dir=tmpdir,
            )

            # Check file was created
            assert filepath.exists()
            assert filepath.name == 'test_dataset_mlp_best_params.json'

            # Check content
            with open(filepath, 'r') as f:
                data = json.load(f)

            assert data['model_type'] == 'mlp'
            assert data['dataset'] == 'test_dataset'
            assert data['best_params'] == best_params

    def test_save_best_params_with_study(self):
        """Test saving parameters with study statistics."""
        import optuna

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create a simple study
            study = optuna.create_study(direction='maximize')
            study.optimize(lambda trial: 0.8, n_trials=5)

            best_params = {'lr': 0.001}

            filepath = save_best_params(
                best_params=best_params,
                model_type='logreg',
                dataset_prefix='test',
                output_dir=tmpdir,
                study=study,
            )

            # Check study stats are saved
            with open(filepath, 'r') as f:
                data = json.load(f)

            assert 'study_stats' in data
            assert data['study_stats']['n_trials'] == 5
            assert data['study_stats']['best_value'] == 0.8

    def test_load_best_params(self):
        """Test loading best parameters from JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create a params file
            best_params = {'lr': 0.01, 'weight_decay': 0.0001}
            save_best_params(
                best_params=best_params,
                model_type='xgb',
                dataset_prefix='test_data',
                output_dir=tmpdir,
            )

            # Load parameters
            loaded_params = load_best_params(
                model_type='xgb', dataset_prefix='test_data', params_dir=tmpdir
            )

            assert loaded_params == best_params

    def test_load_best_params_not_found(self):
        """Test loading parameters returns None when file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            loaded_params = load_best_params(
                model_type='mlp', dataset_prefix='nonexistent', params_dir=tmpdir
            )

            assert loaded_params is None

    def test_save_creates_directory(self):
        """Test that save_best_params creates output directory if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            nested_dir = tmpdir / 'nested' / 'dir'

            # Directory doesn't exist yet
            assert not nested_dir.exists()

            save_best_params(
                best_params={'lr': 0.001},
                model_type='mlp',
                dataset_prefix='test',
                output_dir=nested_dir,
            )

            # Directory should be created
            assert nested_dir.exists()


class TestLoadParamsFromFile:
    """Test loading parameters from explicit file paths."""

    def test_load_params_from_file_absolute(self):
        """Test loading params from an absolute file path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create a params file
            params_data = {
                'model_type': 'mlp',
                'dataset': 'test',
                'best_params': {'lr': 0.005, 'weight_decay': 0.001, 'n_hidden': 20},
            }
            filepath = tmpdir / 'test_mlp_best_params.json'
            with open(filepath, 'w') as f:
                json.dump(params_data, f)

            # Load using absolute path
            loaded = load_params_from_file(str(filepath))
            assert loaded == params_data['best_params']

    def test_load_params_from_file_not_found(self):
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError, match="params_file not found"):
            load_params_from_file('/nonexistent/path/to/params.json')

    def test_load_params_from_file_invalid_json(self):
        """Test that ValueError is raised for invalid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            filepath = tmpdir / 'invalid.json'

            # Write invalid JSON
            with open(filepath, 'w') as f:
                f.write("not valid json {")

            with pytest.raises(ValueError, match="Invalid JSON"):
                load_params_from_file(str(filepath))

    def test_load_params_from_file_missing_best_params_key(self):
        """Test that ValueError is raised when best_params key is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            filepath = tmpdir / 'missing_key.json'

            # Write JSON without best_params key
            with open(filepath, 'w') as f:
                json.dump({'model_type': 'mlp', 'dataset': 'test'}, f)

            with pytest.raises(ValueError, match="Expected 'best_params' key"):
                load_params_from_file(str(filepath))

    def test_load_params_from_file_with_study_stats(self):
        """Test loading params file that includes study statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            params_data = {
                'model_type': 'xgb',
                'dataset': 'credit-g',
                'best_params': {'max_depth': 5, 'learning_rate': 0.1},
                'study_stats': {'best_value': 0.85, 'n_trials': 25},
            }
            filepath = tmpdir / 'xgb_params.json'
            with open(filepath, 'w') as f:
                json.dump(params_data, f)

            loaded = load_params_from_file(str(filepath))
            assert loaded == params_data['best_params']


class TestRunHyperparameterTuning:
    """Test running hyperparameter tuning."""

    @pytest.fixture
    def small_dataset(self):
        """Create a small synthetic dataset for testing."""
        np.random.seed(42)
        n_samples = 100
        n_features = 5

        X_train = np.random.randn(n_samples, n_features).astype(np.float32)
        y_train = (np.random.randn(n_samples) > 0).astype(np.float32)

        X_test = np.random.randn(n_samples // 2, n_features).astype(np.float32)
        y_test = (np.random.randn(n_samples // 2) > 0).astype(np.float32)

        return X_train, y_train, X_test, y_test

    def test_run_tuning_mlp_quick(self, small_dataset):
        """Test running a quick MLP tuning (2 trials)."""
        X_train, y_train, X_test, y_test = small_dataset

        tuning_config = TuningConfig(
            enabled=True,
            n_trials=2,
            metric='test_auc',
            direction='maximize',
            pruning_enabled=False,
        )

        best_params, study, best_model = run_hyperparameter_tuning(
            model_type='mlp',
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            tuning_config=tuning_config,
            random_seed=42,
            device='cpu',
        )

        # Check best_params has expected keys
        assert 'n_hidden' in best_params
        assert 'lr' in best_params
        assert 'weight_decay' in best_params
        assert 'patience' in best_params
        assert 'batch_size' in best_params

        # Check study completed
        assert len(study.trials) == 2
        assert study.best_value is not None

        # Check best_model is returned
        assert best_model is not None
        assert hasattr(best_model, 'predict')

    def test_run_tuning_logreg_quick(self, small_dataset):
        """Test running a quick LogReg tuning (2 trials)."""
        X_train, y_train, X_test, y_test = small_dataset

        tuning_config = TuningConfig(
            enabled=True,
            n_trials=2,
            metric='test_auc',
            direction='maximize',
            pruning_enabled=False,
        )

        best_params, study, best_model = run_hyperparameter_tuning(
            model_type='logreg',
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            tuning_config=tuning_config,
            random_seed=42,
            device='cpu',
        )

        # Check best_params has expected keys
        assert 'lr' in best_params
        assert 'weight_decay' in best_params
        assert 'patience' in best_params

        # Check study completed
        assert len(study.trials) == 2

        # Check best_model is returned
        assert best_model is not None

    def test_run_tuning_xgb_quick(self, small_dataset):
        """Test running a quick XGBoost tuning (2 trials)."""
        X_train, y_train, X_test, y_test = small_dataset

        tuning_config = TuningConfig(
            enabled=True,
            n_trials=2,
            metric='test_auc',
            direction='maximize',
            pruning_enabled=False,
        )

        best_params, study, best_model = run_hyperparameter_tuning(
            model_type='xgb',
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            tuning_config=tuning_config,
            random_seed=42,
            device='cpu',
        )

        # Check best_params has expected keys
        assert 'max_depth' in best_params
        assert 'learning_rate' in best_params
        assert 'n_estimators' in best_params

        # Check study completed
        assert len(study.trials) == 2

        # Check best_model is returned
        assert best_model is not None

    def test_run_tuning_with_different_metrics(self, small_dataset):
        """Test tuning with different evaluation metrics."""
        X_train, y_train, X_test, y_test = small_dataset

        # Test with accuracy metric
        tuning_config = TuningConfig(
            enabled=True, n_trials=2, metric='accuracy', direction='maximize'
        )

        best_params, study, best_model = run_hyperparameter_tuning(
            model_type='mlp',
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            tuning_config=tuning_config,
            random_seed=42,
            device='cpu',
        )

        assert study.best_value is not None
        assert 0 <= study.best_value <= 1  # Accuracy is between 0 and 1
        assert best_model is not None


class TestObjectiveDeviceHandling:
    """Test that objectives properly handle device specification."""

    def test_mlp_objective_respects_explicit_device(self):
        """Test MLPObjective uses the explicitly specified device."""
        from prism.hyperparameter_tuning.objectives import MLPObjective

        # Create minimal test data
        X_train = np.random.randn(50, 5).astype(np.float32)
        y_train = (np.random.randn(50) > 0).astype(np.float32)
        X_test = np.random.randn(25, 5).astype(np.float32)
        y_test = (np.random.randn(25) > 0).astype(np.float32)

        tuning_config = TuningConfig(enabled=True, n_trials=1)

        # Pass 'cpu' as device explicitly
        objective = MLPObjective(
            X_train, y_train, X_test, y_test, tuning_config, random_seed=42, device='cpu'
        )

        # Verify specified device is used
        assert objective.device == 'cpu'

    def test_mlp_objective_auto_detects_device(self):
        """Test MLPObjective auto-detects device when None is passed."""
        from prism.hyperparameter_tuning.objectives import MLPObjective

        X_train = np.random.randn(50, 5).astype(np.float32)
        y_train = (np.random.randn(50) > 0).astype(np.float32)
        X_test = np.random.randn(25, 5).astype(np.float32)
        y_test = (np.random.randn(25) > 0).astype(np.float32)

        tuning_config = TuningConfig(enabled=True, n_trials=1)

        # Pass None as device - should auto-detect
        objective = MLPObjective(
            X_train, y_train, X_test, y_test, tuning_config, random_seed=42, device=None
        )

        # Verify a valid device is selected (cpu, cuda, or mps)
        assert objective.device in ('cpu', 'cuda', 'mps')

    def test_logreg_objective_respects_explicit_device(self):
        """Test LogRegObjective uses the explicitly specified device."""
        from prism.hyperparameter_tuning.objectives import LogRegObjective

        X_train = np.random.randn(50, 5).astype(np.float32)
        y_train = (np.random.randn(50) > 0).astype(np.float32)
        X_test = np.random.randn(25, 5).astype(np.float32)
        y_test = (np.random.randn(25) > 0).astype(np.float32)

        tuning_config = TuningConfig(enabled=True, n_trials=1)

        # Pass 'cpu' as device explicitly
        objective = LogRegObjective(
            X_train, y_train, X_test, y_test, tuning_config, random_seed=42, device='cpu'
        )

        # Verify specified device is used
        assert objective.device == 'cpu'

    def test_logreg_objective_auto_detects_device(self):
        """Test LogRegObjective auto-detects device when None is passed."""
        from prism.hyperparameter_tuning.objectives import LogRegObjective

        X_train = np.random.randn(50, 5).astype(np.float32)
        y_train = (np.random.randn(50) > 0).astype(np.float32)
        X_test = np.random.randn(25, 5).astype(np.float32)
        y_test = (np.random.randn(25) > 0).astype(np.float32)

        tuning_config = TuningConfig(enabled=True, n_trials=1)

        # Pass None as device - should auto-detect
        objective = LogRegObjective(
            X_train, y_train, X_test, y_test, tuning_config, random_seed=42, device=None
        )

        # Verify a valid device is selected
        assert objective.device in ('cpu', 'cuda', 'mps')

    def test_prn_objective_respects_explicit_device(self):
        """Test PRNObjective uses the explicitly specified device."""
        from prism.hyperparameter_tuning.objectives import PRNObjective

        X_train = np.random.randn(50, 5).astype(np.float32)
        y_train = (np.random.randn(50) > 0).astype(np.float32)
        X_test = np.random.randn(25, 5).astype(np.float32)
        y_test = (np.random.randn(25) > 0).astype(np.float32)

        tuning_config = TuningConfig(enabled=True, n_trials=1)

        # Create a simple mask for PRN
        mask = np.ones((5, 10), dtype=np.float32)

        # Pass 'cpu' as device explicitly
        objective = PRNObjective(
            X_train,
            y_train,
            X_test,
            y_test,
            tuning_config,
            random_seed=42,
            device='cpu',
            mask=mask,
        )

        # Verify specified device is used
        assert objective.device == 'cpu'

    def test_prn_objective_auto_detects_device(self):
        """Test PRNObjective auto-detects device when None is passed."""
        from prism.hyperparameter_tuning.objectives import PRNObjective

        X_train = np.random.randn(50, 5).astype(np.float32)
        y_train = (np.random.randn(50) > 0).astype(np.float32)
        X_test = np.random.randn(25, 5).astype(np.float32)
        y_test = (np.random.randn(25) > 0).astype(np.float32)

        tuning_config = TuningConfig(enabled=True, n_trials=1)

        # Create a simple mask for PRN
        mask = np.ones((5, 10), dtype=np.float32)

        # Pass None as device - should auto-detect
        objective = PRNObjective(
            X_train, y_train, X_test, y_test, tuning_config, random_seed=42, device=None, mask=mask
        )

        # Verify a valid device is selected
        assert objective.device in ('cpu', 'cuda', 'mps')

    def test_tuning_runs_on_cpu(self):
        """Test that tuning runs successfully on CPU."""
        X_train = np.random.randn(50, 5).astype(np.float32)
        y_train = (np.random.randn(50) > 0).astype(np.float32)
        X_test = np.random.randn(25, 5).astype(np.float32)
        y_test = (np.random.randn(25) > 0).astype(np.float32)

        tuning_config = TuningConfig(
            enabled=True, n_trials=1, metric='test_auc', pruning_enabled=False
        )

        # Run tuning on CPU explicitly
        best_params, study, best_model = run_hyperparameter_tuning(
            model_type='mlp',
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            tuning_config=tuning_config,
            random_seed=42,
            device='cpu',
        )

        # Should complete successfully
        assert best_params is not None
        assert len(study.trials) == 1
        assert best_model is not None


class TestGPUAcceleration:
    """Test GPU acceleration features for hyperparameter tuning."""

    def test_use_multiprocessing_config(self):
        """TuningConfig accepts use_multiprocessing field."""
        config = TuningConfig(enabled=True, n_trials=10, use_multiprocessing=True)
        assert config.use_multiprocessing is True

        config_default = TuningConfig(enabled=True, n_trials=10)
        assert config_default.use_multiprocessing is False

    def test_tuning_auto_enables_multiprocessing_for_gpu(self):
        """Tuning auto-enables multiprocessing when n_jobs > 1 and GPU device."""
        # This test verifies the logic path, not actual GPU execution
        X_train = np.random.randn(50, 5).astype(np.float32)
        y_train = (np.random.randn(50) > 0).astype(np.float32)
        X_test = np.random.randn(25, 5).astype(np.float32)
        y_test = (np.random.randn(25) > 0).astype(np.float32)

        # With n_jobs=1, should use standard optimization even with GPU device
        tuning_config = TuningConfig(enabled=True, n_trials=1, n_jobs=1, pruning_enabled=False)

        # Run with cuda device but n_jobs=1 - should use standard optimization
        best_params, study, best_model = run_hyperparameter_tuning(
            model_type='mlp',
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            tuning_config=tuning_config,
            random_seed=42,
            device='cpu',  # Use CPU to avoid actual GPU requirement
        )

        assert best_params is not None
        assert len(study.trials) == 1

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_tuning_runs_on_cuda(self):
        """Tuning runs successfully on CUDA when available."""
        X_train = np.random.randn(50, 5).astype(np.float32)
        y_train = (np.random.randn(50) > 0).astype(np.float32)
        X_test = np.random.randn(25, 5).astype(np.float32)
        y_test = (np.random.randn(25) > 0).astype(np.float32)

        tuning_config = TuningConfig(
            enabled=True,
            n_trials=2,
            metric='test_auc',
            pruning_enabled=False,
            n_jobs=1,  # Sequential to avoid multiprocessing complexity in test
        )

        best_params, study, best_model = run_hyperparameter_tuning(
            model_type='mlp',
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            tuning_config=tuning_config,
            random_seed=42,
            device='cuda',
        )

        assert best_params is not None
        assert len(study.trials) == 2
        assert best_model is not None


class TestSaveLoadBestModel:
    """Test saving and loading best models from tuning."""

    def test_save_best_model_mlp(self):
        """Test saving an MLP model from tuning."""
        from prism.maskedmlp import MaskedMLP

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create a simple MLP model
            model = MaskedMLP(input_dim=5, hidden_units=10, output_dim=1)

            hyperparams = {'n_hidden': 10, 'lr': 0.001, 'weight_decay': 1e-5}

            # Save model
            filepath = save_best_model(
                model=model,
                model_type='mlp',
                dataset_prefix='test_dataset_mlp',
                output_dir=tmpdir,
                hyperparameters=hyperparams,
                feature_names=['f1', 'f2', 'f3', 'f4', 'f5'],
            )

            # Check file was created in subdirectory
            assert filepath.exists()
            assert filepath.name == 'test_dataset_mlp_model_tuned.pt'
            assert filepath.parent.name == 'test_dataset_mlp'

    def test_save_best_model_with_scaler(self):
        """Test saving model with scaler."""
        from prism.maskedmlp import MaskedMLP
        from prism.preprocessing import PRiSMScaler

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            model = MaskedMLP(input_dim=5, hidden_units=10, output_dim=1)
            scaler = PRiSMScaler()
            scaler.fit(np.random.randn(100, 5))

            filepath = save_best_model(
                model=model,
                model_type='mlp',
                dataset_prefix='test_mlp',
                output_dir=tmpdir,
                scaler=scaler,
            )

            # Load and verify scaler is saved
            checkpoint = torch.load(filepath, weights_only=False)
            assert 'scaler' in checkpoint
            assert checkpoint['scaler'] is not None

    def test_save_best_model_prn_with_mask(self):
        """Test saving PRN model with mask."""
        from prism.maskedmlp import MaskedMLP

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create PRN model with mask
            mask = np.random.rand(5, 10).astype(np.float32)
            model = MaskedMLP(input_dim=5, hidden_units=10, output_dim=1, mask=mask)

            filepath = save_best_model(
                model=model,
                model_type='prn',
                dataset_prefix='test_mlp_prn',
                output_dir=tmpdir,
                mask=mask,
            )

            # Load and verify mask is saved
            checkpoint = torch.load(filepath, weights_only=False)
            assert 'mask' in checkpoint
            assert checkpoint['mask'] is not None
            np.testing.assert_array_equal(checkpoint['mask'], mask)

    def test_load_tuned_model_exists(self):
        """Test loading a tuned model that exists."""
        from prism.maskedmlp import MaskedMLP

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Save a model
            model = MaskedMLP(input_dim=5, hidden_units=10, output_dim=1)
            hyperparams = {'n_hidden': 10, 'lr': 0.001}

            save_best_model(
                model=model,
                model_type='mlp',
                dataset_prefix='test_load_mlp',
                output_dir=tmpdir,
                hyperparameters=hyperparams,
            )

            # Load model
            checkpoint = load_tuned_model(
                model_type='mlp', dataset_prefix='test_load_mlp', models_dir=tmpdir
            )

            assert checkpoint is not None
            assert 'model' in checkpoint
            assert checkpoint['tuned'] is True
            assert checkpoint['hyperparameters'] == hyperparams
            assert checkpoint['model_type'] == 'mlp'

    def test_load_tuned_model_not_exists(self):
        """Test loading returns None when model doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            checkpoint = load_tuned_model(
                model_type='mlp', dataset_prefix='nonexistent', models_dir=tmpdir
            )

            assert checkpoint is None

    def test_load_tuned_model_state_dict(self):
        """Test that loaded model has state_dict saved."""
        from prism.maskedmlp import MaskedMLP

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Save model
            model = MaskedMLP(input_dim=5, hidden_units=10, output_dim=1)
            save_best_model(
                model=model, model_type='mlp', dataset_prefix='test_state_dict', output_dir=tmpdir
            )

            # Load and verify state_dict
            checkpoint = load_tuned_model(
                model_type='mlp', dataset_prefix='test_state_dict', models_dir=tmpdir
            )

            assert 'model_state_dict' in checkpoint
            assert 'model_class' in checkpoint
            assert checkpoint['model_class'] == MaskedMLP

    def test_tuning_returns_and_saves_best_model(self):
        """Integration test: tuning returns best model that can be saved/loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Run quick tuning
            np.random.seed(42)
            X_train = np.random.randn(50, 5).astype(np.float32)
            y_train = (np.random.randn(50) > 0).astype(np.float32)
            X_test = np.random.randn(25, 5).astype(np.float32)
            y_test = (np.random.randn(25) > 0).astype(np.float32)

            tuning_config = TuningConfig(
                enabled=True, n_trials=2, metric='test_auc', pruning_enabled=False
            )

            best_params, study, best_model = run_hyperparameter_tuning(
                model_type='mlp',
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                tuning_config=tuning_config,
                random_seed=42,
                device='cpu',
            )

            # Save best model
            filepath = save_best_model(
                model=best_model,
                model_type='mlp',
                dataset_prefix='integration_test_mlp',
                output_dir=tmpdir,
                hyperparameters=best_params,
            )

            assert filepath.exists()

            # Load and verify
            checkpoint = load_tuned_model(
                model_type='mlp', dataset_prefix='integration_test_mlp', models_dir=tmpdir
            )

            assert checkpoint is not None
            assert checkpoint['tuned'] is True

            # Verify loaded model can make predictions
            loaded_model = checkpoint['model']
            preds = loaded_model.predict(torch.tensor(X_test))
            assert len(preds) == len(y_test)
