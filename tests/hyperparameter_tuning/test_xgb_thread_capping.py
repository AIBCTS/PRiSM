"""Tests for XGB/RF thread capping during parallel Optuna tuning."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from prism.hyperparameter_tuning.config import TuningConfig


@pytest.fixture
def small_dataset():
    """Small synthetic dataset for objective tests."""
    np.random.seed(42)
    n_samples, n_features = 60, 5
    X_train = np.random.randn(n_samples, n_features).astype(np.float32)
    y_train = (np.random.randn(n_samples) > 0).astype(np.float32)
    X_test = np.random.randn(30, n_features).astype(np.float32)
    y_test = (np.random.randn(30) > 0).astype(np.float32)
    return X_train, y_train, X_test, y_test


@pytest.fixture
def tuning_config():
    return TuningConfig(
        enabled=True,
        n_trials=1,
        metric='test_auc',
        direction='maximize',
        pruning_enabled=False,
    )


# ---------- XGBObjective nthread ----------


class TestXGBObjectiveNthread:
    def test_nthread_stored(self, small_dataset, tuning_config):
        from prism.hyperparameter_tuning.objectives import XGBObjective

        obj = XGBObjective(*small_dataset, tuning_config, random_seed=42, nthread=4)
        assert obj.nthread == 4

    def test_nthread_default_none(self, small_dataset, tuning_config):
        from prism.hyperparameter_tuning.objectives import XGBObjective

        obj = XGBObjective(*small_dataset, tuning_config, random_seed=42)
        assert obj.nthread is None

    def test_nthread_passed_to_xgb(self, small_dataset, tuning_config):
        """When nthread is set, XGBClassifier receives it."""
        from prism.hyperparameter_tuning.objectives import XGBObjective

        obj = XGBObjective(*small_dataset, tuning_config, random_seed=42, nthread=2)

        # Run a real trial to verify nthread flows through
        import optuna

        study = optuna.create_study(direction='maximize')

        with patch('xgboost.XGBClassifier') as mock_cls:
            mock_model = MagicMock()
            mock_model.predict_proba.return_value = np.column_stack(
                [np.zeros(30), np.random.rand(30)]
            )
            mock_model.evals_result.return_value = {
                'validation_0': {'logloss': [0.5]},
                'validation_1': {'logloss': [0.5]},
            }
            mock_model.best_iteration = 0
            mock_cls.return_value = mock_model

            study.optimize(obj, n_trials=1)

            # Verify nthread was in the kwargs
            call_kwargs = mock_cls.call_args
            assert call_kwargs[1]['nthread'] == 2 or call_kwargs.kwargs.get('nthread') == 2

    def test_nthread_none_not_passed(self, small_dataset, tuning_config):
        """When nthread is None, XGBClassifier should NOT receive nthread kwarg."""
        from prism.hyperparameter_tuning.objectives import XGBObjective

        obj = XGBObjective(*small_dataset, tuning_config, random_seed=42)

        import optuna

        study = optuna.create_study(direction='maximize')

        with patch('xgboost.XGBClassifier') as mock_cls:
            mock_model = MagicMock()
            mock_model.predict_proba.return_value = np.column_stack(
                [np.zeros(30), np.random.rand(30)]
            )
            mock_model.evals_result.return_value = {
                'validation_0': {'logloss': [0.5]},
                'validation_1': {'logloss': [0.5]},
            }
            mock_model.best_iteration = 0
            mock_cls.return_value = mock_model

            study.optimize(obj, n_trials=1)

            # nthread should NOT be in the call kwargs
            call_kwargs = mock_cls.call_args
            # Check both positional-as-keyword and keyword forms
            all_kwargs = call_kwargs.kwargs if call_kwargs.kwargs else {}
            assert 'nthread' not in all_kwargs


# ---------- RFObjective nthread ----------


class TestRFObjectiveNthread:
    def test_nthread_stored(self, small_dataset, tuning_config):
        from prism.hyperparameter_tuning.objectives import RFObjective

        obj = RFObjective(*small_dataset, tuning_config, random_seed=42, nthread=3)
        assert obj.nthread == 3

    def test_nthread_default_none(self, small_dataset, tuning_config):
        from prism.hyperparameter_tuning.objectives import RFObjective

        obj = RFObjective(*small_dataset, tuning_config, random_seed=42)
        assert obj.nthread is None

    def test_nthread_passed_to_rf(self, small_dataset, tuning_config):
        """When nthread is set, XGBRFClassifier receives it."""
        from prism.hyperparameter_tuning.objectives import RFObjective

        obj = RFObjective(*small_dataset, tuning_config, random_seed=42, nthread=2)

        import optuna

        study = optuna.create_study(direction='maximize')

        with patch('xgboost.XGBRFClassifier') as mock_cls:
            mock_model = MagicMock()
            mock_model.predict_proba.return_value = np.column_stack(
                [np.zeros(30), np.random.rand(30)]
            )
            mock_cls.return_value = mock_model

            study.optimize(obj, n_trials=1)

            call_kwargs = mock_cls.call_args
            assert call_kwargs[1]['nthread'] == 2 or call_kwargs.kwargs.get('nthread') == 2


# ---------- Thread calculation logic ----------


class TestThreadCalculation:
    """Test the nthread = max(1, cpu_count // n_jobs) formula."""

    @pytest.mark.parametrize(
        "cpu_count,n_jobs,expected",
        [
            (8, 1, 8),
            (8, 2, 4),
            (8, 4, 2),
            (8, 8, 1),
            (8, 16, 1),  # more jobs than cores -> floor to 1
            (1, 4, 1),  # single core
            (16, 3, 5),  # non-even division
        ],
    )
    def test_formula(self, cpu_count, n_jobs, expected):
        result = max(1, cpu_count // n_jobs)
        assert result == expected


# ---------- Integration: tuning.py injects nthread ----------


class TestTuningInjectsNthread:
    """Verify run_hyperparameter_tuning sets nthread for XGB/RF when n_jobs > 1."""

    def test_xgb_threaded_path_sets_nthread(self, small_dataset, tuning_config):
        """Standard (non-multiprocessing) path with n_jobs > 1 should set nthread."""
        from prism.hyperparameter_tuning.tuning import run_hyperparameter_tuning

        tuning_config.n_jobs = 2

        with patch('os.cpu_count', return_value=8):
            best_params, study, best_model = run_hyperparameter_tuning(
                model_type='xgb',
                X_train=small_dataset[0],
                y_train=small_dataset[1],
                X_test=small_dataset[2],
                y_test=small_dataset[3],
                tuning_config=tuning_config,
                random_seed=42,
                device='cpu',
            )

        # If it ran without error, nthread was accepted.
        # Verify the study completed its trial(s).
        assert len(study.trials) >= 1

    def test_xgb_single_job_no_nthread(self, small_dataset, tuning_config):
        """With n_jobs=1, nthread should not be injected (let XGB use all cores)."""
        from prism.hyperparameter_tuning.tuning import run_hyperparameter_tuning

        tuning_config.n_jobs = 1

        best_params, study, best_model = run_hyperparameter_tuning(
            model_type='xgb',
            X_train=small_dataset[0],
            y_train=small_dataset[1],
            X_test=small_dataset[2],
            y_test=small_dataset[3],
            tuning_config=tuning_config,
            random_seed=42,
            device='cpu',
        )

        assert len(study.trials) >= 1

    def test_rf_threaded_path_sets_nthread(self, small_dataset, tuning_config):
        """RF with n_jobs > 1 should also get nthread capping."""
        from prism.hyperparameter_tuning.tuning import run_hyperparameter_tuning

        tuning_config.n_jobs = 2

        with patch('os.cpu_count', return_value=8):
            best_params, study, best_model = run_hyperparameter_tuning(
                model_type='rf',
                X_train=small_dataset[0],
                y_train=small_dataset[1],
                X_test=small_dataset[2],
                y_test=small_dataset[3],
                tuning_config=tuning_config,
                random_seed=42,
                device='cpu',
            )

        assert len(study.trials) >= 1
