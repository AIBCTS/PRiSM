"""Nomogram additivity round-trip.

The deployed nomogram is purely additive on the log-odds scale:

    logit = intercept + sum_k (beta_k * phi_k(x))

and the exported JSON stores each beta-scaled contribution on a grid keyed by the
feature's x-values (NomogramData documents: "sum all response contributions and add
the model intercept"). This test drives the *real* export -> load path on an additive
ground-truth model and asserts that scoring the reloaded nomogram reproduces the
selected LASSO model's logit.

There is no points axis in the output (the nomogram is rendered/exported on the
log-odds scale), so "1 point = 0.1 log-odds" is a hand-reading convention only and is
not a code path to pin here. The additive-on-logit identity below is the testable
content of the additive nomogram contract.
"""

import numpy as np
import pytest
import torch

from prism.lasso import LassoComputation, LassoConfig
from prism.partial_responses import partial_responses
from prism.plotting import load_nomogram_json, save_nomogram_json
from prism.plotting.pipeline import PlottingPipeline
from prism.preprocessing import NoScaler
from tests.conftest import LinearTestModel


@pytest.mark.integration
class TestNomogramAdditivityRoundTrip:
    def test_logodds_round_trip_reproduces_lasso_logit(self, tmp_path):
        torch.manual_seed(0)
        n, d = 200, 4
        feature_names = ['f0', 'f1', 'f2', 'f3']

        # Purely additive source model: only f0 and f1 carry signal, no interactions.
        x = torch.randn(n, d)
        model = LinearTestModel([0.8, 0.6, 0.0, 0.0], bias=0.0)
        y = ((0.8 * x[:, 0] + 0.6 * x[:, 1]) > 0).numpy().astype(float)

        # Partial responses (the LASSO design matrix), univariate-first.
        pr = partial_responses(x=x, model=model, method='dirac', device='cpu')
        pr_np = pr.numpy()

        cfg = LassoConfig(
            max_lambda=5.0,
            min_lambda=1e-3,
            nlambda=25,
            batch_size=64,
            max_workers=1,
            real_time_plot=False,
        )
        results, _ = LassoComputation(cfg).fit(
            partial_responses_train=pr[:150],
            partial_responses_test=pr[150:],
            y_train=y[:150],
            y_test=y[150:],
            feature_names=feature_names,
        )
        results.select_lambda_max_test_auc()

        # Additive model -> no genuine interaction -> no bivariate term should survive,
        # so the deployed nomogram is a sum of univariate contributions only.
        assert (
            results.get_selected_bivariate_index_pairs() == []
        ), "additive model unexpectedly selected a bivariate term"
        selected_univ = set(results.get_selected_univariate_indices())
        assert {0, 1}.issubset(selected_univ)

        # --- Real export -> load round-trip ---
        pipeline = PlottingPipeline(results, group_manager=None, label_manager=None)
        bundle = pipeline.prepare_plotting_bundle(
            x=x,
            model=model,
            scaler=NoScaler(),
            n_steps=200,
            method='dirac',
            feature_names=feature_names,
            categorical_threshold=15,
            trim_quantile=None,
        )
        bundle = pipeline.apply_beta_scaling(bundle)
        json_path = save_nomogram_json(
            bundle=bundle,
            lasso_results=results,
            file_path=tmp_path / "nomogram.json",
            method='dirac',
        )
        data = load_nomogram_json(json_path)

        # Score every sample by the documented contract: intercept + sum of each
        # feature's beta-scaled response interpolated at the sample's x-value.
        total_logit = np.full(n, float(data.intercept))
        for name, feat in data.univariate.items():
            col = feature_names.index(name)
            xv = np.asarray(feat['x_values'], dtype=float)
            resp = np.asarray(feat['response'], dtype=float)  # already beta-scaled
            total_logit += np.interp(x[:, col].numpy(), xv, resp)

        # Reference: the selected (beta-thresholded) LASSO model's own logit.
        selected_model = results.get_selected_model()
        lasso_logit = selected_model.decision_function(pr_np)

        max_diff = np.abs(total_logit - lasso_logit).max()
        print(f"\nNomogram round-trip vs LASSO logit: max |diff| = {max_diff:.3e}")
        print(
            f"  intercept = {data.intercept:.6f}, "
            f"selected univariate = {sorted(selected_univ)}"
        )

        # Additive model + linear partial responses => grid interpolation is near-exact.
        assert max_diff < 1e-2, (
            f"reloaded nomogram does not reproduce the LASSO logit (max diff {max_diff:.3e}); "
            f"additivity on the log-odds scale is broken"
        )

    def test_responses_are_logodds_not_probabilities(self, tmp_path):
        """Guard: exported responses live on the log-odds scale, never squashed to [0,1]."""
        torch.manual_seed(1)
        n, d = 150, 3
        feature_names = ['a', 'b', 'c']
        x = torch.randn(n, d) * 2.0  # wide spread -> large-magnitude log-odds contributions
        model = LinearTestModel([1.5, -1.2, 0.0], bias=0.0)
        y = ((1.5 * x[:, 0] - 1.2 * x[:, 1]) > 0).numpy().astype(float)

        pr = partial_responses(x=x, model=model, method='dirac', device='cpu')
        cfg = LassoConfig(
            max_lambda=5.0,
            min_lambda=1e-3,
            nlambda=25,
            batch_size=64,
            max_workers=1,
            real_time_plot=False,
        )
        results, _ = LassoComputation(cfg).fit(
            partial_responses_train=pr[:110],
            partial_responses_test=pr[110:],
            y_train=y[:110],
            y_test=y[110:],
            feature_names=feature_names,
        )
        results.select_lambda_max_test_auc()

        pipeline = PlottingPipeline(results, group_manager=None, label_manager=None)
        bundle = pipeline.prepare_plotting_bundle(
            x=x,
            model=model,
            scaler=NoScaler(),
            n_steps=100,
            method='dirac',
            feature_names=feature_names,
            categorical_threshold=15,
            trim_quantile=None,
        )
        bundle = pipeline.apply_beta_scaling(bundle)
        data = load_nomogram_json(
            save_nomogram_json(
                bundle=bundle,
                lasso_results=results,
                file_path=tmp_path / "n.json",
                method='dirac',
            )
        )

        # At least one contribution must exceed the [0,1] probability band, which is
        # impossible if the responses had been sigmoid-transformed.
        max_abs = max(
            float(np.abs(np.asarray(feat['response'], dtype=float)).max())
            for feat in data.univariate.values()
        )
        assert max_abs > 1.0, (
            f"all exported responses fit within +/-1 (max {max_abs:.3f}); responses may "
            f"have been put on the probability scale instead of log-odds"
        )
