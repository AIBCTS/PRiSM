"""
Visual Validation Tests with Synthetic Data.

This test generates plots with varied, realistic medical data to properly
visualize all rendering phases without requiring production fixtures.

Generated plots:
- Phase 1: Univariate categorical and continuous (single/two-column, odds ratio)
- Phase 2: Mixed bivariate features (cat x cont)
- Phase 3: Heatmaps and contours
- Phase 4: Response plots with histograms

DATA: Synthetic medical dataset with varied distributions to properly test rendering.

Output Directory Structure:
    tests/fixtures/output/visual_validation/
    ├── production/           # From real production data (test_visual_production.py)
    │   ├── with_labels/      # Tests with FeatureLabelManager
    │   └── no_labels/        # Tests without labels (fallback behavior)
    └── synthetic/            # From mock/synthetic data (this file)

Usage:
    pytest tests/plotting/test_visual_synthetic.py -v -s
"""

import matplotlib
import numpy as np
import pytest
import torch

matplotlib.use('Agg')  # Non-interactive backend
from pathlib import Path

import matplotlib.pyplot as plt

from prism.plotting.formatter import PlotFormatter
from prism.plotting.pipeline import PlottingPipeline
from prism.plotting.renderer import NomogramRenderer

# Unified output directory structure
OUTPUT_BASE = Path(__file__).parent.parent / "fixtures" / "output" / "visual_validation"
OUTPUT_DIR = OUTPUT_BASE / "synthetic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class TestRealisticVisualVerification:
    """Visual verification with realistic medical data."""

    @pytest.fixture
    def realistic_lasso_results(self):
        """Mock LassoResultsManager with varied features."""
        from itertools import combinations

        class MockLassoResults:
            def __init__(self):
                # Realistic medical features with various types
                self.univariate_feature_names = [
                    'age_years',  # 0: continuous (20-80)
                    'bmi',  # 1: continuous (18-45)
                    'systolic_bp',  # 2: continuous (90-180)
                    'diabetes',  # 3: binary
                    'smoking_status',  # 4: categorical (3 categories)
                    'cholesterol_ldl',  # 5: continuous (50-250)
                    'exercise_level',  # 6: categorical (4 categories)
                    'family_history',  # 7: binary
                ]
                self.n_univ = len(self.univariate_feature_names)

                # All possible bivariate pairs (8 choose 2 = 28 pairs)
                self.bivariate_inputs = list(combinations(range(self.n_univ), 2))
                self.n_biv = len(self.bivariate_inputs)

                # Selected bivariate pairs (subset with non-zero beta)
                self._selected_bivariate_pairs = [(0, 1), (1, 3), (2, 5), (3, 4), (3, 7), (4, 5)]

                # all_feature_names includes univariate + ALL bivariate pair names
                self.all_feature_names = self.univariate_feature_names.copy()
                for i, j in self.bivariate_inputs:
                    self.all_feature_names.append(
                        f"{self.univariate_feature_names[i]} : {self.univariate_feature_names[j]}"
                    )

            def get_selected_univariate_indices(self):
                # All univariate features selected
                return list(range(self.n_univ))

            def get_selected_bivariate_index_pairs(self):
                # Selected bivariate pairs (non-zero beta)
                return self._selected_bivariate_pairs

            def get_selected_beta(self):
                # Full beta array: 8 univariate + 28 bivariate = 36 coefficients
                # Only selected pairs have non-zero values
                beta = np.zeros(self.n_univ + self.n_biv)

                # Univariate betas (all selected)
                beta[: self.n_univ] = [0.8, 1.2, 0.5, 2.1, 1.5, 0.9, 0.7, 1.8]

                # Bivariate betas (only selected pairs)
                bivariate_betas = [0.5, 0.6, 0.7, 0.9, 0.3, 0.4]
                for (i, j), b in zip(self._selected_bivariate_pairs, bivariate_betas):
                    # Find position in bivariate_inputs
                    pair_idx = self.bivariate_inputs.index((i, j))
                    beta[self.n_univ + pair_idx] = b

                return beta

        return MockLassoResults()

    @pytest.fixture
    def realistic_model(self):
        """Realistic mock model with varied predictions."""

        class RealisticModel:
            def predict_proba(self, x, device='cpu'):
                # Create realistic predictions based on features
                batch_size = x.shape[0] if hasattr(x, 'shape') else len(x)
                if hasattr(x, 'shape') and x.shape[1] > 0:
                    # Combine multiple features for realistic variation
                    # Normalize to [0, 1] range roughly
                    age_effect = torch.sigmoid((x[:, 0] - 50) / 15)
                    bmi_effect = torch.sigmoid((x[:, 1] - 25) / 5)
                    diabetes_effect = x[:, 3] * 0.3

                    # Combine effects
                    logit = (
                        age_effect * 0.3
                        + bmi_effect * 0.3
                        + diabetes_effect
                        + torch.randn(batch_size, device=device) * 0.1
                    )
                    return torch.sigmoid(logit).unsqueeze(1)
                else:
                    return torch.ones(batch_size, 1, device=device) * 0.5

            def __call__(self, x):
                return self.predict_proba(x)

        return RealisticModel()

    @pytest.fixture
    def realistic_data(self):
        """Create realistic medical test data with varied distributions."""
        np.random.seed(42)
        n_samples = 250

        # Age: Normal distribution 20-80, centered at 55
        age = np.clip(np.random.normal(55, 12, n_samples), 20, 80)

        # BMI: Skewed distribution 18-45, centered at 27
        bmi = np.clip(np.random.gamma(3, 3, n_samples) + 18, 18, 45)

        # Systolic BP: Normal distribution 90-180, centered at 125
        systolic_bp = np.clip(np.random.normal(125, 18, n_samples), 90, 180)

        # Diabetes: Binary with 30% prevalence
        diabetes = np.random.binomial(1, 0.3, n_samples).astype(float)

        # Smoking status: 3 categories (never=0, former=1, current=2)
        # Distribution: 50% never, 30% former, 20% current
        smoking_status = np.random.choice([0, 1, 2], size=n_samples, p=[0.5, 0.3, 0.2])

        # LDL Cholesterol: Skewed distribution 50-250, centered at 130
        cholesterol_ldl = np.clip(np.random.gamma(4, 20, n_samples) + 50, 50, 250)

        # Exercise level: 4 categories (sedentary=0, light=1, moderate=2, vigorous=3)
        # Distribution: 25% each category
        exercise_level = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.25, 0.25, 0.25, 0.25])

        # Family history: Binary with 40% prevalence
        family_history = np.random.binomial(1, 0.4, n_samples).astype(float)

        data = np.column_stack(
            [
                age,
                bmi,
                systolic_bp,
                diabetes,
                smoking_status,
                cholesterol_ldl,
                exercise_level,
                family_history,
            ]
        )

        return torch.from_numpy(data).float()

    @pytest.fixture
    def realistic_bundle(self, realistic_lasso_results, realistic_model, realistic_data):
        """Create bundle with realistic medical data."""
        pipeline = PlottingPipeline(
            lasso_results=realistic_lasso_results,
            group_manager=None,
            label_manager=None,
        )

        bundle = pipeline.prepare_plotting_bundle(
            x=realistic_data,
            model=realistic_model,
            feature_names=[
                'Age (years)',
                'BMI',
                'Systolic BP',
                'Diabetes',
                'Smoking Status',
                'LDL Cholesterol',
                'Exercise Level',
                'Family History',
            ],
            categorical_threshold=5,  # Binary and low-cardinality will be categorical
        )

        bundle = pipeline.apply_beta_scaling(bundle)
        return bundle

    def test_phase1_single_column_nomogram(self, realistic_bundle):
        """Phase 1: Single-column nomogram with realistic data."""
        formatter = PlotFormatter()
        renderer = NomogramRenderer(realistic_bundle, formatter)

        fig = renderer.render_nomogram()

        output_path = OUTPUT_DIR / "phase1_single_column_nomogram.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"\n[PHASE 1] Saved: {output_path}")
        print("  Check: Univariate features render clearly with varied data")
        assert output_path.exists()

    def test_phase1_two_column_nomogram(self, realistic_bundle):
        """Phase 1: Two-column nomogram layout."""
        formatter = PlotFormatter()
        renderer = NomogramRenderer(realistic_bundle, formatter)

        fig = renderer.render_nomogram(two_column=True)

        output_path = OUTPUT_DIR / "phase1_two_column_nomogram.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"[PHASE 1] Saved: {output_path}")
        print("  Check: Two-column layout balances features properly")
        assert output_path.exists()

    def test_phase1_odds_ratio(self, realistic_bundle):
        """Phase 1: Nomogram with odds ratio conversion."""
        formatter = PlotFormatter()
        renderer = NomogramRenderer(realistic_bundle, formatter, use_odds_ratio=True)

        fig = renderer.render_nomogram()

        output_path = OUTPUT_DIR / "phase1_odds_ratio_nomogram.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"[PHASE 1] Saved: {output_path}")
        print("  Check: Y-axis shows odds ratios (>1 or <1), log scale")
        assert output_path.exists()

    def test_phase2_mixed_bivariate(self, realistic_bundle):
        """Phase 2: Nomogram with mixed bivariate features."""
        formatter = PlotFormatter()
        renderer = NomogramRenderer(realistic_bundle, formatter)

        # This includes univariate AND mixed bivariate
        fig = renderer.render_nomogram()

        output_path = OUTPUT_DIR / "phase2_with_mixed_bivariate.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"[PHASE 2] Saved: {output_path}")
        print("  Check: Mixed bivariate features show multiple lines grouped by category")
        assert output_path.exists()

    def test_phase2_mixed_legend_right(self, realistic_bundle):
        """Phase 2: Mixed bivariate with legend on right side."""
        formatter = PlotFormatter()
        renderer = NomogramRenderer(realistic_bundle, formatter)

        fig = renderer.render_nomogram(legend_on_right=True)

        output_path = OUTPUT_DIR / "phase2_mixed_legend_right.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"[PHASE 2] Saved: {output_path}")
        print("  Check: Legends appear on right side of mixed bivariate plots")
        assert output_path.exists()

    def test_phase3_bivariate_heatmaps(self, realistic_bundle):
        """Phase 3: Bivariate heatmaps (cat x cat and cont x cont)."""
        formatter = PlotFormatter()
        renderer = NomogramRenderer(realistic_bundle, formatter)

        fig = renderer.render_bivariate_heatmaps()

        if fig is not None:
            output_path = OUTPUT_DIR / "phase3_bivariate_heatmaps.png"
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            print(f"[PHASE 3] Saved: {output_path}")
            print("  Check: Heatmaps for cat x cat, contours for cont x cont")
            print("  Check: Colorbars show value ranges clearly")
            assert output_path.exists()

    def test_phase3_heatmaps_odds_ratio(self, realistic_bundle):
        """Phase 3: Heatmaps with odds ratio conversion."""
        formatter = PlotFormatter()
        renderer = NomogramRenderer(realistic_bundle, formatter, use_odds_ratio=True)

        fig = renderer.render_bivariate_heatmaps()

        if fig is not None:
            output_path = OUTPUT_DIR / "phase3_heatmaps_odds_ratio.png"
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            print(f"[PHASE 3] Saved: {output_path}")
            print("  Check: Colorbar shows odds ratios instead of log odds")
            assert output_path.exists()

    def test_phase4_response_plots(self, realistic_bundle):
        """Phase 4: Response plots with histograms."""
        formatter = PlotFormatter()
        renderer = NomogramRenderer(realistic_bundle, formatter)

        fig = renderer.render_response_plots()

        output_path = OUTPUT_DIR / "phase4_response_plots.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"[PHASE 4] Saved: {output_path}")
        print("  Check: Response plots show partial responses with data histograms")
        print("  Check: Histograms show realistic data distributions")
        assert output_path.exists()

    def test_phase4_response_plots_odds_ratio(self, realistic_bundle):
        """Phase 4: Response plots with odds ratio."""
        formatter = PlotFormatter()
        renderer = NomogramRenderer(realistic_bundle, formatter, use_odds_ratio=True)

        fig = renderer.render_response_plots()

        output_path = OUTPUT_DIR / "phase4_response_plots_odds_ratio.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"[PHASE 4] Saved: {output_path}")
        print("  Check: Y-axes show odds ratios with log scale")
        assert output_path.exists()

    def test_complete_suite(self, realistic_bundle):
        """Complete: All plot types together."""
        formatter = PlotFormatter()
        renderer = NomogramRenderer(realistic_bundle, formatter)

        # Main nomogram (two-column for compactness)
        fig_main = renderer.render_nomogram(two_column=True)
        output_main = OUTPUT_DIR / "complete_main_nomogram.png"
        fig_main.savefig(output_main, dpi=150, bbox_inches='tight')
        plt.close(fig_main)
        print(f"[COMPLETE] Saved: {output_main}")

        # Bivariate heatmaps
        fig_heatmaps = renderer.render_bivariate_heatmaps()
        if fig_heatmaps is not None:
            output_heatmaps = OUTPUT_DIR / "complete_bivariate_heatmaps.png"
            fig_heatmaps.savefig(output_heatmaps, dpi=150, bbox_inches='tight')
            plt.close(fig_heatmaps)
            print(f"[COMPLETE] Saved: {output_heatmaps}")

        # Response plots
        fig_response = renderer.render_response_plots()
        output_response = OUTPUT_DIR / "complete_response_plots.png"
        fig_response.savefig(output_response, dpi=150, bbox_inches='tight')
        plt.close(fig_response)
        print(f"[COMPLETE] Saved: {output_response}")

        assert output_main.exists()
        assert output_response.exists()


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("VISUAL VERIFICATION TEST - REALISTIC DATA")
    print("=" * 80)
    print("\nGenerating plots with realistic medical data...")
    print(f"Output directory: {OUTPUT_DIR}\n")

    pytest.main([__file__, "-v", "-s"])

    print("\n" + "=" * 80)
    print("GENERATED FILES FOR MANUAL REVIEW:")
    print("=" * 80)
    for png_file in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  {png_file.name}")
    print("\nPlease review these images to verify plot quality!")
    print("Data distributions are now realistic and varied for proper visualization.")
    print("=" * 80 + "\n")
