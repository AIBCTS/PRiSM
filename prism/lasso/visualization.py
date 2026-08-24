"""
Visualization module for LASSO regression results.

This module provides plotting functions for LASSO regression results.
"""

import matplotlib.pyplot as plt

try:
    from IPython.display import clear_output, display
except ImportError:
    display = None
    clear_output = None
from typing import Dict, List, Optional

import numpy as np
from rich import box
from rich.console import Console
from rich.table import Table


class LassoVisualizer:
    """Class for handling all LASSO visualization."""

    def __init__(self, base_model_name: Optional[str] = None):
        """
        Initialize LassoVisualizer.

        Parameters
        ----------
        base_model_name : Optional[str], optional
            Base model name to include in titles and tables
        """
        self.base_model_name = base_model_name

    def _get_annotation_indices(
        self, lambda_values: np.ndarray, max_annotations: int = 5
    ) -> List[int]:
        """
        Get indices for annotations at major x-axis gridpoints.

        Parameters
        ----------
        lambda_values : np.ndarray
            Array of lambda values
        max_annotations : int, optional
            Maximum number of annotations to show (default is 5)

        Returns
        -------
        List[int]
            Indices of lambda values to annotate
        """
        if len(lambda_values) <= max_annotations:
            return list(range(len(lambda_values)))

        # Select evenly spaced indices including first and last
        indices = np.linspace(0, len(lambda_values) - 1, max_annotations, dtype=int)
        return list(indices)

    def plot_path(
        self,
        lambda_values: np.ndarray,
        metrics: Dict[str, np.ndarray],
        title: str = "LASSO Path",
    ) -> plt.Figure:
        """
        Plot LASSO path showing various metrics.

        Parameters
        ----------
        lambda_values : np.ndarray
            Array of lambda values
        metrics : Dict[str, np.ndarray]
            Dictionary containing arrays for different metrics
        title : str, optional
            Plot title

        Returns
        -------
        plt.Figure
            The generated figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

        # Create title with base model name if provided
        full_title = f"{title} ({self.base_model_name})" if self.base_model_name else title

        # Plot losses
        ax1.semilogx(lambda_values, metrics["train_loss"], marker="o", label="Train Loss")
        ax1.semilogx(lambda_values, metrics["test_loss"], marker="o", label="Test Loss")
        ax1.set_xlabel("Lambda")
        ax1.set_ylabel("Log Loss")
        ax1.set_title(f"{full_title} - Loss")
        ax1.legend()
        ax1.invert_xaxis()

        # Plot AUC
        ax2.semilogx(lambda_values, metrics["train_auc"], marker="o", label="Train AUC")
        ax2.semilogx(lambda_values, metrics["test_auc"], marker="o", label="Test AUC")
        ax2.set_xlabel("Lambda")
        ax2.set_ylabel("AUC")
        ax2.set_title(f"{full_title} - AUC")
        ax2.legend()
        ax2.invert_xaxis()

        # Add padding to bottom of AUC plot for footnote
        if len(metrics["train_auc"]) > 0:
            min_auc = min(np.min(metrics["train_auc"]), np.min(metrics["test_auc"]))
            max_auc = max(np.max(metrics["train_auc"]), np.max(metrics["test_auc"]))
            y_range = max_auc - min_auc
            # If range is 0 (e.g. single point or constant), use a default small range
            if y_range == 0:
                y_range = 0.1

            # Extend bottom limit by 20% of range to make room for footnote
            ax2.set_ylim(bottom=min_auc - (y_range * 0.20))

        # Add feature count annotations at major gridpoints
        beta_counts_univ = metrics.get("beta_counts_univ")
        beta_counts_biv = metrics.get("beta_counts_biv")

        if beta_counts_univ is not None and beta_counts_biv is not None:
            annotation_indices = self._get_annotation_indices(lambda_values)

            # Find max test AUC index and ensure it's included
            max_auc_idx = int(np.argmax(metrics["test_auc"]))
            if max_auc_idx not in annotation_indices:
                annotation_indices.append(max_auc_idx)

            # Sort indices for consistent left-to-right processing
            annotation_indices = sorted(annotation_indices)

            # Calculate vertical offsets to avoid overlapping annotations
            # Check proximity in log space since x-axis is logarithmic
            log_lambdas = np.log10(lambda_values)
            log_range = log_lambdas.max() - log_lambdas.min() if len(log_lambdas) > 1 else 1
            proximity_threshold = log_range * 0.15  # 15% of log range

            offsets = []
            base_offset = -15
            stagger_step = -12
            for i, idx in enumerate(annotation_indices):
                offset = base_offset
                # Check if too close to previous annotation
                if i > 0:
                    prev_idx = annotation_indices[i - 1]
                    log_dist = abs(log_lambdas[idx] - log_lambdas[prev_idx])
                    if log_dist < proximity_threshold:
                        # Stagger: alternate between levels
                        prev_offset = offsets[-1]
                        if prev_offset == base_offset:
                            offset = base_offset + stagger_step
                        else:
                            offset = base_offset
                offsets.append(offset)

            # Add annotations to the AUC plot (bottom plot)
            for i, idx in enumerate(annotation_indices):
                if idx < len(lambda_values):
                    x_pos = lambda_values[idx]
                    y_pos = metrics["test_auc"][idx]
                    n_univ = int(beta_counts_univ[idx])
                    n_biv = int(beta_counts_biv[idx])

                    # Check if this is the max AUC point
                    is_max_auc = idx == max_auc_idx

                    # Create compact annotation text
                    annotation_text = f"U:{n_univ} B:{n_biv}"

                    if is_max_auc:
                        box_color = "lightyellow"
                        edge_color = "goldenrod"
                        text_color = "darkgoldenrod"
                        line_color = "goldenrod"
                    else:
                        box_color = "white"
                        edge_color = "lightgray"
                        text_color = "dimgray"
                        line_color = "gray"

                    # Add annotation with arrow pointing to the data point
                    ax2.annotate(
                        annotation_text,
                        xy=(x_pos, y_pos),
                        xytext=(0, offsets[i]),
                        textcoords="offset points",
                        fontsize=8,
                        ha="center",
                        va="top",
                        color=text_color,
                        bbox=dict(
                            boxstyle="round,pad=0.2",
                            facecolor=box_color,
                            edgecolor=edge_color,
                            alpha=0.9,
                        ),
                        arrowprops=dict(
                            arrowstyle="-",
                            color=line_color,
                            alpha=0.6,
                            lw=0.8,
                        ),
                    )

            # Add footnote explaining the annotations
            ax2.text(
                0.98,
                0.02,
                "U: Univariate effects (|beta|>0.1). B: Bivariate effects (|beta|>0.1). Yellow = max Test AUC",
                transform=ax2.transAxes,
                fontsize=7,
                color="dimgray",
                verticalalignment="bottom",
                horizontalalignment="right",
                fontstyle="italic",
            )

        plt.tight_layout()
        return fig

    def display_progress(
        self,
        lambda_values: np.ndarray,
        metrics: Dict[str, np.ndarray],
        batch_end: int,
        is_final: bool = False,
        return_fig: bool = False,
    ) -> Optional[plt.Figure]:
        """
        Display real-time progress of LASSO computation.

        Parameters
        ----------
        lambda_values : np.ndarray
            Array of lambda values up to current batch
        metrics : Dict[str, np.ndarray]
            Dictionary containing metric arrays up to current batch
        batch_end : int
            Current batch end index
        is_final : bool, optional
            Whether this is the final display (default is False)
        return_fig : bool, optional
            Whether to return the figure object (default is False)

        Returns
        -------
        Optional[plt.Figure]
            The generated figure if return_fig is True, None otherwise
        """
        if clear_output is not None:
            clear_output(wait=True)

        # Plot the path
        fig = self.plot_path(
            lambda_values[:batch_end],
            {k: v[:batch_end] for k, v in metrics.items()},
        )
        if display is not None:
            display(fig)
        plt.close(fig)

        # Create a Rich table for displaying the results
        console = Console()

        # Find the row with max test AUC if this is the final display
        max_test_auc_idx = None
        if is_final and batch_end > 0:
            max_test_auc_idx = np.argmax(metrics["test_auc"][:batch_end])

        # Display the results table
        table = self.create_results_table(
            lambda_values[:batch_end],
            {k: v[:batch_end] for k, v in metrics.items()},
            metrics.get("beta_counts_univ", np.zeros(batch_end)),
            metrics.get("beta_counts_biv", np.zeros(batch_end)),
            highlight_idx=max_test_auc_idx,
        )

        console.print(table)

        if return_fig:
            return fig
        return None

    def create_results_table(
        self,
        lambda_values: np.ndarray,
        metrics: Dict[str, np.ndarray],
        beta_counts_univ: np.ndarray,
        beta_counts_biv: np.ndarray,
        highlight_idx: Optional[int] = None,
    ) -> Table:
        """
        Create a Rich table for the LASSO results.

        Parameters
        ----------
        lambda_values : np.ndarray
            Array of lambda values
        metrics : Dict[str, np.ndarray]
            Dictionary containing metric arrays
        beta_counts_univ : np.ndarray
            Array of univariate beta counts
        beta_counts_biv : np.ndarray
            Array of bivariate beta counts
        highlight_idx : Optional[int], optional
            Index to highlight (e.g., best model) (default is None)

        Returns
        -------
        Table
            Rich Table with formatted results
        """
        # Create title with base model name if provided
        table_title = "LASSO Regression Results"
        if self.base_model_name:
            table_title = f"LASSO Regression Results ({self.base_model_name})"

        table = Table(
            box=box.ROUNDED,
            title=table_title,
            title_style="bold blue",
            header_style="bold",
            caption=(
                "* in Index column indicates row with maximum Test AUC"
                if highlight_idx is not None
                else None
            ),
            caption_style="italic",
        )

        # Add columns with appropriate formatting
        table.add_column("Index", justify="right", style="cyan")
        table.add_column("Lambda", justify="right", style="cyan")
        table.add_column("Train\nAUC", justify="right", style="green")
        table.add_column("Test\nAUC", justify="right", style="green")
        table.add_column("Train\nLoss", justify="right", style="yellow")
        table.add_column("Test\nLoss", justify="right", style="yellow")
        table.add_column("Univ.\nbeta>0.1", justify="right", style="magenta")
        table.add_column("Biv.\nbeta>0.1", justify="right", style="magenta")

        # Add rows
        for i in range(len(lambda_values)):
            row_style = "bold on grey85" if i == highlight_idx else None
            index_prefix = "* " if i == highlight_idx else ""

            table.add_row(
                f"{index_prefix}{i}",
                f"{lambda_values[i]:.4g}",
                f"{metrics['train_auc'][i]:.4f}",
                f"{metrics['test_auc'][i]:.4f}",
                f"{metrics['train_loss'][i]:.4f}",
                f"{metrics['test_loss'][i]:.4f}",
                f"{int(beta_counts_univ[i])}" if len(beta_counts_univ) > 0 else "N/A",
                f"{int(beta_counts_biv[i])}" if len(beta_counts_biv) > 0 else "N/A",
                style=row_style,
            )

        return table

    def format_output_row(
        self,
        idx: int,
        lambda_val: float,
        metrics: Dict[str, float],
        beta_counts: Dict[str, int],
        highlight_metrics: Optional[Dict[str, float]] = None,
    ) -> str:
        """
        Format a row for the output log table.

        Parameters
        ----------
        idx : int
            Lambda index
        lambda_val : float
            Lambda value
        metrics : Dict[str, float]
            Dictionary of metric values
        beta_counts : Dict[str, int]
            Dictionary containing counts of non-zero coefficients
        highlight_metrics : Optional[Dict[str, float]]
            Dictionary of best metric values to highlight

        Returns
        -------
        str
            Formatted row string
        """

        # This method is kept for backward compatibility but is no longer the primary display method
        def format_metric(value: float, best_value: Optional[float] = None) -> str:
            if best_value is not None and abs(value - best_value) < 1e-10:
                return f"{value:<9.4f}*"
            return f"{value:<10.4f}"

        row = [
            f"{idx:<6d}",
            f"{lambda_val:<10.5f}",
        ]

        # Add metrics with optional highlighting
        for metric_name in ["auc", "loss"]:
            for prefix in ["train_", "test_"]:
                key = f"{prefix}{metric_name}"
                best_value = highlight_metrics.get(key) if highlight_metrics else None
                row.append(format_metric(metrics[key], best_value))

        # Add beta counts
        row.extend(
            [
                f"{beta_counts['univ']:<11d}",
                f"{beta_counts['biv']:<10d}",
            ]
        )

        return " ".join(row)
