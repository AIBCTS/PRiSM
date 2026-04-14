"""PRiSM CLI -- command-line interface for the PRiSM pipeline.

Entry point registered as ``prism`` console script via pyproject.toml.
Delegates to the pipeline, parallel, and tuning runners.
"""

import sys

import typer

app = typer.Typer(
    name="prism",
    help="PRiSM: Partial Responses in Structured Models -- pipeline CLI tools.",
    add_completion=False,
)


@app.command(
    context_settings={
        "allow_extra_args": True,
        "allow_interspersed_args": False,
        "ignore_unknown_options": True,
    },
)
def run(ctx: typer.Context):
    """Run the PRiSM notebook pipeline (sequential).

    All arguments and flags are forwarded to the pipeline runner.

    Examples:
        prism run htx_example
        prism run htx_example --skip-preprocessing
        prism run -f batch.yaml
        prism run --list-configs
    """
    sys.argv = ["prism run"] + ctx.args
    from prism.cli.pipeline import main

    raise SystemExit(main() or 0)


@app.command(
    name="run-parallel",
    context_settings={
        "allow_extra_args": True,
        "allow_interspersed_args": False,
        "ignore_unknown_options": True,
    },
)
def run_parallel(ctx: typer.Context):
    """Run the PRiSM notebook pipeline in parallel across multiple GPUs.

    All arguments and flags are forwarded to the parallel runner.

    Examples:
        prism run-parallel htx_example --gpus 0,1,2,3
        prism run-parallel -f batch.yaml --gpus 0,1
        prism run-parallel htx_example
    """
    sys.argv = ["prism run-parallel"] + ctx.args
    from prism.cli.parallel import main

    raise SystemExit(main() or 0)


@app.command(
    context_settings={
        "allow_extra_args": True,
        "allow_interspersed_args": False,
        "ignore_unknown_options": True,
    },
)
def tune(ctx: typer.Context):
    """Run Optuna-based hyperparameter tuning.

    All arguments and flags are forwarded to the tuning runner.

    Examples:
        prism tune htx_example
        prism tune htx_example --models mlp xgb --trials 30
    """
    sys.argv = ["prism tune"] + ctx.args
    from prism.cli.tune import main

    raise SystemExit(main() or 0)


@app.command(name="list-configs")
def list_configs():
    """List available YAML configuration files."""
    sys.argv = ["prism list-configs", "--list-configs"]
    from prism.cli.pipeline import main

    raise SystemExit(main() or 0)


if __name__ == "__main__":
    app()
