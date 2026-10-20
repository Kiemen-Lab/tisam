"""Console entry point with optional workflows loaded only when invoked."""

import importlib
import json
from pathlib import Path
from typing import Annotated, Optional

import typer

from tisam.checkpointing.weights import load_model
from tisam.config.run import load_config

app = typer.Typer(no_args_is_help=True, help="Dual-encoder TiSAM training and inference.")
eval_app = typer.Typer(no_args_is_help=True)
app.add_typer(eval_app, name="eval")


def workflow(module: str, extra: str):
    """Report the required extra without importing it for unrelated commands."""
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as error:
        raise typer.BadParameter(
            f"This command requires tisam[{extra}]; missing {error.name}"
        ) from error


@app.command("train")
def train_command(
    config: Annotated[Path, typer.Option("--config")],
    resume: Annotated[bool, typer.Option("--resume/--no-resume")] = True,
    checkpoint: Optional[Path] = None,
    initialize_from: Optional[Path] = None,
    trusted: bool = False,
):
    """Train one configuration; resume its latest checkpoint by default."""
    workflow("tisam.training.runner", "train").train(
        load_config(config),
        resume=resume,
        checkpoint=checkpoint,
        initialize_from=initialize_from,
        trusted=trusted,
    )


def run_evaluation(config: Path, checkpoint: Path, output: Path, split: str, trusted: bool):
    """Share config and model loading between validation and test commands."""
    cfg = load_config(config)
    module = workflow("tisam.inference", "inference")
    model = load_model(
        checkpoint,
        config=cfg.model,
        device=cfg.device,
        sam3_checkpoint=cfg.sam3_checkpoint,
        trusted=trusted,
    )
    result = module.evaluate(model, cfg, split=split)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    typer.echo(json.dumps(result, indent=2))


@eval_app.command("validate")
def validate_command(
    config: Annotated[Path, typer.Option("--config")],
    checkpoint: Annotated[Path, typer.Option("--checkpoint")],
    output: Path = Path("validation.json"),
    trusted: bool = False,
):
    """Evaluate the configured validation tiles and write foreground metrics."""
    run_evaluation(config, checkpoint, output, "validation", trusted)


@eval_app.command("test")
def test_command(
    config: Annotated[Path, typer.Option("--config")],
    checkpoint: Annotated[Path, typer.Option("--checkpoint")],
    output: Path = Path("test.json"),
    trusted: bool = False,
):
    """Evaluate the configured held-out test tiles."""
    run_evaluation(config, checkpoint, output, "test", trusted)


@eval_app.command("segment")
def segment_command(
    config: Annotated[Path, typer.Option("--config")],
    checkpoint: Annotated[Path, typer.Option("--checkpoint")],
    source: Annotated[Path, typer.Option("--source")],
    output: Annotated[Path, typer.Option("--output")],
    check_classify: bool = False,
    mini_check_classify: bool = False,
    patch_size: int = 1024,
    effective_size: int = 800,
    resume: Annotated[bool, typer.Option("--resume/--no-resume")] = True,
    trusted: bool = False,
):
    """Segment a file or directory into integer masks and optional JPG previews."""
    cfg = load_config(config)
    module = workflow("tisam.inference.batch", "inference")
    results = module.segment_sources(
        cfg,
        checkpoint,
        source,
        output,
        check_classify=check_classify,
        mini_check_classify=mini_check_classify,
        patch_size=patch_size,
        effective_size=effective_size,
        resume=resume,
        trusted=trusted,
    )
    for result in results:
        typer.echo(str(result))
