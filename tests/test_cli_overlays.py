"""Exercise explicit CLI options and preview-only repair without pretrained models."""

from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image
from typer.testing import CliRunner

from tisam.cli import app
from tisam.config import DataConfig, ModelConfig, TrainConfig
from tisam.inference import batch


@pytest.mark.parametrize(
    "regular,mini", [(False, False), (True, False), (False, True), (True, True)]
)
def test_completed_mask_previews(tmp_path, monkeypatch, regular, mini):
    source = tmp_path / "slide.png"
    Image.fromarray(np.full((20, 30, 3), 100, dtype=np.uint8)).save(source)
    output = tmp_path / "out"
    output.mkdir()
    tifffile.imwrite(
        output / "slide.tif", np.ones((20, 30), dtype=np.uint8), tile=(256, 256), compression="zstd"
    )
    cfg = TrainConfig(
        model=ModelConfig(num_classes=1, extra_encoder="hf-hub:MahmoodLab/UNI2-h"),
        data=DataConfig(
            class_names=["void", "tissue"], colormap={"void": (0, 0, 0), "tissue": (200, 0, 0)}
        ),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Completed masks must not load a model")

    monkeypatch.setattr(batch, "load_model", forbidden)
    batch.segment_sources(
        cfg,
        Path("unused.safetensors"),
        source,
        output,
        check_classify=regular,
        mini_check_classify=mini,
    )
    full = output / "check_classification/slide.jpg"
    small = output / "check_classification/mini/slide.jpg"
    assert full.exists() == regular
    assert small.exists() == mini
    for path, size in [(full, (30, 20)), (small, (3, 2))]:
        if path.exists():
            with Image.open(path) as image:
                assert image.size == size
                assert np.allclose(np.asarray(image)[0, 0], (130, 70, 70), atol=3)


def test_cli_explicit_options(monkeypatch):
    seen = {}
    monkeypatch.setattr("tisam.cli.load_config", lambda path: path)

    def segment(*args, **kwargs):
        seen.update(kwargs)
        assert args == (Path("cfg.yaml"), Path("model.safetensors"), Path("slides"), Path("out"))
        return []

    monkeypatch.setattr(batch, "segment_sources", segment)
    runner = CliRunner()
    args = [
        "eval",
        "segment",
        "--config",
        "cfg.yaml",
        "--checkpoint",
        "model.safetensors",
        "--source",
        "slides",
        "--output",
        "out",
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert not seen["check_classify"] and not seen["mini_check_classify"]
    result = runner.invoke(app, args + ["--mini-check-classify"])
    assert result.exit_code == 0, result.output
    assert not seen["check_classify"] and seen["mini_check_classify"]
    assert (
        runner.invoke(
            app, ["eval", "segment", "cfg.yaml", "model.safetensors", "slides", "out"]
        ).exit_code
        != 0
    )


def test_palette_and_collisions_fail_before_loading(tmp_path, monkeypatch):
    cfg = TrainConfig(
        model=ModelConfig(num_classes=1, extra_encoder="hf-hub:MahmoodLab/UNI2-h"),
        data=DataConfig(class_names=["void", "tissue"]),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid request must not load a model")

    monkeypatch.setattr(batch, "load_model", forbidden)
    with pytest.raises(ValueError, match="missing colors"):
        batch.segment_sources(
            cfg, Path("unused"), tmp_path / "image.png", tmp_path / "out", mini_check_classify=True
        )
    (tmp_path / "same.png").touch()
    (tmp_path / "same.tif").touch()
    with pytest.raises(ValueError, match="colliding"):
        batch.segment_sources(cfg, Path("unused"), tmp_path, tmp_path / "out")


@pytest.mark.parametrize("command,split", [("validate", "validation"), ("test", "test")])
def test_eval_named_arguments(monkeypatch, command, split):
    calls = []
    monkeypatch.setattr("tisam.cli.run_evaluation", lambda *args: calls.append(args))
    result = CliRunner().invoke(
        app,
        [
            "eval",
            command,
            "--config",
            "cfg.yaml",
            "--checkpoint",
            "weights.safetensors",
            "--output",
            "metrics.json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls == [
        (Path("cfg.yaml"), Path("weights.safetensors"), Path("metrics.json"), split, False)
    ]


def test_directory_infers_once_per_source_with_one_model(tmp_path, monkeypatch):
    source = tmp_path / "images"
    source.mkdir()
    for name in ["a", "b"]:
        Image.fromarray(np.zeros((20, 30, 3), dtype=np.uint8)).save(source / f"{name}.png")
    cfg = TrainConfig(
        model=ModelConfig(num_classes=1, extra_encoder="hf-hub:MahmoodLab/UNI2-h"),
        data=DataConfig(class_names=["void", "tissue"]),
    )
    loads, masks = [], []
    monkeypatch.setattr(batch, "load_model", lambda *a, **kw: loads.append(a) or object())
    monkeypatch.setattr(
        batch, "prepare_wsi", lambda model, src, dest, **kw: masks.append((src, dest))
    )
    output = tmp_path / "out"
    assert batch.segment_sources(cfg, Path("weights"), source, output) == [
        output / "a.tif",
        output / "b.tif",
    ]
    assert len(loads) == 1
    assert masks == [(source / "a.png", output / "a.tif"), (source / "b.png", output / "b.tif")]
