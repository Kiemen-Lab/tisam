"""Check inference overlap, bounded handoffs, and resumable background failures."""

from contextlib import closing
from threading import Event

import numpy as np
import portalocker
import pytest
import tifffile
import torch
from mammoth.core import BackgroundPipelineError
from PIL import Image

from tisam.config import DataConfig, TrainConfig
from tisam.inference import batch, wsi
from tisam.inference.transfers import BatchPrefetcher, PredictionHandoff, transfer_predictions


def image(path):
    Image.fromarray(np.full((19, 23, 3), 100, dtype=np.uint8)).save(path)


def test_next_slide_inference_overlaps_publication(small_model, tmp_path, monkeypatch):
    model = small_model()
    cfg = TrainConfig(
        model=model.config, data=DataConfig(class_names=["void", "a", "b"]), device="cpu"
    )
    source = tmp_path / "sources"
    source.mkdir()
    image(source / "a.png")
    image(source / "b.png")
    started, inferred_next = Event(), Event()
    original_publish, original_prepare = wsi.PreparedMask.publish, batch.prepare_wsi

    def publish(prepared):
        if prepared.output.stem == "a":
            started.set()
            assert inferred_next.wait(10), "Next slide did not infer during publication"
        return original_publish(prepared)

    def prepare(model, path, target, **kwargs):
        if path.stem == "b":
            assert started.wait(10)
            try:
                return original_prepare(model, path, target, **kwargs)
            finally:
                inferred_next.set()
        return original_prepare(model, path, target, **kwargs)

    monkeypatch.setattr(batch, "load_model", lambda *a, **kw: model)
    monkeypatch.setattr(batch, "prepare_wsi", prepare)
    monkeypatch.setattr(wsi.PreparedMask, "publish", publish)
    results = batch.segment_sources(
        cfg, tmp_path / "unused", source, tmp_path / "output", patch_size=16, effective_size=8
    )
    assert len(results) == 2
    for result in results:
        assert tifffile.imread(result).shape == (19, 23)
        assert not result.with_name(result.name + ".work").exists()


def test_prediction_writer_overlaps_next_forward(small_model, tmp_path, monkeypatch):
    model = small_model()
    source = tmp_path / "image.png"
    image(source)
    writing, next_forward = Event(), Event()
    original_resolve, original_forward = PredictionHandoff.resolve, model.forward
    resolved, forwarded = 0, 0

    def resolve(handoff):
        nonlocal resolved
        resolved += 1
        if resolved == 1:
            writing.set()
            assert next_forward.wait(10), "Writer blocked the next forward"
        return original_resolve(handoff)

    def forward(images):
        nonlocal forwarded
        forwarded += 1
        if forwarded == 2:
            assert writing.wait(10)
            next_forward.set()
        return original_forward(images)

    monkeypatch.setattr(PredictionHandoff, "resolve", resolve)
    monkeypatch.setattr(model, "forward", forward)
    wsi.segment_wsi(model, source, tmp_path / "mask.tif", patch_size=16, effective_size=8)
    assert resolved == forwarded and resolved > 1


def test_failed_publication_resumes_without_forward(small_model, tmp_path, monkeypatch):
    model = small_model()
    source, target = tmp_path / "image.png", tmp_path / "mask.tif"
    image(source)
    prepared = wsi.prepare_wsi(model, source, target, patch_size=16, effective_size=8)
    original_write = wsi.tifffile.imwrite

    def fail(*args, **kwargs):
        raise OSError("TIFF publication failed")

    monkeypatch.setattr(wsi.tifffile, "imwrite", fail)
    with pytest.raises(OSError, match="publication failed"):
        prepared.publish()
    assert not target.exists()
    with portalocker.Lock(prepared.work / "writer.lock", timeout=0):
        pass
    monkeypatch.setattr(wsi.tifffile, "imwrite", original_write)

    def forbidden(*args):
        raise AssertionError("Completed Zarr mask must not be inferred again")

    monkeypatch.setattr(model, "forward", forbidden)
    wsi.segment_wsi(model, source, target, patch_size=16, effective_size=8)
    assert target.exists() and not prepared.work.exists()


def test_writer_failure_releases_lock_and_preserves_resume(small_model, tmp_path, monkeypatch):
    model = small_model()
    source, target = tmp_path / "image.png", tmp_path / "mask.tif"
    image(source)
    resolve = PredictionHandoff.resolve

    def fail(handoff):
        raise OSError("Zarr handoff failed")

    monkeypatch.setattr(PredictionHandoff, "resolve", fail)
    with pytest.raises(BackgroundPipelineError):
        wsi.segment_wsi(model, source, target, patch_size=16, effective_size=8)
    assert not target.exists()
    with portalocker.Lock(target.with_name(target.name + ".work") / "writer.lock", timeout=0):
        pass
    monkeypatch.setattr(PredictionHandoff, "resolve", resolve)
    wsi.segment_wsi(model, source, target, patch_size=16, effective_size=8)
    assert target.exists()


def test_failed_previous_publication_releases_unaccepted_slide(small_model, tmp_path, monkeypatch):
    model = small_model()
    cfg = TrainConfig(
        model=model.config, data=DataConfig(class_names=["void", "a", "b"]), device="cpu"
    )
    source = tmp_path / "sources"
    source.mkdir()
    image(source / "a.png")
    image(source / "b.png")
    prepared_next = Event()
    original_publish, original_prepare = wsi.PreparedMask.publish, batch.prepare_wsi

    def publish(prepared):
        if prepared.output.stem == "a":
            try:
                assert prepared_next.wait(10)
                raise OSError("deliberate publication failure")
            finally:
                prepared.release()
        return original_publish(prepared)

    def prepare(model, path, target, **kwargs):
        prepared = original_prepare(model, path, target, **kwargs)
        if path.stem == "b":
            prepared_next.set()
        return prepared

    monkeypatch.setattr(batch, "load_model", lambda *a, **kw: model)
    monkeypatch.setattr(batch, "prepare_wsi", prepare)
    monkeypatch.setattr(wsi.PreparedMask, "publish", publish)
    output = tmp_path / "output"
    with pytest.raises(BackgroundPipelineError):
        batch.segment_sources(
            cfg, tmp_path / "unused", source, output, patch_size=16, effective_size=8
        )
    for stem in ["a", "b"]:
        with portalocker.Lock(output / f"{stem}.tif.work/writer.lock", timeout=0):
            assert not (output / f"{stem}.tif").exists()
    monkeypatch.setattr(wsi.PreparedMask, "publish", original_publish)
    monkeypatch.setattr(batch, "prepare_wsi", original_prepare)
    batch.segment_sources(cfg, tmp_path / "unused", source, output, patch_size=16, effective_size=8)
    assert all((output / f"{stem}.tif").exists() for stem in ["a", "b"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA transfer check requires a device")
def test_cuda_prefetch_and_prediction_handoff():
    """Exercise real copy streams/events without constructing pretrained encoders."""
    device = torch.device("cuda")
    inputs = [
        torch.full((2, 3, 16, 16), value, dtype=torch.float32, pin_memory=True)
        for value in range(6)
    ]
    prefetcher = BatchPrefetcher(iter(inputs), device)
    stream = torch.cuda.Stream(device=device)
    handoffs = []
    with closing(prefetcher.__iter__()) as batches, torch.inference_mode():
        for index, images in enumerate(batches):
            cores = (images[:, 0] + 1).to(torch.uint8)
            handoffs.append(transfer_predictions((2 * index, 2 * index + 1), cores, stream))
    for index, handoff in enumerate(handoffs):
        assert handoff.predictions.is_pinned()
        assert handoff.indices == (2 * index, 2 * index + 1)
        assert torch.equal(handoff.resolve(), torch.full((2, 16, 16), index + 1, dtype=torch.uint8))
