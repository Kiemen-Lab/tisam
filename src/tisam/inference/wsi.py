"""Single-process, resumable WSI inference with bounded-memory TIFF output."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import portalocker
import tifffile
import torch
import zarr
from mammoth.core import BoundedBackgroundPipeline
from torch.utils.data import DataLoader, Subset

from tisam.data.wsi_dataset import WSIDataset, read_segmentation_source_shape
from tisam.inference.transfers import (
    BatchPrefetcher,
    PredictionHandoff,
    observe_pipeline,
    transfer_predictions,
)
from tisam.model.segmentor import TiSAM


def model_fingerprint(model: TiSAM) -> str:
    """Identify exact weights before reusing previously computed WSI patches."""
    digest = hashlib.sha256(model.config.model_dump_json().encode())
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy())
    return digest.hexdigest()


@dataclass
class PreparedMask:
    """Own a completed Zarr mask and its lock until CPU publication finishes.

    prepare_wsi returns this to batch's Mammoth worker; no model or CUDA tensors
    cross this handoff. Failed publication retains scratch for a later resume.
    """

    work: Path
    output: Path
    shape: tuple[int, int]
    dtype: str
    lock: portalocker.Lock

    def release(self) -> None:
        """Release exclusive scratch ownership if a handoff is not accepted."""
        self.lock.release()

    def publish(self) -> Path:
        """Fuse Zarr chunks into a TIFF and atomically publish the final mask."""
        try:
            group = zarr.open_group(self.work / "patches.zarr", mode="r")
            mask = group["mask"]
            height, width = self.shape

            def tiles():
                for y in range(0, height, 256):
                    for x in range(0, width, 256):
                        yield np.asarray(mask[y : min(y + 256, height), x : min(x + 256, width)])

            temporary = self.work / "mask.tif"
            tifffile.imwrite(
                temporary,
                data=tiles(),
                shape=self.shape,
                dtype=self.dtype,
                tile=(256, 256),
                compression="zstd",
                photometric="minisblack",
                bigtiff=True,
            )
            os.link(temporary, self.output)
        finally:
            self.release()
        shutil.rmtree(self.work)
        return self.output


def segment_wsi(
    model: TiSAM,
    source: str | Path,
    output: str | Path,
    *,
    patch_size: int = 1024,
    effective_size: int = 800,
    batch_size: int = 1,
    num_workers: int = 0,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    resume: bool = True,
) -> Path:
    """Infer and publish one mask synchronously; batch uses prepare_wsi to overlap slides."""
    prepared = prepare_wsi(
        model,
        source,
        output,
        patch_size=patch_size,
        effective_size=effective_size,
        batch_size=batch_size,
        num_workers=num_workers,
        mean=mean,
        std=std,
        resume=resume,
    )
    return prepared.publish()


def prepare_wsi(
    model: TiSAM,
    source: str | Path,
    output: str | Path,
    *,
    patch_size: int = 1024,
    effective_size: int = 800,
    batch_size: int = 1,
    num_workers: int = 0,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    resume: bool = True,
) -> PreparedMask:
    """Prepare a completed Zarr mask for segment_wsi or batch publication.

    A sibling .work directory contains a chunked class mask and completion bitmap.
    Resume checks input identity, preprocessing, geometry and model weights.
    The caller owns the returned lock and must publish or release the handoff.
    Existing final outputs are never overwritten; failures retain resume scratch.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or output.suffix.lower() not in (".tif", ".tiff"):
        raise ValueError("output must be a distinct TIFF path")
    if effective_size <= 0 or patch_size < effective_size or (patch_size - effective_size) % 2:
        raise ValueError("patch_size-effective_size must be nonnegative and even")
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    if any(v <= 0 for v in std):
        raise ValueError("std must be positive")
    if output.exists():
        raise FileExistsError(output)
    height, width, _ = read_segmentation_source_shape(source)
    stat = source.stat()
    identity = {
        "schema": 1,
        "source": str(source),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "shape": [height, width],
        "model": model_fingerprint(model),
        "patch_size": patch_size,
        "effective_size": effective_size,
        "mean": list(mean),
        "std": list(std),
    }
    work = output.with_name(output.name + ".work")
    output.parent.mkdir(parents=True, exist_ok=True)
    if work.is_symlink():
        raise ValueError("WSI scratch directory cannot be a symlink")
    if work.exists():
        if not resume:
            raise FileExistsError(f"Scratch exists; resume it or choose another output: {work}")
        saved = json.loads((work / "identity.json").read_text())
        if saved != identity:
            raise ValueError("WSI resume identity differs; choose another output")
    else:
        work.mkdir()
        (work / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    # An exclusive lock prevents concurrent writers from corrupting recovery state.
    lock = work / "writer.lock"
    writer_lock = portalocker.Lock(lock, mode="a+b", timeout=0)
    writer_lock.acquire()
    was_training = model.training
    handed_off = False
    try:
        with WSIDataset(source, mean, std, patch_size, effective_size) as dataset:
            format_key = "zarr_version" if zarr.__version__.split(".")[0] == "2" else "zarr_format"
            format_options: dict[str, Any] = {format_key: 2}
            group: Any = zarr.open_group(work / "patches.zarr", mode="a", **format_options)
            require_array = getattr(group, "require_array", None) or group.require_dataset
            dtype = "uint8" if model.total_classes <= 256 else "uint16"
            mask = require_array(
                "mask",
                shape=(height, width),
                chunks=(effective_size, effective_size),
                dtype=dtype,
                fill_value=0,
            )
            completed = require_array(
                "completed", shape=(len(dataset),), chunks=(1,), dtype="bool", fill_value=False
            )
            pending = np.flatnonzero(~np.asarray(completed[:], dtype=bool)).tolist()
            device = next(model.parameters()).device
            loader = DataLoader(
                Subset(dataset, pending),
                batch_size=batch_size,
                num_workers=num_workers,
                persistent_workers=False,
                pin_memory=device.type == "cuda",
                prefetch_factor=2 if num_workers else None,
                multiprocessing_context="spawn" if num_workers else None,
            )
            padding = (patch_size - effective_size) // 2
            model_input_hw = getattr(model, "input_hw", None)
            model.eval()
            offset = 0

            def write_batch(handoff: PredictionHandoff) -> None:
                cores = handoff.resolve().numpy().astype(dtype, copy=False)
                for index, core in zip(handoff.indices, cores):
                    y, x = map(int, dataset.positions[index])
                    h, w = min(effective_size, height - y), min(effective_size, width - x)
                    mask[y : y + h, x : x + w] = core[:h, :w]
                    completed[index] = True

            d2h = torch.cuda.Stream(device=device) if device.type == "cuda" else None
            prefetcher = BatchPrefetcher(iter(loader), device)
            with (
                closing(prefetcher.__iter__()) as batches,
                BoundedBackgroundPipeline(
                    write_batch, max_pending=1, thread_name_prefix="tisam-zarr"
                ) as writes,
                torch.inference_mode(),
            ):
                for images in batches:
                    observe_pipeline(writes)
                    if model_input_hw is not None and tuple(images.shape[-2:]) != tuple(
                        model_input_hw
                    ):
                        images = torch.nn.functional.interpolate(
                            images,
                            size=model_input_hw,
                            mode="bilinear",
                            align_corners=False,
                        )
                    logits = model(images)
                    logits = torch.nn.functional.interpolate(
                        logits, size=(patch_size, patch_size), mode="bilinear", align_corners=False
                    )
                    cores = (
                        logits[
                            :,
                            :,
                            padding : padding + effective_size,
                            padding : padding + effective_size,
                        ]
                        .argmax(1)
                        .to(torch.uint8 if dtype == "uint8" else torch.int32)
                    )
                    handoff = transfer_predictions(
                        tuple(pending[offset : offset + len(cores)]), cores, d2h
                    )
                    try:
                        writes.submit(handoff)
                        observe_pipeline(writes)
                    except BaseException:
                        if not writes.owns_input(handoff):
                            handoff.resolve()
                        raise
                    offset += len(cores)
            if not bool(np.all(completed[:])):
                raise RuntimeError("WSI publication requires every patch")
            prepared = PreparedMask(work, output, (height, width), dtype, writer_lock)
        handed_off = True
        return prepared
    finally:
        model.train(was_training)
        if not handed_off:
            writer_lock.release()
