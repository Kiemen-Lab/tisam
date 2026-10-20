"""Bounded CUDA transfers used by WSI inference and its Mammoth Zarr worker.

CPU inference follows the same handoff contract without CUDA streams or events.
"""

from __future__ import annotations

from collections.abc import Generator, Iterator
from dataclasses import dataclass
from typing import Any

import torch
from mammoth.core import BoundedBackgroundPipeline


def observe_pipeline(pipeline: BoundedBackgroundPipeline[Any, Any]) -> None:
    """Release observed results and propagate errors on the inference thread."""
    for result in pipeline.take_completed():
        pipeline.acknowledge(result.submission)
    interrupt = pipeline.take_deferred_interrupt()
    if interrupt is not None:
        raise interrupt


@dataclass
class DeviceBatch:
    """Keep host storage alive until the prefetch stream finishes reading it."""

    images: torch.Tensor
    source: torch.Tensor
    ready: torch.cuda.Event | None = None


class BatchPrefetcher:
    """Stage one next batch for wsi on a separate H2D CUDA stream."""

    def __init__(self, iterator: Iterator[torch.Tensor], device: torch.device):
        self.iterator = iterator
        self.device = device
        self.stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
        self.pending: DeviceBatch | None = None
        self.error: Exception | None = None

    def _preload(self) -> None:
        try:
            host = next(self.iterator)
        except StopIteration:
            return
        except Exception as error:
            self.error = error
            return
        if self.stream is None:
            self.pending = DeviceBatch(host.to(self.device), host)
        else:
            with torch.cuda.stream(self.stream):
                images = host.to(self.device, non_blocking=host.is_pinned())
                ready = torch.cuda.Event()
                ready.record(self.stream)
                self.pending = DeviceBatch(images, host, ready)

    def __iter__(self) -> Generator[torch.Tensor, None, None]:
        try:
            self._preload()
            while self.pending is not None or self.error is not None:
                if self.error is not None:
                    raise self.error
                current = self.pending
                assert current is not None
                self.pending = None
                self._preload()
                if current.ready is not None:
                    stream = torch.cuda.current_stream(self.device)
                    stream.wait_event(current.ready)
                    current.images.record_stream(stream)
                yield current.images
                # Keep the pinned host alive through consumption of this batch.
        finally:
            if self.stream is not None:
                self.stream.synchronize()
            self.pending = None


@dataclass
class PredictionHandoff:
    """Own predictions until a background writer resolves their D2H event."""

    indices: tuple[int, ...]
    predictions: torch.Tensor
    ready: torch.cuda.Event | None = None
    source: torch.Tensor | None = None

    def resolve(self) -> torch.Tensor:
        """Wait on the writer thread before reading host predictions."""
        if self.ready is not None:
            self.ready.synchronize()
        return self.predictions


def transfer_predictions(
    indices: tuple[int, ...], cores: torch.Tensor, stream: torch.cuda.Stream | None
) -> PredictionHandoff:
    """Stage integer cores in pinned memory; retain CUDA storage until completion."""
    if stream is None:
        return PredictionHandoff(indices, cores.cpu().contiguous())
    host = torch.empty(cores.shape, dtype=cores.dtype, device="cpu", pin_memory=True)
    producer = torch.cuda.current_stream(cores.device)
    with torch.cuda.stream(stream):
        stream.wait_stream(producer)
        host.copy_(cores, non_blocking=True)
        ready = torch.cuda.Event()
        ready.record(stream)
        cores.record_stream(stream)
    return PredictionHandoff(indices, host, ready, cores)
