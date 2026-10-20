"""CLI file/directory inference orchestration over the single-WSI backend.

Consumes TrainConfig and delegates integer masks to wsi and previews to overlays.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from mammoth.core import BoundedBackgroundPipeline

from tisam.checkpointing.weights import load_model
from tisam.config.run import TrainConfig
from tisam.data.wsi_dataset import read_segmentation_source_shape
from tisam.inference.overlays import overlay_palette, write_overlays
from tisam.inference.transfers import observe_pipeline
from tisam.inference.wsi import PreparedMask, prepare_wsi


@dataclass
class SlidePublication:
    """CPU-only slide handoff owned by the bounded publication worker."""

    source: Path
    target: Path
    prepared: Optional[PreparedMask]
    palette: Optional[np.ndarray]
    check_classify: bool
    mini_check_classify: bool

    def publish(self) -> Path:
        """Publish a prepared TIFF followed by independently requested JPGs."""
        if self.prepared is not None:
            self.prepared.publish()
        if self.palette is not None:
            write_overlays(
                self.source,
                self.target,
                self.palette,
                check_classify=self.check_classify,
                mini_check_classify=self.mini_check_classify,
            )
        return self.target

    def release(self) -> None:
        """Keep scratch resumable if the pipeline rejects this handoff."""
        if self.prepared is not None:
            self.prepared.release()


def segment_sources(
    cfg: TrainConfig,
    checkpoint: Path,
    source: Path,
    output: Path,
    *,
    check_classify: bool = False,
    mini_check_classify: bool = False,
    patch_size: int = 1024,
    effective_size: int = 800,
    resume: bool = True,
    trusted: bool = False,
) -> list[Path]:
    """Serve CLI segmentation, repairing requested previews from existing masks.

    With resume enabled, existing masks are treated as user-selected completed
    results. They are never overwritten or claimed to match a new checkpoint.
    """
    suffixes = {".tif", ".tiff", ".png"}
    sources = (
        sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in suffixes)
        if source.is_dir()
        else [source]
    )
    if not sources:
        raise ValueError("No TIFF/PNG sources found")
    if len({p.stem.casefold() for p in sources}) != len(sources):
        raise ValueError("Source filenames have colliding output stems")
    if output.exists() and not output.is_dir():
        raise ValueError("output must be a directory")
    if effective_size <= 0 or patch_size < effective_size or (patch_size - effective_size) % 2:
        raise ValueError("patch_size-effective_size must be nonnegative and even")
    palette = overlay_palette(cfg.data) if check_classify or mini_check_classify else None
    outputs = [output / f"{p.stem}.tif" for p in sources]
    for path, target in zip(sources, outputs):
        read_segmentation_source_shape(path)
        if target.resolve() == path.resolve():
            raise ValueError("Output mask must not replace its source")
        if target.exists() and not resume:
            raise FileExistsError(target)
    model = None
    with BoundedBackgroundPipeline[SlidePublication, Path](
        lambda slide: slide.publish(), max_pending=1, thread_name_prefix="tisam-publication"
    ) as publications:
        for path, target in zip(sources, outputs):
            observe_pipeline(publications)
            prepared = None
            if not target.exists():
                if model is None:
                    model = load_model(
                        checkpoint,
                        config=cfg.model,
                        device=cfg.device,
                        sam3_checkpoint=cfg.sam3_checkpoint,
                        trusted=trusted,
                    )
                prepared = prepare_wsi(
                    model,
                    path,
                    target,
                    patch_size=patch_size,
                    effective_size=effective_size,
                    batch_size=cfg.batch_size,
                    num_workers=cfg.num_workers,
                    mean=cfg.data.mean,
                    std=cfg.data.std,
                    resume=resume,
                )
            slide = SlidePublication(
                path, target, prepared, palette, check_classify, mini_check_classify
            )
            try:
                publications.submit(slide)
                observe_pipeline(publications)
            except BaseException:
                if not publications.owns_input(slide):
                    slide.release()
                raise
    return outputs
