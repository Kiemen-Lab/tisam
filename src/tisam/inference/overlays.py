"""Render config-colored JPG previews from completed integer masks for batch inference."""

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import tifffile
from PIL import Image

from tisam.config.data import DataConfig
from tisam.data.wsi_dataset import open_segmentation_source


def overlay_palette(data: DataConfig) -> np.ndarray:
    """Validate the CLI's requested palette before loading model weights."""
    missing = [name for name in data.class_names if name not in data.colormap]
    if missing:
        raise ValueError(f"data.colormap is missing colors for: {', '.join(missing)}")
    return np.asarray([data.colormap[name] for name in data.class_names], dtype=np.uint8)


def write_overlays(
    source: Path,
    mask_path: Path,
    palette: np.ndarray,
    *,
    check_classify: bool = False,
    mini_check_classify: bool = False,
) -> None:
    """Publish independently requested previews; batch calls this after mask creation.

    Rendering materializes one slide and its mask. Mini previews use area
    downsampling of the blended image, not interpolated class identifiers.
    """
    directory = mask_path.parent / "check_classification"
    targets = []
    if check_classify:
        targets.append((directory / f"{mask_path.stem}.jpg", False))
    if mini_check_classify:
        targets.append((directory / "mini" / f"{mask_path.stem}.jpg", True))
    targets = [(path, mini) for path, mini in targets if not path.exists()]
    if not targets:
        return
    mask = tifffile.imread(mask_path)
    if (
        mask.ndim != 2
        or mask.dtype.kind not in "ui"
        or mask.min() < 0
        or mask.max() >= len(palette)
    ):
        raise ValueError("Mask must contain valid integer class IDs")
    levels, image = open_segmentation_source(source)
    try:
        rgb = np.asarray(image[:])
    finally:
        if levels is not None:
            levels.close()
    if rgb.shape[:2] != mask.shape:
        raise ValueError("Mask and source dimensions differ")
    overlay = cv2.addWeighted(palette[mask], 0.3, rgb, 0.7, 0)
    for path, mini in targets:
        pixels = (
            cv2.resize(
                overlay,
                (max(1, mask.shape[1] // 10), max(1, mask.shape[0] // 10)),
                interpolation=cv2.INTER_AREA,
            )
            if mini
            else overlay
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".overlay-", suffix=".jpg", dir=path.parent)
        os.close(descriptor)
        try:
            Image.fromarray(pixels).save(temporary, format="JPEG")
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
