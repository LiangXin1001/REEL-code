"""Exact Qwen visual-token counting using the released processor."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable


@lru_cache(maxsize=8)
def _load_processor(model_path: str, max_pixels: int | None):
    from transformers import AutoProcessor

    kwargs = (
        {"size": {"shortest_edge": 65_536, "longest_edge": max_pixels}}
        if max_pixels is not None else {}
    )
    return AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        **kwargs,
    )


def qwen_visual_tokens(
    images: Iterable[object],
    model_path: str | Path,
    max_pixels: int | None = None,
) -> int:
    """Return visual tokens after the model's own image preprocessing.

    Qwen processors expose ``image_grid_thw`` in patch-grid coordinates.  The
    language model receives the grid after spatial merging, so each image/video
    grid contributes ``T * (H / merge_size) * (W / merge_size)`` tokens.
    """

    image_list = list(images)
    if not image_list:
        return 0
    processor = _load_processor(str(model_path), max_pixels)
    image_processor = processor.image_processor
    encoded = image_processor(images=image_list, return_tensors="pt")
    grid = encoded.get("image_grid_thw")
    if grid is None:
        raise RuntimeError("Qwen processor did not return image_grid_thw")
    merge_size = int(getattr(image_processor, "merge_size", 1))
    if merge_size <= 0:
        raise RuntimeError(f"Invalid Qwen merge_size={merge_size}")
    total = 0
    for temporal, height, width in grid.tolist():
        if height % merge_size or width % merge_size:
            raise RuntimeError(
                f"Grid {(temporal, height, width)} is not divisible by merge_size={merge_size}"
            )
        total += int(temporal) * (int(height) // merge_size) * (int(width) // merge_size)
    return total
