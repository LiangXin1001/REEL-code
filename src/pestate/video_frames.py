from __future__ import annotations

from pathlib import Path

import cv2

from .time_utils import index_to_time


def time_to_frame_index(time_s: float, total_frames: int, duration: float) -> int:
    if total_frames <= 1 or duration <= 0:
        return 0
    idx = round(float(time_s) * float(total_frames - 1) / float(duration))
    return max(0, min(total_frames - 1, idx))


def extract_frames_at_times(
    video_path: str,
    times_s: list[float],
    out_dir: str | Path,
    total_frames: int,
    duration: float,
    max_side: int | None = None,
) -> list[str]:
    if max_side is not None and max_side <= 0:
        raise ValueError(f"max_side must be positive, got {max_side}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    paths: list[str] = []
    for j, time_s in enumerate(times_s):
        frame_idx = time_to_frame_index(time_s, total_frames, duration)
        frame_path = out / f"frame_{j:03d}_{frame_idx}_{time_s:.3f}s.jpg"
        if frame_path.exists():
            paths.append(str(frame_path))
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        if max_side is not None:
            height, width = frame.shape[:2]
            longest = max(height, width)
            if longest > max_side:
                scale = max_side / longest
                resized_width = max(1, round(width * scale))
                resized_height = max(1, round(height * scale))
                frame = cv2.resize(
                    frame,
                    (resized_width, resized_height),
                    interpolation=cv2.INTER_AREA,
                )
        if not cv2.imwrite(str(frame_path), frame):
            raise RuntimeError(f"Could not write extracted frame: {frame_path}")
        paths.append(str(frame_path))

    cap.release()
    if not paths:
        raise ValueError(f"No frames extracted from {video_path}")
    return paths


def uniform_times_in_window(start: float, end: float, count: int) -> list[float]:
    if count <= 0:
        return []
    if count == 1:
        return [(start + end) / 2.0]
    if end <= start:
        return [start] * count
    return [start + (i + 0.5) * (end - start) / count for i in range(count)]


def oracle_window_times(windows: list[dict], budget: int) -> list[float]:
    if budget <= 0 or not windows:
        return []
    cleaned = [
        {"start": float(w["start"]), "end": float(w["end"]), "duration": float(w["end"]) - float(w["start"])}
        for w in windows
        if float(w["end"]) > float(w["start"])
    ]
    if not cleaned:
        return []
    total_duration = sum(w["duration"] for w in cleaned)
    allocations = []
    remaining = budget
    for idx, w in enumerate(cleaned):
        if idx == len(cleaned) - 1:
            n = remaining
        else:
            raw = budget * w["duration"] / total_duration if total_duration > 0 else budget / len(cleaned)
            n = max(1, int(round(raw)))
            n = min(n, remaining - (len(cleaned) - idx - 1))
        allocations.append(n)
        remaining -= n
    times: list[float] = []
    for w, n in zip(cleaned, allocations):
        times.extend(uniform_times_in_window(w["start"], w["end"], n))
    return sorted(times[:budget])
