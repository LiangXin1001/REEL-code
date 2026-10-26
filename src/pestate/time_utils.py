from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Window:
    start: float
    end: float

    @property
    def center(self) -> float:
        return (self.start + self.end) / 2.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def normalize_windows(raw_windows: Iterable[Iterable[float]], duration: float) -> list[Window]:
    windows: list[Window] = []
    for raw in raw_windows or []:
        vals = list(raw)
        if len(vals) < 2:
            continue
        start = max(0.0, min(float(vals[0]), duration))
        end = max(0.0, min(float(vals[1]), duration))
        if end < start:
            start, end = end, start
        if end > start:
            windows.append(Window(start=start, end=end))
    return windows


def frame_indices(total_frames: int, num_frames: int, sampling: str = "uniform") -> list[int]:
    if total_frames <= 0:
        return []
    if num_frames <= 1:
        return [total_frames // 2]
    if sampling == "uniform":
        return [round(i * (total_frames - 1) / (num_frames - 1)) for i in range(num_frames)]
    if sampling == "offset":
        step = total_frames / num_frames
        return [
            min(total_frames - 1, max(0, round((i + 0.5) * step - 0.5)))
            for i in range(num_frames)
        ]
    if sampling in {"end", "terminal_raw"}:
        start = max(0, total_frames - num_frames)
        return list(range(start, total_frames))
    raise ValueError(f"unknown sampling mode: {sampling}")


def index_to_time(index: int, total_frames: int, duration: float) -> float:
    if total_frames <= 1:
        return duration / 2.0
    return float(index) * float(duration) / float(total_frames - 1)


def selected_times(total_frames: int, duration: float, num_frames: int, sampling: str = "uniform") -> list[float]:
    if sampling == "end_region":
        return end_region_times(duration, num_frames)
    return [index_to_time(i, total_frames, duration) for i in frame_indices(total_frames, num_frames, sampling)]


def end_region_times(duration: float, num_frames: int, end_fraction: float = 0.2) -> list[float]:
    duration = float(duration)
    if duration <= 0 or num_frames <= 0:
        return []
    start = duration * max(0.0, min(1.0, 1.0 - float(end_fraction)))
    if num_frames == 1:
        return [(start + duration) / 2.0]
    return [start + i * (duration - start) / (num_frames - 1) for i in range(num_frames)]


def point_cells(times: Iterable[float], duration: float, single_radius_s: float = 0.5) -> list[Window]:
    selected = sorted(set(float(t) for t in times))
    if not selected:
        return []
    duration = float(duration)
    if len(selected) == 1:
        time_s = selected[0]
        return [
            Window(
                max(0.0, time_s - single_radius_s),
                min(duration, time_s + single_radius_s),
            )
        ]

    cells: list[Window] = []
    for idx, time_s in enumerate(selected):
        if idx == 0:
            start = time_s - (selected[1] - time_s) / 2.0
        else:
            start = (selected[idx - 1] + time_s) / 2.0
        if idx == len(selected) - 1:
            end = time_s + (time_s - selected[idx - 1]) / 2.0
        else:
            end = (time_s + selected[idx + 1]) / 2.0
        cells.append(Window(max(0.0, start), min(duration, end)))
    return cells


def point_radius_windows(times: Iterable[float], radius_s: float, duration: float) -> list[Window]:
    windows: list[Window] = []
    for time_s in sorted(set(float(t) for t in times)):
        start = max(0.0, time_s - radius_s)
        end = min(float(duration), time_s + radius_s)
        if end > start:
            windows.append(Window(start, end))
    return windows


def point_hits_window(time_s: float, window: Window, tolerance: float = 0.0) -> bool:
    return (window.start - tolerance) <= time_s <= (window.end + tolerance)


def interval_intersection(a: Window, b: Window) -> float:
    return max(0.0, min(a.end, b.end) - max(a.start, b.start))


def temporal_iou(pred: Window, gold: Window) -> float:
    intersection = interval_intersection(pred, gold)
    union = pred.duration + gold.duration - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def temporal_iop(pred: Window, gold: Window) -> float:
    if pred.duration <= 0:
        return 1.0 if point_hits_window(pred.center, gold) else 0.0
    return interval_intersection(pred, gold) / pred.duration


def max_temporal_overlap(preds: Iterable[Window], golds: Iterable[Window]) -> tuple[float, float]:
    pred_list = list(preds)
    gold_list = list(golds)
    if not pred_list or not gold_list:
        return 0.0, 0.0
    max_iou = 0.0
    max_iop = 0.0
    for pred in pred_list:
        for gold in gold_list:
            max_iou = max(max_iou, temporal_iou(pred, gold))
            max_iop = max(max_iop, temporal_iop(pred, gold))
    return max_iou, max_iop


def interval_window_hit_rate(
    preds: Iterable[Window],
    golds: list[Window],
    threshold: float,
    metric: str = "iop",
) -> float:
    if not golds:
        return 0.0
    pred_list = list(preds)
    if not pred_list:
        return 0.0
    hits = 0
    for gold in golds:
        score = 0.0
        for pred in pred_list:
            if metric == "iop":
                score = max(score, temporal_iop(pred, gold))
            elif metric == "iou":
                score = max(score, temporal_iou(pred, gold))
            else:
                raise ValueError(f"unknown temporal overlap metric: {metric}")
        if score >= threshold:
            hits += 1
    return hits / len(golds)


def window_hit_rate(times: Iterable[float], windows: list[Window], tolerance: float = 0.0) -> float:
    if not windows:
        return 0.0
    selected = list(times)
    hits = 0
    for window in windows:
        if any(point_hits_window(t, window, tolerance=tolerance) for t in selected):
            hits += 1
    return hits / len(windows)


def selected_point_precision(times: Iterable[float], windows: list[Window], tolerance: float = 0.0) -> float:
    selected = list(times)
    if not selected:
        return 0.0
    hits = 0
    for t in selected:
        if any(point_hits_window(t, window, tolerance=tolerance) for window in windows):
            hits += 1
    return hits / len(selected)


def any_window_hit(times: Iterable[float], windows: list[Window], tolerance: float = 0.0) -> bool:
    return window_hit_rate(times, windows, tolerance=tolerance) > 0


def all_windows_hit(times: Iterable[float], windows: list[Window], tolerance: float = 0.0) -> bool:
    return bool(windows) and window_hit_rate(times, windows, tolerance=tolerance) >= 1.0


def nearest_window_distance(time_s: float, windows: list[Window]) -> float:
    if not windows:
        return 0.0
    distances = []
    for w in windows:
        if w.start <= time_s <= w.end:
            distances.append(0.0)
        else:
            distances.append(min(abs(time_s - w.start), abs(time_s - w.end)))
    return min(distances)


def mean_nearest_window_distance(times: Iterable[float], windows: list[Window]) -> float:
    selected = list(times)
    if not selected or not windows:
        return 0.0
    return sum(nearest_window_distance(t, windows) for t in selected) / len(selected)


def relative_position_bucket(center_s: float, duration: float) -> str:
    if duration <= 0:
        return "unknown"
    rel = center_s / duration
    if rel < 1 / 3:
        return "early"
    if rel < 2 / 3:
        return "middle"
    if rel < 0.85:
        return "late"
    return "end"


def windows_to_dicts(windows: list[Window]) -> list[dict]:
    return [{"start": w.start, "end": w.end, "center": w.center, "duration": w.duration} for w in windows]
