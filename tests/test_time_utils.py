from pestate.time_utils import (
    end_region_times,
    interval_window_hit_rate,
    max_temporal_overlap,
    normalize_windows,
    point_cells,
    selected_times,
    window_hit_rate,
)


def test_uniform_selected_times_hit_middle_window():
    times = selected_times(total_frames=101, duration=10.0, num_frames=11)
    windows = normalize_windows([(4.9, 5.1)], duration=10.0)
    assert window_hit_rate(times, windows) == 1.0


def test_window_miss_without_tolerance():
    windows = normalize_windows([(4.9, 5.1)], duration=10.0)
    assert window_hit_rate([4.0, 6.0], windows) == 0.0


def test_frame_cell_overlap_recovers_nearby_sparse_frame():
    times = [0.0, 5.862, 11.724, 17.586, 23.414, 29.276, 35.138, 41.0]
    cells = point_cells(times, duration=41.0)
    gold = normalize_windows([(24.5, 27.9)], duration=41.0)
    assert window_hit_rate([29.276], gold, tolerance=0.5) == 0.0

    max_iou, max_iop = max_temporal_overlap([cells[5]], gold)
    assert max_iou > 0.0
    assert max_iop > 0.0
    assert max_iop < 0.3
    assert interval_window_hit_rate([cells[5]], gold, threshold=0.3, metric="iop") == 0.0


def test_point_cells_do_not_expand_clustered_end_frames_to_full_video():
    cells = point_cells([59.73, 59.77, 59.80, 59.97], duration=60.0)
    assert cells[0].start > 59.70
    assert cells[0].end < 59.79
    assert cells[-1].start > 59.88
    assert cells[-1].end <= 60.0


def test_single_point_cell_is_local():
    cells = point_cells([30.0], duration=60.0)
    assert cells[0].start == 29.5
    assert cells[0].end == 30.5


def test_end_region_times_sample_final_twenty_percent():
    times = end_region_times(duration=100.0, num_frames=5)
    assert times == [80.0, 85.0, 90.0, 95.0, 100.0]
