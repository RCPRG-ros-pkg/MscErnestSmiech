import numpy
import streamlit as st
from shapely import Polygon

from analysis.utils import calculate_overlaps
from data.state_locator import StateLocator

state_locator = StateLocator()


def gather_time_in_overlaps(
        times: list[int],
        trajectory: list[Polygon | None],
        groundtruth: list[Polygon],
        ignore_invisible: bool = False,
        threshold: float = -1
) -> numpy.ndarray:
    _times = numpy.array(times)
    overlaps = numpy.array(calculate_overlaps(trajectory, groundtruth))
    mask = numpy.ones(len(overlaps), dtype=bool)

    for i, (region_tr, region_gt) in enumerate(zip(trajectory, groundtruth)):
        # Skip if groundtruth is unknown
        if ignore_invisible and region_gt.area == 0.0:
            mask[i] = False
        # Skip if predicted is initialization frame
        elif region_tr is None:
            mask[i] = False
        elif overlaps[i] <= threshold:
            mask[i] = False

    return _times[mask]


def sequence_time(
        times: list[int],
        trajectory: list[Polygon | None],
        groundtruth: list[Polygon],
        ignore_invisible: bool = False,
        threshold: float = -1
) -> float:
    cummulative = 0
    overlaps = gather_time_in_overlaps(times, trajectory, groundtruth, ignore_invisible, threshold)

    if overlaps.size > 0:
        cummulative += numpy.mean(overlaps)

    return cummulative


def average_time(
        tracker: str,
        dataset: str,
        times: list[list[int]],
        trajectories: list[list[Polygon | None]],
        groundtruths: list[list[Polygon]]
):
    g = state_locator.provide_results()
    df = g.loc[(g['tracker'] == tracker) & (g['dataset'] == dataset), ['trajectory', 'groundtruth', 'times']]

    _times = times + df['times'].tolist()
    _trajectories = trajectories + df['trajectory'].tolist()
    _groundtruths = groundtruths + df['groundtruth'].tolist()

    quality = 0
    count = 0

    for time, trajectory, groundtruth in zip(_times, _trajectories, _groundtruths):
        quality += sequence_time(time, trajectory, groundtruth)
        count += 1

    state_locator.provide_cache()['average_time'].loc[(tracker, dataset), :] = quality / count


def count_time_frames(
        times: list[int],
        trajectory: list[Polygon | None],
        groundtruth: list[Polygon]
) -> tuple[[], [], [], [], []]:
    overlaps = numpy.array(calculate_overlaps(trajectory, groundtruth))

    # Tracking, Failure, Miss, Hallucination, Notice
    T, F, M, H, N = [], [], [], [], []

    for i, (time, region_tr, region_gt) in enumerate(zip(times, trajectory, groundtruth)):
        if region_gt.area == 0:
            if not region_tr:
                N.append(time)
            else:
                H.append(time)
        else:
            if overlaps[i] > 0:
                T.append(time)
            else:
                if not region_tr:
                    M.append(time)
                else:
                    F.append(time)

    return T, F, M, H, N


def time_quality_auxiliary(
        times: list[int],
        trajectory: list[Polygon | None],
        groundtruth: list[Polygon]
) -> tuple[int, int, int, int]:
    T, F, M, H, N = count_time_frames(times, trajectory, groundtruth)

    if M:
        not_reported_error = numpy.array(M).mean()
    else:
        not_reported_error = 0
    if F:
        drift_rate_error = numpy.array(F).mean()
    else:
        drift_rate_error = 0
    if T:
        robustness = numpy.array(T).mean()
    else:
        robustness = 0

    if len(N + H) > 10:
        absence_detection = numpy.array(N).mean()
    else:
        absence_detection = 0

    return robustness, not_reported_error, drift_rate_error, absence_detection


def average_time_quality_auxiliary(
        tracker: str,
        dataset: str,
        times: list[list[int]],
        trajectories: list[list[Polygon | None]],
        groundtruths: list[list[Polygon]]
):
    g = state_locator.provide_results()
    df = g.loc[(g['tracker'] == tracker) & (g['dataset'] == dataset), ['trajectory', 'groundtruth', 'times']]
    print(g)
    print(df)

    _times = times + df['times'].tolist()
    _trajectories = trajectories + df['trajectory'].tolist()
    _groundtruths = groundtruths + df['groundtruth'].tolist()

    not_reported_error = 0
    drift_rate_error = 0
    absence_detection = 0
    absence_count = 0
    robustness = 0
    count = 0

    for time, trajectory, groundtruth in zip(_times, _trajectories, _groundtruths):
        r, nre, dre, ad = time_quality_auxiliary(time, trajectory, groundtruth)
        not_reported_error += nre
        drift_rate_error += dre
        robustness += r
        if ad is not None:
            absence_count += 1
            absence_detection += ad

        count += 1

    if absence_count > 0:
        absence_detection /= absence_count

    state_locator.provide_cache()['average_time_quality_auxiliary'].loc[(tracker, dataset), :] = [robustness / count, not_reported_error / count, drift_rate_error / count, absence_detection]
