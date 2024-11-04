import numpy
from shapely import Polygon

from analysis.utils import calculate_overlaps


def gather_time_in_overlaps(
        times: list[int],
        trajectory: list[Polygon | None],
        groundtruth: list[Polygon],
        ignore_invisible: bool = False,
        threshold: float = -1
) -> numpy.ndarray:
    """
    Gather time in overlaping regions.

    :param trajectory: List of regions predicted by the tracker.
    :param groundtruth: List of groundtruth regions.
    :param ignore_invisible: Ignore invisible regions in the groundtruth.
    :param threshold: Minimum overlap to consider.
    :return: np.ndarray: List of times.
    """
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
    """
    Gather average time of overlaping frames.

    :param times: list of times
    :param trajectory: List of regions predicted by the tracker.
    :param groundtruth: List of groundtruth regions.
    :param ignore_invisible: Ignore invisible regions in the groundtruth.
    :param threshold: Minimum overlap to consider.
    :return: average time
    """
    cummulative = 0
    overlaps = gather_time_in_overlaps(times, trajectory, groundtruth, ignore_invisible, threshold)

    if overlaps.size > 0:
        cummulative += numpy.mean(overlaps)

    return cummulative


def average_time(
        times: list[list[int]],
        trajectories: list[list[Polygon | None]],
        groundtruths: list[list[Polygon]]
) -> float:
    """
    Gather average times from all sequences.

    :param times: list of times of sequences.
    :param trajectories: list of sequences of regions predicted by the tracker.
    :param groundtruths: list of sequences of groundtruth regions.
    :return: average times.
    """
    quality = 0
    count = 0

    for time, trajectory, groundtruth in zip(times, trajectories, groundtruths):
        quality += sequence_time(time, trajectory, groundtruth)
        count += 1

    return quality / count


def count_time_frames(
        times: list[int],
        trajectory: list[Polygon | None],
        groundtruth: list[Polygon]
) -> tuple[[], [], [], [], []]:
    """
    gathers times when the tracker is correct, fails, misses, hallucinates or notices an object.

    :param trajectory: Trajectory of the tracker.
    :param groundtruth: Groundtruth trajectory.
    :return: Times when the tracker is correct, fails, misses, hallucinates or notices an object.
    """
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
    """
    Computes the times spent in robustness, non-reported error, drift-rate error and absence-detection quality.

    :param trajectory: Trajectory of the tracker.
    :param groundtruth: Groundtruth trajectory.
    :return: tuple of times of (robustness, nre, dre, ad)
    """
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
        times: list[list[int]],
        trajectories: list[list[Polygon | None]],
        groundtruths: list[list[Polygon]]
) -> [float, float, float, float]:
    """
    Computes the average times spent in robustness, non-reported error, drift-rate error and absence-detection quality.

    :param trajectories: list of sequences of regions predicted by the tracker.
    :param groundtruths: list of sequences of groundtruth regions.
    :return: tuple of average times of (robustness, nre, dre, ad)
    """
    not_reported_error = 0
    drift_rate_error = 0
    absence_detection = 0
    absence_count = 0
    robustness = 0
    count = 0

    for time, trajectory, groundtruth in zip(times, trajectories, groundtruths):
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

    return [robustness / count, not_reported_error / count, drift_rate_error / count, absence_detection]
