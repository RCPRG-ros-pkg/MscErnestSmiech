import numpy
from shapely import Polygon

from analysis.accuracy import sequence_accuracy
from analysis.utils import calculate_overlaps


def count_frames(trajectory: list[Polygon | None], groundtruth: list[Polygon | None]):
    """
    Counts the number of frames where the tracker is correct, fails, misses, hallucinates or notices an object.

    :param trajectory: Trajectory of the tracker.
    :param groundtruth: Groundtruth trajectory.
    :return: Number of frames where the tracker is correct, fails, misses, hallucinates or notices an object.
    """
    overlaps = numpy.array(calculate_overlaps(trajectory, groundtruth))

    # Tracking, Failure, Miss, Hallucination, Notice
    T, F, M, H, N = 0, 0, 0, 0, 0

    for i, (region_tr, region_gt) in enumerate(zip(trajectory, groundtruth)):
        if region_gt is None or region_gt.area == 0:
            if not region_tr:
                N += 1
            else:
                H += 1
        else:
            if overlaps[i] > 0:
                T += 1
            else:
                if not region_tr:
                    M += 1
                else:
                    F += 1

    return T, F, M, H, N


def accuracy_robustness(trajectories: list[list[Polygon | None]], groundtruths: list[list[Polygon | None]]) -> [float, float]:
    """
    Longterm multi-object accuracy-robustness measure.

    :param trajectories: list of sequences of regions predicted by the tracker.
    :param groundtruths: list of sequences of groundtruth regions.
    :return: tuple (robustness, accuracy)
    """
    accuracy = 0
    robustness = 0
    count = 0

    for trajectory, groundtruth in zip(trajectories, groundtruths):
        accuracy += sequence_accuracy(trajectory, groundtruth, True, 0.0)
        T, F, M, _, _ = count_frames(trajectory, groundtruth)

        robustness += T / (T + F + M)
        count += 1

    return [robustness / count, accuracy / count]


def quality_auxiliary(trajectory: list[Polygon | None], groundtruth: list[Polygon | None]) -> tuple[float, float, float]:
    """
    Computes the non-reported error, drift-rate error and absence-detection quality.

    :param trajectory: Trajectory of the tracker.
    :param groundtruth: Groundtruth trajectory.
    :return: tuple of (nre, dre, ad)
    """
    T, F, M, H, N = count_frames(trajectory, groundtruth)

    not_reported_error = M / (T + F + M)
    drift_rate_error = F / (T + F + M)

    if N + H > 10:
        absence_detection = N / (N + H)
    else:
        absence_detection = 0

    return not_reported_error, drift_rate_error, absence_detection


def average_quality_auxiliary(trajectories: list[list[Polygon | None]], groundtruths: list[list[Polygon | None]]) -> [float, float, float]:
    """
    Computes the average non-reported error, drift-rate error and absence-detection quality.

    :param trajectories: list of sequences of regions predicted by the tracker.
    :param groundtruths: list of sequences of groundtruth regions.
    :return: tuple of average (nre, dre, ad)
    """
    not_reported_error = 0
    drift_rate_error = 0
    absence_detection = 0
    absence_count = 0
    count = 0

    for trajectory, groundtruth in zip(trajectories, groundtruths):
        nre, dre, ad = quality_auxiliary(trajectory, groundtruth)
        not_reported_error += nre
        drift_rate_error += dre
        if ad is not None:
            absence_count += 1
            absence_detection += ad

        count += 1

    if absence_count > 0:
        absence_detection /= absence_count

    return [not_reported_error / count, drift_rate_error / count, absence_detection]
