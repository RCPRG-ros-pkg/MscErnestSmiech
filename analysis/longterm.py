import numpy
from shapely import Polygon

from analysis.accuracy import sequence_accuracy
from analysis.utils import calculate_overlaps


def count_frames(trajectory: list[Polygon | None], groundtruth: list[Polygon]):
    overlaps = numpy.array(calculate_overlaps(trajectory, groundtruth))

    # Tracking, Failure, Miss, Hallucination, Notice
    T, F, M, H, N = 0, 0, 0, 0, 0

    for i, (region_tr, region_gt) in enumerate(zip(trajectory, groundtruth)):
        if region_gt.area == 0:
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


def accuracy_robustness(trajectories: list[list[Polygon | None]], groundtruths: list[list[Polygon]]) -> [float, float]:
    accuracy = 0
    robustness = 0
    count = 0

    for trajectory, groundtruth in zip(trajectories, groundtruths):
        accuracy += sequence_accuracy(trajectory, groundtruth, True, 0.0)
        T, F, M, _, _ = count_frames(trajectory, groundtruth)

        robustness += T / (T + F + M)
        count += 1

    return [robustness / count, accuracy / count]


def quality_auxiliary(trajectory: list[Polygon | None], groundtruth: list[Polygon]) -> tuple[float, float, float]:
    T, F, M, H, N = count_frames(trajectory, groundtruth)

    not_reported_error = M / (T + F + M)
    drift_rate_error = F / (T + F + M)

    if N + H > 10:
        absence_detection = N / (N + H)
    else:
        absence_detection = 0

    return not_reported_error, drift_rate_error, absence_detection


def average_quality_auxiliary(trajectories: list[list[Polygon | None]], groundtruths: list[list[Polygon]]) -> [float, float, float]:
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
