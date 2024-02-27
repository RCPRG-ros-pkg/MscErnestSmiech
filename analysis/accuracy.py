import numpy
from shapely import Polygon

from analysis.utils import calculate_overlaps


def gather_overlaps( # jakość - default, dokładność - custom
        trajectory: list[Polygon | None],
        groundtruth: list[Polygon],
        ignore_invisible: bool = False,
        threshold: float = -1
) -> numpy.ndarray:
    overlaps = numpy.array(calculate_overlaps(trajectory, groundtruth))
    mask = numpy.ones(len(overlaps), dtype=bool)

    for i, (region_tr, region_gt) in enumerate(zip(trajectory, groundtruth)):
        if ignore_invisible and region_gt.area == 0.0:
            mask[i] = False
        elif region_tr is None:
            mask[i] = False
        elif overlaps[i] <= threshold:
            mask[i] = False

    return overlaps[mask]


def success_plot(trajectory: list[Polygon | None], groundtruth: list[Polygon]) -> list[tuple[float, float]]:
    axis_x = numpy.linspace(0, 1, 100)
    axis_y = numpy.zeros_like(axis_x)

    overlaps = gather_overlaps(trajectory, groundtruth)
    if overlaps.size > 0:
        for i, threshold in enumerate(axis_x):
            if threshold == 1:
                # Nicer handling of the edge case
                axis_y[i] += numpy.sum(overlaps >= threshold) / len(overlaps)
            else:
                axis_y[i] += numpy.sum(overlaps > threshold) / len(overlaps)

    return [(x, y) for x, y in zip(axis_x, axis_y)]


def average_success_plot(trajectories: list[list[Polygon | None]], groundtruths: list[list[Polygon]]) -> (numpy.ndarray, numpy.ndarray):
    axis_x = numpy.linspace(0, 1, 100)
    axis_y = numpy.zeros_like(axis_x)
    count = 0

    for trajectory, groundtruth in zip(trajectories, groundtruths):
        for j, (_, y) in enumerate(success_plot(trajectory, groundtruth)):
            axis_y[j] += y

        count += 1

    axis_y /= count

    return axis_x, axis_y


def sequence_accuracy(trajectory: list[Polygon | None], groundtruth: list[Polygon], ignore_invisible: bool = False, threshold: float = -1) -> float:
    cummulative = 0
    overlaps = gather_overlaps(trajectory, groundtruth, ignore_invisible, threshold)

    if overlaps.size > 0:
        cummulative += numpy.mean(overlaps)

    return cummulative


def average_accuracy(trajectories: list[list[Polygon | None]], groundtruths: list[list[Polygon]]) -> float:
    accuracy = 0
    count = 0

    for trajectory, groundtruth in zip(trajectories, groundtruths):
        accuracy += sequence_accuracy(trajectory, groundtruth)
        count += 1

    return accuracy / count
