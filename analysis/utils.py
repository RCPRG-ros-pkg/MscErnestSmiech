import numpy
from shapely import Polygon


def polygon_to_floatarray(polygon: Polygon) -> list[float]:
    _xx, _yy = polygon.exterior.coords.xy
    _coords = numpy.array(tuple(zip(_xx, _yy))).astype(float)

    return [x for xs in _coords for x in xs]


def polygon_to_intarray(polygon: Polygon) -> list[int]:
    _xx, _yy = polygon.exterior.coords.xy
    _coords = numpy.array(tuple(zip(_xx, _yy))).astype(int)

    return [x for xs in _coords for x in xs]


def calculate_overlap(first: Polygon | None, second: Polygon | None) -> float:
    if (first is None or first.area == 0.0) and (second is None or second.area == 0.0):
        return 1.0
    elif (first is None or first.area == 0.0) or (second is None or second.area == 0.0):
        return 0.0
    intersect = first.intersection(second).area
    union = first.union(second).area
    return intersect / union


def calculate_overlaps(trajectory: list[Polygon | None], groundtruth: list[Polygon | None]) -> list[float | None]:
    """
    Calculate the overlap between two lists of regions.

    :param trajectory: Trajectory of the tracker.
    :param groundtruth: Groundtruth trajectory.
    :return: list of floats with the overlap between the two regions. Note that overlap is one by definition if both regions are empty.
    :raises: Exception: if the lists are not of the same size
    """
    if not len(trajectory) == len(groundtruth):
        raise Exception("List not of the same size {} != {}".format(len(trajectory), len(groundtruth)))
    return [calculate_overlap(pairs[0], pairs[1]) for i, pairs in enumerate(zip(trajectory, groundtruth))]
