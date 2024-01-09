import numpy
import pandas
from shapely import Polygon


def polygon_to_array(polygon: Polygon) -> list[float]:
    _xx, _yy = polygon.exterior.coords.xy
    _coords = numpy.array(tuple(zip(_xx, _yy))).astype(float) # todo unnecessary?

    return [x for xs in _coords for x in xs]


def calculate_overlap(first: Polygon | None, second: Polygon | None) -> float:
    if first is None or first.area == 0.0 or second is None or second.area == 0.0:
        return 0.0
    intersect = first.intersection(second).area
    union = first.union(second).area
    return intersect / union


def calculate_overlaps(trajectory: list[Polygon | None], groundtruth: list[Polygon]) -> list[float | None]:
    """ Calculate the overlap between two lists of regions. The function first rasterizes both regions to 2-D binary masks and calculates overlap between them

    Args:
        trajectory: first list of regions
        groundtruth: second list of regions

    Returns:
        list of floats with the overlap between the two regions. Note that overlap is one by definition if both regions are empty.

    Raises:
        RegionException: if the lists are not of the same size
    """
    if not len(trajectory) == len(groundtruth):
        raise Exception("List not of the same size {} != {}".format(len(trajectory), len(groundtruth)))
    return [calculate_overlap(pairs[0], pairs[1]) for i, pairs in enumerate(zip(trajectory, groundtruth))]


def create_polygon(_points: list[int | float] | tuple[int | float]) -> Polygon:
    if len(_points) == 4:
        _x, _y, _width, _height = _points
        _polygon = Polygon([
            (_x, _y),
            (_x + _width, _y),
            (_x + _width, _y + _height),
            (_x, _y + _height)
        ])
    elif len(_points) >= 6:
        _polygon = Polygon(zip(_points[::2], _points[1::2]))
    else:
        raise Exception("Incorrect number of points")

    return _polygon
