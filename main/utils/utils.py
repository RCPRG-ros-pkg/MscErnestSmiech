from typing import Any

import numpy
from numpy._typing import _64Bit
from scipy.spatial import ConvexHull
from shapely import Polygon


# https://gis.stackexchange.com/questions/22895/finding-minimum-area-rectangle-for-given-points/169633#169633
def minimum_bounding_rectangle(points) -> numpy.ndarray[Any, numpy.dtype[numpy.floating[_64Bit] | numpy.float_]] | None:
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.

    :param points: a nx2 matrix of coordinates
    :rval: a nx2 matrix of coordinates
    """
    if not all([int(j) != 0 for i in points for j in i]):
        return None

    pi2 = numpy.pi / 2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = numpy.zeros((len(hull_points) - 1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = numpy.zeros((len(edges)))
    angles = numpy.arctan2(edges[:, 1], edges[:, 0])

    angles = numpy.abs(numpy.mod(angles, pi2))
    angles = numpy.unique(angles)

    # find rotation matrices
    # XXX both work
    rotations = numpy.vstack([
        numpy.cos(angles),
        numpy.cos(angles - pi2),
        numpy.cos(angles + pi2),
        numpy.cos(angles)]).T
    #     rotations = numpy.vstack([
    #         numpy.cos(angles),
    #         -numpy.sin(angles),
    #         numpy.sin(angles),
    #         numpy.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = numpy.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = numpy.nanmin(rot_points[:, 0], axis=1)
    max_x = numpy.nanmax(rot_points[:, 0], axis=1)
    min_y = numpy.nanmin(rot_points[:, 1], axis=1)
    max_y = numpy.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = numpy.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = numpy.zeros((4, 2))
    rval[0] = numpy.dot([x1, y2], r)
    rval[1] = numpy.dot([x2, y2], r)
    rval[2] = numpy.dot([x2, y1], r)
    rval[3] = numpy.dot([x1, y1], r)

    return rval


def polygon_to_tuple(polygon: Polygon) -> tuple[int, int, int, int] | None:
    _xx, _yy = polygon.exterior.coords.xy
    _coords = tuple(zip(_xx, _yy))
    _l = minimum_bounding_rectangle(numpy.array(_coords).astype(int))

    if _l is None:
        return None

    _max_x = max(0, int(max([i[0] for i in _l])))
    _max_y = max(0, int(max([i[1] for i in _l])))
    _min_x = max(0, int(min([i[0] for i in _l])))
    _min_y = max(0, int(min([i[1] for i in _l])))

    return _min_x, _min_y, _max_x - _min_x, _max_y - _min_y
