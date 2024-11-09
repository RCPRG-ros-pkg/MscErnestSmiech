import csv
import inspect
from typing import Any

import numpy
import pandas
import streamlit as st
from numpy._typing import _64Bit
from scipy.spatial import ConvexHull
from shapely import Polygon

from analysis.utils import polygon_to_floatarray
from stack import results_file, cache_dir


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


def polygon_to_tuple(polygon: Polygon | None) -> tuple[int, int, int, int] | None:
    """
    Changes shapely Polygon into tuple that's representing the corners of the bounding box.

    :param polygon:
    :return: (x, y, width, height)
    """
    if polygon is None:
        return None
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


def get_concrete_classes(cls):
    """
    Convenient function used to get all non-abstract classes that inherit from abstract parent. Used to get all trackers
    that extend Tracker abstract class.
    :param cls:
    :return:
    """
    for subclass in cls.__subclasses__():
        yield from get_concrete_classes(subclass)
        if not inspect.isabstract(subclass):
            yield subclass


def create_polygon(_points: list[int | float] | tuple[int | float]) -> Polygon | None:
    """
    Creates list of polygons given list of points. If given are 4 points they are treated as (x, y, w, h) otherwise
    it's treated as list of (x, y) points.

    :param _points: list of points
    :return: list of polygons
    """
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
    elif (len(_points) == 1 and _points[0] == 0) or _points == []:
        _polygon = None
    else:
        raise Exception("Incorrect number of points")

    return _polygon

def get_ground_truth_positions(_file_name: str) -> list[Polygon]:
    """
    Convenience functions for reading ground truth from file.
    :param _file_name:
    :return: list of shapely Polygon
    """
    with open(_file_name) as csvfile:
        # _ground_truth_pos = [[int(x) for x in y] for y in csv.reader(csvfile, delimiter='\t')]
        _ground_truth_pos = [create_polygon([abs(int(float(x))) for x in y]) for y in csv.reader(csvfile)]

    return _ground_truth_pos
