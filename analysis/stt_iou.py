import json

import pandas
import streamlit as st
from Geometry3D import *
from shapely import Polygon

from analysis.utils import create_polygon, polygon_to_intarray
from stack import results_file


def create_cubes(_tr: list[list[int]]):
    polygons = []
    for z, t in enumerate(_tr):
        points = []
        for x,y in zip(t[:-2:2], t[1:-2:2]):
            points.append(Point(x,y,z))
        polygons.append(points)

    cubes = []
    for i in range(len(_tr)-1):
        if not polygons[i + 1] or not polygons[i]:
            continue
        makarena = [ConvexPolygon(polygons[i])]
        makarena.append(ConvexPolygon((polygons[i][0], polygons[i][1], polygons[i+1][1], polygons[i+1][0])))
        makarena.append(ConvexPolygon((polygons[i][1], polygons[i+1][1], polygons[i+1][2], polygons[i][2])))
        makarena.append(ConvexPolygon((polygons[i][2], polygons[i+1][2], polygons[i+1][3], polygons[i][3])))
        makarena.append(ConvexPolygon((polygons[i][3], polygons[i+1][3], polygons[i+1][0], polygons[i][0])))
        makarena.append(ConvexPolygon(polygons[i+1]))
        cubes.append(ConvexPolyhedron(makarena))

    return cubes


def stt_iou(_tr: list[Polygon], _gt: list[Polygon]):
    tr_polygons = [polygon_to_intarray(polygon) if polygon is not None else [] for polygon in _tr]
    gt_polygons = [polygon_to_intarray(polygon) if polygon is not None else [] for polygon in _gt]

    tr_cubes = create_cubes(tr_polygons)
    gt_cubes = create_cubes(gt_polygons)

    tr_volume = 0
    gt_volume = 0
    intersection_volume = 0

    for tr_cube, gt_cube in zip(tr_cubes, gt_cubes):
        tr_volume += tr_cube.volume()
        gt_volume += gt_cube.volume()
        _intersection = intersection(tr_cube, gt_cube)
        if _intersection is None or type(_intersection) is not ConvexPolyhedron:
            continue
        intersection_volume += _intersection.volume()

    if tr_volume + gt_volume - intersection_volume == 0:
        return 0

    return intersection_volume / (tr_volume + gt_volume - intersection_volume)


def average_stt_iou(
        tracker: str,
        dataset: str,
        trajectories: list[list[Polygon | None]],
        groundtruths: list[list[Polygon]]):
    g = st.session_state.results
    df = g.loc[(g['tracker'] == tracker) & (g['dataset'] == dataset), ['trajectory', 'groundtruth']]

    _trajectories = trajectories + df['trajectory'].tolist()
    _groundtruths = groundtruths + df['groundtruth'].tolist()

    cumulative = 0
    count = 0

    for trajectory, groundtruth in zip(_trajectories, _groundtruths):
        try:
            cumulative += stt_iou(trajectory, groundtruth)
        except Exception as e:
            print(e)
            continue
        count += 1

    st.session_state.cache['average_stt_iou'].loc[(tracker, dataset), :] = cumulative / count


if __name__ == '__main__':
    results = pandas.read_csv(f"../{results_file}")
    trajectories = results.trajectory.apply(json.loads)
    groundtruths = results.groundtruth.apply(json.loads)
    results.trajectory = [[create_polygon(points) if points != [] else None for points in trajectory] for trajectory in trajectories]
    results.groundtruth = [[create_polygon(points) if points != [] else None for points in groundtruth] for groundtruth in groundtruths]

    df = results.loc[(results['tracker'] == 'MedianFlow') & (results['dataset'] == 'VOT Basic Test Stack') & (results['sequence'] == 'sunshade'), ['trajectory', 'groundtruth']]

    print(stt_iou(
        df['trajectory'].tolist()[0],
        df['groundtruth'].tolist()[0]
    ))

    r = Renderer()

    tr_cubes = create_cubes([polygon_to_intarray(polygon) if polygon is not None else [] for polygon in df['trajectory'].tolist()[0]])
    gt_cubes = create_cubes([polygon_to_intarray(polygon) if polygon is not None else [] for polygon in df['groundtruth'].tolist()[0]])
    for tr_cube, gt_cube in zip(tr_cubes, gt_cubes):
        r.add((tr_cube, 'r', 1))
        r.add((gt_cube, 'g', 1))
        cph2=intersection(tr_cube, gt_cube)
        if cph2 is None or type(cph2) is not ConvexPolyhedron:
            continue
        r.add((cph2, 'b', 1))

    r.show()