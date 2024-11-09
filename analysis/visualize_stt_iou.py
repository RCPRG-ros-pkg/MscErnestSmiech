import json

import pandas
from Geometry3D import Point, ConvexPolygon, ConvexPolyhedron, Renderer, intersection

from analysis.utils import polygon_to_intarray
from utils.utils import create_polygon
from stack import results_file


def create_cubes(_tr: list[list[int]]) -> list[ConvexPolyhedron]:
    """
    Function returns list of Polyhedrons later used by renderer to draw example of stt_iou

    :param _tr: list of trajectories
    :return: list of Polyhedrons
    """
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


if __name__ == '__main__':
    """
    Function draws stt_iou of ground truth, tracker and their intersection.
    
    You may need to tweak show function to properly display tubes.
    """
    results = pandas.read_csv(f"../{results_file}")
    trajectories = results.trajectory.apply(json.loads)
    groundtruths = results.groundtruth.apply(json.loads)
    results.trajectory = [[create_polygon(points) if points != [] else None for points in trajectory] for trajectory in trajectories]
    results.groundtruth = [[create_polygon(points) if points != [] else None for points in groundtruth] for groundtruth in groundtruths]

    df = results.loc[(results['tracker'] == 'TLD') & (results['dataset'] == 'VOT Basic Test Stack') & (results['sequence'] == 'surfing'), ['trajectory', 'groundtruth']]

    r = Renderer()

    tr_cubes = create_cubes([polygon_to_intarray(polygon) if polygon is not None else [] for polygon in df['trajectory'].tolist()[0]])
    gt_cubes = create_cubes([polygon_to_intarray(polygon) if polygon is not None else [] for polygon in df['groundtruth'].tolist()[0]])
    for tr_cube, gt_cube in zip(tr_cubes, gt_cubes):
        r.add((tr_cube, 'r', 1))
        # r.add((gt_cube, 'g', 1))
        cph2=intersection(tr_cube, gt_cube)
        if cph2 is None or type(cph2) is not ConvexPolyhedron:
            continue
        r.add((cph2, 'b', 1))

    # 25/-80
    # 5/-85
    r.show() # linie 28 i 29