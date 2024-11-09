import json

import pandas
from shapely import Polygon

from utils.utils import create_polygon
from stack import results_file


def stt_iou(_tr: list[Polygon], _gt: list[Polygon]) -> float:
    """
    Calculates average precision based on spatio-temporal tubes.

    :param _tr: Trajectory of the tracker.
    :param _gt: Groundtruth trajectory.
    :return: average precision
    """
    intersect = 0
    union = 0

    for t, g in zip(_tr, _gt):
        if (t is None or t.area == 0.0) and (g is None or g.area == 0.0):
            continue
        elif t is None or t.area == 0.0:
            union += g.area
        elif g is None or g.area == 0.0:
            union += t.area
        else:
            intersect += t.intersection(g).area
            union += t.union(g).area

    return intersect / union


def average_stt_iou(trajectories: list[list[Polygon | None]], groundtruths: list[list[Polygon]]) -> float:
    """
    Averages stt-ap from all sequences.

    :param trajectories: list of sequences of regions predicted by the tracker.
    :param groundtruths: list of sequences of groundtruth regions.
    :return: average ap from all sequences
    """
    cumulative = 0
    count = 0

    for trajectory, groundtruth in zip(trajectories, groundtruths):
        try:
            cumulative += stt_iou(trajectory, groundtruth)
        except Exception as e:
            continue
        count += 1

    return cumulative / count


if __name__ == '__main__':
    results = pandas.read_csv(f"../{results_file}")
    trajectories = results.trajectory.apply(json.loads)
    groundtruths = results.groundtruth.apply(json.loads)
    results.trajectory = [[create_polygon(points) if points != [] else None for points in trajectory] for trajectory in trajectories]
    results.groundtruth = [[create_polygon(points) if points != [] else None for points in groundtruth] for groundtruth in groundtruths]

    df = results.loc[(results['tracker'] == 'TLD') & (results['dataset'] == 'VOT Basic Test Stack') & (results['sequence'] == 'surfing'), ['trajectory', 'groundtruth']]