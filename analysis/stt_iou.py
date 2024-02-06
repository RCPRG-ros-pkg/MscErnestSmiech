import json

import pandas
import streamlit as st
from shapely import Polygon

from analysis.utils import create_polygon
from stack import results_file


def stt_iou(_tr: list[Polygon], _gt: list[Polygon]) -> float:
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

    df = results.loc[(results['tracker'] == 'TLD') & (results['dataset'] == 'VOT Basic Test Stack') & (results['sequence'] == 'surfing'), ['trajectory', 'groundtruth']]

    print(stt_iou(
        df['trajectory'].tolist()[0],
        df['groundtruth'].tolist()[0]
    ))