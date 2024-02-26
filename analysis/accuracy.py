import numpy
import pandas
import streamlit as st
from shapely import Polygon

from analysis.utils import calculate_overlaps
from data.state_locator import StateLocator

state_locator = StateLocator()


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


def average_success_plot(
        tracker: str,
        dataset: str,
        trajectories: list[list[Polygon | None]],
        groundtruths: list[list[Polygon]]
):
    g = state_locator.provide_results()
    df = g.loc[(g['tracker'] == tracker) & (g['dataset'] == dataset), ['trajectory', 'groundtruth']]

    _trajectories = trajectories + df['trajectory'].tolist()
    _groundtruths = groundtruths + df['groundtruth'].tolist()

    axis_x = numpy.linspace(0, 1, 100)
    axis_y = numpy.zeros_like(axis_x)
    count = 0

    for trajectory, groundtruth in zip(_trajectories, _groundtruths):
        for j, (_, y) in enumerate(success_plot(trajectory, groundtruth)):
            axis_y[j] += y

        count += 1

    axis_y /= count

    df = state_locator.provide_cache()['average_success_plot']
    try:
        df.drop(df[(df['Tracker'] == tracker) & (df['Dataset'] == dataset)].index, inplace=True)
    except KeyError:
        pass
    state_locator.provide_cache()['average_success_plot'] = pandas.concat([
        df,
        pandas.DataFrame({
            'Tracker': [tracker] * len(axis_x),
            'Dataset': [dataset] * len(axis_x),
            'Threshold': axis_x,
            'Success': axis_y
        })
    ], ignore_index=True)


def sequence_accuracy(trajectory: list[Polygon | None], groundtruth: list[Polygon], ignore_invisible: bool = False, threshold: float = -1) -> float:
    cummulative = 0
    overlaps = gather_overlaps(trajectory, groundtruth, ignore_invisible, threshold)

    if overlaps.size > 0:
        cummulative += numpy.mean(overlaps)

    return cummulative


def average_accuracy(
        tracker: str,
        dataset: str,
        trajectories: list[list[Polygon | None]],
        groundtruths: list[list[Polygon]]
):
    g = state_locator.provide_results()
    df = g.loc[(g['tracker'] == tracker) & (g['dataset'] == dataset), ['trajectory', 'groundtruth']]

    _trajectories = trajectories + df['trajectory'].tolist()
    _groundtruths = groundtruths + df['groundtruth'].tolist()

    accuracy = 0
    count = 0

    for trajectory, groundtruth in zip(_trajectories, _groundtruths):
        accuracy += sequence_accuracy(trajectory, groundtruth)
        count += 1

    state_locator.provide_cache()['average_accuracy'].loc[(tracker, dataset), :] = accuracy / count
