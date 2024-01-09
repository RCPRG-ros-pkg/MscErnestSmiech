import numpy
import pandas
import streamlit as st
from shapely import Polygon

from analysis.utils import calculate_overlaps


def gather_overlaps(trajectory: "pandas.Series[Polygon]", groundtruth: "pandas.Series[Polygon]",
                    ignore_invisible: bool = False, threshold: float = -1) -> numpy.ndarray:
    overlaps = numpy.array(calculate_overlaps(trajectory, groundtruth))
    mask = numpy.ones(len(overlaps), dtype=bool)

    for i, (region_tr, region_gt) in enumerate(zip(trajectory, groundtruth)):
        # Skip if groundtruth is unknown
        if ignore_invisible and region_gt.area == 0.0:
            mask[i] = False
        # Skip if predicted is initialization frame
        elif region_tr is None:
            mask[i] = False
        elif overlaps[i] <= threshold:
            mask[i] = False

    return overlaps[mask]


def success_plot(tracker: str, dataset: str, sequence: str) -> list[tuple[float, float]]:
    g = st.session_state.results
    trajectories_groundtruths = g.loc[(g['tracker'] == tracker) & (g['dataset'] == dataset) & (g['sequence'] == sequence), ["trajectory", "groundtruth"]]
    axis_x = numpy.linspace(0, 1, 100)
    axis_y = numpy.zeros_like(axis_x)

    for trajectory, groundtruth in zip(trajectories_groundtruths['trajectory'], trajectories_groundtruths['groundtruth']):
        overlaps = gather_overlaps(trajectory, groundtruth)
        if overlaps.size > 0:
            for i, threshold in enumerate(axis_x):
                if threshold == 1:
                    # Nicer handling of the edge case
                    axis_y[i] += numpy.sum(overlaps >= threshold) / len(overlaps)
                else:
                    axis_y[i] += numpy.sum(overlaps > threshold) / len(overlaps)

    axis_y /= len(trajectories_groundtruths)

    return [(x, y) for x, y in zip(axis_x, axis_y)]


def average_success_plot():
    ret_df = pandas.DataFrame(columns=['Tracker', 'Dataset', 'Threshold', 'Success'])

    for (tracker, dataset), g in st.session_state.results.groupby(['tracker', 'dataset']):
        sequences = g.loc[(g['tracker'] == tracker) & (g['dataset'] == dataset), 'sequence'].unique()
        axis_x = numpy.linspace(0, 1, 100)
        axis_y = numpy.zeros_like(axis_x)

        for sequence in sequences:
            for j, (_, y) in enumerate(success_plot(tracker, dataset, sequence)):
                axis_y[j] += y

        axis_y /= len(sequences)

        ret_df = pandas.concat([ret_df, pandas.DataFrame(
            {'Tracker': [tracker] * len(axis_x), 'Dataset': [dataset] * len(axis_x), 'Threshold': axis_x,
             'Success': axis_y})])

    return ret_df


def sequence_accuracy(tracker: str, dataset: str, sequence: str, ignore_invisible: bool = False, threshold: float = -1) -> float:
    g = st.session_state.results
    trajectories_groundtruths = g.loc[(g['tracker'] == tracker) & (g['dataset'] == dataset) & (g['sequence'] == sequence), ["trajectory", "groundtruth"]]

    cummulative = 0
    for trajectory, groundtruth in zip(trajectories_groundtruths['trajectory'], trajectories_groundtruths['groundtruth']):
        overlaps = gather_overlaps(trajectory, groundtruth, ignore_invisible, threshold)

        if overlaps.size > 0:
            cummulative += numpy.mean(overlaps)

    return cummulative / len(trajectories_groundtruths)


def average_accuracy() -> pandas.DataFrame:
    ret_df = pandas.DataFrame(columns=['Tracker', 'Dataset', 'Quality'])

    for (tracker, dataset), g in st.session_state.results.groupby(['tracker', 'dataset']):
        accuracy = 0
        frames = 0
        sequences = g.loc[(g['tracker'] == tracker) & (g['dataset'] == dataset), 'sequence'].unique()

        for sequence in sequences:
            accuracy += sequence_accuracy(tracker, dataset, sequence)
            frames += 1

        ret_df.loc[len(ret_df)] = [tracker, dataset, accuracy / frames]

    return ret_df
