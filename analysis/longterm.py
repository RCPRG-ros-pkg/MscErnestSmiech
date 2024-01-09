import numpy
import streamlit as st
from shapely import Polygon

from analysis.accuracy import sequence_accuracy
from analysis.utils import calculate_overlaps


def count_frames(trajectory: list[Polygon | None], groundtruth: list[Polygon]):
    overlaps = numpy.array(calculate_overlaps(trajectory, groundtruth))

    # Tracking, Failure, Miss, Hallucination, Notice
    T, F, M, H, N = 0, 0, 0, 0, 0

    for i, (region_tr, region_gt) in enumerate(zip(trajectory, groundtruth)):
        if not region_gt:
            if not region_tr:
                N += 1
            else:
                H += 1
        else:
            if overlaps[i] > 0:
                T += 1
            else:
                if not region_tr:
                    M += 1
                else:
                    F += 1

    return T, F, M, H, N


def accuracy_robustness(
        tracker: str,
        dataset: str,
        trajectories: list[list[Polygon | None]],
        groundtruths: list[list[Polygon]]
):
    g = st.session_state.results
    df = g.loc[(g['tracker'] == tracker) & (g['dataset'] == dataset), ['trajectory', 'groundtruth']]

    _trajectories = trajectories + df['trajectory'].tolist()
    _groundtruths = groundtruths + df['groundtruth'].tolist()

    accuracy = 0
    robustness = 0
    count = 0

    for trajectory, groundtruth in zip(_trajectories, _groundtruths):
        accuracy += sequence_accuracy(trajectory, groundtruth, True, 0.0)
        T, F, M, _, _ = count_frames(trajectory, groundtruth)

        robustness += T / (T + F + M)
        count += 1

    st.session_state.cache['accuracy_robustness'].loc[(tracker, dataset), :] = [robustness / count, accuracy / count]


def quality_auxiliary(trajectory: list[Polygon | None], groundtruth: list[Polygon]) -> tuple[float, float, float]:
    T, F, M, H, N = count_frames(trajectory, groundtruth)

    not_reported_error = M / (T + F + M)
    drift_rate_error = F / (T + F + M)

    if N + H > 10:
        absence_detection = N / (N + H)
    else:
        absence_detection = None

    return not_reported_error, drift_rate_error, absence_detection


def average_quality_auxiliary(
        tracker: str,
        dataset: str,
        trajectories: list[list[Polygon | None]],
        groundtruths: list[list[Polygon]]
):
    g = st.session_state.results
    df = g.loc[(g['tracker'] == tracker) & (g['dataset'] == dataset), ['trajectory', 'groundtruth']]

    _trajectories = trajectories + df['trajectory'].tolist()
    _groundtruths = groundtruths + df['groundtruth'].tolist()

    not_reported_error = 0
    drift_rate_error = 0
    absence_detection = 0
    absence_count = 0
    count = 0

    for trajectory, groundtruth in zip(_trajectories, _groundtruths):
        nre, dre, ad = quality_auxiliary(trajectory, groundtruth)
        not_reported_error += nre
        drift_rate_error += dre
        if ad is not None:
            absence_count += 1
            absence_detection += ad

        count += 1

    if absence_count > 0:
        absence_detection /= absence_count

    st.session_state.cache['average_quality_auxiliary'].loc[(tracker, dataset), :] = [not_reported_error / count, drift_rate_error / count, absence_detection]
