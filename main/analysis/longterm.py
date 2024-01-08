import numpy
import pandas
import streamlit as st

from analysis.accuracy import sequence_accuracy
from analysis.utils import calculate_overlaps


def count_frames(tracker: str, dataset: str, sequence: str):
    g = st.session_state.results
    trajectories_groundtruths = g.loc[(g['tracker'] == tracker) & (g['dataset'] == dataset) & (g['sequence'] == sequence), ["trajectory", "groundtruth"]]

    CN, CF, CM, CH, CT = 0, 0, 0, 0, 0
    for trajectory, groundtruth in zip(trajectories_groundtruths['trajectory'], trajectories_groundtruths['groundtruth']):
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

        CN += N
        CF += F
        CM += M
        CH += H
        CT += T

    CN /= len(trajectories_groundtruths)
    CF /= len(trajectories_groundtruths)
    CM /= len(trajectories_groundtruths)
    CH /= len(trajectories_groundtruths)
    CT /= len(trajectories_groundtruths)

    return CT, CF, CM, CH, CN


def accuracy_robustness() -> pandas.DataFrame:
    ret_df = pandas.DataFrame(columns=['Tracker', 'Dataset', 'Robustness', 'Accuracy'])

    for (tracker, dataset), g in st.session_state.results.groupby(['tracker', 'dataset']):
        sequences = g.loc[(g['tracker'] == tracker) & (g['dataset'] == dataset), 'sequence'].unique()

        accuracy = 0
        robustness = 0
        count = 0 # of sequences

        for sequence in sequences:
            accuracy += sequence_accuracy(tracker, dataset, sequence, True, 0.0)
            T, F, M, _, _ = count_frames(tracker, dataset, sequence)

            robustness += T / (T + F + M)
            count += 1

        ret_df.loc[len(ret_df)] = [tracker, dataset, robustness / count, accuracy / count]

    return ret_df


def quality_auxiliary(tracker: str, dataset: str, sequence: str) -> tuple[float, float, float]:
    T, F, M, H, N = count_frames(tracker, dataset, sequence)

    not_reported_error = M / (T + F + M)
    drift_rate_error = F / (T + F + M)

    if N + H > 10:
        absence_detection = N / (N + H)
    else:
        absence_detection = None

    return not_reported_error, drift_rate_error, absence_detection


def average_quality_auxiliary():
    ret_df = pandas.DataFrame(columns=['Tracker', 'Dataset', 'NRE', 'DRE', 'AD'])

    for (tracker, dataset), g in st.session_state.results.groupby(['tracker', 'dataset']):
        # todo .unique() potncjalnie zbędne
        sequences = g.loc[(g['tracker'] == tracker) & (g['dataset'] == dataset), 'sequence'].unique()

        not_reported_error = 0
        drift_rate_error = 0
        absence_detection = 0
        absence_count = 0

        for sequence in sequences:
            nre, dre, ad = quality_auxiliary(tracker, dataset, sequence)
            not_reported_error += nre
            drift_rate_error += dre
            if ad is not None:
                absence_count += 1
                absence_detection += ad

        if absence_count > 0:
            absence_detection /= absence_count

        ret_df.loc[len(ret_df)] = [tracker, dataset, not_reported_error / len(sequences), drift_rate_error / len(sequences), absence_detection]

    return ret_df
