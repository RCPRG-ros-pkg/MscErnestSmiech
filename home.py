import json
import os

import numpy
import pandas
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from analysis.stt_iou import average_stt_iou
from analysis.time import average_time_quality_auxiliary, average_time
from utils.utils import save_results, get_or_create_cache
from stack import datasets, results_dir, results_file, cache_dir, tests_dir
from utils.tracker_test import get_ground_truth_positions

from analysis.accuracy import average_success_plot, average_accuracy
from analysis.longterm import accuracy_robustness, average_quality_auxiliary
from analysis.utils import create_polygon

import tracker as tracker


class TestsPage:
    submitted = False
    sequence = ""

    current_frame_image: DeltaGenerator | None = None
    current_example_bar: DeltaGenerator | None = None
    all_examples_bar: DeltaGenerator | None = None

    def __init__(self) -> None:
        super().__init__()
        if 'results' not in st.session_state:
            try:
                os.makedirs(results_dir)
                st.session_state.results = pandas.DataFrame(
                    columns=['date', 'tracker', 'dataset', 'sequence', 'selected', 'trajectory', 'groundtruth', 'times'],
                    dtype=object
                )
                _results = pandas.DataFrame(
                    columns=['date', 'tracker', 'dataset', 'sequence', 'trajectory', 'groundtruth', 'times'],
                    dtype=object
                )
                _results.to_csv(results_file, mode='x', index=False)
            except FileExistsError:
                # directory already exists
                _results = pandas.read_csv(results_file)
                _results.trajectory = [[create_polygon(points) if points != [] else None for points in trajectory] for trajectory in _results['trajectory'].map(json.loads)]
                _results.groundtruth = [[create_polygon(points) if points != [] else None for points in groundtruth] for groundtruth in _results['groundtruth'].map(json.loads)]
                _results.times = _results.times.map(json.loads)
                _results['selected'] = False
                st.session_state.results = _results
        if 'cache' not in st.session_state:
            index=pandas.MultiIndex.from_tuples([], names=['Tracker', 'Dataset'])
            st.session_state.cache = {
                'average_stt_iou': pandas.DataFrame(index=index, columns=['STT-IOU']),
                'average_accuracy': pandas.DataFrame(index=index, columns=['Quality']),
                'average_time': pandas.DataFrame(index=index, columns=['Quality']),
                'average_success_plot': pandas.DataFrame(columns=['Tracker', 'Dataset', 'Threshold', 'Success']),
                'average_quality_auxiliary': pandas.DataFrame(index=index, columns=['NRE', 'DRE', 'AD']),
                'average_time_quality_auxiliary': pandas.DataFrame(index=index, columns=['Robustness', 'NRE', 'DRE', 'AD']),
                'accuracy_robustness': pandas.DataFrame(index=index, columns=['Robustness', 'Accuracy']),
            }
            os.makedirs(cache_dir, exist_ok=True)
            names = [
                'average_stt_iou',
                'average_accuracy',
                'average_time',
                'average_success_plot',
                'average_quality_auxiliary',
                'average_time_quality_auxiliary',
                'accuracy_robustness'
            ]
            for n in names:
                get_or_create_cache(n, n != 'average_success_plot')
        if 'selection' not in st.session_state:
            st.session_state["selection"]: [(str, str)] = []
        if 'table_selected_trackers' not in st.session_state:
            indexes = st.session_state.cache['average_accuracy'].index
            st.session_state.table_selected_trackers = pandas.DataFrame([False]*indexes.size,index=indexes,columns=['Selected'])
        if 'trackers' not in st.session_state:
            st.session_state.trackers = [_tracker() for _tracker in tracker.get_concrete_classes(tracker.Tracker)]

        self.sidebar()

        if self.submitted:
            self.current_frame_image = st.empty()
            self.current_example_bar = st.progress(0)
            self.all_examples_bar = st.progress(0)

        self.handle_submitted()

        st.header("Results")
        st.subheader("IoU")
        self.draw_iou_scores()
        self.draw_iou_table()
        st.subheader("Time")
        self.draw_time_table()

    def sidebar(self) -> None:
        with st.sidebar.form("Options"):
            tracker = st.selectbox('Trackers', [tracker.name for tracker in st.session_state.trackers])
            selected_dataset = st.selectbox("Datasets", datasets.keys())

            self.submitted = st.form_submit_button("Submit", use_container_width=True, type="primary")

            if self.submitted:
                st.session_state.selection.append((tracker, selected_dataset))
                st.session_state.table_selected_trackers.loc[(tracker, selected_dataset), :] = True


    @staticmethod
    def draw_iou_scores():
        iou_quality, iou_ar = st.columns(2)

        selected = st.session_state.table_selected_trackers
        selected = selected.loc[selected['Selected'] == True]
        if selected.empty:
            return
        ts = st.session_state.cache['average_success_plot']
        ts = ts.set_index(['Tracker', 'Dataset'])
        ts = ts.loc[selected.index]
        ts = ts.reset_index()
        ts['TrackerDataset'] = ts[['Tracker', 'Dataset']].agg(' - '.join, axis=1)
        if len(ts) > 0:
            iou_quality.text("Quality plot")
            iou_quality.line_chart(ts, x='Threshold', y='Success', color='TrackerDataset', height=300)

        ra = st.session_state.cache['accuracy_robustness']
        ra = ra.loc[selected.index]
        ra = ra.reset_index()
        ra['TrackerDataset'] = ra[['Tracker', 'Dataset']].agg(' - '.join, axis=1)
        if len(ra) > 0:
            iou_ar.text("AR plot")
            iou_ar.scatter_chart(ra, x='Robustness', y='Accuracy', color='TrackerDataset', height=300)

    @staticmethod
    def on_iou_table_change(df_tracker_dataset): # fixme something wrong with selecting
        if 'show_iou_trackers' in st.session_state:
            edited = st.session_state.show_iou_trackers['edited_rows']

            for i in edited:
                st.session_state.table_selected_trackers.loc[*df_tracker_dataset.iloc[i].tolist()] = edited[i]['Selected']

    def draw_iou_table(self):
        ts = st.session_state.cache['average_success_plot']
        ra = st.session_state.cache['accuracy_robustness']
        qa = st.session_state.cache['average_quality_auxiliary']
        ac = st.session_state.cache['average_accuracy']
        stt = st.session_state.cache['average_stt_iou']

        if len(ts) <= 0 or len(ra) <= 0 or len(qa) <= 0 or len(ac) <= 0:
            return

        # todo performance
        success = ts.groupby(['Tracker', 'Dataset']).apply(lambda x: x['Success'].tolist())

        df = pandas.concat(
            [stt, ac, ra, qa, success, st.session_state.table_selected_trackers],
            axis=1,
        ).rename(columns={0: 'success'}).reset_index()

        st.data_editor(
            df,
            column_config={
                "tracker": "Tracker",
                "dataset": "Dataset",
                "stt_iou": "STT-IOU",
                "quality": "Quality",
                "accuracy": "Accuracy",
                "robustness": "Robustness",
                "nre": "NRE",
                "dre": "DRE",
                "ad": "AD",
                "success": st.column_config.LineChartColumn(
                    "Success", y_min=0.0, y_max=1.0, width='small'
                ),
                "selected": st.column_config.CheckboxColumn(
                    "Selected",
                    help="Select tracker to display in charts",
                    default=True,
                ),
            },
            disabled=['tracker', "dataset", "robustness", "accuracy", "success", 'nre', 'dre', 'ad', 'quality', "stt_iou"],
            on_change=self.on_iou_table_change,
            use_container_width=True,
            key='show_iou_trackers',
            column_order=["Tracker", "Dataset", "STT-IOU", "Quality", "Accuracy", "Robustness", 'NRE', 'DRE', 'AD', "success", "Selected"],
            hide_index=True,
            args=[df[['Tracker', 'Dataset']]]
        )

    @staticmethod
    def draw_time_table():
        qa = st.session_state.cache['average_time_quality_auxiliary']
        ac = st.session_state.cache['average_time']

        if len(qa) <= 0 or len(ac) <= 0:
            return

        df = pandas.concat(
            [ac, qa, st.session_state.table_selected_trackers],
            axis=1,
        ).reset_index()

        st.data_editor(
            df,
            column_config={
                "tracker": "Tracker",
                "dataset": "Dataset",
                "quality": "Quality",
                "accuracy": "Accuracy",
                "robustness": "Robustness",
                "nre": "NRE",
                "dre": "DRE",
                "ad": "AD"
            },
            disabled=['tracker', "dataset", "robustness", 'nre', 'dre', 'ad', 'quality', 'accuracy'],
            use_container_width=True,
            key='show_time_trackers',
            column_order=["Tracker", "Dataset", "Quality", "Accuracy", "Robustness", 'NRE', 'DRE', 'AD'],
            hide_index=True
        )

    def handle_submitted(self) -> None:
        if self.submitted:
            # tracker, dataset
            for _selection in st.session_state.selection:
                sequences = []
                dates = []
                with open(f'{tests_dir}/{datasets[_selection[1]]}/sequences/list.txt') as f:
                    for line in f.readlines():
                        sequences.append(line.strip())

                trajectories = []
                groundtruths = []
                detection_times = []
                for index, sequence in enumerate(sequences):
                    self.all_examples_bar.progress(index / len(sequences), text=f"Testing {index + 1} out of {len(sequences)} films")
                    self.sequence = sequence
                    _tracker: tracker.Tracker = next(x for x in st.session_state.trackers if x.name == _selection[0])
                    date, detection_time, trajectory = _tracker.test(
                        f'{tests_dir}/{datasets[_selection[1]]}/sequences/{sequence}',
                        listener=self.display_bar,
                        frame_listener=self.display_frame,
                        iou_threshold_for_correction=-1
                    )
                    dates.append(date)
                    detection_times.append(detection_time)
                    trajectories.append(trajectory)
                    # [1:] because first frame is consumed for init of tracker
                    groundtruths.append(
                        get_ground_truth_positions(f"{tests_dir}/{datasets[_selection[1]]}/sequences/{sequence}/groundtruth.txt")[1:]
                    )

                accuracy_robustness(_selection[0], _selection[1], trajectories, groundtruths)
                average_quality_auxiliary(_selection[0], _selection[1], trajectories, groundtruths)
                average_accuracy(_selection[0], _selection[1], trajectories, groundtruths)
                average_success_plot(_selection[0], _selection[1], trajectories, groundtruths)
                average_stt_iou(_selection[0], _selection[1], trajectories, groundtruths)
                average_time_quality_auxiliary(_selection[0], _selection[1], detection_times, trajectories, groundtruths)
                average_time(_selection[0], _selection[1], detection_times, trajectories, groundtruths)
                save_results(dates, _selection[0], _selection[1], sequences, detection_times, trajectories, groundtruths)
            st.session_state.selection = []
            self.all_examples_bar.empty()
            self.current_example_bar.empty()
            self.current_frame_image.empty()

    def display_bar(self, total_frames: int, current_frame: int) -> None:
        if self.current_example_bar is not None:
            self.current_example_bar.progress(current_frame / total_frames, text=f"Testing with {self.sequence}")

    def display_frame(self, frame: numpy.ndarray):
        self.current_frame_image.image(frame, channels="BGR")


testsPage = TestsPage()
