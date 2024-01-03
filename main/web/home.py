import inspect
import json
import os
import sys
from json import JSONEncoder
from pathlib import Path

import math
import numpy
import pandas
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))

import main.tracker as tracker


def get_concrete_classes(cls):
    for subclass in cls.__subclasses__():
        yield from get_concrete_classes(subclass)
        if not inspect.isabstract(subclass):
            yield subclass


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

datasets = {
    'VOT Basic Test Stack': './raw/tests/basic/sequences',
    'VOT-ST2022 bounding-box challenge': './raw/tests/vot2022stb/sequences'
}
# vot2022/sts
results_dir = './raw/results'
results_file = f'{results_dir}/results.csv'


# fixme after first test with empty table, table has extra row
class TestsPage: # todo output to raw
    submitted = False
    sequence = ""

    current_frame_image: DeltaGenerator | None = None
    current_example_bar: DeltaGenerator | None = None
    all_examples_bar: DeltaGenerator | None = None
    iou_quality_chart: DeltaGenerator | None = None
    iou_ar_chart: DeltaGenerator | None = None
    result_table: DeltaGenerator | None = None

    def __init__(self) -> None:
        super().__init__()
        if 'results' not in st.session_state:
            try:
                os.makedirs(results_dir)
                st.session_state.results = pandas.DataFrame(
                    columns=['tracker', 'dataset', 'supervised', 'sequence', 'ious', 'stt-iou'],
                    dtype=object
                )
                st.session_state.results.to_csv(results_file, mode='x', index=False)
            except FileExistsError:
                # directory already exists
                st.session_state.results = pandas.read_csv(results_file)
                # todo performance
                st.session_state.results.ious = st.session_state.results.ious.apply(json.loads).apply(numpy.array)
        if 'selection' not in st.session_state:
            st.session_state["selection"]: [(str, str, bool)] = []
        if 'table_selected_trackers' not in st.session_state:
            indexes = st.session_state.results[['tracker', 'dataset', 'supervised']].value_counts().index.rename(['Tracker', 'Dataset', "Supervised"])
            st.session_state['table_selected_trackers'] = pandas.DataFrame([False]*indexes.size,index=indexes,columns=['Selected'])
        if 'trackers' not in st.session_state:
            st.session_state.trackers = [_tracker() for _tracker in get_concrete_classes(tracker.Tracker)]

        self.sidebar()
        self.main_page()  # fixme znikające wykresy przy starcie
        self.handle_submitted()
        self.draw_scores()
        self.draw_table()

    def sidebar(self) -> None:
        with st.sidebar.form("Options"):
            tracker = st.selectbox('Trackers', [tracker.name for tracker in st.session_state.trackers])
            selected_dataset = st.selectbox("Datasets", datasets.keys())
            supervised = st.checkbox('Supervised')

            self.submitted = st.form_submit_button("Submit", use_container_width=True, type="primary")

            if self.submitted:
                st.session_state.selection.append((tracker, selected_dataset, supervised))
                st.session_state.table_selected_trackers.loc[tracker, selected_dataset, supervised] = True

    def main_page(self) -> None:
        self.current_frame_image = st.empty()

        self.current_example_bar = st.progress(0)
        self.current_example_bar.empty()

        self.all_examples_bar = st.progress(0)
        self.all_examples_bar.empty()

        st.header("Results")

        st.subheader("IoU")

        iou_quality, iou_ar = st.columns(2)
        iou_quality.text("Quality plot")
        iou_ar.text("AR plot")
        self.iou_quality_chart = iou_quality.empty()
        self.iou_ar_chart = iou_ar.empty()
        self.result_table = st.empty()

    def draw_scores(self):
        selected = st.session_state.table_selected_trackers
        selected = selected.loc[selected['Selected'] == True]
        if selected.empty:
            return
        ts = self.get_threshold_success()
        ts = ts.set_index(['Tracker', 'Dataset', 'Supervised'])
        ts = ts.loc[selected.index]
        ts = ts.reset_index()
        ts['Supervised'] = ts['Supervised'].apply(lambda x: "True" if x else "False")
        ts['TrackerDatasetSupervised'] = ts[['Tracker', 'Dataset', 'Supervised']].agg(' - '.join, axis=1)
        if len(ts) > 0:
            self.iou_quality_chart.line_chart(ts, x='Threshold', y='Success', color='TrackerDatasetSupervised', height=300)

        ra = self.get_robustness_accuracy()
        ra = ra.set_index(['Tracker', 'Dataset', 'Supervised'])
        ra = ra.loc[selected.index]
        ra = ra.reset_index()
        ra['Supervised'] = ra['Supervised'].apply(lambda x: "True" if x else "False")
        ra['TrackerDatasetSupervised'] = ra[['Tracker', 'Dataset', 'Supervised']].agg(' - '.join, axis=1)
        if len(ra) > 0:
            self.iou_ar_chart.scatter_chart(ra, x='Robustness', y='Accuracy', color='TrackerDatasetSupervised', height=300)

    def on_table_change(self, foo): # fixme something wrong with selecting
        if 'show_trackers' in st.session_state:
            edited = st.session_state.show_trackers['edited_rows']

            for i in edited:
                st.session_state.table_selected_trackers.loc[*foo.iloc[i].tolist()] = edited[i]['Selected']

            self.draw_scores()

    def draw_table(self):
        ts = self.get_threshold_success()
        ra = self.get_robustness_accuracy()

        if len(ts) <= 0 or len(ra) <= 0:
            return

        # todo performance
        success = ts.groupby(['Tracker', 'Dataset', 'Supervised']).apply(lambda x: x['Success'].tolist())
        auc = ts.groupby(['Tracker', 'Dataset', 'Supervised']).apply(lambda x: numpy.trapz(y=x['Success'].tolist(), x=x['Threshold'].tolist()))
        ra = ra.set_index(['Tracker', 'Dataset', 'Supervised'])

        df = pandas.concat(
            [ra, success, auc, st.session_state.table_selected_trackers],
            axis=1,
        ).rename(columns={0: 'success', 1: 'auc'}).reset_index()

        self.result_table = st.data_editor(
            df,
            column_config={
                "tracker": "Tracker",
                "dataset": "Dataset",
                "supervised": st.column_config.CheckboxColumn(
                    "Supervised",
                    help="Was tracker supervised during testing",
                    disabled=True,
                ),
                "robustness": "Robustness",
                "accuracy": "Accuracy",
                "auc": "AUC",
                "success": st.column_config.LineChartColumn(
                    "Success", y_min=0.0, y_max=1.0, width='small'
                ),
                "selected": st.column_config.CheckboxColumn(
                    "Selected",
                    help="Select tracker to display in charts",
                    default=True,
                ),
            },
            disabled=['Tracker', "Dataset", "Supervised", "Robustness", "Accuracy", "AUC", "Success"],
            on_change=self.on_table_change(df[['Tracker', 'Dataset', 'Supervised']]),
            use_container_width=True,
            key='show_trackers',
            column_order=["Tracker", "Dataset", "Supervised", "Robustness", "Accuracy", "auc", "success", "Selected"],
            hide_index=True
        )

    @staticmethod
    def save_results(tracker: str, dataset: str, supervised: bool, sequence: str, ious: numpy.ndarray):
        df = pandas.DataFrame({'tracker': tracker, 'dataset': dataset, 'supervised': supervised, 'sequence': sequence, 'ious': [ious], 'stt-iou': pandas.NA})

        st.session_state.results = pandas.concat([
            st.session_state.results,
            df
        ])

        df.ious = json.dumps(ious, cls=NumpyArrayEncoder)
        df.to_csv(results_file, mode='a', header=False, index=False)

    @staticmethod
    def get_robustness_accuracy() -> pandas.DataFrame:
        ret_df = pandas.DataFrame(columns=['Tracker', 'Dataset', 'Supervised', 'Robustness', 'Accuracy'])

        for (tracker, dataset, supervised), g in st.session_state.results.groupby(['tracker', 'dataset', 'supervised']):
            _failures = 0
            _accuracy = 0
            _weights = 0
            for tracker, dataset, supervised, _, ious, _ in g.itertuples(name=None, index=False):
                _weights += len(ious)
                _failures += len(ious[ious == 0])
                _accuracy += numpy.mean(ious[ious > 0]) * len(ious)
            robustness = math.exp(- (_failures / _weights) * 30)  # todo check
            accuracy = _accuracy / _weights

            ret_df.loc[len(ret_df)] = [tracker, dataset, supervised, robustness, accuracy]

        return ret_df

    @staticmethod
    def get_threshold_success() -> pandas.DataFrame:
        ret_df = pandas.DataFrame(columns=['Tracker', 'Dataset', 'Supervised', 'Threshold', 'Success'])

        for (tracker, dataset, supervised), g in st.session_state.results.groupby(['tracker', 'dataset', 'supervised']):
            axis_x = numpy.linspace(0, 1, 100)
            axis_y = numpy.zeros_like(axis_x)
            object_y = numpy.zeros_like(axis_x)
            for tracker, dataset, supervised, sequence, ious, stt_iou in g.itertuples(name=None, index=False):
                for i, threshold in enumerate(axis_x):
                    if threshold == 1:
                        # Nicer handling of the edge case
                        object_y[i] += numpy.sum(ious >= threshold) / len(ious)
                    else:
                        object_y[i] += numpy.sum(ious > threshold) / len(ious)
            axis_y += object_y / len(g)
            ret_df = pandas.concat([ret_df, pandas.DataFrame(
                {'Tracker': [tracker] * len(axis_x), 'Dataset': [dataset] * len(axis_x), 'Supervised': [supervised] * len(axis_x), 'Threshold': axis_x,
                 'Success': axis_y})])

        return ret_df

    def handle_submitted(self) -> None:
        if self.submitted:
            # tracker, dataset
            for _selection in st.session_state.selection:
                sequences = []
                with open(f'{datasets[_selection[1]]}/list.txt') as f:
                    for line in f.readlines():
                        sequences.append(line.strip())

                for index, sequence in enumerate(sequences):
                    self.all_examples_bar.progress(index / len(sequences),
                                                   text=f"Testing {index + 1} out of {len(sequences)} films")
                    self.sequence = sequence
                    _tracker: tracker.Tracker = next(x for x in st.session_state.trackers if x.name == _selection[0])
                    ious, _ = _tracker.test(
                        f'{datasets[_selection[1]]}/{sequence}',
                        listener=self.display_bar,
                        frame_listener=self.display_frame,
                        _iou_threshold_for_correction=0 if _selection[2] else -1 # if supervised
                    )
                    self.save_results(_selection[0], _selection[1], _selection[2], sequence, ious)
                    self.draw_scores()
            st.session_state.selection = []
            self.all_examples_bar.empty()
            self.current_example_bar.empty()
            self.current_frame_image.empty()

    def display_bar(self, total_frames: int, current_frame: int) -> None:
        if self.current_example_bar is not None:
            self.current_example_bar.progress(current_frame / total_frames, text=f"Testing with {self.sequence}")

    def display_frame(self, frame: numpy.ndarray):
        self.current_frame_image.image(frame)


testsPage = TestsPage()
