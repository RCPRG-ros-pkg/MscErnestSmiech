import numpy
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

import tracker as tracker
from analysis.accuracy import average_success_plot, average_accuracy
from analysis.longterm import accuracy_robustness, average_quality_auxiliary
from analysis.stt_iou import average_stt_iou
from analysis.time import average_time_quality_auxiliary, average_time
from data.state_locator import StateLocator
from data.viewmodel_home import HomeViewModel
from stack import datasets, tests_dir
from utils.tracker_test import get_ground_truth_positions
from utils.utils import save_results


class TestsPage:
    state_locator = StateLocator()
    view_model = HomeViewModel()

    submitted = False
    sequence = ""

    current_frame_image: DeltaGenerator | None = None
    current_example_bar: DeltaGenerator | None = None
    all_examples_bar: DeltaGenerator | None = None

    def __init__(self) -> None:
        super().__init__()
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
            tracker = st.selectbox('Trackers', self.view_model.tracker_names)
            selected_dataset = st.selectbox("Datasets", datasets.keys())

            self.submitted = st.form_submit_button("Submit", use_container_width=True, type="primary")

        if self.submitted: # todo
            self.state_locator.provide_selection().append((tracker, selected_dataset))
            self.state_locator.provide_table_selected_trackers().loc[(tracker, selected_dataset), :] = True

    def draw_iou_scores(self):
        iou_quality, iou_ar = st.columns(2)

        ts = self.view_model.get_quality_plot()
        if ts is not None and len(ts) > 0:
            iou_quality.text("Quality plot")
            iou_quality.line_chart(ts, x='Threshold', y='Success', color='TrackerDataset', height=300)

        ra = self.view_model.get_ar_plot()
        if ra is not None and len(ra) > 0:
            iou_ar.text("AR plot")
            iou_ar.scatter_chart(ra, x='Robustness', y='Accuracy', color='TrackerDataset', height=300)

    def draw_iou_table(self):
        df = self.view_model.get_iou_table()

        if df is not None:
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
                on_change=self.view_model.on_iou_table_change,
                use_container_width=True,
                key='show_iou_trackers',
                column_order=["Tracker", "Dataset", "STT-IOU", "Quality", "Accuracy", "Robustness", 'NRE', 'DRE', 'AD', "success", "Selected"],
                hide_index=True,
                args=[df[['Tracker', 'Dataset']]]
            )

    def draw_time_table(self):
        df = self.view_model.get_time_table()

        if df is not None:
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

    def handle_submitted(self) -> None: # todo popraw zapisywanie - powinno być bez stanowe. Następnie spróbuj ogarnąć ten handle
        if self.submitted:
            # tracker, dataset
            for _selection in self.state_locator.provide_selection():
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
                    _tracker: tracker.Tracker = next(x for x in self.state_locator.provide_trackers() if x.name == _selection[0])
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
            self.state_locator.clear_selection()
            self.all_examples_bar.empty()
            self.current_example_bar.empty()
            self.current_frame_image.empty()

    def display_bar(self, total_frames: int, current_frame: int) -> None:
        if self.current_example_bar is not None:
            self.current_example_bar.progress(current_frame / total_frames, text=f"Testing with {self.sequence}")

    def display_frame(self, frame: numpy.ndarray):
        self.current_frame_image.image(frame, channels="BGR")


testsPage = TestsPage()
