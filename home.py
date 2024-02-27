import numpy
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from data.state_locator import StateLocator
from data.viewmodel_home import HomeViewModel
from stack import datasets


class TestsPage:
    state_locator = StateLocator()
    view_model = HomeViewModel()

    current_frame_image: DeltaGenerator = st.image([])
    current_example_bar: DeltaGenerator | None = None
    all_examples_bar: DeltaGenerator | None = None

    def __init__(self) -> None:
        super().__init__()
        self.sidebar()

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

            submitted = st.form_submit_button("Submit", use_container_width=True, type="primary")

            if submitted:
                self.current_example_bar = st.progress(0)
                self.all_examples_bar = st.progress(0)

        if submitted:
            self.state_locator.provide_selection().append((tracker, selected_dataset))
            self.state_locator.provide_table_selected_trackers().loc[(tracker, selected_dataset), :] = True

            self.view_model.handle_submitted(
                self.handle_all_examples_bar,
                self.handle_current_example_bar,
                self.handle_current_frame_image
            )

            self.all_examples_bar.empty()
            self.current_example_bar.empty()
            self.current_frame_image.empty()

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

    def handle_all_examples_bar(self, index: int, sequences: [str]):
        if self.all_examples_bar is not None:
            self.all_examples_bar.progress(index / len(sequences), text=f"Testing {index + 1} out of {len(sequences)} films")

    def handle_current_example_bar(self, total_frames: int, current_frame: int, sequence: str) -> None:
        if self.current_example_bar is not None:
            self.current_example_bar.progress(current_frame / total_frames, text=f"Testing with {sequence}")

    def handle_current_frame_image(self, frame: numpy.ndarray):
        self.current_frame_image.image(frame, channels="BGR")


testsPage = TestsPage()
