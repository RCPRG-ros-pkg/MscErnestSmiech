from datetime import datetime, time

import numpy
import pandas
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from data.viewmodel_errors import ErrorsViewModel


# todo przycisk clear
class ErrorsPage:
    """
    Second page that allows displaying frames where tracker failed to track as well as what was tracker's performance in
    selected test. User can filter through trackers, datasets, sequences and dates when test was performed.
    """
    errors_view_model = ErrorsViewModel()

    image_displayer: DeltaGenerator | None = None

    def __init__(self) -> None:
        super().__init__()

        self.errors_view_model.set_page_name()

        st.set_page_config(page_title="Error frames viewer", page_icon="📈")

        self.sidebar()

        st.markdown("# Error frames viewer")

        selected_rows = self.errors_view_model.get_selected_rows()
        if selected_rows is not None:
            if len(selected_rows) == 1:
                self.display_picture()

            self.display_scores()

        self.display_selector()

    def display_scores(self):
        df_success_plot = self.errors_view_model.get_success_plot()
        if df_success_plot is None or df_success_plot.empty:
            return
        df_ar_plot = self.errors_view_model.get_ar_plot()
        if df_ar_plot is None or df_ar_plot.empty:
            return
        df_iou_table = self.errors_view_model.get_iou_table()
        if df_iou_table is None or df_iou_table.empty:
            return
        df_time_table = self.errors_view_model.get_time_table()
        if df_time_table is None or df_time_table.empty:
            return

        st.markdown("## Scores")

        self.display_plots(df_success_plot, df_ar_plot)
        self.display_iou_table(df_iou_table)
        self.display_time_table(df_time_table)

    def display_selector(self):
        """
        Displays selector that allows to select trackers, datasets, sequences and dates when test was performed
        """
        st.markdown("## Selector")
        st.data_editor(
            self.errors_view_model.get_df_table(),
            column_config={
                "tracker": "Tracker",
                "dataset": "Dataset",
                "sequence": "Sequence",
                "date": "Date",
                "selected": st.column_config.CheckboxColumn(
                    "Selected",
                    help="Select tracker to display in charts",
                    default=True,
                ),
            },
            disabled=["tracker", "dataset", "sequence", "date"],
            on_change=self.errors_view_model.on_table_change,
            use_container_width=True,
            key='table',
            column_order=["tracker", "dataset", "sequence", "date", "selected"],
            hide_index=True,
            args=[self.errors_view_model.get_df_table()]
        )

    @staticmethod
    def display_time_table(table: pandas.DataFrame):
        st.text("Times table")
        st.data_editor(
            table,
            column_config={
                "tracker": "Tracker",
                "dataset": "Dataset",
                "sequence": "Sequence",
                "date": "Date",
                "success": "Success",
                "robustness": "Robustness",
                "nre": "NRE",
                "dre": "DRE",
                "ad": "AD"
            },
            disabled=['tracker', "dataset", 'sequence', "robustness", 'nre', 'dre', 'ad', 'success', 'date'],
            use_container_width=True,
            key='bar',
            column_order=["tracker", "dataset", "sequence", 'date', "success", "robustness", 'nre', 'dre', 'ad'],
            hide_index=True
        )

    @staticmethod
    def display_iou_table(table: pandas.DataFrame):
        st.text("Iou table")
        st.data_editor(
            table,
            column_config={
                "tracker": "Tracker",
                "dataset": "Dataset",
                "sequence": "Sequence",
                "date": "Date",
                "accuracy": "Accuracy",
                "robustness": "Robustness",
                "stt_iou": "STT-IoU",
                "nre": "NRE",
                "dre": "DRE",
                "ad": "AD"
            },
            disabled=['tracker', "dataset", "robustness", 'nre', 'dre', 'ad', 'sequence', 'accuracy', 'date',
                      'stt_iou'],
            use_container_width=True,
            key='foo',
            column_order=["tracker", "dataset", "sequence", 'date', "accuracy", "robustness", "stt_iou", 'nre', 'dre',
                          'ad'],
            hide_index=True
        )

    @staticmethod
    def display_plots(success_plot: pandas.DataFrame, ar_plot: pandas.DataFrame):
        iou_quality, iou_ar = st.columns(2)
        with iou_quality:
            st.text("Quality plot")
            st.line_chart(
                success_plot,
                x='threshold',
                y='success',
                color='date',
                height=300
            )
        with iou_ar:
            st.text("AR plot")
            st.scatter_chart(
                ar_plot,
                x='robustness',
                y='accuracy',
                color='date',
                height=300
            )

    def display_picture(self):
        """
        Displays pictures of failed frames. Allows to move to next or previous picture.
        """
        self.errors_view_model.load_images()

        try:
            image = self.errors_view_model.get_current_image()
        except IndexError:
            self.errors_view_model.zero_selected_image_index()
            image = self.errors_view_model.get_current_image()

        st.markdown("## Errors")

        _, col, _ = st.columns((2, 11, 2))
        with col:
            st.image(image, width=500)

            _, col1, col2, col3, _ = st.columns((3, 1, 1, 1, 3))
            with col1:
                st.button("←", on_click=self.errors_view_model.decrease_selected_image_index)
            with col2:
                st.write(f"{self.errors_view_model.get_image_index()}/{len(self.errors_view_model.get_images())}")
            with col3:
                st.button("→", on_click=self.errors_view_model.increase_selected_image_index)

    def sidebar(self) -> None:
        """
        Sidebar for filtering records in table.
        """
        with st.sidebar.form("Options"):
            trackers = st.multiselect('Trackers', self.errors_view_model.df['tracker'].unique())
            datasets = st.multiselect("Datasets", self.errors_view_model.df['dataset'].unique())
            sequences = st.multiselect("Sequence", self.errors_view_model.df['sequence'].unique())

            available_dates = self.errors_view_model.get_available_dates()

            if available_dates.size > 1:
                start_date, end_date = self.sidebar_date_selector(available_dates)
            else:
                start_date, end_date = self.sidebar_time_selector(available_dates)

            if st.form_submit_button("Submit", use_container_width=True, type="primary"):
                self.errors_view_model.set_tracker_selection(trackers, datasets, sequences, start_date, end_date)

    def sidebar_time_selector(self, available_dates: numpy.ndarray[datetime.date]) -> (datetime, datetime):
        available_times = self.errors_view_model.get_available_times()

        times = st.select_slider(
            label="Time",
            options=available_times,
            value=(available_times[0], available_times[-1]),
            format_func=lambda d: d.strftime('%H:%M:%S')
        )

        start_date = datetime.combine(available_dates[0], times[0])
        end_date = datetime.combine(available_dates[0], times[1])

        return start_date, end_date

    @staticmethod
    def sidebar_date_selector(available_dates: numpy.ndarray[datetime.date]) -> (datetime, datetime):
        dates = st.select_slider(
            label="Date",
            options=available_dates,
            value=(available_dates[0], available_dates[-1]),
            format_func=lambda d: d.strftime('%d %b %Y')
        )
        start_time = st.slider(
            label="Start time",
            min_value=time(0, 0, 0),
            max_value=time(23, 59, 59)
        )
        end_time = st.slider(
            label="End time",
            min_value=time(0, 0, 0),
            max_value=time(23, 59, 59),
            value=time(23, 59, 59)
        )

        start_date = datetime.combine(dates[0], start_time)
        end_date = datetime.combine(dates[1], end_time)

        return start_date, end_date


ErrorsPage()

