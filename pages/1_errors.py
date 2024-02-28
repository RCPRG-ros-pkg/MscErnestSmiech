import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from data.viewmodel_errors import ErrorsViewModel


# todo przycisk clear
# todo wybór przedziału czasu
class ErrorsPage:
    errors_view_model = ErrorsViewModel()

    image_displayer: DeltaGenerator | None = None

    def __init__(self) -> None:
        super().__init__()

        st.set_page_config(page_title="Error frames viewer", page_icon="📈")

        self.sidebar()

        st.markdown("# Error frames viewer")

        selected_rows = self.errors_view_model.get_selected_rows()
        if selected_rows is not None:
            if len(selected_rows) == 1:
                self.display_picture()

            st.markdown("## Scores")

            self.display_plots()
            self.display_iou_table()
            self.display_time_table()

        self.display_selector()

    def display_selector(self):
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

    def display_time_table(self):
        data = self.errors_view_model.get_time_tables()
        if data is None or data.empty:
            return

        st.text("Times table")
        st.data_editor(
            data,
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

    def display_iou_table(self):
        data = self.errors_view_model.get_iou_table()
        if data is None or data.empty:
            return

        st.text("Iou table")
        st.data_editor(
            data,
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

    def display_plots(self):
        data = self.errors_view_model.get_success_plot()
        if data is None or data.empty:
            return

        iou_quality, iou_ar = st.columns(2)
        with iou_quality:
            st.text("Quality plot")
            st.line_chart(
                data,
                x='threshold',
                y='success',
                color='date',
                height=300
            )
        with iou_ar:
            st.text("AR plot")
            st.scatter_chart(
                self.errors_view_model.get_ar_plot(),
                x='robustness',
                y='accuracy',
                color='date',
                height=300
            )

    def display_picture(self):
        self.errors_view_model.load_images()

        try:
            image = self.errors_view_model.get_image(self.errors_view_model.get_image_count())
        except IndexError:
            self.errors_view_model.zero_image_count()
            image = self.errors_view_model.get_image(self.errors_view_model.get_image_count())

        st.markdown("## Errors")

        _, col, _ = st.columns((2, 11, 2))
        with col:
            st.image(image, width=500)

            _, col1, col2, col3, _ = st.columns((3, 1, 1, 1, 3))
            with col1:
                st.button("←", on_click=self.errors_view_model.decrease_image_count)
            with col2:
                st.write(f"{self.errors_view_model.get_image_count()}/{len(self.errors_view_model.get_images())}")
            with col3:
                st.button("→", on_click=self.errors_view_model.increase_image_count)

    def sidebar(self) -> None:
        with st.sidebar.form("Options"):
            trackers = st.multiselect('Trackers', self.errors_view_model.df['tracker'].unique())
            datasets = st.multiselect("Datasets", self.errors_view_model.df['dataset'].unique())
            sequences = st.multiselect("Sequence", self.errors_view_model.df['sequence'].unique())
            dates = st.multiselect("Date", self.errors_view_model.df['date'].unique())

            if st.form_submit_button("Submit", use_container_width=True, type="primary"):
                self.errors_view_model.set_tracker_selection(trackers, datasets, sequences, dates)


ErrorsPage()

