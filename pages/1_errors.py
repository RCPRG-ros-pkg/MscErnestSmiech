import glob

import numpy
import pandas
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from analysis.stt_iou import stt_iou
from analysis.accuracy import sequence_accuracy, success_plot
from analysis.longterm import quality_auxiliary, count_frames
from analysis.time import time_quality_auxiliary, sequence_time
from data.data_locator import DataLocator
from data.state_locator import StateLocator
from data.viewmodel_errors import ErrorsViewModel
from stack import datasets


class ErrorsPage:
    errors_view_model = ErrorsViewModel()
    state_locator = StateLocator()
    data_locator = DataLocator()
    
    df = errors_view_model.df
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

            _df = self.data_locator.provide_results()[self.data_locator.provide_results()['date'].isin(selected_rows['date'])]
            if not _df.empty:
                quality = pandas.DataFrame(
                    numpy.vectorize(quality_auxiliary)(_df['trajectory'], _df['groundtruth'])
                ).T
                accuracy = pandas.DataFrame(
                    numpy.vectorize(sequence_accuracy)(_df['trajectory'], _df['groundtruth']),
                    columns=("accuracy",)
                )
                stt_ious = pandas.DataFrame(
                    numpy.vectorize(stt_iou)(_df['trajectory'], _df['groundtruth']),
                    columns=("stt_iou",)
                )

                T, F, M, _, _ = numpy.vectorize(count_frames)(_df['trajectory'], _df['groundtruth'])
                robustness = pandas.DataFrame(T / (T + F + M), columns=['robustness'])

                success = pandas.DataFrame(map(success_plot, _df['trajectory'], _df['groundtruth']), index=_df['date']) \
                    .unstack() \
                    .reset_index()[['date', 0]] \
                    .sort_values(by=['date', 0]) \
                    .reset_index(drop=True)

                arr = numpy.array([item for row in success[0] for item in row])
                arr = arr.reshape((len(arr)//2, 2))
                tmp = pandas.DataFrame(arr, columns=['threshold', 'success'])
                tmp['date'] = success['date']
                success = tmp

                times_quality_auxiliary = pandas.DataFrame(
                    numpy.vectorize(time_quality_auxiliary)(_df['times'], _df['trajectory'], _df['groundtruth'])
                ).T
                times_sequence = pandas.DataFrame(
                    numpy.vectorize(sequence_time)(_df['times'], _df['trajectory'], _df['groundtruth']),
                    columns=["success"]
                )

                times = pandas.concat(
                    [
                        _df[['tracker', 'dataset', 'sequence', 'date']],
                        times_quality_auxiliary.rename(columns={0: 'robustness', 1: "nre", 2: "dre", 3: "ad"}),
                        times_sequence,
                    ],
                    axis=1,
                )

                _df = pandas.concat(
                    [
                        _df[['tracker', 'dataset', 'sequence', 'date']],
                        quality.rename(columns={0: "nre", 1: "dre", 2: "ad"}),
                        accuracy,
                        stt_ious,
                        robustness,
                    ],
                    axis=1,
                ).rename(columns={0: "success"})

                st.markdown("## Scores")

                iou_quality, iou_ar = st.columns(2)

                with iou_quality:
                    st.text("Quality plot")
                    st.line_chart(success, x='threshold', y='success', color='date', height=300)

                with iou_ar:
                    st.text("AR plot")
                    st.scatter_chart(_df[['accuracy', 'robustness', 'date']], x='robustness', y='accuracy', color='date', height=300)

                st.text("Iou table")
                st.data_editor(
                    _df,
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
                    disabled=['tracker', "dataset", "robustness", 'nre', 'dre', 'ad', 'sequence', 'accuracy', 'date', 'stt_iou'],
                    use_container_width=True,
                    key='foo',
                    column_order=["tracker", "dataset", "sequence", 'date', "accuracy", "robustness", "stt_iou", 'nre', 'dre', 'ad'],
                    hide_index=True
                )

                st.text("Times table")
                st.data_editor(
                    times,
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
            on_change=self.on_table_change,
            use_container_width=True,
            key='table',
            column_order=["tracker", "dataset", "sequence", "date", "selected"],
            hide_index=True,
            args=[self.errors_view_model.get_df_table()]
        )

    def display_picture(self):
        d = self.errors_view_model.get_selected_rows().head(1)
        d = d[['tracker', 'dataset', 'sequence', 'date']].astype(str).values.flatten().tolist()
        d[1] = datasets[d[1]]
        self.errors_view_model.set_images(glob.glob(f"./raw/errors/{'/'.join(d)}/*"))
        try:
            image = self.errors_view_model.get_image(self.state_locator.provide_image_count())
        except IndexError:
            self.state_locator.zero_image_count()
            image = self.errors_view_model.get_image(self.state_locator.provide_image_count())

        st.markdown("## Errors")

        _, col, _ = st.columns((2, 11, 2))
        with col:
            st.image(image, width=500)

            _, col1, col2, col3, _ = st.columns((3, 1, 1, 1, 3))
            with col1:
                st.button("←", on_click=self.state_locator.decrease_image_count)
            with col2:
                st.write(f"{self.state_locator.provide_image_count()}/{len(self.errors_view_model.get_images())}")
            with col3:
                st.button("→", on_click=self.state_locator.increase_image_count)

    def sidebar(self) -> None:
        with st.sidebar.form("Options"):
            trackers = st.multiselect('Trackers', self.errors_view_model.df['tracker'].unique())
            datasets = st.multiselect("Datasets", self.errors_view_model.df['dataset'].unique())
            sequences = st.multiselect("Sequence", self.errors_view_model.df['sequence'].unique())
            dates = st.multiselect("Date", self.errors_view_model.df['date'].unique())

            if st.form_submit_button("Submit", use_container_width=True, type="primary"):
                self.errors_view_model.set_tracker_selection(
                    {
                        'tracker': trackers if trackers else self.errors_view_model.df['tracker'].unique().tolist(),
                        'dataset': datasets if datasets else self.errors_view_model.df['dataset'].unique().tolist(),
                        'sequence': sequences if sequences else self.errors_view_model.df['sequence'].unique().tolist(),
                        'date': dates if dates else self.errors_view_model.df['date'].unique().tolist(),
                    }
                )

    def on_table_change(self, df_table):
        if 'table' in st.session_state:
            edited = st.session_state.table['edited_rows']

            selected_indexes = []
            for i in edited:
                if edited[i]['selected']:
                    selected_indexes.append(i)

            self.errors_view_model.set_selected_rows(df_table.iloc[selected_indexes])


ErrorsPage()

