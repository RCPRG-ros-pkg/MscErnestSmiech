import glob

import numpy
import pandas
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from analysis.stt_iou import stt_iou
from analysis.accuracy import sequence_accuracy, success_plot
from analysis.longterm import quality_auxiliary, count_frames
from analysis.time import time_quality_auxiliary, sequence_time
from stack import datasets


class ErrorsPage:
    df = st.session_state.results[['tracker', 'dataset', 'sequence', 'date', 'selected']]
    image_displayer: DeltaGenerator | None = None

    def __init__(self) -> None:
        super().__init__()

        if 'count' not in st.session_state:
            st.session_state.count = 0

        if 'images' not in st.session_state:
            st.session_state.images = None

        if 'tracker_selection' not in st.session_state:
            st.session_state.tracker_selection = {
                'tracker': [],
                'dataset': [],
                'sequence': [],
                'date': [],
            }

        if 'selected_rows' not in st.session_state:
            st.session_state.selected_rows = None

        st.set_page_config(page_title="Error frames viewer", page_icon="📈")

        self.sidebar()

        st.markdown("# Error frames viewer")

        if st.session_state.selected_rows is not None:
            if len(st.session_state.selected_rows) == 1:
                self.display_picture()

            _df = st.session_state.results[st.session_state.results['date'].isin(st.session_state.selected_rows['date'])]
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

        df_table = self.df[
            self.df['tracker'].isin(st.session_state.tracker_selection['tracker']) &
            self.df['dataset'].isin(st.session_state.tracker_selection['dataset']) &
            self.df['sequence'].isin(st.session_state.tracker_selection['sequence']) &
            self.df['date'].isin(st.session_state.tracker_selection['date'])
            ]

        st.markdown("## Selector")
        st.data_editor(
            df_table,
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
            args=[df_table]
        )

    def display_picture(self):
        d = st.session_state.selected_rows.head(1)
        d = d[['tracker', 'dataset', 'sequence', 'date']].astype(str).values.flatten().tolist()
        d[1] = datasets[d[1]]
        st.session_state.images = glob.glob(f"./raw/errors/{'/'.join(d)}/*")
        try:
            image = st.session_state.images[st.session_state.count]
        except IndexError:
            st.session_state.count = 0
            image = st.session_state.images[st.session_state.count]

        st.markdown("## Errors")

        _, col, _ = st.columns((2, 11, 2))
        with col:
            st.image(image, width=500)

            _, col1, col2, col3, _ = st.columns((3, 1, 1, 1, 3))
            with col1:
                st.button("←", on_click=self.decrement_counter)
            with col2:
                st.write(f"{st.session_state.count}/{len(st.session_state.images)}")
            with col3:
                st.button("→", on_click=self.increment_counter)

    def sidebar(self) -> None:
        with st.sidebar.form("Options"):
            trackers = st.multiselect('Trackers', self.df['tracker'].unique())
            datasets = st.multiselect("Datasets", self.df['dataset'].unique())
            sequences = st.multiselect("Sequence", self.df['sequence'].unique())
            dates = st.multiselect("Date", self.df['date'].unique())

            if st.form_submit_button("Submit", use_container_width=True, type="primary"):
                if not trackers:
                    pass
                st.session_state.tracker_selection = {
                    'tracker': trackers if trackers else self.df['tracker'].unique().tolist(),
                    'dataset': datasets if datasets else self.df['dataset'].unique().tolist(),
                    'sequence': sequences if sequences else self.df['sequence'].unique().tolist(),
                    'date': dates if dates else self.df['date'].unique().tolist(),
                }

    @staticmethod
    def increment_counter():
        st.session_state.count += 1

    @staticmethod
    def decrement_counter():
        st.session_state.count -= 1

    @staticmethod
    def on_table_change(df_table):
        if 'table' in st.session_state:
            edited = st.session_state.table['edited_rows']

            selected_indexes = []
            for i in edited:
                if edited[i]['selected']:
                    selected_indexes.append(i)

            st.session_state.selected_rows = df_table.iloc[selected_indexes]


ErrorsPage()

