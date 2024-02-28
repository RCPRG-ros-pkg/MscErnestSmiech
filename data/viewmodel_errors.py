import glob

import numpy
import pandas

import streamlit as st

from analysis.accuracy import success_plot, sequence_accuracy
from analysis.longterm import quality_auxiliary, count_frames
from analysis.stt_iou import stt_iou
from analysis.time import sequence_time, time_quality_auxiliary
from data.data_locator import DataLocator
from data.singleton_meta import SingletonMeta
from data.state_locator import StateLocator
from stack import datasets


class ErrorsViewModel(metaclass=SingletonMeta):
    _data_locator = DataLocator()
    _state_locator = StateLocator()

    _images: [str] = []
    _selected_rows: None | pandas.DataFrame = None

    def __init__(self):
        super().__init__()

        self._tracker_selection = {
            'tracker': self.df['tracker'].unique().tolist(),
            'dataset': self.df['dataset'].unique().tolist(),
            'sequence': self.df['sequence'].unique().tolist(),
            'date': self.df['date'].unique().tolist(),
        }

    def get_image_count(self) -> int:
        return self._state_locator.provide_image_count()

    def set_selected_rows(self, df: pandas.DataFrame):
        self._selected_rows = df

    def get_selected_rows(self) -> None | pandas.DataFrame:
        return self._selected_rows

    def get_images(self) -> [str]:
        return self._images

    def get_image(self, index) -> str:
        return self._images[index]

    def set_images(self, images: [str]):
        self._images = images

    def load_images(self):
        d = self.get_selected_rows().head(1)
        d = d[['tracker', 'dataset', 'sequence', 'date']].astype(str).values.flatten().tolist()
        d[1] = datasets[d[1]]
        self.set_images(glob.glob(f"./raw/errors/{'/'.join(d)}/*"))

    def set_tracker_selection(
            self,
            trackers: numpy.ndarray,
            datasets: numpy.ndarray,
            sequences: numpy.ndarray,
            dates: numpy.ndarray
    ):
        self._tracker_selection = {
            'tracker': trackers if trackers else self.df['tracker'].unique().tolist(),
            'dataset': datasets if datasets else self.df['dataset'].unique().tolist(),
            'sequence': sequences if sequences else self.df['sequence'].unique().tolist(),
            'date': dates if dates else self.df['date'].unique().tolist(),
        }

    def get_df_table(self):
        return self.df[
            self.df['tracker'].isin(self._tracker_selection['tracker']) &
            self.df['dataset'].isin(self._tracker_selection['dataset']) &
            self.df['sequence'].isin(self._tracker_selection['sequence']) &
            self.df['date'].isin(self._tracker_selection['date'])
            ]

    @property
    def df(self) -> pandas.DataFrame:
        return self._data_locator.provide_results()[['tracker', 'dataset', 'sequence', 'date', 'selected']]

    def on_table_change(self, df_table):
        if 'table' in st.session_state:
            edited = st.session_state.table['edited_rows']

            selected_indexes = []
            for i in edited:
                if edited[i]['selected']:
                    selected_indexes.append(i)

            self.set_selected_rows(df_table.iloc[selected_indexes])

    def increase_image_count(self):
        self._state_locator.increase_image_count()

    def decrease_image_count(self):
        self._state_locator.decrease_image_count()

    def zero_image_count(self):
        self._state_locator.zero_image_count()

    def get_success_plot(self) -> None | pandas.DataFrame:
        selected_rows = self.get_selected_rows()
        if selected_rows is None:
            return None

        _df = self._data_locator.provide_results()[self._data_locator.provide_results()['date'].isin(selected_rows['date'])]

        success = pandas.DataFrame(map(success_plot, _df['trajectory'], _df['groundtruth']), index=_df['date']) \
            .unstack() \
            .reset_index()[['date', 0]] \
            .sort_values(by=['date', 0]) \
            .reset_index(drop=True)

        arr = numpy.array([item for row in success[0] for item in row])
        arr = arr.reshape((len(arr)//2, 2))
        tmp = pandas.DataFrame(arr, columns=['threshold', 'success'])
        tmp['date'] = success['date']

        return tmp

    def get_time_tables(self) -> None | pandas.DataFrame:
        selected_rows = self.get_selected_rows()
        if selected_rows is None:
            return None

        _df = self._data_locator.provide_results()[self._data_locator.provide_results()['date'].isin(selected_rows['date'])]

        times_quality_auxiliary = pandas.DataFrame(
            numpy.vectorize(time_quality_auxiliary)(_df['times'], _df['trajectory'], _df['groundtruth'])
        ).T
        times_sequence = pandas.DataFrame(
            numpy.vectorize(sequence_time)(_df['times'], _df['trajectory'], _df['groundtruth']),
            columns=["success"]
        )

        return pandas.concat(
            [
                _df[['tracker', 'dataset', 'sequence', 'date']],
                times_quality_auxiliary.rename(columns={0: 'robustness', 1: "nre", 2: "dre", 3: "ad"}),
                times_sequence,
            ],
            axis=1,
        )

    def get_iou_table(self) -> None | pandas.DataFrame:
        selected_rows = self.get_selected_rows()
        if selected_rows is None:
            return None

        _df = self._data_locator.provide_results()[self._data_locator.provide_results()['date'].isin(selected_rows['date'])]

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

        return pandas.concat(
            [
                _df[['tracker', 'dataset', 'sequence', 'date']],
                quality.rename(columns={0: "nre", 1: "dre", 2: "ad"}),
                accuracy,
                stt_ious,
                robustness,
            ],
            axis=1,
        ).rename(columns={0: "success"})

    def get_ar_plot(self):
        return self.get_iou_table()[['accuracy', 'robustness', 'date']]

