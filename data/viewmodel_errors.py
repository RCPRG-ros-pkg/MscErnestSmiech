import glob
from datetime import datetime

import cv2
import numpy
import pandas
import pandas as pd

import streamlit as st

from analysis.accuracy import success_plot, sequence_accuracy
from analysis.longterm import quality_auxiliary, count_frames
from analysis.stt_iou import stt_iou
from analysis.time import sequence_time, time_quality_auxiliary
from analysis.utils import calculate_overlaps
from data.data_locator import DataLocator
from data.singleton_meta import SingletonMeta
from data.state_locator import StateLocator
from stack import datasets, tests_dir


class ErrorsViewModel(metaclass=SingletonMeta):
    """
    ViewModel for ErrorsPage. It's a single source for handling data.
    """
    _data_locator = DataLocator()
    _state_locator = StateLocator()

    _pd_images: pd.DataFrame | None = None
    _selected_rows: None | pandas.DataFrame = None

    def __init__(self):
        super().__init__()

        self._tracker_selection = {
            'tracker': self.df['tracker'].unique().tolist(),
            'dataset': self.df['dataset'].unique().tolist(),
            'sequence': self.df['sequence'].unique().tolist(),
            'date': self.df['date'].unique().tolist(),
        }

    def get_image_index(self) -> int:
        return self._state_locator.provide_selected_image_index()

    def set_selected_rows(self, df: pandas.DataFrame):
        self._selected_rows = df

    def get_selected_rows(self) -> None | pandas.DataFrame:
        return self._selected_rows

    def get_images(self) -> [str]:
        return self._pd_images['pictures'].tolist()

    def get_current_image(self) -> cv2.Mat | numpy.ndarray:
        trajectory, groundtruth, image_url = self._pd_images.iloc[self._state_locator.provide_selected_image_index()]
        image = cv2.imread(image_url, cv2.IMREAD_COLOR)

        if groundtruth is not None:
            _xx, _yy = groundtruth.exterior.coords.xy
            _coords = numpy.array(tuple(zip(_xx, _yy))).astype(int)
            cv2.polylines(image, [_coords], False, (0, 255, 0), 3)

        if trajectory is not None:
            _xx, _yy = trajectory.exterior.coords.xy
            _coords = numpy.array(tuple(zip(_xx, _yy))).astype(int)
            cv2.polylines(image, [_coords], False, (255, 0, 0), 3)

        return image

    def set_pd_images(self, pd_images: pandas.DataFrame):
        self._pd_images = pd_images

    def load_images(self):
        """
        Loads images for selected options.
        """
        selected_rows = self.get_selected_rows()
        if selected_rows is None:
            return None

        _df = self._data_locator.provide_results()[self._data_locator.provide_results()['date'].isin(selected_rows['date'])]

        overlaps = calculate_overlaps(_df['trajectory'].iloc[0], _df['groundtruth'].iloc[0])
        overlaps = numpy.array(overlaps)
        indexes = numpy.argwhere(overlaps == 0).flatten()

        selected_dataset, selected_sequence = selected_rows.head(1)[['dataset', 'sequence']].astype(str).values.flatten().tolist()
        pictures = numpy.array(glob.glob(f"{tests_dir}/{datasets[selected_dataset]}/sequences/{selected_sequence}/color/*.jpg"))
        pictures.sort()

        _pd = pandas.DataFrame(
            # index=indexes+2,
            columns=['trajectory', 'groundtruth', 'pictures'],
            data={
                'trajectory': numpy.array(_df['trajectory'].iloc[0])[indexes],
                'groundtruth': numpy.array(_df['groundtruth'].iloc[0])[indexes],
                'pictures': pictures[indexes + 1]
            }
        )

        self.set_pd_images(_pd)

    def set_tracker_selection(
            self,
            trackers: numpy.ndarray,
            datasets: numpy.ndarray,
            sequences: numpy.ndarray,
            start_date: datetime,
            end_date: datetime,
    ):
        """
        Method sets selected options. If any wasn't set then it fallbacks to last selection.

        :param trackers:
        :param datasets:
        :param sequences:
        :param start_date:
        :param end_date:
        """
        foo = self.df['date'].unique()
        self._tracker_selection = {
            'tracker': trackers if trackers else self.df['tracker'].unique().tolist(),
            'dataset': datasets if datasets else self.df['dataset'].unique().tolist(),
            'sequence': sequences if sequences else self.df['sequence'].unique().tolist(),
            'date': foo[(foo > start_date) & (foo < end_date)],
        }

    def get_df_table(self) -> pandas.DataFrame:
        """
        :return: DataFrame based on selection.
        """
        return self.df[
            self.df['tracker'].isin(self._tracker_selection['tracker']) &
            self.df['dataset'].isin(self._tracker_selection['dataset']) &
            self.df['sequence'].isin(self._tracker_selection['sequence']) &
            self.df['date'].isin(self._tracker_selection['date'])
            ]

    @property
    def df(self) -> pandas.DataFrame:
        """
        :return: DataFrame of raw results
        """
        return self._data_locator.provide_results()[['tracker', 'dataset', 'sequence', 'date', 'selected']]

    def on_table_change(self, df_table):
        """
        Method that handles Streamlit table changes.

        :param df_table:
        """
        if 'table' in st.session_state:
            edited = st.session_state.table['edited_rows']

            selected_indexes = []

            for i in edited:
                if edited[i]['selected']:
                    selected_indexes.append(i)

            if len(selected_indexes) == 1:
                self.zero_selected_image_index()

            self.set_selected_rows(df_table.iloc[selected_indexes])

    def increase_selected_image_index(self):
        self._state_locator.increase_selected_image_index()

    def decrease_selected_image_index(self):
        self._state_locator.decrease_selected_image_index()

    def zero_selected_image_index(self):
        self._state_locator.zero_selected_image_index()

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
        tmp['date'] = success['date'].map(lambda x: x.strftime('%Y-%m-%d-%H-%M-%S-%f'))

        return tmp

    def get_time_table(self) -> None | pandas.DataFrame:
        selected_rows = self.get_selected_rows()
        if selected_rows is None:
            return None

        _df = self._data_locator.provide_results()[self._data_locator.provide_results()['date'].isin(selected_rows['date'])]

        if _df is None or _df.empty:
            return None

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
                times_quality_auxiliary.rename(columns={0: 'robustness', 1: "nre", 2: "dre", 3: "ad"}).set_index(_df.index),
                times_sequence.set_index(_df.index),
            ],
            axis=1,
        ).reset_index(drop=True)

    def get_iou_table(self) -> None | pandas.DataFrame:
        selected_rows = self.get_selected_rows()
        if selected_rows is None:
            return None

        _df = self._data_locator.provide_results()[self._data_locator.provide_results()['date'].isin(selected_rows['date'])]

        if _df is None or _df.empty:
            return None

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

        wut = pandas.concat(
            [
                _df[['tracker', 'dataset', 'sequence', 'date']],
                quality.rename(columns={0: "nre", 1: "dre", 2: "ad"}).set_index(_df.index),
                accuracy.set_index(_df.index),
                stt_ious.set_index(_df.index),
                robustness.set_index(_df.index),
            ],
            axis=1,
        ).rename(columns={0: "success"}).reset_index(drop=True)

        return wut

    def get_ar_plot(self) -> pandas.DataFrame | None:
        df = self.get_iou_table()
        if df is None:
            return None
        df = df[['accuracy', 'robustness', 'date']]
        df['date'] = df['date'].map(lambda x: x.strftime('%Y-%m-%d-%H-%M-%S-%f'))
        return df

    def get_available_dates(self) -> numpy.ndarray[datetime.date]:
        return self.df['date'].unique().map(lambda d: d.date()).unique()

    def get_available_times(self) -> numpy.ndarray[datetime.time]:
        return self.df['date'].unique().map(lambda d: d.time()).unique()

    def set_page_name(self):
        if self._data_locator.provide_current_page() != "errors":
            self._selected_rows = None
            self._pd_images = None
            self._data_locator.modify_current_page("errors")
            self.zero_selected_image_index()
