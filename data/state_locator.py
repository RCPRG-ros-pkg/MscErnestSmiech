import json
import os

import pandas
import streamlit as st

import tracker
from analysis.utils import create_polygon
from data.singleton_meta import SingletonMeta
from stack import results_dir, results_file, cache_dir
from utils.utils import get_or_create_cache


class StateLocator(metaclass=SingletonMeta):

    def provide_trackers(self) -> list[tracker.Tracker]:
        if 'trackers' not in st.session_state:
            self._create_trackers()

        return st.session_state.trackers

    @staticmethod
    def _create_trackers():
        st.session_state.trackers = [_tracker() for _tracker in tracker.get_concrete_classes(tracker.Tracker)]

    def provide_table_selected_trackers(self) -> pandas.DataFrame:
        if 'table_selected_trackers' not in st.session_state:
            self._create_table_selected_trackers()

        return st.session_state.table_selected_trackers

    def _create_table_selected_trackers(self):
        indexes = self.provide_cache()['average_accuracy'].index
        st.session_state.table_selected_trackers = pandas.DataFrame([False]*indexes.size,index=indexes,columns=['Selected'])

    @staticmethod
    def clear_selection():
        st.session_state.selection = []

    def provide_selection(self) -> [(str, str)]:
        if 'selection' not in st.session_state:
            self._create_selection()

        return st.session_state.selection

    @staticmethod
    def _create_selection():
        st.session_state["selection"]: [(str, str)] = []

    def provide_cache(self) -> pandas.DataFrame:
        if 'cache' not in st.session_state:
            self._create_cache()

        return st.session_state.cache

    @staticmethod
    def _create_cache():
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
            get_or_create_cache(n, n != 'average_success_plot') # todo move here

    def provide_results(self) -> pandas.DataFrame:
        if 'results' not in st.session_state:
            self._create_results()

        return st.session_state.results

    @staticmethod
    def _create_results():
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
