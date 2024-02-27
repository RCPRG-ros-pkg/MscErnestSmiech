import pandas
import streamlit as st

import tracker
from data.data_locator import DataLocator
from data.singleton_meta import SingletonMeta


class StateLocator(metaclass=SingletonMeta):
    _data_locator = DataLocator()

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
        indexes = self._data_locator.provide_cache()['average_accuracy'].index
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
