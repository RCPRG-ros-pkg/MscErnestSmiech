import pandas
import streamlit as st

import tracker
from data.data_locator import DataLocator
from data.singleton_meta import SingletonMeta


class StateLocator(metaclass=SingletonMeta):
    """
    Locator pattern for streamlit state. Streamlit creates global states. It means that different pages can override
    their states if they use same names. To prevent that all of them are held here as single source of truth,
    """
    _data_locator = DataLocator()

    def provide_trackers(self) -> list[tracker.Tracker]:
        """
        Used in TestsPage to get trackers.
        """
        if 'trackers' not in st.session_state:
            self._create_trackers()

        return st.session_state.trackers

    @staticmethod
    def _create_trackers():
        st.session_state.trackers = [_tracker() for _tracker in tracker.get_concrete_classes(tracker.Tracker)]

    def provide_table_selected_trackers(self) -> pandas.DataFrame:
        """
        Used in TestsPage to get selected trackers.
        """
        if 'table_selected_trackers' not in st.session_state:
            self._create_table_selected_trackers()

        return st.session_state.table_selected_trackers

    def _create_table_selected_trackers(self):
        indexes = self._data_locator.provide_cache()['average_accuracy'].index
        st.session_state.table_selected_trackers = pandas.DataFrame([False]*indexes.size,index=indexes,columns=['Selected'])

    @staticmethod
    def clear_selection():
        """
        Used in TestsPage to clear selected tracker and dataset for testing.
        """
        st.session_state.selection = []

    def provide_selection(self) -> [(str, str)]:
        """
        Used in TestsPage to provide selected tracker and dataset for testing.
        """
        if 'selection' not in st.session_state:
            self._create_selection()

        return st.session_state.selection

    @staticmethod
    def _create_selection():
        st.session_state["selection"]: [(str, str)] = []

    def provide_selected_image_index(self) -> int:
        """
        Used in ErrorsPage to provide current picture index.
        """
        if 'selected_image_index' not in st.session_state:
            self._create_selected_image_index()

        return st.session_state.selected_image_index

    @staticmethod
    def increase_selected_image_index():
        """
        Used in ErrorsPage to select next picture
        """
        st.session_state.selected_image_index += 1

    @staticmethod
    def decrease_selected_image_index():
        """
        Used in ErrorsPage to select previous picture
        """
        st.session_state.selected_image_index -= 1

    @staticmethod
    def zero_selected_image_index():
        """
        Used in ErrorsPage to return to first picture
        """
        st.session_state.selected_image_index = 0

    def _create_selected_image_index(self):
        self.zero_selected_image_index()
