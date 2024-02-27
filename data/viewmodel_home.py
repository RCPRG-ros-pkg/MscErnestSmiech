import pandas
import streamlit as st

from data.singleton_meta import SingletonMeta
from data.state_locator import StateLocator


class HomeViewModel(metaclass=SingletonMeta):

    state_locator = StateLocator()

    tracker_names = [tracker.name for tracker in state_locator.provide_trackers()]

    def on_iou_table_change(self, df_tracker_dataset):
        if 'show_iou_trackers' in st.session_state:
            edited = st.session_state.show_iou_trackers['edited_rows']

            for i in edited:
                self.state_locator.provide_table_selected_trackers().loc[*df_tracker_dataset.iloc[i].tolist()] = edited[i]['Selected']

    def get_selected_trackers(self) -> pandas.DataFrame:
        selected = self.state_locator.provide_table_selected_trackers()
        return selected.loc[selected['Selected'] == True]

    def get_quality_plot(self) -> None | pandas.DataFrame:
        selected = self.get_selected_trackers()
        if selected.empty:
            return None

        ts = self.state_locator.provide_cache()['average_success_plot']
        ts = ts.set_index(['Tracker', 'Dataset'])
        ts = ts.loc[selected.index]
        ts = ts.reset_index()
        ts['TrackerDataset'] = ts[['Tracker', 'Dataset']].agg(' - '.join, axis=1)

        return ts

    def get_ar_plot(self) -> None | pandas.DataFrame:
        selected = self.get_selected_trackers()
        if selected.empty:
            return None

        ra = self.state_locator.provide_cache()['accuracy_robustness']
        ra = ra.loc[selected.index]
        ra = ra.reset_index()
        ra['TrackerDataset'] = ra[['Tracker', 'Dataset']].agg(' - '.join, axis=1)

        return ra

    def get_iou_table(self) -> None | pandas.DataFrame:
        ts = self.state_locator.provide_cache()['average_success_plot']
        ra = self.state_locator.provide_cache()['accuracy_robustness']
        qa = self.state_locator.provide_cache()['average_quality_auxiliary']
        ac = self.state_locator.provide_cache()['average_accuracy']
        stt = self.state_locator.provide_cache()['average_stt_iou']

        if len(ts) <= 0 or len(ra) <= 0 or len(qa) <= 0 or len(ac) <= 0:
            return None

        # todo performance
        success = ts.groupby(['Tracker', 'Dataset']).apply(lambda x: x['Success'].tolist())

        df = pandas.concat(
            [stt, ac, ra, qa, success, self.state_locator.provide_table_selected_trackers()],
            axis=1,
        ).rename(columns={0: 'success'}).reset_index()

        return df

    def get_time_table(self) -> None | pandas.DataFrame:
        qa = self.state_locator.provide_cache()['average_time_quality_auxiliary']
        ac = self.state_locator.provide_cache()['average_time']

        if len(qa) <= 0 or len(ac) <= 0:
            return None

        df = pandas.concat(
            [ac, qa, self.state_locator.provide_table_selected_trackers()],
            axis=1,
        ).reset_index()

        return df
