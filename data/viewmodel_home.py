import pandas
import streamlit as st
from shapely import Polygon

from analysis.accuracy import average_accuracy, average_success_plot
from analysis.longterm import accuracy_robustness, average_quality_auxiliary
from analysis.stt_iou import average_stt_iou
from analysis.time import average_time_quality_auxiliary, average_time
from analysis.utils import polygon_to_floatarray
from data.data_locator import DataLocator
from data.singleton_meta import SingletonMeta
from data.state_locator import StateLocator
from stack import cache_dir, results_file


class HomeViewModel(metaclass=SingletonMeta):

    state_locator = StateLocator()
    data_locator = DataLocator()

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

        ts = self.data_locator.provide_cache()['average_success_plot']
        ts = ts.set_index(['Tracker', 'Dataset'])
        ts = ts.loc[selected.index]
        ts = ts.reset_index()
        ts['TrackerDataset'] = ts[['Tracker', 'Dataset']].agg(' - '.join, axis=1)

        return ts

    def get_ar_plot(self) -> None | pandas.DataFrame:
        selected = self.get_selected_trackers()
        if selected.empty:
            return None

        ra = self.data_locator.provide_cache()['accuracy_robustness']
        ra = ra.loc[selected.index]
        ra = ra.reset_index()
        ra['TrackerDataset'] = ra[['Tracker', 'Dataset']].agg(' - '.join, axis=1)

        return ra

    def get_iou_table(self) -> None | pandas.DataFrame:
        ts = self.data_locator.provide_cache()['average_success_plot']
        ra = self.data_locator.provide_cache()['accuracy_robustness']
        qa = self.data_locator.provide_cache()['average_quality_auxiliary']
        ac = self.data_locator.provide_cache()['average_accuracy']
        stt = self.data_locator.provide_cache()['average_stt_iou']

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
        qa = self.data_locator.provide_cache()['average_time_quality_auxiliary']
        ac = self.data_locator.provide_cache()['average_time']

        if len(qa) <= 0 or len(ac) <= 0:
            return None

        df = pandas.concat(
            [ac, qa, self.state_locator.provide_table_selected_trackers()],
            axis=1,
        ).reset_index()

        return df

    def accuracy_robustness(
            self,
            tracker: str,
            dataset: str,
            trajectories: list[list[Polygon | None]],
            groundtruths: list[list[Polygon]]
    ):
        g = self.data_locator.provide_results()
        df = g.loc[(g['tracker'] == tracker) & (g['dataset'] == dataset), ['trajectory', 'groundtruth']]

        _trajectories = trajectories + df['trajectory'].tolist()
        _groundtruths = groundtruths + df['groundtruth'].tolist()

        value = accuracy_robustness(_trajectories, _groundtruths)

        self.data_locator.provide_cache()['accuracy_robustness'].loc[(tracker, dataset), :] = value

    def average_quality_auxiliary(
            self,
            tracker: str,
            dataset: str,
            trajectories: list[list[Polygon | None]],
            groundtruths: list[list[Polygon]]
    ):
        g = self.data_locator.provide_results()
        df = g.loc[(g['tracker'] == tracker) & (g['dataset'] == dataset), ['trajectory', 'groundtruth']]

        _trajectories = trajectories + df['trajectory'].tolist()
        _groundtruths = groundtruths + df['groundtruth'].tolist()

        value = average_quality_auxiliary(_trajectories, _groundtruths)

        self.data_locator.provide_cache()['average_quality_auxiliary'].loc[(tracker, dataset), :] = value

    def average_accuracy(
            self,
            tracker: str,
            dataset: str,
            trajectories: list[list[Polygon | None]],
            groundtruths: list[list[Polygon]]
    ):
        g = self.data_locator.provide_results()
        df = g.loc[(g['tracker'] == tracker) & (g['dataset'] == dataset), ['trajectory', 'groundtruth']]

        _trajectories = trajectories + df['trajectory'].tolist()
        _groundtruths = groundtruths + df['groundtruth'].tolist()

        value = average_accuracy(_trajectories, _groundtruths)

        self.data_locator.provide_cache()['average_accuracy'].loc[(tracker, dataset), :] = value

    def average_success_plot(
            self,
            tracker: str,
            dataset: str,
            trajectories: list[list[Polygon | None]],
            groundtruths: list[list[Polygon]]
    ):
        g = self.data_locator.provide_results()
        df = g.loc[(g['tracker'] == tracker) & (g['dataset'] == dataset), ['trajectory', 'groundtruth']]

        _trajectories = trajectories + df['trajectory'].tolist()
        _groundtruths = groundtruths + df['groundtruth'].tolist()

        axis_x, axis_y = average_success_plot(_trajectories, _groundtruths)

        df = self.data_locator.provide_cache()['average_success_plot']
        try:
            df.drop(df[(df['Tracker'] == tracker) & (df['Dataset'] == dataset)].index, inplace=True)
        except KeyError:
            pass
        self.data_locator.provide_cache()['average_success_plot'] = pandas.concat([
            df,
            pandas.DataFrame({
                'Tracker': [tracker] * len(axis_x),
                'Dataset': [dataset] * len(axis_x),
                'Threshold': axis_x,
                'Success': axis_y
            })
        ], ignore_index=True)

    def average_stt_iou(
            self,
            tracker: str,
            dataset: str,
            trajectories: list[list[Polygon | None]],
            groundtruths: list[list[Polygon]]):
        g = self.data_locator.provide_results()
        df = g.loc[(g['tracker'] == tracker) & (g['dataset'] == dataset), ['trajectory', 'groundtruth']]

        _trajectories = trajectories + df['trajectory'].tolist()
        _groundtruths = groundtruths + df['groundtruth'].tolist()

        value = average_stt_iou(_trajectories, _groundtruths)

        self.data_locator.provide_cache()['average_stt_iou'].loc[(tracker, dataset), :] = value

    def average_time_quality_auxiliary(
            self,
            tracker: str,
            dataset: str,
            times: list[list[int]],
            trajectories: list[list[Polygon | None]],
            groundtruths: list[list[Polygon]]
    ):
        g = self.data_locator.provide_results()
        df = g.loc[(g['tracker'] == tracker) & (g['dataset'] == dataset), ['trajectory', 'groundtruth', 'times']]

        _times = times + df['times'].tolist()
        _trajectories = trajectories + df['trajectory'].tolist()
        _groundtruths = groundtruths + df['groundtruth'].tolist()

        value = average_time_quality_auxiliary(_times, _trajectories, _groundtruths)

        self.data_locator.provide_cache()['average_time_quality_auxiliary'].loc[(tracker, dataset), :] = value

    def average_time(
            self,
            tracker: str,
            dataset: str,
            times: list[list[int]],
            trajectories: list[list[Polygon | None]],
            groundtruths: list[list[Polygon]]
    ):
        g = self.data_locator.provide_results()
        df = g.loc[(g['tracker'] == tracker) & (g['dataset'] == dataset), ['trajectory', 'groundtruth', 'times']]

        _times = times + df['times'].tolist()
        _trajectories = trajectories + df['trajectory'].tolist()
        _groundtruths = groundtruths + df['groundtruth'].tolist()

        value = average_time(_times, _trajectories, _groundtruths)

        self.data_locator.provide_cache()['average_time'].loc[(tracker, dataset), :] = value

    def save_results(
            self,
            date: list[str],
            tracker: str,
            dataset: str,
            sequences: list[str],
            times: list[list[int]],
            trajectories: list[list[Polygon | None]],
            groundtruths: list[list[Polygon]],
    ):
        df = pandas.DataFrame({
            'date': date,
            'tracker': tracker,
            'dataset': dataset,
            'sequence': sequences,
            'selected': [False]*len(sequences),
            'trajectory': trajectories,
            'groundtruth': groundtruths,
            'times': times
        })

        self.data_locator.concat_results(df)

        self.data_locator.provide_cache()['average_accuracy'].to_csv(f"{cache_dir}/average_accuracy.csv")
        self.data_locator.provide_cache()['average_stt_iou'].to_csv(f"{cache_dir}/average_stt_iou.csv")
        self.data_locator.provide_cache()['average_time'].to_csv(f"{cache_dir}/average_time.csv")
        self.data_locator.provide_cache()['average_success_plot'].to_csv(f"{cache_dir}/average_success_plot.csv", index=False)
        self.data_locator.provide_cache()['average_quality_auxiliary'].to_csv(f"{cache_dir}/average_quality_auxiliary.csv")
        self.data_locator.provide_cache()['average_time_quality_auxiliary'].to_csv(f"{cache_dir}/average_time_quality_auxiliary.csv")
        self.data_locator.provide_cache()['accuracy_robustness'].to_csv(f"{cache_dir}/accuracy_robustness.csv")
    
        df = pandas.DataFrame({
            'date': date,
            'tracker': tracker,
            'dataset': dataset,
            'sequence': sequences,
            'trajectory': [[[] if polygon is None else polygon_to_floatarray(polygon) for polygon in trajectory] for trajectory in trajectories],
            'groundtruth': [[[] if polygon is None else polygon_to_floatarray(polygon) for polygon in groundtruth] for groundtruth in groundtruths],
            'times': times
        })
    
        df.to_csv(results_file, mode='a', header=False, index=False)
