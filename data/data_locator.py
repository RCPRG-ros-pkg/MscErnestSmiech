import json
import os

import pandas

from utils.utils import create_polygon
from data.singleton_meta import SingletonMeta
from stack import cache_dir, results_dir, results_file


class DataLocator(metaclass=SingletonMeta):
    """
    Locator pattern for data. It makes sure there's single source of truth for data. It holds raw data from tests in
    results as well as calculated values in cache.
    """

    _cache: None | dict[str, pandas.DataFrame] = None
    _results: None | pandas.DataFrame = None
    _current_page: str | None = None

    def provide_results(self) -> pandas.DataFrame:
        if self._results is None:
            self._create_results()

        return self._results

    def _create_results(self):
        try:
            os.makedirs(results_dir)
            self._results = pandas.DataFrame(
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
            _results.date = _results.date.map(lambda f: pandas.to_datetime(f, format='%Y-%m-%d-%H-%M-%S-%f'))
            _results['selected'] = False
            self._results = _results

    def concat_results(self, df: pandas.DataFrame):
        """
        Adds new results to existing ones.
        :param df: results in DataFrame
        :return:
        """
        self._results = pandas.concat([self.provide_results(), df])

    def provide_cache(self) -> dict[str, pandas.DataFrame]:
        if self._cache is None:
            self._create_cache()

        return self._cache

    def _create_cache(self):
        index=pandas.MultiIndex.from_tuples([], names=['Tracker', 'Dataset'])
        self._cache = {
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
            self._get_or_create_cache(n, n != 'average_success_plot')

    def _get_or_create_cache(self, name: str, set_index: bool):
        try:
            self._cache[name].to_csv(f"{cache_dir}/{name}.csv", mode='x', index=False)
        except:
            self._cache[name] = pandas.read_csv(f"{cache_dir}/{name}.csv")
            if set_index:
                self._cache[name] = self._cache[name].set_index(['Tracker', 'Dataset'])

    def provide_current_page(self) -> str | None:
        return self._current_page

    def modify_current_page(self, page_name: str):
        self._current_page = page_name
