import pandas

from data.data_locator import DataLocator
from data.singleton_meta import SingletonMeta


class ErrorsViewModel(metaclass=SingletonMeta):
    _data_locator = DataLocator()

    _images: [str] = []
    _tracker_selection = {
        'tracker': [],
        'dataset': [],
        'sequence': [],
        'date': [],
    }
    _selected_rows: None | pandas.DataFrame = None

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

    def set_tracker_selection(self, d: dict[str, list]):
        self._tracker_selection = d

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
