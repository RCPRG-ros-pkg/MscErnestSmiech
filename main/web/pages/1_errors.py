import glob
import os

import pandas
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder
from streamlit.delta_generator import DeltaGenerator

# todo move to file
datasets = {
    'VOT Basic Test Stack': './raw/tests/basic/sequences/',
    'VOT-ST2022 bounding-box challenge': './raw/tests/vot2022stb/sequences/',
}


class ErrorsPage:
    image_displayer: DeltaGenerator | None = None
    button_previous: DeltaGenerator | None = None
    button_next: DeltaGenerator | None = None
    text_image_number: DeltaGenerator | None = None

    def __init__(self) -> None:
        super().__init__()

        if 'count' not in st.session_state:
            st.session_state.count = 0

        if 'images' not in st.session_state:
            st.session_state.images = None

        st.set_page_config(page_title="Error frames viewer", page_icon="📈")

        st.markdown("# Error frames viewer")

        if st.session_state.images is not None:
            try:
                image = st.session_state.images[st.session_state.count]
            except IndexError:
                st.session_state.count = 0
                image = st.session_state.images[st.session_state.count]

            self.image_displayer = st.image(image, use_column_width=True)

            _, col1, col2, col3, _ = st.columns((3, 1, 1, 1, 3))
            with col1:
                self.button_previous = st.button("←", on_click=self.decrement_counter)
            with col2:
                self.text_image_number = st.write(f"{st.session_state.count}/{len(st.session_state.images)}")
            with col3:
                self.button_next = st.button("→", on_click=self.increment_counter)
        else:
            self.image_displayer = st.empty()
            _, col1, col2, col3, _ = st.columns((3, 1, 1, 1, 3))
            with col1:
                self.button_previous = st.button("←", on_click=self.decrement_counter)
            with col2:
                self.text_image_number = st.write(f"")
            with col3:
                self.button_next = st.button("→", on_click=self.increment_counter)

        df = get_errors_dataframe()
        return_value = AgGrid(
            data=df,
            gridOptions=self.get_grid_style(df),
            fit_columns_on_grid_load=True,
        )

        if return_value['selected_rows']:
            st.session_state.images = glob.glob(f"{return_value['selected_rows'][0]['path']}/*")
            try:
                image = st.session_state.images[st.session_state.count]
            except IndexError:
                st.session_state.count = 0
                image = st.session_state.images[st.session_state.count]
            self.image_displayer.image(image, use_column_width=True)

    @staticmethod
    def increment_counter():
        st.session_state.count += 1

    @staticmethod
    def decrement_counter():
        st.session_state.count -= 1

    @staticmethod
    def get_grid_style(df: pandas.DataFrame):
        builder = GridOptionsBuilder.from_dataframe(df)
        builder.configure_selection(selection_mode='single', use_checkbox=False)
        grid_options = builder.build()

        column_defs = grid_options["columnDefs"]
        columns_to_hide = ["path"]

        # update the column definitions to hide the specified columns
        for col in column_defs:
            if col["headerName"] in columns_to_hide:
                col["hide"] = True

        return grid_options


def get_errors_dataframe() -> pandas.DataFrame:
    data = {
        'tracker': [],
        'dataset': [],
        'sequence': [],
        'date': [],
        'path': []
    }
    for tracker in get_sub_dirs("output"):
        for dataset in get_sub_dirs(tracker):
            for sequence in get_sub_dirs(dataset):
                for date in get_sub_dirs(sequence):
                    _, _tracker, _dataset, _sequence, _date = date.split('/')
                    for key in datasets:
                        print(_dataset, _dataset.replace('_', '/'))
                        if _dataset.replace('_', '/') in datasets[key]:
                            data['tracker'].append(_tracker)
                            data['dataset'].append(key)
                            data['sequence'].append(_sequence)
                            data['date'].append(_date)
                            data['path'].append(f'{date}/errors')
                            break

    return pandas.DataFrame(data)


def get_sub_dirs(folder: str) -> [str]:
    return [f.path for f in os.scandir(folder) if f.is_dir()]


ErrorsPage()
