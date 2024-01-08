import csv
import inspect
import json
import os
import sys
from json import JSONEncoder, JSONDecoder
from pathlib import Path
from pprint import pprint

import numpy
import pandas
from shapely import Polygon
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))

import main.tracker as tracker


def get_concrete_classes(cls):
    for subclass in cls.__subclasses__():
        yield from get_concrete_classes(subclass)
        if not inspect.isabstract(subclass):
            yield subclass


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


class PolygonEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Polygon):
            _xx, _yy = obj.exterior.coords.xy
            _coords = numpy.array(tuple(zip(_xx, _yy))).astype(float) # todo unnecessary?
            return [x for xs in _coords for x in xs]

        return JSONEncoder.default(self, obj)
    

class PolygonDecoder(JSONDecoder): # todo zapis i odczyt z pliku 'results'. Później liczenie samych wyników i zapis do 'cache'
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    @staticmethod
    def object_hook(obj):
        if isinstance(obj, list):
            return Polygon(zip(obj[::2], obj[1::2]))

        return obj



datasets = {
    'VOT Basic Test Stack': './raw/tests/basic/sequences',
    'VOT-ST2022 bounding-box challenge': './raw/tests/vot2022stb/sequences'
}
# vot2022/sts
results_dir = './raw/results'
results_file = f'{results_dir}/results.csv'


# fixme after first test with empty table, table has extra row
def get_ground_truth_positions(_file_name: str) -> list[Polygon]:
    with open(_file_name) as csvfile:
        # _ground_truth_pos = [[int(x) for x in y] for y in csv.reader(csvfile, delimiter='\t')]
        _ground_truth_pos = [create_polygon([abs(int(float(x))) for x in y]) for y in csv.reader(csvfile)]

    return _ground_truth_pos


def save_results(
        tracker: str,
        dataset: str,
        sequences: list[str],
        trajectories: list[list[Polygon | None]],
        groundtruths: list[list[Polygon]],
):
    df = pandas.DataFrame({
        'tracker': tracker,
        'dataset': dataset,
        'sequence': sequences,
        'trajectory': trajectories,
        'groundtruth': groundtruths,
    })

    st.session_state.results = pandas.concat([
        st.session_state.results,
        df
    ])

    df = pandas.DataFrame({
        'tracker': tracker,
        'dataset': dataset,
        'sequence': sequences,
        'trajectory': [[[] if polygon is None else polygon_to_array(polygon) for polygon in trajectory] for trajectory in trajectories],
        'groundtruth': [[[] if polygon is None else polygon_to_array(polygon) for polygon in groundtruth] for groundtruth in groundtruths],
    })

    df.to_csv(results_file, mode='a', header=False, index=False)


class TestsPage: # todo output to raw
    submitted = False
    sequence = ""

    current_frame_image: DeltaGenerator | None = None
    current_example_bar: DeltaGenerator | None = None
    all_examples_bar: DeltaGenerator | None = None
    iou_quality_chart: DeltaGenerator | None = None
    iou_ar_chart: DeltaGenerator | None = None
    result_table: DeltaGenerator | None = None

    def __init__(self) -> None:
        super().__init__()
        if 'results' not in st.session_state:
            try:
                os.makedirs(results_dir)
                st.session_state.results = pandas.DataFrame(
                    columns=['tracker', 'dataset', 'sequence', 'trajectory', 'groundtruth'],
                    dtype=object
                )
                st.session_state.results.to_csv(results_file, mode='x', index=False)
            except FileExistsError:
                # directory already exists
                st.session_state.results = pandas.read_csv(results_file)
                trajectories = st.session_state.results.trajectory.apply(json.loads)
                groundtruths = st.session_state.results.groundtruth.apply(json.loads)
                st.session_state.results.trajectory = [[create_polygon(points) if points != [] else None for points in trajectory] for trajectory in trajectories]
                st.session_state.results.groundtruth = [[create_polygon(points) if points != [] else None for points in groundtruth] for groundtruth in groundtruths]
        if 'selection' not in st.session_state:
            st.session_state["selection"]: [(str, str)] = []
        if 'table_selected_trackers' not in st.session_state:
            indexes = st.session_state['results'][['tracker', 'dataset']].value_counts().index.rename(['Tracker', 'Dataset'])
            st.session_state['table_selected_trackers'] = pandas.DataFrame([False]*indexes.size,index=indexes,columns=['Selected'])
        if 'trackers' not in st.session_state:
            st.session_state.trackers = [_tracker() for _tracker in get_concrete_classes(tracker.Tracker)]

        self.sidebar()
        self.main_page()  # fixme znikające wykresy przy starcie
        self.handle_submitted()
        self.draw_scores()
        self.draw_table()

    def sidebar(self) -> None:
        with st.sidebar.form("Options"):
            tracker = st.selectbox('Trackers', [tracker.name for tracker in st.session_state.trackers])
            selected_dataset = st.selectbox("Datasets", datasets.keys())

            self.submitted = st.form_submit_button("Submit", use_container_width=True, type="primary")

            if self.submitted:
                st.session_state.selection.append((tracker, selected_dataset))
                st.session_state.table_selected_trackers.loc[tracker, selected_dataset] = True

    def main_page(self) -> None:
        self.current_frame_image = st.empty()

        self.current_example_bar = st.progress(0)
        self.current_example_bar.empty()

        self.all_examples_bar = st.progress(0)
        self.all_examples_bar.empty()

        st.header("Results")

        st.subheader("IoU")

        iou_quality, iou_ar = st.columns(2)
        iou_quality.text("Quality plot")
        iou_ar.text("AR plot")
        self.iou_quality_chart = iou_quality.empty()
        self.iou_ar_chart = iou_ar.empty()
        self.result_table = st.empty()

    def draw_scores(self):
        selected = st.session_state.table_selected_trackers
        selected = selected.loc[selected['Selected'] == True]
        if selected.empty:
            return
        ts = average_success_plot()
        ts = ts.set_index(['Tracker', 'Dataset'])
        ts = ts.loc[selected.index]
        ts = ts.reset_index()
        ts['TrackerDataset'] = ts[['Tracker', 'Dataset']].agg(' - '.join, axis=1)
        if len(ts) > 0:
            self.iou_quality_chart.line_chart(ts, x='Threshold', y='Success', color='TrackerDataset', height=300)

        ra = accuracy_robustness()
        ra = ra.set_index(['Tracker', 'Dataset'])
        ra = ra.loc[selected.index]
        ra = ra.reset_index()
        ra['TrackerDataset'] = ra[['Tracker', 'Dataset']].agg(' - '.join, axis=1)
        if len(ra) > 0:
            self.iou_ar_chart.scatter_chart(ra, x='Robustness', y='Accuracy', color='TrackerDataset', height=300)

    def on_table_change(self, foo): # fixme something wrong with selecting
        if 'show_trackers' in st.session_state:
            edited = st.session_state.show_trackers['edited_rows']

            for i in edited:
                st.session_state.table_selected_trackers.loc[*foo.iloc[i].tolist()] = edited[i]['Selected']

            self.draw_scores()

    def draw_table(self):
        ts = average_success_plot()
        ra = accuracy_robustness()
        qa = average_quality_auxiliary()
        ac = average_accuracy()

        if len(ts) <= 0 or len(ra) <= 0 or len(qa) <= 0 or len(ac) <= 0:
            return

        # todo performance
        success = ts.groupby(['Tracker', 'Dataset']).apply(lambda x: x['Success'].tolist())
        ra = ra.set_index(['Tracker', 'Dataset'])
        qa = qa.set_index(['Tracker', 'Dataset'])
        ac = ac.set_index(['Tracker', 'Dataset'])

        df = pandas.concat(
            [ac, ra, qa, success, st.session_state.table_selected_trackers],
            axis=1,
        ).rename(columns={0: 'success'}).reset_index()

        self.result_table = st.data_editor(
            df,
            column_config={
                "tracker": "Tracker",
                "dataset": "Dataset",
                "quality": "Quality",
                "accuracy": "Accuracy",
                "robustness": "Robustness",
                "nre": "NRE",
                "dre": "DRE",
                "ad": "AD",
                "success": st.column_config.LineChartColumn(
                    "Success", y_min=0.0, y_max=1.0, width='small'
                ),
                "selected": st.column_config.CheckboxColumn(
                    "Selected",
                    help="Select tracker to display in charts",
                    default=True,
                ),
            },
            disabled=['tracker', "dataset", "robustness", "accuracy", "success", 'nre', 'dre', 'ad', 'quality'],
            on_change=self.on_table_change(df[['Tracker', 'Dataset']]),
            use_container_width=True,
            key='show_trackers',
            column_order=["Tracker", "Dataset", "Quality", "Accuracy", "Robustness", 'NRE', 'DRE', 'AD', "success", "Selected"],
            hide_index=True
        )

    def handle_submitted(self) -> None:
        if self.submitted:
            # tracker, dataset
            for _selection in st.session_state.selection:
                sequences = []
                with open(f'{datasets[_selection[1]]}/list.txt') as f:
                    for line in f.readlines():
                        sequences.append(line.strip())

                trajectories = []
                groundtruths = []
                for index, sequence in enumerate(sequences):
                    self.all_examples_bar.progress(index / len(sequences), text=f"Testing {index + 1} out of {len(sequences)} films")
                    self.sequence = sequence
                    _tracker: tracker.Tracker = next(x for x in st.session_state.trackers if x.name == _selection[0])
                    _, _, trajectory = _tracker.test( # todo time when no object isn't saved
                        f'{datasets[_selection[1]]}/{sequence}',
                        listener=self.display_bar,
                        frame_listener=self.display_frame,
                        iou_threshold_for_correction=-1
                    )
                    trajectories.append(trajectory)
                    # [1:] because first frame is consumed for init of tracker
                    groundtruths.append(get_ground_truth_positions(f"{datasets[_selection[1]]}/{sequence}/groundtruth.txt")[1:])

                save_results(_selection[0], _selection[1], sequences, trajectories, groundtruths)
                self.draw_scores()
            st.session_state.selection = []
            self.all_examples_bar.empty()
            self.current_example_bar.empty()
            self.current_frame_image.empty()

    def display_bar(self, total_frames: int, current_frame: int) -> None:
        if self.current_example_bar is not None:
            self.current_example_bar.progress(current_frame / total_frames, text=f"Testing with {self.sequence}")

    def display_frame(self, frame: numpy.ndarray):
        self.current_frame_image.image(frame)


def polygon_to_array(polygon: Polygon) -> list[float]:
    _xx, _yy = polygon.exterior.coords.xy
    _coords = numpy.array(tuple(zip(_xx, _yy))).astype(float) # todo unnecessary?

    return [x for xs in _coords for x in xs]


def calculate_overlap(first: Polygon | None, second: Polygon | None) -> float:
    if first is None or first.area == 0.0 or second is None or second.area == 0.0:
        return 0.0
    intersect = first.intersection(second).area
    union = first.union(second).area
    return intersect / union


def calculate_overlaps(first: "pandas.Series[Polygon]", second: "pandas.Series[Polygon]") -> list[float | None]:
    """ Calculate the overlap between two lists of regions. The function first rasterizes both regions to 2-D binary masks and calculates overlap between them

    Args:
        first: first list of regions
        second: second list of regions

    Returns:
        list of floats with the overlap between the two regions. Note that overlap is one by definition if both regions are empty.

    Raises:
        RegionException: if the lists are not of the same size
    """
    if not len(first) == len(second):
        raise Exception("List not of the same size {} != {}".format(len(first), len(second)))
    return [calculate_overlap(pairs[0], pairs[1]) for i, pairs in enumerate(zip(first, second))]


def gather_overlaps(trajectory: "pandas.Series[Polygon]", groundtruth: "pandas.Series[Polygon]",
                    ignore_invisible: bool = False, threshold: float = -1) -> numpy.ndarray:
    overlaps = numpy.array(calculate_overlaps(trajectory, groundtruth))
    mask = numpy.ones(len(overlaps), dtype=bool)

    for i, (region_tr, region_gt) in enumerate(zip(trajectory, groundtruth)):
        # Skip if groundtruth is unknown
        if ignore_invisible and region_gt.area == 0.0:
            mask[i] = False
        # Skip if predicted is initialization frame
        elif region_tr is None:
            mask[i] = False
        elif overlaps[i] <= threshold:
            mask[i] = False

    return overlaps[mask]


def success_plot(tracker: str, dataset: str, sequence: str) -> list[tuple[float, float]]:
    g = st.session_state.results
    trajectories_groundtruths = g.loc[(g['tracker'] == tracker) & (g['dataset'] == dataset) & (g['sequence'] == sequence), ["trajectory", "groundtruth"]]
    axis_x = numpy.linspace(0, 1, 100)
    axis_y = numpy.zeros_like(axis_x)

    for trajectory, groundtruth in zip(trajectories_groundtruths['trajectory'], trajectories_groundtruths['groundtruth']):
        overlaps = gather_overlaps(trajectory, groundtruth)
        if overlaps.size > 0:
            for i, threshold in enumerate(axis_x):
                if threshold == 1:
                    # Nicer handling of the edge case
                    axis_y[i] += numpy.sum(overlaps >= threshold) / len(overlaps)
                else:
                    axis_y[i] += numpy.sum(overlaps > threshold) / len(overlaps)

    axis_y /= len(trajectories_groundtruths)

    return [(x, y) for x, y in zip(axis_x, axis_y)]


def average_success_plot():
    ret_df = pandas.DataFrame(columns=['Tracker', 'Dataset', 'Threshold', 'Success'])

    for (tracker, dataset), g in st.session_state.results.groupby(['tracker', 'dataset']):
        sequences = g.loc[(g['tracker'] == tracker) & (g['dataset'] == dataset), 'sequence'].unique()
        axis_x = numpy.linspace(0, 1, 100)
        axis_y = numpy.zeros_like(axis_x)

        for sequence in sequences:
            for j, (_, y) in enumerate(success_plot(tracker, dataset, sequence)):
                axis_y[j] += y

        axis_y /= len(sequences)

        ret_df = pandas.concat([ret_df, pandas.DataFrame(
            {'Tracker': [tracker] * len(axis_x), 'Dataset': [dataset] * len(axis_x), 'Threshold': axis_x,
             'Success': axis_y})])

    return ret_df


def create_polygon(_points: list[int | float] | tuple[int | float]) -> Polygon:
    if len(_points) == 4:
        _x, _y, _width, _height = _points
        _polygon = Polygon([
            (_x, _y),
            (_x + _width, _y),
            (_x + _width, _y + _height),
            (_x, _y + _height)
        ])
    elif len(_points) >= 6:
        _polygon = Polygon(zip(_points[::2], _points[1::2]))
    else:
        raise Exception("Incorrect number of points")

    return _polygon


def sequence_accuracy(tracker: str, dataset: str, sequence: str, ignore_invisible: bool = False, threshold: float = -1) -> float:
    g = st.session_state.results
    trajectories_groundtruths = g.loc[(g['tracker'] == tracker) & (g['dataset'] == dataset) & (g['sequence'] == sequence), ["trajectory", "groundtruth"]]

    cummulative = 0
    for trajectory, groundtruth in zip(trajectories_groundtruths['trajectory'], trajectories_groundtruths['groundtruth']):
        overlaps = gather_overlaps(trajectory, groundtruth, ignore_invisible, threshold)

        if overlaps.size > 0:
            cummulative += numpy.mean(overlaps)

    return cummulative / len(trajectories_groundtruths)


def average_accuracy() -> pandas.DataFrame:
    ret_df = pandas.DataFrame(columns=['Tracker', 'Dataset', 'Quality'])

    for (tracker, dataset), g in st.session_state.results.groupby(['tracker', 'dataset']):
        accuracy = 0
        frames = 0
        sequences = g.loc[(g['tracker'] == tracker) & (g['dataset'] == dataset), 'sequence'].unique()

        for sequence in sequences:
            accuracy += sequence_accuracy(tracker, dataset, sequence)
            frames += 1

        ret_df.loc[len(ret_df)] = [tracker, dataset, accuracy / frames]

    return ret_df


def count_frames(tracker: str, dataset: str, sequence: str):
    g = st.session_state.results
    trajectories_groundtruths = g.loc[(g['tracker'] == tracker) & (g['dataset'] == dataset) & (g['sequence'] == sequence), ["trajectory", "groundtruth"]]

    CN, CF, CM, CH, CT = 0, 0, 0, 0, 0
    for trajectory, groundtruth in zip(trajectories_groundtruths['trajectory'], trajectories_groundtruths['groundtruth']):
        overlaps = numpy.array(calculate_overlaps(trajectory, groundtruth))

        # Tracking, Failure, Miss, Hallucination, Notice
        T, F, M, H, N = 0, 0, 0, 0, 0

        for i, (region_tr, region_gt) in enumerate(zip(trajectory, groundtruth)):
            if not region_gt:
                if not region_tr:
                    N += 1
                else:
                    H += 1
            else:
                if overlaps[i] > 0:
                    T += 1
                else:
                    if not region_tr:
                        M += 1
                    else:
                        F += 1

        CN += N
        CF += F
        CM += M
        CH += H
        CT += T

    CN /= len(trajectories_groundtruths)
    CF /= len(trajectories_groundtruths)
    CM /= len(trajectories_groundtruths)
    CH /= len(trajectories_groundtruths)
    CT /= len(trajectories_groundtruths)

    return CT, CF, CM, CH, CN


def accuracy_robustness() -> pandas.DataFrame:
    ret_df = pandas.DataFrame(columns=['Tracker', 'Dataset', 'Robustness', 'Accuracy'])

    for (tracker, dataset), g in st.session_state.results.groupby(['tracker', 'dataset']):
        sequences = g.loc[(g['tracker'] == tracker) & (g['dataset'] == dataset), 'sequence'].unique()

        accuracy = 0
        robustness = 0
        count = 0 # of sequences

        for sequence in sequences:
            accuracy += sequence_accuracy(tracker, dataset, sequence, True, 0.0)
            T, F, M, _, _ = count_frames(tracker, dataset, sequence)

            robustness += T / (T + F + M)
            count += 1

        ret_df.loc[len(ret_df)] = [tracker, dataset, robustness / count, accuracy / count]

    return ret_df


def quality_auxiliary(tracker: str, dataset: str, sequence: str) -> tuple[float, float, float]:
    T, F, M, H, N = count_frames(tracker, dataset, sequence)

    not_reported_error = M / (T + F + M)
    drift_rate_error = F / (T + F + M)

    if N + H > 10:
        absence_detection = N / (N + H)
    else:
        absence_detection = None

    return not_reported_error, drift_rate_error, absence_detection


def average_quality_auxiliary():
    ret_df = pandas.DataFrame(columns=['Tracker', 'Dataset', 'NRE', 'DRE', 'AD'])

    for (tracker, dataset), g in st.session_state.results.groupby(['tracker', 'dataset']):
        # todo .unique() potncjalnie zbędne
        sequences = g.loc[(g['tracker'] == tracker) & (g['dataset'] == dataset), 'sequence'].unique()

        not_reported_error = 0
        drift_rate_error = 0
        absence_detection = 0
        absence_count = 0

        for sequence in sequences:
            nre, dre, ad = quality_auxiliary(tracker, dataset, sequence)
            not_reported_error += nre
            drift_rate_error += dre
            if ad is not None:
                absence_count += 1
                absence_detection += ad

        if absence_count > 0:
            absence_detection /= absence_count

        ret_df.loc[len(ret_df)] = [tracker, dataset, not_reported_error / len(sequences), drift_rate_error / len(sequences), absence_detection]

    return ret_df


testsPage = TestsPage()
