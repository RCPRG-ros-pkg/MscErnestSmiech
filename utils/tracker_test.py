import csv
import os
import sys
from datetime import datetime
from glob import glob
from typing import Callable, List

import cv2
import numpy as np
import numpy.typing
import pandas
from pandas import DataFrame
from shapely.geometry import Polygon

from utils.capture import ImageCapture, VideoCapture


def prepare_dirs(_tracker_name: str, _dataset_name: List[str]):
    _now = datetime.now()
    _dt_string = _now.strftime("%Y-%m-%d-%H-%M-%S-%f")
    _raw_index = _dataset_name.index("raw")
    _errors_dir = f"raw/errors/{_tracker_name}/{_dataset_name[_raw_index + 2]}/{_dataset_name[-1]}/{_dt_string}"
    try:
        os.makedirs(_errors_dir)
    except FileExistsError:
        # directory already exists
        pass

    return _errors_dir


def init_video_capture(_file_name: str):
    if glob(f"{_file_name}/*.jpg"):
        _cap = ImageCapture(_file_name)
    elif glob(f"{_file_name}/color/*.jpg"):
        _cap = ImageCapture(f"{_file_name}/color")
    else:
        _cap = VideoCapture(f'{_file_name}/video.mp4')

    # Read first frame.
    _ok, _frame = _cap.read()
    if not _ok:
        print('Cannot read video file')
        sys.exit()

    return _cap, _frame


def get_ground_truth_positions(_file_name: str) -> List[Polygon]:
    with open(_file_name) as csvfile:
        # _ground_truth_pos = [[int(x) for x in y] for y in csv.reader(csvfile, delimiter='\t')]
        _ground_truth_pos = [create_polygon([abs(int(float(x))) for x in y]) for y in csv.reader(csvfile)]

    return _ground_truth_pos

# todo może support dla polygonów. Zależy czy docelowy tracker będzie miał takie zdolności
def handle_tracker_result(_ok, _bbox, _frame):
    # Draw bounding box
    _p1, _p2 = None, None
    if _ok:
        # Tracking success
        _p1 = (int(_bbox[0]), int(_bbox[1]))
        _p2 = (int(_bbox[0] + _bbox[2]), int(_bbox[1] + _bbox[3]))
        cv2.rectangle(_frame, _p1, _p2, (255, 0, 0), 2, 1)
    else:
        # Tracking failure
        cv2.putText(_frame, "Tracking failure detected", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    return _p1, _p2


def display_tracker_name_and_fps(_tracker_name, _fps, _timer, _frame):
    # Display tracker type on frame
    cv2.putText(_frame, _tracker_name + " Tracker", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

    # Display FPS on frame
    cv2.putText(_frame, f"Detection time: {int(_fps)} ms", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)


def display_ground_truth_box(_ground_truth: Polygon, _frame):
    _xx, _yy = _ground_truth.exterior.coords.xy
    _coords = np.array(tuple(zip(_xx, _yy))).astype(int)
    cv2.polylines(_frame, [_coords], False, (0, 255, 0), 3)


def create_polygon(_points: List[int | float] | tuple[int | float]) -> Polygon:
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


def compute_and_display_iou(_ground_truth: Polygon, _bbox: tuple[float | int], _ok: bool, _frame):
    if _ground_truth.area == 0. and all([x == 0. for x in _bbox]):
        _iou = 1
    elif _ground_truth.area == 0. and not all([x == 0. for x in _bbox]):
        _iou = 0
    else:
        # compute the intersection over union and display it
        if _ok:
            bb_polygon = create_polygon(_bbox)

            intersect = bb_polygon.intersection(_ground_truth).area
            union = bb_polygon.union(_ground_truth).area
            _iou = intersect / union
        else:
            _iou = 0
        cv2.putText(_frame, "IoU: {:.4f}".format(_iou), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 170, 50), 2)

    return _iou


def handle_recording_on_error(_iou, _threshold, _errors_dir, _cap, _frame):
    if _iou <= _threshold:
        cv2.imwrite(
            f'{_errors_dir}/failure-{int(_cap.get(cv2.CAP_PROP_POS_FRAMES))}.jpg',
            _frame
        )


def test_tracker(
        _tracker,
        _dataset_dir: str,
        _show_tracking=True,
        _iou_threshold_for_correction=.0,
        _iou_threshold_for_recording=.0,
        listener: Callable[[int, int], None] = None,
        frame_listener: Callable[[numpy.ndarray], None] = None
) -> tuple[numpy.ndarray, DataFrame, list[Polygon|None]]:
    cap, frame = init_video_capture(_dataset_dir)
    ground_truth_polygons = get_ground_truth_positions(f'{_dataset_dir}/groundtruth.txt')
    _tracker.init(ground_truth_polygons[0], frame)
    errors_dir = prepare_dirs(_tracker.name, _dataset_dir.split('/'))

    trajectories: list[Polygon|None] = []

    scores = {
        "iou": [],
        "detection_time": [],
        'failure_rate': 0
    }
    while cap.isOpened():
        if listener is not None:
            listener(cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = cap.read()

        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        try:
            ok, bbox = _tracker.eval(frame)
            trajectories.append(create_polygon(bbox) if ok else None)
        except cv2.error:
            break

        # Calculate detection time in milliseconds
        _detection_time = (cv2.getTickCount() - timer) / cv2.getTickFrequency() * 1000

        # if frame is read correctly ret is True
        if not ret and not ok:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        display_tracker_name_and_fps(_tracker.name, _detection_time, timer, frame)
        ground_truth = ground_truth_polygons.pop(0)
        display_ground_truth_box(ground_truth, frame)

        p1, p2 = handle_tracker_result(ok, bbox, frame)

        iou = compute_and_display_iou(ground_truth, bbox, ok, frame)
        scores['iou'].append(iou)
        if p1 is not None and p2 is not None:
            scores['detection_time'].append(_detection_time)

        if _show_tracking:
            cv2.imshow('frame', frame)
        if frame_listener is not None:
            frame_listener(frame)

        handle_recording_on_error(iou, _iou_threshold_for_recording, errors_dir, cap, frame)

        if iou <= _iou_threshold_for_correction:
            if ground_truth.area != 0.:
                _tracker.init(ground_truth, frame)
            scores['failure_rate'] += 1

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return numpy.asarray(scores['iou']), pandas.DataFrame(data=scores['detection_time'], columns=['detection_time']), trajectories
