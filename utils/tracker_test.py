import csv
import os
import sys
from datetime import datetime
from glob import glob
from typing import Callable, List

import cv2
import numpy as np
import numpy.typing
from shapely.geometry import Polygon

from utils.capture import ImageCapture, VideoCapture


def prepare_dirs(_dt_string: str, _tracker_name: str, _dataset_name: List[str]):
    """
    Creates directory for errors - frames where tracker failed to track within threshold. If directory already exists
    it won't be created again.

    :param _dt_string: date string
    :param _tracker_name:
    :param _dataset_name:
    :return: path to errors dir
    """
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


def get_ground_truth_positions(_file_name: str) -> List[Polygon]: # todo move to util
    """
    Reads data from file and transforms it into polygons that define bounding boxes.

    :param _file_name:
    :return: list of bounding box polygons
    """
    with open(_file_name) as csvfile:
        # _ground_truth_pos = [[int(x) for x in y] for y in csv.reader(csvfile, delimiter='\t')]
        _ground_truth_pos = [create_polygon([abs(int(float(x))) for x in y]) for y in csv.reader(csvfile)]

    return _ground_truth_pos


# todo może support dla polygonów. Zależy czy docelowy tracker będzie miał takie zdolności
def handle_tracker_result(_ok, _bbox, _frame):
    """
    If tracking was success then displays bounding box, otherwise it displays error message. Additionally, it returns
    bounding box.

    :param _ok: whether tracking is successful
    :param _bbox: bounding box
    :param _frame: current cv2 frame
    :return: (x, y, width, height)
    """
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
    """
    Draws bounding box on frame.

    :param _ground_truth:
    :param _frame:
    """
    _xx, _yy = _ground_truth.exterior.coords.xy
    _coords = np.array(tuple(zip(_xx, _yy))).astype(int)
    cv2.polylines(_frame, [_coords], False, (0, 255, 0), 3)


def create_polygon(_points: List[int | float] | tuple[int | float]) -> Polygon:
    """
    Helper function to correctly handle bounding box formats. If there are 4 points then it assumes (x, y, width,
    height) format. Otherwise, list of (x, y, x, y...)

    :param _points: list of points either in (x, y, width, height) or list of (x, y, x, y...)
    :return: Polygon defying bounding box
    """
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


def compute_and_display_iou(_ground_truth: Polygon, _bbox: tuple[float | int], _ok: bool, _frame) -> float:
    """
    Returns computed iou and adds it to the frame

    :param _ground_truth: Polygon that defines correct bounding box
    :param _bbox: bounding box
    :param _ok: if frame is read correctly set True
    :param _frame: cv2 frame
    :return: iou
    """
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
    """
    If iou gets below threshold method starts saving frames.

    :param _iou:
    :param _threshold:
    :param _errors_dir:
    :param _cap:
    :param _frame:
    :return:
    """
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
) -> tuple[str, list[int], list[Polygon|None]]:
    """
    Method used to perform test of a tracker. listener and frame_listener are used in streamlit to nicely show
    tracking progress. Returned value consists of date string, list of times in milliseconds tracker needed to
    finish its work, list of trajectories which are polygons defined by Polygon from shapely.

    :param _tracker:
    :param _dataset_dir: path to directory with dataset
    :param _show_tracking: shows window with current state of tracker
    :param _iou_threshold_for_correction: threshold when tracker should be reinitialized
    :param _iou_threshold_for_recording: threshold when frames should be saved to memory
    :param listener: used for progress bar - returns current frame number and number of all frames
    :param frame_listener: returns current frame with tracker bounding box, ground truth box, IoU and few other informations
    :return: date when test was performed, time for detection, trajectories returned by tracker
    """
    now = datetime.now()
    date_string = now.strftime("%Y-%m-%d-%H-%M-%S-%f")

    cap, frame = init_video_capture(_dataset_dir)
    ground_truth_polygons = get_ground_truth_positions(f'{_dataset_dir}/groundtruth.txt')
    _tracker.init(ground_truth_polygons[0], frame)
    errors_dir = prepare_dirs(date_string, _tracker.name, _dataset_dir.split('/'))

    trajectories: list[Polygon|None] = []
    detection_time: list[int] = []

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
        detection_time.append(int(_detection_time))

        if _show_tracking:
            cv2.imshow('frame', frame)
        if frame_listener is not None:
            frame_listener(frame)

        handle_recording_on_error(iou, _iou_threshold_for_recording, errors_dir, cap, frame)

        if iou <= _iou_threshold_for_correction:
            if ground_truth.area != 0.:
                _tracker.init(ground_truth, frame)

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return date_string, detection_time, trajectories
