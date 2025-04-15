import sys
from datetime import datetime
from glob import glob
from typing import Callable

import cv2
import numpy as np
import numpy.typing
from shapely.geometry import Polygon

from utils.capture import ImageCapture, VideoCapture
from utils.utils import create_polygon, get_ground_truth_positions


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


def display_ground_truth_box(_ground_truth: Polygon | None, _frame):
    """
    Draws bounding box on frame.

    :param _ground_truth:
    :param _frame:
    """
    if _ground_truth is None:
        return None
    _xx, _yy = _ground_truth.exterior.coords.xy
    _coords = np.array(tuple(zip(_xx, _yy))).astype(int)
    cv2.polylines(_frame, [_coords], False, (0, 255, 0), 3)


def compute_and_display_iou(_ground_truth: Polygon | None, _bbox: tuple[float | int], _ok: bool, _frame) -> float:
    """
    Returns computed iou and adds it to the frame

    :param _ground_truth: Polygon that defines correct bounding box
    :param _bbox: bounding box
    :param _ok: if frame is read correctly set True
    :param _frame: cv2 frame
    :return: iou
    """
    if (_ground_truth is None or _ground_truth.area == 0.) and all([x == 0. for x in _bbox]):
        _iou = 1
    elif (_ground_truth is None or _ground_truth.area == 0.) and not all([x == 0. for x in _bbox]):
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


def test_tracker(
        _tracker,
        _dataset_dir: str,
        _show_tracking=True,
        _iou_threshold_for_correction=.0,
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
    :param listener: used for progress bar - returns current frame number and number of all frames
    :param frame_listener: returns current frame with tracker bounding box, ground truth box, IoU and few other informations
    :return: date when test was performed, time for detection, trajectories returned by tracker
    """
    now = datetime.now()
    date_string = now.strftime("%Y-%m-%d-%H-%M-%S-%f")

    cap, frame = init_video_capture(_dataset_dir)
    ground_truth_polygons = get_ground_truth_positions(f'{_dataset_dir}/groundtruth.txt')
    _tracker.init(ground_truth_polygons[0], frame)

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
        except cv2.error as e:
            ok = False
            bbox = None
            print(repr(e))
            total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            print(f"cv2.error in dataset {_dataset_dir} in frame {total - cap.get(cv2.CAP_PROP_POS_FRAMES)} out of {total}")

        trajectories.append(create_polygon(bbox) if ok else None)


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

        if iou <= _iou_threshold_for_correction:
            if ground_truth.area != 0.:
                _tracker.init(ground_truth, frame)

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return date_string, detection_time, trajectories
