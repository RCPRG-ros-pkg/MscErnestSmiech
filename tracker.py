import inspect
from abc import ABC, abstractmethod
from typing import Callable, List, Tuple

import cv2
import numpy
from shapely import Polygon

from utils.tracker_test import test_tracker
from utils.utils import polygon_to_tuple


class Tracker(ABC):
    """
    Wrapper for tracker that provides unified interface for later use in tracker_test.py
    """

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def init(self, ground_truth_polygon: Polygon, frame: numpy.ndarray):
        """
        Should be used to initialize tracker with first frame.

        :param ground_truth_polygon: ground truth position of object in frame to track
        :param frame: frame of video with object to track
        """
        pass

    @abstractmethod
    def eval(self, frame: numpy.ndarray) -> (bool, list[float]):
        """
        Supplies a frame with an object to track.

        :param frame:
        :return:
        """
        pass

    def test(
            self,
            dataset_dir: str,
            show_tracking=False,
            iou_threshold_for_correction=.0,
            listener: Callable[[int, int], None] = None,
            frame_listener: Callable[[numpy.ndarray], None] = None
    ) -> tuple[str, list[int], list[Polygon|None]]:
        """
        Method used to perform test of a tracker. listener and frame_listener are used in streamlit to nicely show
        tracking progress. Returned value consists of date string, list of times in milliseconds tracker needed to
        finish its work, list of trajectories which are polygons defined by Polygon from shapely.

        :param dataset_dir: path to directory with dataset
        :param show_tracking: shows window with current state of tracker
        :param iou_threshold_for_correction: threshold when tracker should be reinitialized
        :param listener: used for progress bar - returns current frame number and number of all frames
        :param frame_listener: returns current frame with tracker bounding box, ground truth box, IoU and few other informations
        :return: date when test was performed, time for detection, trajectories returned by tracker
        """
        return test_tracker(
            self,
            dataset_dir,
            show_tracking,
            iou_threshold_for_correction,
            listener,
            frame_listener
        )


class TrackerCV2(Tracker, ABC):
    _tracker: cv2.Tracker | None = None

    @property
    @abstractmethod
    def tracker(self) -> cv2.Tracker:
        """
        Convenience method for OpenCV tracker object. Should only return object.

        :return: cv2 Tracker object
        """
        pass

    def init(self, ground_truth_polygon: Polygon, frame: numpy.ndarray):
        self._tracker = self.tracker
        _minimum_bounding_rectangle = polygon_to_tuple(ground_truth_polygon)

        if _minimum_bounding_rectangle is not None:
            # Initialize tracker with first frame and bounding box
            self._tracker.init(frame, _minimum_bounding_rectangle)

    def eval(self, frame: numpy.ndarray) -> (bool, list[float]):
        if self._tracker is None:
            raise Exception("Execute init() first")
        return self._tracker.update(frame)


class TrackerMedianFlow(TrackerCV2):

    @property
    def name(self) -> str:
        return "MedianFlow"

    @property
    def tracker(self) -> cv2.Tracker:
        return cv2.legacy.TrackerMedianFlow().create()


class TrackerBoosting(TrackerCV2):

    @property
    def name(self) -> str:
        return "Boosting"

    @property
    def tracker(self) -> cv2.Tracker:
        return cv2.legacy.TrackerBoosting().create()


class TrackerMIL(TrackerCV2):

    @property
    def name(self) -> str:
        return "MIL"

    @property
    def tracker(self) -> cv2.Tracker:
        return cv2.TrackerMIL().create()


class TrackerKCF(TrackerCV2):

    @property
    def name(self) -> str:
        return "KCF"

    @property
    def tracker(self) -> cv2.Tracker:
        return cv2.legacy.TrackerKCF().create()


class TrackerTLD(TrackerCV2):

    @property
    def name(self) -> str:
        return "TLD"

    @property
    def tracker(self) -> cv2.Tracker:
        return cv2.legacy.TrackerTLD().create()


class TrackerGOTURN(TrackerCV2):

    @property
    def name(self) -> str:
        return "GOTURN"

    @property
    def tracker(self) -> cv2.Tracker:
        return cv2.TrackerGOTURN().create()


class TrackerDaSiamRPN(TrackerCV2):

    @property
    def name(self) -> str:
        return "DaSiamRPN"

    @property
    def tracker(self) -> cv2.Tracker:
        return cv2.TrackerDaSiamRPN().create()


class TrackerMOSSE(TrackerCV2):

    @property
    def name(self) -> str:
        return "MOSSE"

    @property
    def tracker(self) -> cv2.Tracker:
        return cv2.legacy.TrackerMOSSE().create()


class TrackerCSRT(TrackerCV2):

    @property
    def name(self) -> str:
        return "CSRT"

    @property
    def tracker(self) -> cv2.Tracker:
        return cv2.legacy.TrackerCSRT().create()


class DummyTracker(Tracker):
    """
    Tracker always returning first bounding box it received. Useful for making sure everything works.
    """

    _ground_truth: Tuple[int, int, int, int] | None = None

    @property
    def name(self) -> str:
        return "Dummy"

    def init(self, ground_truth_polygon: Polygon, frame: numpy.ndarray):
        _minimum_bounding_rectangle = polygon_to_tuple(ground_truth_polygon)

        if _minimum_bounding_rectangle is not None:
            # Initialize tracker with first frame and bounding box
            self._ground_truth = _minimum_bounding_rectangle

    def eval(self, frame: numpy.ndarray) -> (bool, List[float]):
        if self._ground_truth is None:
            raise Exception("Execute init() first")
        return True, self._ground_truth


def get_concrete_classes(cls):
    for subclass in cls.__subclasses__():
        yield from get_concrete_classes(subclass)
        if not inspect.isabstract(subclass):
            yield subclass


if __name__ == '__main__':
    for _tracker in get_concrete_classes(Tracker):
        tracker = _tracker()
