import inspect
from abc import ABC, abstractmethod
from typing import Callable, List, Tuple

import cv2
import numpy
from pandas import DataFrame
from shapely import Polygon

from utils.tracker_test import test_tracker
from utils.utils import polygon_to_tuple


class Tracker(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def init(self, ground_truth_polygon: Polygon, frame: numpy.ndarray):
        """
        :param ground_truth_polygon: ground truth position of object in frame to track
        :param frame: frame of video with object to track
        """
        pass

    @abstractmethod
    def eval(self, frame: numpy.ndarray) -> (bool, list[float]):
        """
        :param frame:
        :return:
        """
        pass

    def test(
            self,
            dataset_dir: str,
            show_tracking=False,
            iou_threshold_for_correction=.0,
            iou_threshold_for_recording=.0,
            listener: Callable[[int, int], None] = None,
            frame_listener: Callable[[numpy.ndarray], None] = None
    ) -> tuple[str, list[int], list[Polygon|None]]:
        return test_tracker(
            self,
            dataset_dir,
            show_tracking,
            iou_threshold_for_correction,
            iou_threshold_for_recording,
            listener,
            frame_listener
        )


class TrackerCV2(Tracker, ABC):
    _tracker: cv2.Tracker | None = None

    @property
    @abstractmethod
    def tracker(self) -> cv2.Tracker:
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
