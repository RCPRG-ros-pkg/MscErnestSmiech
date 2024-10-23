from abc import abstractmethod, ABC
from glob import glob

import cv2
import numpy
from numpy.typing import ArrayLike


class Capture(ABC):
    """
    Wrapper to simplify usage of sequence that's based on either pictures on video.
    """

    @abstractmethod
    def __init__(self, _file_name: str) -> None:
        pass

    @abstractmethod
    def read(self) -> (bool, ArrayLike):
        """
        Reads a frame of sequence. With each read it provides one frame starting from the first till there are frames
        to return.

        :return: frame of a video
        """
        pass

    @abstractmethod
    def get(self, i: int) -> int:
        """
        Used to indicate number of frames:
         - cv2.CAP_PROP_FRAME_COUNT - returns the number of all the frames in sequence
         - cv2.CAP_PROP_POS_FRAMES - returns the number of frames left to show (after using read)

        :param i: cv2.CAP_PROP_FRAME_COUNT or cv2.CAP_PROP_POS_FRAMES
        :return: number of frames
        """
        pass

    @abstractmethod
    def isOpened(self) -> bool:
        """
        Indicates if there are frames to read left.
        :return:
        """
        pass

    @abstractmethod
    def release(self) -> None:
        """
        Clears frames to read. Should be used after finished reading sequence.
        :return:
        """
        pass


class VideoCapture(Capture):

    def __init__(self, _file_name: str) -> None:
        self.cap = cv2.VideoCapture(_file_name)

    def read(self) -> (bool, ArrayLike):
        return self.cap.read()

    def get(self, i: int) -> float:
        return self.cap.get(i)

    def isOpened(self) -> bool:
        return self.cap.isOpened()

    def release(self) -> None:
        self.cap.release()


class ImageCapture(Capture):

    def __init__(self, _file_name: str):
        self.images = sorted(glob(f"{_file_name}/*.jpg"))
        self.frame_count = len(self.images)

    def read(self) -> (bool, ArrayLike):
        try:
            return True, cv2.imread(self.images.pop(0), cv2.IMREAD_COLOR)
        except IndexError:
            return False, numpy.array([])

    def get(self, i: int) -> int:
        if i == cv2.CAP_PROP_FRAME_COUNT:
            return self.frame_count
        elif i == cv2.CAP_PROP_POS_FRAMES:
            return len(self.images)
        else:
            raise TypeError(f"{i} is not supported")

    def isOpened(self) -> bool:
        return self.images != []

    def release(self) -> None:
        self.images = []