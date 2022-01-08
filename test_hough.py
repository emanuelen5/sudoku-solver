import cv2
import numpy as np
import pytest
from pathlib import Path
import hough
from hough_accumulator import HoughAccumulator


@pytest.fixture
def example_image():
    image_path = Path(__file__).parent.joinpath("example.png")
    assert image_path.is_file()
    img = cv2.imread(str(image_path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def test_single_line():
    img = np.zeros((10, 10), dtype=np.uint8)
    img[:, 5] = 1
    accum = hough.houghaccum(img)
    assert type(accum) == np.ndarray
    assert np.amax(accum) == 10


def test_return_type():
    img = np.zeros((10, 10), dtype=np.uint8)
    accum = hough.houghaccum(img)
    assert type(accum) == np.ndarray


def test_houghaccumulator():
    img = np.zeros((10, 10), dtype=np.uint8)
    img[:, 5] = 1
    acc = HoughAccumulator(img)
    assert acc[0, 5] == 10
