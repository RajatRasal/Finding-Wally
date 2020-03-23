"""
Tests for intersection over union implementation
"""
import pytest

from dataset_synthesis import BBox, iou

@pytest.mark.parametrize('box1, box2', [
    # X axis not overlapping
    (BBox(x=0, y=10, w=0, h=0), BBox(x=5, y=0, w=0, h=0)),
    (BBox(x=5, y=0, w=0, h=0), BBox(x=0, y=10, w=0, h=0)),
    # Y axis not overlapping
    (BBox(x=0, y=0, w=10, h=10), BBox(x=5, y=11, w=0, h=0)),
    (BBox(x=5, y=11, w=10, h=10), BBox(x=6, y=5, w=0, h=0)),
    # X and Y coords not overlapping
    (BBox(x=0, y=0, w=0, h=0), BBox(x=5, y=5, w=0, h=0)),
    (BBox(x=5, y=5, w=0, h=0), BBox(x=0, y=0, w=0, h=0)),
])
def test_no_intersection(box1, box2):
    assert iou(box1, box2) == 0 

@pytest.mark.parametrize('box1, box2, score', [
    (BBox(x=0, y=10, w=10, h=5), BBox(x=5, y=0, w=20, h=20), 1/17),
    (BBox(x=5, y=0, w=20, h=20), BBox(x=0, y=10, w=10, h=5), 1/17),
    (BBox(x=0, y=0, w=10, h=10), BBox(x=2, y=8, w=5, h=5), 2/23),
    (BBox(x=2, y=8, w=5, h=5), BBox(x=0, y=0, w=10, h=10), 2/23),
    (BBox(x=2, y=0, w=5, h=5), BBox(x=0, y=2, w=10, h=10), 3/22),
    (BBox(x=0, y=2, w=10, h=10), BBox(x=2, y=0, w=5, h=5), 3/22),
])
def test_intersection_on_edge(box1, box2, score):
    assert iou(box1, box2) == score

@pytest.mark.parametrize('box1, box2, score', [
    (BBox(x=0, y=8, w=10, h=10), BBox(x=8, y=0, w=10, h=10), 1/49),
    (BBox(x=8, y=0, w=10, h=10), BBox(x=0, y=8, w=10, h=10), 1/49),
    (BBox(x=0, y=0, w=10, h=10), BBox(x=5, y=5, w=10, h=10), 1/7),
    (BBox(x=5, y=5, w=10, h=10), BBox(x=0, y=0, w=10, h=10), 1/7),
])
def test_intersection_on_corner(box1, box2, score):
    assert iou(box1, box2) == score

def test_one_inside_another():
    box1 = BBox(x=0, y=0, w=10, h=10)
    box2 = BBox(x=2, y=2, w=5, h=5)
    assert iou(box1, box2) == 0.25 


