"""
File should accept as input:
    - the big image
    - tightly bounded wally
    - output dest

If expected result already exists in output dest then don't do anything.
"""
import os
import cloudpickle as pickle
import random
from collections import namedtuple

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2 as cv

cv.setUseOptimized(True)
cv.setNumThreads(4)

BBox = namedtuple('BBox', 'x y w h')


def scale_image_in_aspect_ratio(full_img, scale_width=1000):
    scale = scale_width / full_img.shape[1]
    scaled_shape = (scale_width, int(full_img.shape[0] * scale))
    full_img_scaled = cv.resize(full_img, scaled_shape)
    return full_img_scaled

def find_candidates(full_img, quality=True):
    """
    Apply segmented search to the input image to find a set of bounding
    boxes. Each bounding box is a region of interest which may contain an
    object that we want to detect at a later stage.
    """
    ss = cv.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(full_img)
    if quality:
        ss.switchToSelectiveSearchQuality()
    else:
        ss.switchToSelectiveSearchFast()
    regions = ss.process()

    candidates = set()
    for x, y, w, h in regions:
        bbox = BBox(x=x, y=y, w=w, h=h)
        if bbox in candidates:
            continue
        candidates.add(bbox)

    return list(candidates)

def iou(a, b):
    """
    Intersection over Union = (a ⋂ b) / (a ⋃ b)
    """
    width = None
    if b.x <= a.x <= b.x + b.w:
        width = min(a.w, b.w - (a.x - b.x)) 
    if a.x <= b.x <= a.x + a.w:
        width = min(b.w, a.w - (b.x - a.x)) 
    if not width:
        return 0

    height = None
    if b.y <= a.y <= b.y + b.h:
        height = min(a.h, b.h - (a.y - b.y)) 
    if a.y <= b.y <= a.y + a.h:
        height = min(b.h, a.h - (b.y - a.y)) 
    if not height:
        return 0

    intersection = height * width
    union = a.h * a.w + b.h * b.w - intersection

    return intersection / union

def find_image_coordinate(full_img, bounded_img):
    res = cv.matchTemplate(full_img, bounded_img, cv.TM_CCOEFF_NORMED)
    loc = np.where(res >= 0.9)
    x, y = loc[::-1]
    return BBox(x=x[0], y=y[0],
        w=bounded_img.shape[1],
        h=bounded_img.shape[0]
    )

def iou_thresholding(candidates, actual, lower=0.2, upper=1):
    true_positives = []
    true_negatives = []

    for bbox in candidates:
        score = iou(actual, bbox)
        if upper >= score > lower:
            true_positives.append(bbox)
        else:
            true_negatives.append(bbox)

    return true_positives, true_negatives

def calculate_offsets(actual, proposal):
    t_x = (actual.x - proposal.x) / proposal.x
    t_y = (actual.y - proposal.y) / proposal.y
    t_w = np.log2(actual.w / proposal.w)
    t_h = np.log2(actual.h / proposal.h)
    return (t_x, t_y, t_w, t_h)

def center_bbox(bbox):
    return BBox(x=bbox.x + bbox.w // 2,
        y=bbox.y + bbox.h // 2,
        w=bbox.w,
        h=bbox.h
    )

def apply_offset(centered_bbox, t_x, t_y, t_w, t_h):
    return BBox(x=centered_bbox.x + t_x,
        y=centered_bbox.y + t_y,
        w=int(centered_bbox.w * t_w),
        h=int(centered_bbox.h * t_h)
    )


if __name__ == '__main__':
    COLUMNS = ['actual', 'image_no', 'x', 'y', 'w', 'h', 'x_t', 'y_t', 'w_t', 'h_t', 'fg']
    WIDTH = 1000
    IOU_THRESHOLD = 0.4
    LOWER_WIDTH_SCALE = 0.5
    UPPER_WIDTH_SCALE = 5.0
    LOWER_HEIGHT_SCALE = 0.5
    UPPER_HEIGHT_SCALE = 3.0

    bboxes = []

    for no in range(1, 56):
        full_img_path = f'./data/original-images/{no}.jpg'
        bounded_img_path = f'./data/original-images/{no}_found.jpg'

        if not (os.path.lexists(full_img_path) and os.path.lexists(bounded_img_path)):
            print(f'Not found: {no}')
            continue

        try:
            full_img = cv.imread(full_img_path, cv.IMREAD_COLOR)
            bounded_img = cv.imread(bounded_img_path, cv.IMREAD_COLOR)
        except Exception as e:
            print(f'Not found: {no}')
            continue

        # Ground truth bounding box coordinates - no scaling
        gt_bbox = find_image_coordinate(full_img, bounded_img)
        x_t = gt_bbox.x
        y_t = gt_bbox.y
        w_t = gt_bbox.w
        h_t = gt_bbox.h

        # Scale down and true region bounding box
        # Many images are very big and will result in too many bounding boxes.
        # We resize each image to height * width(=1000) to reduce the number of
        #   region proposals.
        scale = WIDTH / full_img.shape[1]
        full_img_scaled = scale_image_in_aspect_ratio(full_img, WIDTH)

        # Region Proposals
        if os.path.lexists(f'./data/original-images/{no}_candidates'):
            with open(f'./data/original-images/{no}_candidates', 'rb') as f:
                candidates = pickle.load(f)
        else:
            candidates = find_candidates(full_img_scaled)
            with open(f'./data/original-images/{no}_candidates', 'wb') as f:
                pickle.dump(candidates, f)

        # Bounds for dimensions of bounding boxes
        lower_width = LOWER_WIDTH_SCALE * gt_bbox.w
        upper_width = UPPER_WIDTH_SCALE * gt_bbox.w
        lower_height = LOWER_HEIGHT_SCALE * gt_bbox.h
        upper_height = UPPER_HEIGHT_SCALE * gt_bbox.h

        for bbox in candidates:
            # Upscale bounding box coordinates
            x = bbox.x / scale
            y = bbox.y / scale
            w = bbox.w / scale
            h = bbox.h / scale
            if (lower_width <= w <= upper_width) and (lower_height <= h <= upper_height):
                _bbox = BBox(x, y, w, h)
                iou_score = iou(gt_bbox, _bbox)
                fg = int(iou_score >= IOU_THRESHOLD)
                bboxes.append([full_img_path, no, x, y, w, h, x_t, y_t, w_t, h_t, fg])

    data = pd.DataFrame(bboxes, columns=COLUMNS)
    data.to_csv('./data/data.csv')
