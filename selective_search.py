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


def scale_image_in_aspect_ratio(full_img, scale_width):
    scale = scale_width / full_img.shape[1]
    scaled_shape = (scale_width, int(full_img.shape[0] * scale))
    full_img_scaled = cv.resize(full_img, scaled_shape)
    return full_img_scaled

def find_candidates(full_img):
    """
    Apply segmented search to the input image to find a set of bounding
    boxes. Each bounding box is a region of interest which may contain an
    object that we want to detect at a later stage.
    """
    ss = cv.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(full_img)
    ss.switchToSelectiveSearchQuality()
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
    return BBox(x=x[0], y=y[0], w=bounded_img.shape[1], h=bounded_img.shape[0])

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
    t_x = actual.x - proposal.x / proposal.x
    t_y = actual.y - proposal.y / proposal.y
    w = np.log(proposal.w / actual.w)
    h = np.log(proposal.h / actual.h)
    return (t_x, t_y, w, h)

def center_bbox(bbox):
    return BBox(bbox.x + bbox.w // 2, bbox.y + bbox.h // 2, bbox.w, bbox.h)

def apply_offset(bbox, t_x, t_y, t_w, t_h):
    return BBox(t_x, t_y, bbox.w + bbox.w * t_w, bbox.h + bbox.h * t_h)


if __name__ == '__main__':
    # TODO: Remove relative imports
    import time

    start_time = time.time()

    # Region not centralised
    columns = ['actual', 'x', 'y', 'w', 'h', 'fg', 't_x', 't_y', 't_w', 't_h']
    data = pd.DataFrame(columns=columns)

    no = 2
    nos = [12, 14, 15, 18, 19, 1, 20, 21, 22, 24, 25, 26, 27, 2, 4, 5, 7, 8, 9]

    for no in nos:
        print(f'no: {no}')
        full_img_path = f'./data/original-images/{no}.jpg'
        bounded_img_path = f'./data/original-images/{no}_found.jpg'

        try:
            full_img = cv.imread(full_img_path, cv.IMREAD_COLOR)
            bounded_img = cv.imread(bounded_img_path, cv.IMREAD_COLOR)
        except Exception:
            full_img_path[-3:] = 'png'
            bounded_img_path[-3:] = 'png'
            full_img = cv.imread(full_img_path, cv.IMREAD_COLOR)
            bounded_img = cv.imread(bounded_img_path, cv.IMREAD_COLOR)

        print('scaling')
        # Scale down and true region bounding box
        new_width = 1000
        scale = new_width / full_img.shape[1]
        """
        scaled_shape = (new_width, int(full_img.shape[0] * scale))
        full_img_scaled = cv.resize(full_img, scaled_shape)
        """
        full_img_scaled = scale_image_in_aspect_ratio(full_img, new_width)

        original_box = find_image_coordinate(full_img, bounded_img)
        original_box_scaled = BBox(x=int(original_box.x * scale),
                                   y=int(original_box.y * scale),
                                   w=int(original_box.w * scale),
                                   h=int(original_box.h * scale))

        print('regions')
        # Region Proposals
        if os.path.lexists(f'./data/original-images/{no}_candidates'):
            with open(f'./data/original-images/{no}_candidates', 'rb') as f:
                candidates = pickle.load(f)
        else:
            candidates = find_candidates(full_img_scaled)

            with open(f'./data/original-images/{no}_candidates', 'wb') as f:
                pickle.dump(candidates, f)

        print('IOU')
        # Visualising Proposed Regions
        # colour = (0, 0, 225)
        # thickness = 10

        lower = 0.4
        lower2 = 0.1
        tp, _ = iou_thresholding(candidates, original_box_scaled, lower, 1)
        tp2, tn = iou_thresholding(candidates, original_box_scaled, lower2, lower)
        """
        print('True positives:', len(tp))
        print('Slightly True positives:', len(tp2))
        print('True negatives:', len(tn))
        """

        random.shuffle(tp2)
        random.shuffle(tn)

        tp2 = tp2[:max(50, int(len(tp) * 3))]
        tn = tn[:max(50, int(len(tp) * 1))]

        center_original_bbox = center_bbox(original_box_scaled)
        start = (center_original_bbox.x - center_original_bbox.w // 2,
                 center_original_bbox.y - center_original_bbox.h // 2)
        end = (center_original_bbox.x + center_original_bbox.w // 2,
               center_original_bbox.y + center_original_bbox.h // 2)
        # cv.rectangle(full_img_scaled, start, end, colour, thickness)
        print(f'center box: {center_original_bbox}')

        for bbox in tp:
            # Plot Proposed Region
            centered_bbox = center_bbox(bbox)
            """
            cv.rectangle(full_img_scaled, (bbox.x, bbox.y),
                         (bbox.x + bbox.w, bbox.y + bbox.h), (0, 255, 255), thickness)
            print(f'TP Center BBOX: {centered_bbox}')
            """

            # Plot Proposed Region + Offset
            t_x, t_y, t_w, t_h = calculate_offsets(center_original_bbox, centered_bbox)
            """
            shifted_bbox = apply_offset(centered_bbox, t_x, t_y, t_w, t_h)
            start_new = (int(shifted_bbox.x - shifted_bbox.w // 2),
                         int(shifted_bbox.y - shifted_bbox.h // 2))
            end_new = (int(shifted_bbox.x + shifted_bbox.w // 2),
                       int(shifted_bbox.y + shifted_bbox.h // 2))
            cv.rectangle(full_img_scaled, start_new, end_new, (0, 0, 0), thickness)
            break
            """
            data = data.append({'actual': no, 'x': bbox.x, 'y': bbox.y, 
                                'w': bbox.w, 'h': bbox.h, 'fg': 1, 't_x': t_x,
                                't_y': t_y, 't_w': t_w, 't_h': t_h},
                               ignore_index=True)

        # plt.imshow(full_img_scaled)
        # plt.show()

        # print(data.head())

        for bbox in tp2 + tn:
            data = data.append({'actual': no, 'x': bbox.x, 'y': bbox.y, 
                                'w': bbox.w, 'h': bbox.h, 'fg': 0},
                               ignore_index=True)

        print(data.shape)

        # break
        """
        print('Time taken:', time.time() - start_time)
        """

    print('No of Foreground images:', data.fg.sum())

    data.to_csv('./data/data.csv')
