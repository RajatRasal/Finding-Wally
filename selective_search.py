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
    COLUMNS = ['actual', 'x', 'y', 'w', 'h', 'fg']
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
                fg = iou_score >= IOU_THRESHOLD
                bboxes.append([full_img_path, x, y, w, h, fg])
        
    data = pd.DataFrame(bboxes, columns=COLUMNS)
    data.to_csv('./data/data.csv')

"""
        print('IOU')
        # Visualising Proposed Regions
        colour = (0, 0, 225)
        thickness = 1

        # Filter out very large or small candidate images
        print('Proposals:', len(candidates))
        candidates2 = []
        for proposal in candidates:
            if (0.5 * gt_box.w < proposal.w < 2 * gt_box.w) and \
               (0.5 * gt_box.h < proposal.h < 2 * gt_box.h):
                candidates2.append(proposal)
        print('After Proposals:', len(candidates2))
        candidates = candidates2

        lower = 0.4
        lower2 = 0.2
        tp, _ = iou_thresholding(candidates, gt_box_scaled, lower, 1)
        tp2, tn = iou_thresholding(candidates, gt_box_scaled, lower2, lower)
        print('True positives:', len(tp))
        print('Slightly True positives:', len(tp2))
        print('True negatives:', len(tn))

        random.shuffle(tp2)
        random.shuffle(tn)

        tp2 = tp2[:max(50, int(len(tp) * 2))]
        tn = tn[:max(50, int(len(tp) * 2))]

        center_original_bbox = center_bbox(gt_box_scaled)
        # cv.rectangle(full_img_scaled, start, end, colour, thickness)
        # print(f'center box: {center_original_bbox}')
        print(f'tp: {len(tp)}')

        for bbox in tp:
            # Plot Proposed Region
            centered_bbox = center_bbox(bbox)
            cv.rectangle(full_img_scaled, (bbox.x, bbox.y),
                         (bbox.x + bbox.w, bbox.y + bbox.h), (0, 255, 255), thickness)
            # print(f'TP Center BBOX: {centered_bbox}')

            # Plot Proposed Region + Offset
            t_x, t_y, t_w, t_h = calculate_offsets(center_original_bbox, centered_bbox)
            # print(t_x, t_y, t_w, t_h)
            shifted_bbox = apply_offset(centered_bbox, t_x, t_y, t_w, t_h)
            start_new = (int(shifted_bbox.x - shifted_bbox.w // 2),
                         int(shifted_bbox.y - shifted_bbox.h // 2))
            end_new = (int(shifted_bbox.x + shifted_bbox.w // 2),
                       int(shifted_bbox.y + shifted_bbox.h // 2))
            cv.rectangle(full_img_scaled, start_new, end_new, (0, 0, 0), thickness)
            data = data.append({'actual': no, 'x': bbox.x, 'y': bbox.y,
                                'w': bbox.w, 'h': bbox.h, 'fg': 1, 't_x': t_x,
                                't_y': t_y, 't_w': t_w, 't_h': t_h},
                               ignore_index=True)

        # plt.imshow(full_img_scaled)
        # plt.show()

        for bbox in tp2 + tn:
            data = data.append({'actual': no, 'x': bbox.x, 'y': bbox.y,
                                'w': bbox.w, 'h': bbox.h, 'fg': 0, 't_x': None,
                                't_y': None, 't_w': None, 't_h': None},
                               ignore_index=True)

        # print(data.shape)

        # break
        # print('Time taken:', time.time() - start_time)

    print('No of Foreground images:', data.fg.sum())
    print('Total Proposals:', data.shape)

    data.to_csv('./data/data.csv')
"""
