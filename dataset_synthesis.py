"""
File should accept as input:
    - the big image
    - tightly bounded wally
    - output dest

If expected result already exists in output dest then don't do anything.
"""
import os
import cloudpickle as pickle
from collections import namedtuple

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

cv.setUseOptimized(True)
cv.setNumThreads(4)

BBox = namedtuple('BBox', 'x y w h') 


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

def iou_thresholding(candidates, actual, threshold=0.2):
    true_positives = [actual]
    true_negatives = []

    for bbox in candidates:
        score = iou(actual, bbox)
        if score > score_threshold:
            true_positives.append(bbox)
        else:
            true_negatives.append(bbox)

    return true_positives, true_negatives


if __name__ == '__main__':
    import time

    start_time = time.time()
    print(start_time)
    # TODO: Remove relative imports
    full_img_path = './data/original-images/1.jpg'
    bounded_img_path = './data/original-images/1_found.jpg' 

    full_img = cv.imread(full_img_path, cv.IMREAD_COLOR)
    bounded_img = cv.imread(bounded_img_path, cv.IMREAD_COLOR)

    print('scaling')
    # Scale down and true region bounding box
    new_width = 1000
    scale = new_width / full_img.shape[1]
    scaled_shape = (new_width, int(full_img.shape[0] * scale))
    full_img_scaled = cv.resize(full_img, scaled_shape)

    original_box = find_image_coordinate(full_img, bounded_img)
    original_box_scaled = BBox(x=int(original_box.x * scale),
                               y=int(original_box.y * scale),
                               w=int(original_box.w * scale),
                               h=int(original_box.h * scale))

    print('regions')
    # Region Proposals
    if os.path.lexists('./data/original-images/1_candidates'):
        with open('./data/original-images/1_candidates', 'rb') as f:
            candidates = pickle.load(f)
    else:
        print(full_img_scaled.shape)
        candidates = find_candidates(full_img_scaled)

        with open('./data/original-images/1_candidates', 'wb') as f:
            pickle.dump(candidates, f)

    print('IOU')
    # Visualising Proposed Regions
    score_threshold = 0.5
    colour = (0, 0, 225)
    thickness = 2

    tp, tn = iou_thresholding(candidates, original_box_scaled, score_threshold)
    print(len(tp))
    print(len(tn))

    for bbox in tp:
        start = (bbox.x, bbox.y)
        end = (bbox.x + bbox.w, bbox.y + bbox.h)
        cv.rectangle(full_img_scaled, start, end, colour, thickness)

    print(time.time() - start_time)

    plt.imshow(full_img_scaled)
    plt.show()
