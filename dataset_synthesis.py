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


if __name__ == '__main__':
    # TODO: Remove relative imports
    full_img_path = './data/original-images/1.jpg'
    bounded_img_path = './data/original-images/1_found.jpg' 

    if os.path.lexists('./data/original-images/1_candidates'):
        with open('./data/original-images/1_candidates', 'rb') as f:
            candidates = pickle.load(f)
            print(candidates[:10])
    else:
        full_img = np.array(Image.open(full_img_path))
        bounded_img = Image.open(bounded_img_path)
        
        candidates = find_candidates(full_img)
        print(len(candidates))
        print(candidates[:10])

        with open('./data/original-images/1_candidates', 'wb') as f:
            pickle.dump(candidates, f)
        
        plt.imshow(full_img)
        plt.show()
        plt.imshow(bounded_img)
        plt.show()
