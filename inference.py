"""
End-to-end object detection pipeline
"""
import os
import time

import cloudpickle as pickle
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import vgg16
from PIL import Image

from model import load_model
from selective_search import scale_image_in_aspect_ratio, find_candidates


# Run selective search on image
print('Loading image')
image_path = './data/original-images/19.jpg'
image = np.array(Image.open(image_path))
print('Done Loading')

# Scale image to reduce no. of region proposals
print('Scaling Image')
width = 1000
image_scaled = scale_image_in_aspect_ratio(image, width)
height = image_scaled.shape[0]
assert image_scaled.shape[1] == 1000
print('Done scaling')

# Region proposal using selective search
# TODO: Try tensorflow generate bounding box proposals
print('Selective Search')
candidates_path = './data/test_19_candidates'
if os.path.lexists(candidates_path):
    with open(candidates_path, 'rb') as f:
        candidates = pickle.load(f)
else:
    candidates = find_candidates(image_scaled)
    with open(candidates_path, 'wb') as f:
        pickle.dump(candidates, f)
print('Done Selective Search')

print('Preprocess image')
# Normalise candidate box coordindates and put 
#  into tf.image required format -> [y1, x1, y2, x2]
print(len(candidates))
no_candidates = 100  # len(candidates)
norm_candidates = np.zeros((no_candidates, 4), dtype=np.float32)
for i in range(no_candidates):
    bbox = candidates[i]
    y1 = bbox.y / height
    x1 = bbox.x / width
    y2 = (bbox.y + bbox.h) / height
    x2 = (bbox.x + bbox.w) / width
    norm_candidates[i] = [y1, x1, y2, x2]

tf_image = tf.expand_dims(image_scaled, 0)
region_proposals = tf.image.crop_and_resize(tf_image,
    boxes=norm_candidates[:no_candidates],
    box_indices=np.zeros(no_candidates),
    crop_size=[224, 224])
region_proposals = vgg16.preprocess_input(region_proposals)

print('Done preprocessing image')

# Load model
saved_model_path = './saved_model'
restored_model = load_model(saved_model_path)

# Pass each image into model
preds = restored_model.predict_on_batch(region_proposals)
print(preds[0].shape)
print(preds[1].shape)

# Get predictions - select highest regions of confidence (nms)
#  + unmangle bounding box prediction

# Draw boxes onto image 
