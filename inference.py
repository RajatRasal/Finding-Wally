"""
End-to-end object detection pipeline
"""
import os
import time
import math

import cloudpickle as pickle
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications import vgg16
from tensorflow.keras.utils import Sequence

from model import load_model
from selective_search import scale_image_in_aspect_ratio, find_candidates


# bbox indices - TensorFlow format
Y_1, X_1, Y_2, X_2 = 0, 1, 2, 3
# offset indices - OpenCV format
T_X, T_Y, T_W, T_H = 0, 1, 2, 3
# Size of image required for model input
IMG_DIM = 224
# Prediction Threshold
THRESHOLD = 0.4

class RegionProposalSequence(Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x = x_set
        self.y = y_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y


# Run selective search on image
print('Loading image')
image_path = './data/original-images/19.jpg'
image = np.array(Image.open(image_path))
print('Done Loading')

# Scale image to reduce no. of region proposals
print('Scaling Image')
WIDTH = 1000
image_scaled = scale_image_in_aspect_ratio(image, WIDTH)
HEIGHT = image_scaled.shape[0]
assert image_scaled.shape[1] == 1000
print('Done scaling')

# Region proposal using selective search
# TODO: Try tensorflow generate bounding box proposals
print('Selective Search')
candidates_path = './data/test_19_candidates'
if os.path.lexists(candidates_path):
    with open(candidates_path, 'rb') as f:
        _candidates = pickle.load(f)
else:
    _candidates = find_candidates(image_scaled)
    with open(candidates_path, 'wb') as f:
        pickle.dump(candidates, f)
print('Done Selective Search')

candidates = []
for i in range(len(_candidates)):
    bbox = _candidates[i]
    if (50 < bbox.w < 150) and (50 < bbox.h < 150):
        candidates.append(bbox)

print('Preprocess image')
# Normalise candidate box coordindates and put 
#  into tf.image required format -> [y1, x1, y2, x2]
print(len(candidates))
no_candidates = 1000  # len(candidates)
norm_candidates = np.zeros((no_candidates, 4), dtype=np.float32)
for i in range(no_candidates):
    bbox = candidates[i]
    y1 = bbox.y / HEIGHT
    x1 = bbox.x / WIDTH
    y2 = (bbox.y + bbox.h) / HEIGHT
    x2 = (bbox.x + bbox.w) / WIDTH
    norm_candidates[i, Y_1] = y1
    norm_candidates[i, X_1] = x1
    norm_candidates[i, Y_2] = y2
    norm_candidates[i, X_2] = x2 

tf_image = tf.expand_dims(image_scaled, 0)
region_proposals = tf.image.crop_and_resize(tf_image,
    boxes=norm_candidates[:no_candidates],
    box_indices=np.zeros(no_candidates),
    crop_size=[IMG_DIM, IMG_DIM]
)
region_proposals = vgg16.preprocess_input(region_proposals)
print('Done preprocessing image')
BATCH_SIZE = 100
rp_sequence = RegionProposalSequence(
    region_proposals,
    norm_candidates,
    BATCH_SIZE
)

# Load model
saved_model_path = './saved_model'
restored_model = load_model(saved_model_path)

# Pass each image into model
# pred = restored_model.predict_on_batch(
#     region_proposals,
#     batch_size=batch_size
# )
pred = restored_model.predict(rp_sequence, verbose=1, workers=4)
offset = pred[0]
label = pred[1]
print(pred[0].shape)
print(pred[1].shape)

# Filter by labels which have a high prob of being Wally
label[label >= THRESHOLD] = 1
label[label < THRESHOLD] = 0
mask = label.astype('int').flatten()
indices = np.argwhere(mask).flatten()
print(f'Before: {mask.shape}')
print(f'After: {indices.shape}')

# Un-normalise bounding boxes
refined_candidates = norm_candidates.copy()
refined_candidates[:, [X_1, X_2]] *= WIDTH
refined_candidates[:, [Y_1, Y_2]] *= HEIGHT
# Original width
width = refined_candidates[:, X_2] - refined_candidates[:, X_1]
height = refined_candidates[:, Y_2] - refined_candidates[:, Y_1]
# Calculate center 
center_x = refined_candidates[:, X_1] + width / 2
center_y = refined_candidates[:, Y_1] + height / 2

image_scaled_copy = image_scaled.copy()

# Plot original bboxes on image in blue
for i in indices:  # [50:60]:
    x_1 = int(refined_candidates[i, X_1])
    y_1 = int(refined_candidates[i, Y_1])
    x_2 = int(refined_candidates[i, X_2])
    y_2 = int(refined_candidates[i, Y_2])
    start = (x_1, y_1)
    end = (x_2, y_2)
    # print(i, start, end)
    cv.rectangle(image_scaled_copy, start, end, (0, 0, 255), 2)

# Refine the bounding box estimation
# LilianWeng blog
# Refined center coordinate
center_x += width * offset[:, T_X]
center_y += height * offset[:, T_Y]
# Refined Width estimation 
refined_width = width * 2 ** offset[:, T_W]
refined_height = height * 2 ** offset[:, T_H]
# Update top-left coordinate
refined_candidates[:, X_1] = center_x - refined_width / 2
refined_candidates[:, Y_1] = center_y - refined_height / 2
# Update bottom-right coordinate
refined_candidates[:, X_2] = center_x + refined_width / 2
refined_candidates[:, Y_2] = center_y + refined_height / 2

plt.title('Old')
plt.imshow(image_scaled_copy)
plt.show()

for i in indices:  # [50:60]:
    x_1 = int(refined_candidates[i, X_1])
    y_1 = int(refined_candidates[i, Y_1])
    x_2 = int(refined_candidates[i, X_2])
    y_2 = int(refined_candidates[i, Y_2])
    start = (x_1, y_1)
    end = (x_2, y_2)
    # print(i, start, end)
    cv.rectangle(image_scaled_copy, start, end, (255, 0, 0), 3)

plt.title('New')
plt.imshow(image_scaled_copy)
plt.show()
# Get predictions - select highest regions of confidence (nms)
#  + unmangle bounding box prediction
# Add offsets to normalised region proposals

# Non-maximum suppression
refined_candidates[:, [X_1, X_2]] /= WIDTH
refined_candidates[:, [Y_1, Y_2]] /= HEIGHT
selected_indices = tf.image.non_max_suppression(
    boxes=refined_candidates,
    scores=pred[1].flatten(),
    max_output_size=10
)
selected_boxes = tf.gather(refined_candidates, selected_indices)
refined_candidates[:, [X_1, X_2]] *= WIDTH
refined_candidates[:, [Y_1, Y_2]] *= HEIGHT

image_scaled_copy = image_scaled.copy() 

for i in range(len(selected_boxes)):
    x_1 = int(refined_candidates[i, X_1])
    y_1 = int(refined_candidates[i, Y_1])
    x_2 = int(refined_candidates[i, X_2])
    y_2 = int(refined_candidates[i, Y_2])
    start = (x_1, y_1)
    end = (x_2, y_2)
    cv.rectangle(image_scaled_copy, start, end, (255, 0, 0), 3)

plt.title('NMS boxes')
plt.imshow(image_scaled_copy)
plt.show()
