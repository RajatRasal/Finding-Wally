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




rp_sequence = RegionProposalSequence(region_proposals, norm_candidates, 100)

# Load model
saved_model_path = './saved_model'
restored_model = load_model(saved_model_path)

# Pass each image into model
pred = restored_model.predict(rp_sequence, verbose=1)  # , workers=4)
offset = pred[0]
label = pred[1]
print(pred[0].shape)
print(pred[1].shape)

# Refine the bounding box estimation
# norm_candidates[:, X_2] = norm_candidates[:, X_1] + offset[:, T_W]
# norm_candidates[:, Y_2] = norm_candidates[:, Y_1] + offset[:, T_Y]
# norm_candidates[:, X_1] += offset[:, T_X]
# norm_candidates[:, Y_1] += offset[:, T_Y]

# norm_candidates[:, [X_1, X_2]] = norm_candidates[:, [X_2, X_1]]

# Filter by labels which have a high prob of being Wally
label[label >= THRESHOLD] = 1
label[label < THRESHOLD] = 0
mask = label.astype('int').flatten()
indices = np.argwhere(mask).flatten()
print(f'Before: {mask.shape}')
print(f'After: {indices.shape}')
# target_bboxes = norm_candidates[mask, :].astype('int')

# print(tf_image.dtype)
# print(norm_candidates)

# int(image_scaled.shape)

for i in indices[40:50]:
    x_1 = int(norm_candidates[i, X_1] * WIDTH)
    y_1 = int(norm_candidates[i, Y_1] * HEIGHT)
    x_2 = int(norm_candidates[i, X_2] * WIDTH)
    y_2 = int(norm_candidates[i, Y_2] * HEIGHT)
    start = (x_1, y_1)
    end = (x_2, y_2)
    # print(start)
    # print(end)
    cv.rectangle(image_scaled, start, end, (0, 0, 255), 2)

# Refine the bounding box estimation
# norm_candidates[:, X_2] = norm_candidates[:, X_1] + offset[:, T_W]
# norm_candidates[:, Y_2] = norm_candidates[:, Y_1] + offset[:, T_Y]
# norm_candidates[:, X_1] += offset[:, T_X]
# norm_candidates[:, Y_1] += offset[:, T_Y]

refined_candidates = np.zeros(norm_candidates.shape)
width = norm_candidates[:, X_2] - norm_candidates[:, X_1]
height = norm_candidates[:, Y_2] - norm_candidates[:, Y_1]
refined_candidates[:, X_1] = width * offset[:, X_1] + norm_candidates[:, X_1]
refined_candidates[:, Y_1] = height * offset[:, Y_1] + norm_candidates[:, Y_1]
refined_width = width * np.square(offset[:, T_X])
refined_height = height * np.square(offset[:, T_Y])
refined_candidates[:, X_2] = norm_candidates[:, X_1] + refined_width
refined_candidates[:, Y_2] = norm_candidates[:, Y_2] + refined_height

plt.imshow(image_scaled)
plt.show()

for i in indices[40:50]:
    x_1 = int(refined_candidates[i, X_1] * WIDTH)
    y_1 = int(refined_candidates[i, Y_1] * HEIGHT)
    x_2 = int(refined_candidates[i, X_2] * WIDTH)
    y_2 = int(refined_candidates[i, Y_2] * HEIGHT)
    start = (x_1, y_1)
    end = (x_2, y_2)
    # print(start)
    # print(end)
    cv.rectangle(image_scaled, start, end, (0, 255, 0), 2)

plt.imshow(image_scaled)
plt.show()
# Get predictions - select highest regions of confidence (nms)
#  + unmangle bounding box prediction
# Add offsets to normalised region proposals

"""
refined_candidates = np.zeros(norm_candidates.shape)
width = norm_candidates[:, X_2] - norm_candidates[:, X_1]
height = norm_candidates[:, Y_2] - norm_candidates[:, Y_1]
# Centralise norm_candidates
center_x = norm_candidates[:, X_1] + width / 2 
center_y = norm_candidates[:, Y_1] + height / 2
refined_center_x = width * offset[:, X_1] + center_x 
refined_center_y = height * offset[:, Y_1] + center_y 
refined_width = width * np.square(offset[:, T_X])
refined_height = height * np.square(offset[:, T_Y])

refined_candidates[:, X_1] = refined_center_x - refined_width / 2
refined_candidates[:, Y_1] = refined_center_y - refined_height / 2
refined_candidates[:, X_2] = refined_center_x + refined_width / 2
refined_candidates[:, Y_2] = refined_center_y + refined_height / 2
"""
