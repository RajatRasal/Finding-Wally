from datetime import datetime

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
from PIL import Image
from tensorflow.keras.applications import vgg16

from log import log_image, draw_box_on_image, draw_boxes_on_image
from model import load_model, inference_dataset_etl, predict_on_batch
from nms import nms


csv_file_path = './data/data.csv'
image_no = 5
test_image_path = f'./data/original-images/{image_no}.jpg'
saved_model_path = './saved_model'

logdir = f'./logs/results/{image_no}/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
file_writer = tf.summary.create_file_writer(logdir)

image = np.array(Image.open(test_image_path))
model = load_model(saved_model_path)
test = inference_dataset_etl(csv_file_path, image_no)

bbox = []
scores = []

threshold = 0.5

for i, (test_files, test_images) in enumerate(test):
    print(i)
    offset, cls = predict_on_batch(test_images, model, threshold)
    mask = np.argwhere(cls)
    _, _x, _y, _w, _h = test_files

    for i, data in enumerate(zip(_x, _y, _w, _h)):
        if i not in mask:
            continue

        # Cast original bbox coordinates
        x, y, w, h = data
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        w = tf.cast(w, tf.float32)
        h = tf.cast(h, tf.float32)
        # Draw original bounding box
        image = draw_box_on_image(image, (y, x, y + h, x + w))

        # Find center
        x_center = x + (w / 2.0)
        y_center = y + (h / 2.0)
        # Apply offset
        x_center += w * offset[i, 0]
        y_center += h * offset[i, 1]
        _w = np.exp(offset[i, 2]) * w
        _h = np.exp(offset[i, 3]) * h
        # Unapply offsets
        half_width = _w / 2.0
        half_height = _h / 2.0
        y1 = y_center - half_height
        x1 = x_center - half_width
        y2 = y_center + half_height
        x2 = x_center + half_width
        # Draw tigher bounding box
        image = draw_box_on_image(image, (y1, x1, y2, x2), color=(0, 225, 225))

        bbox.append([y1, x1, y2, x2])
        scores.append(cls[i])

with open('bboxes', 'wb') as f:
    pickle.dump(bbox, f)

with open('scores', 'wb') as f:
    pickle.dump(scores, f)

log_image(file_writer, f'{image_no}-result', image, 0)

image2 = np.array(Image.open(f'./data/original-images/{image_no}.jpg'))

bbox = np.array(bbox)
scores = np.array(scores)

bbox[:, [1, 3]] /= image2.shape[1]
bbox[:, [0, 2]] /= image2.shape[0]
selected_boxes, selected_scores = nms(bbox, scores)
selected_boxes[:, [1, 3]] *= image2.shape[1]
selected_boxes[:, [0, 2]] *= image2.shape[0]

image2 = draw_boxes_on_image(image2, selected_boxes, color=(0, 225, 225))

log_image(file_writer, f'{image_no}-result', image2, 1)
