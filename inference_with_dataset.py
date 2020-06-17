import argparse
from datetime import datetime

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
from PIL import Image
from tensorflow.keras.applications import vgg16

from log import log_image, draw_box_on_image, draw_boxes_on_image
from model import (load_model, inference_dataset_etl, predict_on_batch,
    gpu_config, apply_offset
)
from nms import inference_postprocessing


COLORS = {'red': (139, 0, 0), 'black': (0, 0, 0), 'blue': (32, 178, 170)}

description = 'Inference Options'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-f', '--file', help='CSV Dataset')
parser.add_argument('-i', '--image-no', type=int, help='Image Number')
parser.add_argument('-t', '--pred-threshold', type=float,
    help='Prediction threshold'
)
parser.add_argument('-g', '--gpu', type=int, help='GPU memory allocation')
parser.add_argument('-m', '--model', help='Saved model')
parser.add_argument('-r', '--region-proposals', type=int,
    help='Number of region proposals'
)
parser.add_argument('-l', '--logdir', help='Tensorboard logging directory')
parser.add_argument('-b', '--bounding-boxes', type=int,
    help='No. of bounding boxes'
)
parser.add_argument('-c', '--color', help='Bounding Box Colors')

args = parser.parse_args()
image_no = args.image_no
memory_limit = args.gpu
csv_file_path = args.file
saved_model_path = args.model
region_proposal_no = args.region_proposals
threshold = args.pred_threshold 
logdir = args.logdir
max_output_boxes = args.bounding_boxes
color = COLORS[args.color]

test_image_path = f'./data/original-images/{image_no}.jpg'

gpu_config(memory_limit)

if not logdir:
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    logdir = f'./logs/results/{image_no}/{timestamp}'
else:
    logdir = f'./logs/results/{logdir}'
file_writer = tf.summary.create_file_writer(logdir)

image = np.array(Image.open(test_image_path))
model = load_model(saved_model_path)
test = inference_dataset_etl(csv_file_path, image_no, take=region_proposal_no)

bbox = []
scores = []

for i, (test_files, test_images) in enumerate(test):
    print('Batch:', i)
    offset, cls = predict_on_batch(test_images, model)
    mask = np.argwhere(cls > threshold)
    _, _x, _y, _w, _h = test_files
    for i, data in enumerate(zip(_x, _y, _w, _h)):
        if i not in mask:
            continue
        x1, y1, w, h = data
        y2 = y1 + h
        x2 = x1 + w
        data = (y1, x1, y2, x2)
        _offset = offset[i]
        y1_new, x1_new, y2_new, x2_new = apply_offset(*data, *_offset)
        bbox.append([y1_new, x1_new, y2_new, x2_new])
        scores.append(cls[i])

with open('bboxes', 'wb') as f:
    pickle.dump(bbox, f)

with open('scores', 'wb') as f:
    pickle.dump(scores, f)

bbox = np.array(bbox)
scores = np.array(scores)

print(f"Possible Wallys: {bbox.shape}, score: {scores.shape}")

selected_boxes, selected_scores = inference_postprocessing(
    _bbox=bbox,
    _scores=scores,
    image_height=image.shape[0],
    image_width=image.shape[1],
    max_output_boxes=max_output_boxes,
    score_threshold=0.5,
    iou_threshold=0.5,
    nms_repeats=5
)

print(f"Filtered Possible Wallys: {selected_boxes.shape}")

image = draw_boxes_on_image(
    image=image,
    bboxes=selected_boxes,
    scores=selected_scores,
    color=color,  # (0, 5, 5),
    box_thickness=3
)

log_image(file_writer, f'{image_no}-result', image, 0)
