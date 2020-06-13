from datetime import datetime

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
from PIL import Image
from tensorflow.keras.applications import vgg16

from model import load_model, load_csv_dataset


tf.random.set_seed(42)

image_no = 19
logdir = f'./logs/results/{image_no}/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
file_writer = tf.summary.create_file_writer(logdir)

with open('bboxes', 'rb') as f:
    bbox = pickle.load(f)

with open('scores', 'rb') as f:
    scores = pickle.load(f)

image = np.array(Image.open(f'./data/original-images/{image_no}.jpg'))

bbox = np.array(bbox)
scores = np.array(scores)

bbox[:, [1, 3]] /= image.shape[1]
bbox[:, [0, 2]] /= image.shape[0]

repeats = 10

for i in range(repeats):
    print(i)
    image = np.array(Image.open(f'./data/original-images/{image_no}.jpg'))
    
    np.random.shuffle(bbox)

    selected_indices = tf.image.non_max_suppression(
        boxes=tf.convert_to_tensor(bbox),
        scores=tf.convert_to_tensor(scores),
        max_output_size=20,
        iou_threshold=0.8,
        score_threshold=0.5
    )
    selected_boxes = tf.gather(bbox, selected_indices).numpy()
    selected_boxes[:, [1, 3]] *= image.shape[1]
    selected_boxes[:, [0, 2]] *= image.shape[0]
    selected_scores = tf.gather(scores, selected_indices).numpy()
    
    for (y1, x1, y2, x2), score in zip(selected_boxes, selected_scores):
        rect_obj = cv.rectangle(image,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 0, 225), 5
        )
        cv.putText(rect_obj,
            str(score),
            (int(x1), int(y1) - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            0.9, (0, 0, 225), 2
        )
    
    with file_writer.as_default():
        img_summary = np.reshape(image, (-1, *image.shape)) 
        tf.summary.image(f'{image_no}-result', img_summary, step=i)
