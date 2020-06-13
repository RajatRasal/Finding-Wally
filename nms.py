from datetime import datetime

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
from PIL import Image
from tensorflow.keras.applications import vgg16

from log import draw_boxes_on_image, log_image
from model import load_model, load_csv_dataset


def nms(bboxes, scores, max_output_boxes=10, iou_threshold=0.8,
    score_threshold=0.5
):
    selected_indices = tf.image.non_max_suppression(
        boxes=tf.convert_to_tensor(bboxes),
        scores=tf.convert_to_tensor(scores),
        max_output_size=max_output_boxes,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold
    )
    selected_boxes = tf.gather(bboxes, selected_indices).numpy()
    selected_scores = tf.gather(scores, selected_indices).numpy()
    return selected_boxes, selected_scores


if __name__ == '__main__':
    tf.random.set_seed(42)

    image_no = 5
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
    
        selected_boxes, selected_scores = nms(bbox, scores)
        selected_boxes[:, [1, 3]] *= image.shape[1]
        selected_boxes[:, [0, 2]] *= image.shape[0]
    
        image = draw_boxes_on_image(image, selected_boxes, selected_scores)
        log_image(file_writer, f'{image_no}-result', image, i)
