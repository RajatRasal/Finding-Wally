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
    score_threshold=0.5, soft_nms_sigma=0.0
):
    nms_results = tf.image.non_max_suppression_with_scores(
        boxes=tf.convert_to_tensor(bboxes),
        scores=tf.convert_to_tensor(scores),
        max_output_size=max_output_boxes,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        soft_nms_sigma=soft_nms_sigma
    )
    selected_indices, selected_scores = nms_results
    selected_boxes = tf.gather(bboxes, selected_indices).numpy()
    selected_scores = selected_scores.numpy()
    return selected_boxes, selected_scores


def repeated_nms(bboxes, scores, repeats, **kwargs):
    _bboxes = bboxes.copy()
    _scores = scores.copy()
    for i in range(repeats):
        _bboxes, _scores = nms(_bboxes, _scores, **kwargs)
    return _bboxes, _scores

def inference_postprocessing(_bbox, _scores, image_height, image_width,
    max_output_boxes=40, score_threshold=0.5, iou_threshold=0.95
):
    repeats = 1

    bbox = _bbox.copy()
    scores = _scores.copy()

    bbox[:, [1, 3]] /= image_width
    bbox[:, [0, 2]] /= image_height

    # Shuffle bounding boxes
    shuffle_idx = np.random.permutation(bbox.shape[0])
    bbox = bbox[shuffle_idx]
    scores = scores[shuffle_idx]

    selected_boxes, selected_scores = repeated_nms(
        bboxes=bbox,
        scores=scores,
        repeats=repeats,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
        max_output_boxes=max_output_boxes,
    )

    selected_boxes[:, [1, 3]] *= image_width
    selected_boxes[:, [0, 2]] *= image_height

    mask_width = (selected_boxes[:, 2] - selected_boxes[:, 0]) < 200
    mask_height = (selected_boxes[:, 3] - selected_boxes[:, 1]) < 200
    mask = (mask_width & mask_height).flatten()
    selected_boxes = selected_boxes[mask, :]
    selected_scores = selected_scores[mask]

    # TODO: Retrain model so that we can remove this hack. 
    #   Need to improve regression performance
    selected_boxes[:, [1, 3]] -= 10

    return selected_boxes, selected_scores


if __name__ == '__main__':
    tf.random.set_seed(42)

    image_no = 21
    logdir = f'./logs/results/{image_no}/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    file_writer = tf.summary.create_file_writer(logdir)
    
    with open('bboxes', 'rb') as f:
        bbox = pickle.load(f)
    
    with open('scores', 'rb') as f:
        scores = pickle.load(f)
    
    bbox = np.array(bbox)
    scores = np.array(scores)
    image = np.array(Image.open(f'./data/original-images/{image_no}.jpg'))
    print(f"Possible Wallys: {bbox.shape}")

    selected_boxes, selected_scores = inference_postprocessing(bbox, scores, image.shape[0], image.shape[1])

    print(f"Filtered Possible Wallys: {selected_boxes.shape}")
  
    image = draw_boxes_on_image(
        image=image,
        bboxes=selected_boxes,
        scores=selected_scores,
        color=(0, 5, 5),
        box_thickness=1
    )
    log_image(file_writer, f'{image_no}-result', image, 1)
