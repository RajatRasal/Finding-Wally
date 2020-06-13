import io
import itertools
from datetime import datetime

import cloudpickle as pickle
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def plot_to_image(figure):
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image

def image_grid(train_images, titles, figsize=(10, 10)):
    figure = plt.figure(figsize=figsize)

    col_size = 5
    image_count = train_images.shape[0]
    row_count = image_count // col_size + 1

    # print(image_count, row_count)

    for i in range(image_count):
        # print(i)
        plt.subplot(row_count, col_size, i + 1, title=titles[i])
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i])
    
    return figure

def plot_confusion_matrix(cm, class_names):
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
    return figure

def log_confusion_matrix(cm, file_writer):
    figure = plot_confusion_matrix(cm, class_names=class_names)
    cm_image = plot_to_image(figure)

    with file_writer_cm.as_default():
        tf.summary.image("Confusion Matrix", cm_image)

def log_image(file_writer, log_name, image, i):
    with file_writer.as_default():
        _image = np.reshape(image, (-1, *image.shape)) 
        tf.summary.image(log_name, _image, step=i)

def draw_boxes_on_image(image, bboxes, scores=None, color=(0, 0, 225),
    box_thickness=4, text_thickness=2, text_line_type=2,
    text_font=cv.FONT_HERSHEY_SIMPLEX, text_font_scale=0.9
):
    if scores is None:
        scores = np.zeros(bboxes.shape[0])

    for (y1, x1, y2, x2), score in zip(bboxes, scores):
        rect_obj = cv.rectangle(
            img=image,
            pt1=(int(x1), int(y1)),
            pt2=(int(x2), int(y2)),
            color=color,
            thickness=box_thickness
        )
        if score:
            cv.putText(
                img=rect_obj,
                text=str(score),
                org=(int(x1), int(y1) - 10),
                fontFace=text_font,
                fontScale=text_font_scale,
                color=color,
                thickness=text_thickness,
                lineType=text_line_type
            )

    return image


if __name__ == '__main__':
    # Sets up a timestamped log directory.
    logdir = "./logs/train_data/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    # Creates a file writer for the log directory.
    file_writer = tf.summary.create_file_writer(logdir)
    
    import os
    from PIL import Image
    from collections import Counter
    
    # Using the file writer, log the reshaped image.
    with file_writer.as_default():
        # x_train = pickle.load(open('x_train.pkl', 'rb'))
        x_train = np.load('trial_25.npy')
        images = np.reshape(x_train.astype('uint8'), (-1, 224, 224, 3))
        tf.summary.image("Training data", images, max_outputs=25, step=0)
    
        # for img_file in os.listdir('./data/original-images/'):
        #     if 'jpg' in img_file:
        #         og_img = Image.open('./data/original-images/' + img_file)
        #         tf.summary.image(img_file, images, step=0)
