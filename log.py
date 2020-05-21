import io
import itertools
from datetime import datetime

import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import cloudpickle as pickle
import sklearn.metrics

"""
print("TensorFlow version: ", tf.__version__)
assert .parse(tf.__version__).release[0] >= 2, \
            "This notebook requires TensorFlow 2.0 or above."
"""

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

    for i in range(25):
        plt.subplot(5, 5, i + 1, title=titles[i])
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i])
    
    return figure

def plot_confusion_matrix(cm, class_names):
    return None

def log_confusion_matrix(cm, file_writer):
    figure = plot_confusion_matrix(cm, class_names=class_names)
    cm_image = plot_to_image(figure)

    with file_writer_cm.as_default():
        tf.summary.image("Confusion Matrix", cm_image)


# # Prepare the plot
# figure = image_grid()
# # Convert to image and log
# with file_writer.as_default():
# tf.summary.image("Training data", plot_to_image(figure), step=0)

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
