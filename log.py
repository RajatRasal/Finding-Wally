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
"""
logdir = "logs/plots/" + datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(logdir)

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

def image_grid():
    # Create a figure to contain the plot.
    figure = plt.figure(figsize=(10,10))

    for i in range(25):
        # Start next subplot.
        plt.subplot(5, 5, i + 1, title=class_names[train_labels[i]])
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
    
    return figure

# Prepare the plot
figure = image_grid()
# Convert to image and log
with file_writer.as_default():
tf.summary.image("Training data", plot_to_image(figure), step=0)
"""

# Sets up a timestamped log directory.
logdir = "./logs/train_data/" + datetime.now().strftime("%Y%m%d-%H%M%S")
# Creates a file writer for the log directory.
file_writer = tf.summary.create_file_writer(logdir)

import os
from PIL import Image

# Using the file writer, log the reshaped image.
with file_writer.as_default():
    x_train = pickle.load(open('x_train.pkl', 'rb'))
    images = np.reshape(x_train[0:25].astype('int'), (-1, 224, 224, 3))
    print(images[0].mean())
    print(images[0].std())
    from collections import Counter
    print(Counter(images[0].flatten()))
    np.save('trial_25.npy', images)
    tf.summary.image("Training data", images, max_outputs=25, step=0)

    for img_file in os.listdir('./data/original-images/'):
        if 'jpg' in img_file:
            og_img = Image.open('./data/original-images/' + img_file)
            tf.summary.image(img_file, images, step=0)
