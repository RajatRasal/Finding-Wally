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

x_train = pickle.load(open('x_train.pkl', 'rb')).astype('float16')
print(x_train.shape)
img = np.reshape(x_train[0], (-1, 224, 224, 3))
print(x_train.shape)

# Sets up a timestamped log directory.
logdir = "./logs/train_data/" + datetime.now().strftime("%Y%m%d-%H%M%S")
# Creates a file writer for the log directory.
file_writer = tf.summary.create_file_writer(logdir)

# Using the file writer, log the reshaped image.
with file_writer.as_default():
    tf.summary.image("Training data", img, step=0)
