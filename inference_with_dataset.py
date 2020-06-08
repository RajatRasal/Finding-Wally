import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications import vgg16

from model import load_model, load_csv_dataset


csv_file_path = './data/data.csv'
test_images = [19, 31, 49, 20, 56, 21]
image_no = 19


@tf.function
def cast(img_file, x, y, w, h):
    x = tf.cast(x, tf.int32)
    y = tf.cast(y, tf.int32)
    w = tf.cast(w, tf.int32)
    h = tf.cast(h, tf.int32)
    return (img_file, x, y, w, h)

@tf.function
def load_image(img_file, x, y, w, h):
    image_string = tf.io.read_file(img_file)
    image = tf.image.decode_jpeg(image_string)
    image = tf.image.crop_to_bounding_box(image, y, x, h, w)
    image = tf.image.resize(image, [224, 224])
    return image

_, test = load_csv_dataset(csv_file_path, test_images, reader='tf')
test_files = test.shuffle(10000) \
    .filter(lambda *batch: batch[1] == image_no) \
    .map(lambda *batch: (batch[0], *batch[2:6]), tf.data.experimental.AUTOTUNE) \
    .map(cast, tf.data.experimental.AUTOTUNE)
test_images = test_files.map(load_image, tf.data.experimental.AUTOTUNE) \
    .map(vgg16.preprocess_input, tf.data.experimental.AUTOTUNE)
test = tf.data.Dataset.zip((test_files, test_images)).take(500)

image = np.array(Image.open('./data/original-images/19.jpg'))
model = load_model('./saved_model')

for i, (test_file, test_image) in enumerate(test):
    img_file, x, y, w, h = test_file
    pred = model.predict(tf.expand_dims(test_image, 0))
    if pred[-1] >= 0.6:
        print(i)
        cv.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 225), 2)

plt.imshow(image)
plt.show()
