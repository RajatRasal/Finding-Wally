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

csv_file_path = './data/data.csv'
test_images = [31, 49, 20, 56, 21]
image_no = 5

shuffle_buffer = 10000
batch_size = 250
take = 3000


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

# Extract
test, _ = load_and_split_csv_dataset(csv_file_path, test_images, reader='tf')
model = load_model('./saved_model')

# Transform
test_files = test.shuffle(shuffle_buffer) \
    .filter(lambda *batch: batch[1] == image_no) \
    .map(lambda *batch: (batch[0], *batch[2:6]), tf.data.experimental.AUTOTUNE) \
    .map(cast, tf.data.experimental.AUTOTUNE)
test_images = test_files.map(load_image, tf.data.experimental.AUTOTUNE) \
    .map(vgg16.preprocess_input, tf.data.experimental.AUTOTUNE)

# Load
test = tf.data.Dataset.zip((test_files, test_images)) \
    .take(take) \
    .batch(batch_size)

image = np.array(Image.open(f'./data/original-images/{image_no}.jpg'))

bbox = []
scores = []

threshold = 0.5

for i, (test_file, test_image) in enumerate(test):
    print(i)
    pred = model.predict_on_batch(test_image)
    offset = pred[0]
    cls = pred[1].flatten()
    mask = np.argwhere(cls >= threshold)
    _, _x, _y, _w, _h = test_file
    for i, data in enumerate(zip(_x, _y, _w, _h)):
        if i not in mask:
            continue
        x, y, w, h = data
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        w = tf.cast(w, tf.float32)
        h = tf.cast(h, tf.float32)
        # find center
        x_center = x + (w / 2.0)
        y_center = y + (h / 2.0)
        # apply offset
        x_center += w * offset[i, 0]
        y_center += h * offset[i, 1]
        _w = np.exp(offset[i, 2]) * w
        _h = np.exp(offset[i, 3]) * h
        # print(f'OLD: {x}, {y}, {w}, {h}')
        # print(f'NEW: {x_center}, {y_center}, {_w}, {_h}')
        cv.rectangle(image,
            (int(x), int(y)),
            (int(x + w), int(y + h)),
            (0, 0, 225), 4
        )

        half_width = _w / 2.0
        half_height = _h / 2.0
        y1 = y_center - half_height
        x1 = x_center - half_width
        y2 = y_center + half_height
        x2 = x_center + half_width
        cv.rectangle(image,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 225, 225), 2
        )
        bbox.append([y1, x1, y2, x2])
        scores.append(cls[i])
        # print(f'OLD: {(int(x), int(y)), (int(x + w), int(y + h))}')
        # print(f'NEW: {(int(x_center - (_w // 2)), int(y_center - (_h // 2))), (int(x_center + (_w //2)), int(y_center + (_h // 2)))}')

with open('bboxes', 'wb') as f:
    pickle.dump(bbox, f)

with open('scores', 'wb') as f:
    pickle.dump(scores, f)

# plt.imshow(image)
# plt.show()

logdir = f'./logs/results/{image_no}/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
file_writer = tf.summary.create_file_writer(logdir)

with file_writer.as_default():
    img_summary = np.reshape(image, (-1, *image.shape)) 
    tf.summary.image(f'{image_no}-result', img_summary, step=0)

with open('bboxes', 'rb') as f:
    bbox = pickle.load(f)

with open('scores', 'rb') as f:
    scores = pickle.load(f)

image2 = np.array(Image.open(f'./data/original-images/{image_no}.jpg'))

bbox = np.array(bbox)
scores = np.array(scores)

bbox[:, [1, 3]] /= image2.shape[1]
bbox[:, [0, 2]] /= image2.shape[0]
selected_indices = tf.image.non_max_suppression(
    boxes=tf.convert_to_tensor(bbox),
    scores=tf.convert_to_tensor(scores),
    max_output_size=100,
    iou_threshold=0.5,
    score_threshold=0.4
)
selected_boxes = tf.gather(bbox, selected_indices).numpy()
selected_boxes[:, [1, 3]] *= image2.shape[1]
selected_boxes[:, [0, 2]] *= image2.shape[0]

for y1, x1, y2, x2 in selected_boxes:
    cv.rectangle(image2,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        (0, 225, 225), 2
    )

with file_writer.as_default():
    img2_summary = np.reshape(image2, (-1, *image2.shape)) 
    tf.summary.image(f'{image_no}-result', img2_summary, step=1)

# plt.imshow(image2)
# plt.show()
