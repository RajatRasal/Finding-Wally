from datetime import datetime

from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import tensorflow as tf


df = pd.read_csv('./data/data.csv')
df = df[df.fg]
print(df.shape)
images = []
bbox = []

for _file, group in df.groupby(df.actual):
    img = np.array(Image.open(_file), dtype=np.uint8)
    for i, row in group.iterrows():
        img = cv2.rectangle(img,
            (int(row['x']), int(row['y'])),
            (int(row['x'] + row['w']), int(row['y'] + row['h'])),
            (0, 0, 255), 3
        )
    images.append(img)
    img = np.array(Image.open(_file), dtype=np.uint8)
    for i, row in group.iterrows():
        bbox.append(img[int(row['y']):int(row['y'] + row['h']),
            int(row['x']):int(row['x'] + row['w']), :])

logdir = "./logs/train_data/true_positives/img" + datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(logdir)

with file_writer.as_default():
    for i in range(len(images)):
        tf.summary.image(
            "Wally boxes",
            images[i].reshape(-1, *images[i].shape),
            step=i
        )
        img_obj = Image.fromarray(images[i])
        img_obj.save(f'./data/bbox_drawn/{i}.jpg')
    for i, img in enumerate(bbox):
        try:
            tf.summary.image(
                "bboxes",
                img.reshape(-1, *img.shape),
                step=i
            )
            Image.fromarray(img).save(f'./data/bbox/{i}.jpg')
        except Exception as e:
            continue
