import argparse
from datetime import datetime

import cloudpickle as pickle
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score as f1_score_sk

from model import (f1_score, preprocess_data, rcnn_reg_loss,
    rcnn_cls_loss, rcnn_reg_mse, rcnn_cls_f1_score
)
from log import image_grid, plot_to_image


description = 'Make predictions using a model'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-m', '--model', help='Trained model file')
parser.add_argument('-f', '--file', nargs=4, type=argparse.FileType('rb'),
    help='Training files produced by generate_dataset.py'
)
args = parser.parse_args()
x_train_file, y_train_file, x_test_file, y_test_file = args.file
saved_model_path = args.model

# TODO: Change this to load_weights
custom_objects = {
    'rcnn_reg_loss': rcnn_reg_loss,
    'rcnn_cls_loss': rcnn_cls_loss,
    'rcnn_reg_mse': rcnn_reg_mse,
    'rcnn_cls_f1_score': rcnn_cls_f1_score
}
restored_model = tf.keras.models.load_model(saved_model_path,
    custom_objects=custom_objects,
    compile=True,
)
# restored_model.compile(optimizer='adam', loss='binary_crossentropy')

print('Loading data')
try:
    x_train = np.load(x_train_file).astype('float16')
    x_test = np.load(x_test_file).astype('float16')
except Exception:
    x_train = pickle.load(x_train_file).astype('float16')
    x_test = pickle.load(x_test_file).astype('float16')

# TODO: Remove this hack - convert RGB to BGR in selective_search.py
B = x_test[:, :, 0]
R = x_test[:, :, 2]
x_test[:, :, 0] = R
x_test[:, :, 2] = B

y_train = pickle.load(y_train_file)
y_test = pickle.load(y_test_file)

y_train = y_train[1].astype('int8')
y_test = y_test[1].astype('int8')
print('Done loading')

_, test_dataset = preprocess_data(x_train, y_train, x_test, y_test)

result = []
y_test = []

for _x_test, _y_test in test_dataset:
    pred = restored_model.predict(_x_test)
    reg = pred[0]
    cls = pred[1]
    result.append(cls.flatten())
    y_test.append(_y_test.numpy().flatten())

result = np.concatenate(result)
y_test = np.concatenate(y_test)

"""
best = float('-inf') 
best_thres = 0
best_result = []
best_y_test = []
for t in range(10, 90, 5):
    threshold = t / 100  # 0.42
    # for x, y in test_dataset:
    #     pred = restored_model.predict(x)
    #     reg = pred[0]
    #     cls = pred[1]
    #     cls[cls >= threshold] = 1
    #     cls[cls < threshold] = 0
    #     results += cls.flatten().tolist()
    #     y_test += y.numpy().flatten().astype('float16').tolist()


    res = f1_score_sk(y_test, results)
    if res > best:
        best = res
        best_thres = threshold
        best_y_test = y_test
        best_result = results
"""

best_result = result.copy()
best_thres = 0.5
best_result[best_result >= best_thres] = 1
best_result[best_result < best_thres] = 0
best = f1_score_sk(y_test, best_result)

best_y_test = y_test

cm = confusion_matrix(best_y_test, best_result)
print(f'Threshold: {best_thres}')
print(f'f1 score: {best}')
print(f'confusion matrix:\n{cm}')
print(classification_report(best_y_test, best_result))

# Log inference output to Tensorboard
images = []
for x, _ in test_dataset:
    images.append(x)

titles = [f'test: {test}, pred: {pred}' for test, pred in zip(best_y_test, best_result)]

images_in_grid = 25
results_count = len(titles)

logdir = "./logs/test_data/" + datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(logdir)
    
for i in range(0, results_count // images_in_grid):
    bottom = i * images_in_grid
    top = (i + 1) * images_in_grid
    figure = image_grid(images[0][bottom:top].numpy().astype('uint8'), titles[bottom:top])
    grid = plot_to_image(figure)
    
    with file_writer.as_default():
        tf.summary.image("Test Data", grid, step=i)
