import argparse
from datetime import datetime

import cloudpickle as pickle
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score as f1_score_sk

from model import load_model, load_csv_dataset, preprocess_dataset
from log import image_grid, plot_to_image


description = 'Make predictions using a model'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-m', '--model', help='Trained model file')
parser.add_argument('-i', '--images', nargs='+', type=int, help='Testing images')
parser.add_argument('-f', '--file', help='Test CSV dataset')
parser.add_argument('-t', '--test-type', help='In-sample or out-of-sample')

args = parser.parse_args()
csv_file_path = args.file
saved_model_path = args.model
test_images = args.images
test_type = args.test_type

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        gpu = gpus[0]
        config = tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7000)
        tf.config.experimental.set_virtual_device_configuration(gpu, [config])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

train, test = load_csv_dataset(csv_file_path, test_images, reader='tf')
if test_type == 'in':
    test = preprocess_dataset(train, repeat=1)
else:
    test = preprocess_dataset(test, repeat=1)

model = load_model(saved_model_path, False)

result = []
y_test = []

for _x_test, _y_test in test:
    pred = model.predict_on_batch(_x_test)
    cls = pred[1]
    result.extend(list(cls.flatten()))
    y_test.extend(list(_y_test.numpy()[:, -1].flatten()))

result = np.array(result)
y_test = np.array(y_test)

best = float('-inf') 

for i in range(47, 52, 1):
    _result = result.copy()
    thres = i / 100
    _result[_result >= thres] = 1
    _result[_result < thres] = 0
    f1 = f1_score_sk(y_test, _result)
    cm = confusion_matrix(y_test, _result)
    print(f'Threshold: {i}')
    print(f'f1 score: {f1}')
    print(f'confusion matrix:\n{cm}')

"""
    if f1 > best:
        best = f1 
        best_thres = thres
        best_result = _result

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
"""
