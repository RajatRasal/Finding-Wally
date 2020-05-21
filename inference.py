import argparse
from datetime import datetime

import cloudpickle as pickle
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score as f1_score_sk

from model import f1_score, preprocess_data
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
restored_model = tf.keras.models.load_model(saved_model_path, compile=True,
    custom_objects={'f1_score': f1_score}
)
# restored_model.compile(optimizer='adam', loss='binary_crossentropy')

print('Loading data')
try:
    x_train = np.load(x_train_file).astype('float16')
    x_test = np.load(x_test_file).astype('float16')
except Exception:
    x_train = pickle.load(x_train_file).astype('float16')
    x_test = pickle.load(x_test_file).astype('float16')

y_train = pickle.load(y_train_file)
y_test = pickle.load(y_test_file)

y_train = y_train[1].astype('int8')
y_test = y_test[1].astype('int8')
print('Done loading')

_, test_dataset = preprocess_data(x_train, y_train, x_test, y_test)

best = 0
best_thres = 0
best_result = []
best_ytest = []
for t in range(40, 50, 1):
    threshold = t / 100  # 0.42
    results = []
    y_test = []
    for x, y in test_dataset:
        pred = restored_model.predict(x)
        pred[pred >= threshold] = 1
        pred[pred < threshold] = 0
        results += pred.flatten().tolist()
        y_test += y.numpy().flatten().astype('float16').tolist()

    res = f1_score_sk(y_test, results)
    if res > best:
        best = res
        best_thres = threshold
        best_y_test = y_test
        best_result = results

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
