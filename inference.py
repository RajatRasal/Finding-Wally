import argparse

import cloudpickle as pickle
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score as f1_score_sk

from model import f1_score, preprocess_data


description = 'Make predictions using a model'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-m', '--model', nargs=1,
                    help='Trained model file')
parser.add_argument('-f', '--file', nargs=4, type=argparse.FileType('rb'),
                    help='Training files produced by generate_dataset.py')
args = parser.parse_args()
x_train_file, y_train_file, x_test_file, y_test_file = args.file
saved_model_path = args.model[0]

# TODO: Change this to load_weights
restored_model = tf.keras.models.load_model(saved_model_path, compile=True,
        custom_objects={'f1_score': f1_score})
# restored_model.compile(optimizer='adam', loss='binary_crossentropy')

x_train = pickle.load(x_train_file).astype('float16')
x_test = pickle.load(x_test_file).astype('float16')
y_train = pickle.load(y_train_file)
y_test = pickle.load(y_test_file)

y_train = y_train[1].astype('int8')
y_test = y_test[1].astype('int8')

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

print(f'Threshold: {best_thres}')
print(f'f1 score: {best}')
print(f'confusion matrix:\n{confusion_matrix(best_y_test, best_result)}')
print(classification_report(best_y_test, best_result))
