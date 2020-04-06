import argparse

import cloudpickle as pickle
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score as f1_score_sk

from model import f1_score


def get_data(x_train, y_train, x_test, y_test):

    def scale(image, label):
        image = tf.cast(image, tf.float16)
        label = tf.cast(label, tf.float16)
        image /= 255
        return image, label

    BS_PER_GPU = 64
    NUM_GPUS = 2

    tf.random.set_seed(42)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
        .map(scale) \
        .shuffle(x_train.shape[0]) \
        .batch(BS_PER_GPU * NUM_GPUS, drop_remainder=True) \
        .cache()
        # .repeat(epochs) \
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)) \
        .map(scale) \
        .shuffle(x_test.shape[0]) \
        .batch(BS_PER_GPU * NUM_GPUS, drop_remainder=True) \
        .cache()

    return train_dataset, test_dataset


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

_, test_dataset = get_data(x_train, y_train, x_test, y_test)

# tf_x_test = test_dataset.map(lambda image, label: image)
# tf_y_test = test_dataset.map(lambda image, label: label)

threshold = 0.6
results = []
y_test = []
for x, y in test_dataset:
    pred = restored_model.predict(x)
    pred[pred >= threshold] = 1
    pred[pred < threshold] = 0
    results += pred.flatten().tolist()
    y_test += y.numpy().flatten().astype('float16').tolist()

print(len(results))
print(len(y_test))
print(confusion_matrix(y_test, results))
print(classification_report(y_test, results))
print(f1_score_sk(y_test, results))
