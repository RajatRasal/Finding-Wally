import argparse
import json
import os
import socket
from collections import Counter

import cloudpickle as pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score as f1_score_sk
from tensorflow.keras.optimizers import Adam, SGD

from model import f1_score, build_model


strategy = tf.compat.v1.distribute.experimental.MultiWorkerMirroredStrategy()

def train_model(X, y, model, optimizer, random_state=42,
                keras_metrics=[f1_score], epochs=100, batch_size=32):
    # TODO: Seen and unseen should be between seen and unseen images
    class_weight = Counter(y_train.flatten())
    print(class_weight)
    print(class_weight[0])
    print(class_weight[1])
    zeros = class_weight[1] / y_train.shape[0]
    ones = class_weight[0] / y_train.shape[0]

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=keras_metrics)
    # steps_per_epoch=y_train.shape[0] / batch_size,
    history = model.fit(X, y, batch_size=batch_size, shuffle=True,
                        class_weight={0: zeros, 1: ones}, epochs=epochs,
                        use_multiprocessing=True, workers=-1, verbose=1)

    return model

def scale(image, label):
    image = tf.cast(image, tf.float16)
    # = tf.image.per_image_standardization(x)
    #        return x, y
    image /= 255
    return image, label

def distributed_train_model(X, y, model_builder, workers, index, optimizer,
        random_state=42, keras_metrics=[f1_score], epochs=2, batch_size=32):
    os.environ['TF_CONFIG'] = json.dumps({
        'cluster': {'worker': workers},
        'task': {'type': 'worker', 'index': index}
    })

    NUM_GPUS = len(workers)
    BS_PER_GPU = batch_size

    # multiworker_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    strategy = tf.compat.v1.distribute.experimental.MultiWorkerMirroredStrategy()
    with strategy.scope():
        model = model_builder()
        print(X.shape)
        print(y.shape)
        dataset = tf.data.Dataset.from_tensor_slices((X, y)) \
            .map(scale) \
            .shuffle(X.shape[0]) \
            .batch(BS_PER_GPU * NUM_GPUS, drop_remainder=True) \
            .repeat(epochs) \
            .cache() \
            .prefetch(2 * BS_PER_GPU * NUM_GPUS)
        class_weight = Counter(y.flatten())
        zeros = class_weight[1] / y_train.shape[0]
        ones = class_weight[0] / y_train.shape[0]
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=keras_metrics)
        model.fit(dataset, steps_per_epoch=X.shape[0] // (BS_PER_GPU * NUM_GPUS),
                  epochs=epochs, class_weight={0: zeros, 1: ones}, verbose=1)
        return model

def get_data(x_train, y_train, x_test, y_test):

    def scale(image, label):
        image = tf.cast(image, tf.float16)
        label = tf.cast(label, tf.float16)
        image /= 255
        return image, label

    BS_PER_GPU = 64
    NUM_GPUS = 2

    tf.random.seed(42)
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

def build_distributed_model():
    with strategy.scope():
        model = build_model()
        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=0.00001),
                      metrics=[f1_score])
        return model

def predict(X, model, threshold=0.6):
    pred = model.predict(X)
    pred[pred >= threshold] = 1
    pred[pred < threshold] = 0
    return pred


description = 'Choose between local or distributed training'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-f', '--file', nargs=4, type=argparse.FileType('rb'),
                    help='Training files produced by generate_dataset.py')
parser.add_argument('-d', '--distributed', type=argparse.FileType('r'),
                    help='Distributed training config id')
args = parser.parse_args()
x_train_file, y_train_file, x_test_file, y_test_file = args.file
dist_config_file = args.distributed

x_train = pickle.load(x_train_file).astype('float16')
x_test = pickle.load(x_test_file).astype('float16')
y_train = pickle.load(y_train_file)
y_test = pickle.load(y_test_file)

y_train = y_train[1].astype('int8')
y_test = y_test[1].astype('int8')

print('Y train counter:', Counter(list(y_train.flatten())))
print('Y test counter:', Counter(list(y_test.flatten())))

if dist_config_file:
    print('------------------ Distributed Training ------------------')
    dist_config = json.loads(dist_config_file.read())
    print(dist_config)
    ip = socket.gethostbyname(socket.gethostname())
    index = list(map(lambda x: x.split(':')[0], dist_config['worker'])).index(ip)
    workers = dist_config['worker']

    os.environ['TF_CONFIG'] = json.dumps({
        'cluster': {'worker': workers},
        'task': {'type': 'worker', 'index': index}
    })

    # opt = SGD(learning_rate=0.1, momentum=0.9)
    # batch_size = 64

    train_dataset, test_dataset = get_data(x_train, y_train, x_test, y_test)
    model = build_distributed_model()

    """
    opt = Adam(lr=0.00001)
    model = distributed_train_model(x_train, y_train, build_model,
            workers, index, opt, epochs=3, batch_size=batch_size)
    """

    class_weight = Counter(y_train.flatten())
    zeros = class_weight[1] / y_train.shape[0]
    ones = class_weight[0] / y_train.shape[0]
    model.fit(train_dataset, epochs=2, class_weight={0: zeros, 1: ones}, verbose=1)

    saved_model_path = '/tmp/keras_save'
    model.save(saved_model_path)

    # steps_per_epoch=x.shape[0] // (BS_PER_GPU * NUM_GPUS),
    """
    result = predict(test_dataset, model, threshold=0.6)
    """

    """
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)) \
            .map(scale) \
            .batch(batch_size * len(workers), drop_remainder=True)

    tf_x_test = test_dataset.map(lambda image, label: image)
    tf_y_test = test_dataset.map(lambda image, label: label)
    threshold = 0.6
    results = []
    y_test = []
    for x, y in test_dataset:
        print(x.shape)
        pred = model.predict(x)
        print(pred.shape)
        pred[pred >= threshold] = 1
        pred[pred < threshold] = 0
        results += pred.flatten().tolist()
        # y = tf_y_test.take(1)
        y_test += y.numpy().flatten().astype('float16').tolist()

    print(len(results))
    print(len(y_test))
    print(confusion_matrix(y_test, results))
    print(classification_report(y_test, results))
    print(f1_score(y_test, results))
    """
else:
    model = build_model()
    print('***************** Local training *****************')
    # model = train_model(x_train/255, y_train, model, Adam(lr=0.00001), epochs=3, batch_size=200)
    model = train_model(x_train/255, y_train, model, Adam(lr=0.00001), epochs=2, batch_size=128)
    
    result_in_sample = predict(x_train/255, model, threshold=0.6)
    result_out_sample = predict(x_test/255, model, threshold=0.6)
    
    print(f1_score_sk(y_train.astype('int32').flatten(), result_in_sample.astype('int32').flatten()))
    print(classification_report(y_train.astype('int32').flatten(), result_in_sample.astype('int32').flatten()))
    print(confusion_matrix(y_train.astype('int32').flatten(), result_in_sample.astype('int32').flatten()))
    
    print(f1_score_sk(y_test.astype('int32').flatten(), result_out_sample.astype('int32').flatten()))
    print(classification_report(y_test.astype('int32').flatten(), result_out_sample.astype('int32').flatten()))
    print(confusion_matrix(y_test.astype('int32').flatten(), result_out_sample.astype('int32').flatten()))
    
    with open('./result.pkl', 'wb') as f:
        pickle.dump(result_out_sample, f)
