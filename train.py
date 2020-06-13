import argparse
import functools
import json
import os
import socket
from collections import Counter
from datetime import datetime

import cloudpickle as pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score as f1_score_sk

from model import (preprocess_data, build_and_compile_model,
    build_and_compile_distributed_model, save_model, preprocess_dataset,
    load_and_split_csv_dataset, load_model
)


description = 'Choose between local or distributed training'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-f', '--file', help='CSV Dataset')
parser.add_argument('-i', '--images', nargs='+', type=int, help='Test images')
parser.add_argument('-o', '--output', help='Output saved model')
parser.add_argument('-t', '--train-type', help='Training type')
parser.add_argument('-l', '--logdir', help='Tensorboard logging directory')
parser.add_argument('-d', '--distributed',
    type=argparse.FileType('r'),
    help='Distributed training config id'
)
args = parser.parse_args()
csv_file_path = args.file
test_images = args.images
dist_config_file = args.distributed
saved_model_path = args.output
train = args.train_type

# Tensorboard logging callbacks
logdir = args.logdir + datetime.now().strftime('%Y%m%d-%H%M%S')
# tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

GPU_MEMORY = 7000

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        gpu = gpus[0]
        config = tf.config.experimental.VirtualDeviceConfiguration(memory_limit=11000)
        tf.config.experimental.set_virtual_device_configuration(gpu, [config])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

if dist_config_file:
    print('------------------ Distributed Training ------------------')
    dist_config = json.loads(dist_config_file.read())
    ip = socket.gethostbyname(socket.gethostname())
    # workers = dist_config['worker']
    workers = ['146.169.53.219:2222']
    # workers = ['146.169.53.225:2223', '146.169.53.207:2222']
    index = list(map(lambda x: x.split(':')[0], workers)).index(ip)
    print(workers)

    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    os.environ['TF_CONFIG'] = json.dumps({
        'cluster': {'worker': workers},
        'task': {'type': 'worker', 'index': index}
    })
    print(os.environ['TF_CONFIG'])

    BS_PER_GPU = 32
    NUM_GPUS = len(workers)
    train_dataset, test_dataset = preprocess_data(x_train, y_train,
        x_test, y_test,
        NUM_GPUS, BS_PER_GPU
    )

    model = build_and_compile_distributed_model(strategy,
        batch_size=BS_PER_GPU * NUM_GPUS
    )

    steps_per_epoch = x_train.shape[0] // (BS_PER_GPU * NUM_GPUS)
    model.fit(train_dataset,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        verbose=1
    )

    save_model(saved_model_path)
elif train == 'local':
    print('***************** Local training *****************')
    EPOCHS = 200
    # file_writer_cm = tf.summary.create_file_writer(logdir + '-train')
    # tensorboard_callback = keras.callbacks.TensorBoard(logdir)
    train, test = load_and_split_csv_dataset(csv_file_path, test_images, reader='tf')
    train = preprocess_dataset(train, 4)
    test = preprocess_dataset(test, 1)

    model = build_and_compile_model()
    model.fit(train,
        validation_data=test,
        validation_steps=5,
        validation_freq=20,
        epochs=EPOCHS,
        verbose=1
    )

    save_model(model, saved_model_path)
elif train == 'retrain':
    train, _ = load_and_split_csv_dataset(csv_file_path, test_images, reader='tf')
    train = preprocess_dataset(train, 1)

    # load model
    model = load_model(saved_model_path, lr=0.000001)

    X_misclassified = []
    y_misclassified = []

    threshold = 0.50

    # find false positives
    for X, y in train:
        pred = model.predict_on_batch(X)
        cls = pred[1]
        cls[cls >= threshold] = 1
        cls[cls < threshold] = 0
        mask = tf.not_equal(y[:, -1], cls.flatten())
        _X_misclassified = tf.boolean_mask(X, mask)
        _y_misclassified = tf.boolean_mask(y, mask)
        X_misclassified.append(_X_misclassified)
        y_misclassified.append(_y_misclassified)

    X_misclassified = tf.concat(X_misclassified, axis=0)
    y_misclassified = tf.concat(y_misclassified, axis=0)

    print(X_misclassified.shape, X_misclassified.dtype)
    print(y_misclassified.shape, y_misclassified.dtype)

    X_retrain_dataset = tf.data.Dataset.from_tensor_slices(X_misclassified) \
        .map(tf.io.serialize_tensor)
    writer = tf.data.experimental.TFRecordWriter('./data/X_retrain.tfrecord')
    writer.write(X_retrain_dataset)

    y_retrain_dataset = tf.data.Dataset.from_tensor_slices(y_misclassified) \
        .map(tf.io.serialize_tensor)
    writer = tf.data.experimental.TFRecordWriter('./data/y_retrain.tfrecord')
    writer.write(y_retrain_dataset)

    BATCH_SIZE = 64
    EPOCHS = 100

    X_retrain_dataset = tf.data.TFRecordDataset("./data/X_retrain.tfrecord") \
        .map(lambda x: tf.io.parse_tensor(x, tf.float32))

    y_retrain_dataset = tf.data.TFRecordDataset("./data/y_retrain.tfrecord") \
        .map(lambda x: tf.io.parse_tensor(x, tf.float32))

    _zip = (X_retrain_dataset, y_retrain_dataset)

    retrain_dataset = tf.data.Dataset.zip(_zip) \
        .batch(BATCH_SIZE) \
        .prefetch(5)

    # retrain with only misclassified
    model.fit(retrain_dataset, epochs=EPOCHS, verbose=1)

    # save model
    save_model(model, saved_model_path)
