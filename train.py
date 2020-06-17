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

from model import (preprocess_dataset, build_and_compile_model,
    build_and_compile_distributed_model, save_model, load_model,
    load_and_split_csv_dataset, gpu_config, predict_on_batch
)


def train(csv_file_path, test_images, validation_steps=5, validation_freq=20,
    epochs=50, verbose=1, repeat_train=4, repeat_test=1, **kwargs
):
    train, test = load_and_split_csv_dataset(csv_file_path, test_images, reader='tf')
    train = preprocess_dataset(train, repeat_train)
    test = preprocess_dataset(test, repeat_test)

    model = build_and_compile_model()
    model.fit(
        train,
        validation_data=test,
        validation_steps=validation_steps,
        validation_freq=validation_freq,
        epochs=epochs,
        verbose=verbose
    )

    return model

def retrain(csv_file_path, test_images, output, validation_steps=5, epochs=50,
    validation_freq=20, batch_size=64, verbose=1, threshold=0.5, lr=0.000001,
    **kwargs
):
    train, _ = load_and_split_csv_dataset(csv_file_path, test_images, reader='tf')
    train = preprocess_dataset(train, 1)

    # load model
    model = load_model(output, lr=lr)

    X_misclassified = []
    y_misclassified = []

    # find false positives
    for X, y in train:
        _, cls = predict_on_batch(X, model, threshold)
        mask = tf.not_equal(y[:, -1], cls.flatten())
        _X_misclassified = tf.boolean_mask(X, mask)
        _y_misclassified = tf.boolean_mask(y, mask)
        X_misclassified.append(_X_misclassified)
        y_misclassified.append(_y_misclassified)

    X_misclassified = tf.concat(X_misclassified, axis=0)
    y_misclassified = tf.concat(y_misclassified, axis=0)

    X_retrain_dataset = tf.data.Dataset.from_tensor_slices(X_misclassified) \
        .map(tf.io.serialize_tensor)
    writer = tf.data.experimental.TFRecordWriter('./data/X_retrain.tfrecord')
    writer.write(X_retrain_dataset)

    y_retrain_dataset = tf.data.Dataset.from_tensor_slices(y_misclassified) \
        .map(tf.io.serialize_tensor)
    writer = tf.data.experimental.TFRecordWriter('./data/y_retrain.tfrecord')
    writer.write(y_retrain_dataset)

    X_retrain_dataset = tf.data.TFRecordDataset("./data/X_retrain.tfrecord") \
        .map(lambda x: tf.io.parse_tensor(x, tf.float32))

    y_retrain_dataset = tf.data.TFRecordDataset("./data/y_retrain.tfrecord") \
        .map(lambda x: tf.io.parse_tensor(x, tf.float32))

    _zip = (X_retrain_dataset, y_retrain_dataset)

    retrain_dataset = tf.data.Dataset.zip(_zip) \
        .batch(batch_size) \
        .prefetch(5)

    # retrain with only misclassified
    model.fit(retrain_dataset, epochs=epochs, verbose=verbose)

    return model

def distributed_train(csv_file_path, test_images, distributed, verbose=1,
    batch_size=64, validation_steps=5, validation_freq=20, epochs=50,
    repeat_test=1, repeat_train=4, **kwargs
):
    print('------------------ Distributed Training ------------------')
    # dist_config = json.loads(distributed.read())
    ip = socket.gethostbyname(socket.gethostname())
    # workers = dist_config['worker']
    workers = ['146.169.53.215:2222']
    # workers = ['146.169.53.225:2223', '146.169.53.207:2222']
    index = list(map(lambda x: x.split(':')[0], workers)).index(ip)

    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    os.environ['TF_CONFIG'] = json.dumps({
        'cluster': {'worker': workers},
        'task': {'type': 'worker', 'index': index}
    })

    num_gpus = len(workers)
    global_batch_size = batch_size * num_gpus

    train, test = load_and_split_csv_dataset(csv_file_path, test_images, reader='tf')
    train = preprocess_dataset(train, repeat_train, global_batch_size)
    test = preprocess_dataset(test, repeat_test, global_batch_size)

    data = pd.read_csv(csv_file_path, index_col=0)
    data_shape = ((~data.image_no.isin(test_images)) & data.fg).sum() * repeat_train
    steps_per_epoch = data_shape // global_batch_size

    multi_worker_model = build_and_compile_distributed_model(strategy)

    multi_worker_model.fit(
        train,
        steps_per_epoch=steps_per_epoch,
        validation_data=test,
        validation_steps=validation_steps,
        validation_freq=validation_freq,
        verbose=verbose
    )

    return multi_worker_model

def main():
    description = 'Choose between local or distributed training'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-f', '--csv-file-path', help='CSV Annotations')
    parser.add_argument('-i', '--test-images', nargs='+', type=int, help='Test images')
    parser.add_argument('-g', '--gpu', type=int, help='GPU memory allocation')
    parser.add_argument('-o', '--output', help='Output saved model')
    parser.add_argument('-t', '--train-type', help='Training type')
    parser.add_argument('-e', '--epochs', type=int, help='Training epochs')
    parser.add_argument('-b', '--batch-size', type=int, help='Batch size per GPU')
    parser.add_argument('-l', '--logdir', help='Tensorboard logging directory')
    parser.add_argument('-d', '--distributed',
        type=argparse.FileType('r'),
        help='Distributed training config id'
    )
    args = parser.parse_args()
    
    # Tensorboard logging callbacks
    logdir = args.logdir + datetime.now().strftime('%Y%m%d-%H%M%S')
    # tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    gpu_config(args.gpu)
    
    train_types = {'local': train, 'dist': distributed_train, 'retrain': retrain}
    train_method = train_types[args.train_type]
    model = train_method(**vars(args))
    save_model(model, args.output)


if __name__ == '__main__':
    main()
