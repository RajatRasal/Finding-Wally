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
    load_csv_dataset
)


description = 'Choose between local or distributed training'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-f', '--file', nargs=4,
    type=argparse.FileType('rb'),
    help='Training files produced by generate_dataset.py'
)
parser.add_argument('-d', '--distributed',
    type=argparse.FileType('r'),
    help='Distributed training config id'
)
parser.add_argument('-l', '--logdir',
    help='Tensorboard logging directory'
)
parser.add_argument('-o', '--output',
    help='Output saved model'
)
args = parser.parse_args()
x_train_file, y_train_file, x_test_file, y_test_file = args.file
dist_config_file = args.distributed
saved_model_path = args.output

# Tensorboard logging callbacks
logdir = args.logdir + datetime.now().strftime('%Y%m%d-%H%M%S')
# tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

"""
x_train = pickle.load(x_train_file).astype('float64')
x_test = pickle.load(x_test_file).astype('float64')
_y_train = pickle.load(y_train_file)
_y_test = pickle.load(y_test_file)

# TODO: Remove this hack - convert RGB to BGR in selective_search.py
B = x_train[:, :, 0]
R = x_train[:, :, 2]
x_train[:, :, 0] = R
x_train[:, :, 2] = B

# TODO: Remove all of this and put into selective_search.py or generate_dataset.py
y_reg_train = _y_train[0]
y_cls_train = _y_train[1]
y_reg_test = _y_test[0]
y_cls_test = _y_test[1]

y_reg_train[np.isnan(y_reg_train)] = 0
y_reg_test[np.isnan(y_reg_test)] = 0

assert y_reg_train.shape[0] == y_cls_train.shape[0]
assert y_reg_test.shape[0] == y_cls_test.shape[0]

y_train = np.zeros((y_reg_train.shape[0], 5))
y_test = np.zeros((y_reg_test.shape[0], 5))
print(f'Train shape: {y_train.shape}')
print(f'Test shape: {y_test.shape}')

y_train[:, 0:4] = y_reg_train
y_train[:, 4:] = y_cls_train
y_test[:, 0:4] = y_reg_test
y_test[:, 4:] = y_cls_test

# print('Y train counter:', Counter(list(y_train.flatten())))
# print('Y test counter:', Counter(list(y_test.flatten())))
"""

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

    epochs = 50
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
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        verbose=1
    )

    save_model(saved_model_path)
else:
    print('***************** Local training *****************')
    # file_writer_cm = tf.summary.create_file_writer(logdir + '-train')
    # tensorboard_callback = keras.callbacks.TensorBoard(logdir)

    epochs = 50
    
    CSV_FILE_PATH = './data/data.csv'
    TEST_IMAGES = [19, 31, 49, 20, 56, 21]

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            # Currently, memory growth needs to be the same across GPUs
            # for gpu in gpus:
            #     tf.config.experimental.set_memory_growth(gpu, True)
            # logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7000)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    train, test = load_csv_dataset(CSV_FILE_PATH, TEST_IMAGES, reader='tf')
    train = preprocess_dataset(train)

    model = build_and_compile_model()
    model.fit(train, epochs=epochs, verbose=1)

    save_model(model, saved_model_path)
