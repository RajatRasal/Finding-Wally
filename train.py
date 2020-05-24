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
from tensorflow.keras.optimizers import Adam, SGD

from model import (preprocess_data, build_and_compile_model, 
    build_and_compile_distributed_model, save_model
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
    BS_PER_GPU = 64
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
    tensorboard_callback = keras.callbacks.TensorBoard(logdir)

    epochs = 50 
    BS_PER_GPU = 128
    NUM_GPUS = 1

    train_dataset, test_dataset = preprocess_data(x_train, y_train,
        x_test, y_test,
        processors=NUM_GPUS,
        batch_size_per_processor=BS_PER_GPU,
        cache=False
    )

    model = build_and_compile_model()
    steps_per_epoch = x_train.shape[0] // (BS_PER_GPU * NUM_GPUS)

    model.fit(train_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=test_dataset,
        validation_freq=5,
        callbacks=[tensorboard_callback],
        verbose=1,
    )

    save_model(saved_model_path)
