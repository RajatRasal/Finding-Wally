import argparse
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

from model import f1_score, build_model, preprocess_data


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
    history = model.fit(X, y, batch_size=batch_size,
        shuffle=True,
        class_weight={0: zeros, 1: ones},
        epochs=epochs,
        use_multiprocessing=True,
        workers=-1,
        verbose=1
    )

    return model

def build_distributed_model(strategy):
    with strategy.scope():
        model = build_model()
        model.compile(loss='binary_crossentropy',
            optimizer=Adam(lr=0.00001),
            metrics=[f1_score]
        )
        return model

def predict(X, model, threshold=0.6):
    pred = model.predict(X)
    pred[pred >= threshold] = 1
    pred[pred < threshold] = 0
    return pred


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
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
file_writer_cm = tf.summary.create_file_writer(logdir + '/cm')

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
    # strategy = tf.compat.v1.distribute.experimental.MultiWorkerMirroredStrategy()

    epochs = 50
    BS_PER_GPU = 64
    NUM_GPUS = len(workers)
    train_dataset, test_dataset = preprocess_data(x_train, y_train,
        x_test, y_test,
        NUM_GPUS, BS_PER_GPU
    )
    model = build_distributed_model(strategy)

    class_weight = Counter(y_train.flatten())
    zeros = class_weight[1] / y_train.shape[0]
    ones = class_weight[0] / y_train.shape[0]
    steps_per_epoch = x_train.shape[0] // (BS_PER_GPU * NUM_GPUS)
    model.fit(train_dataset, 
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        class_weight={0: zeros, 1: ones},
        verbose=1
    )

    model.save(saved_model_path)
else:
    model = build_model()
    print('***************** Local training *****************')
    # model = train_model(x_train/255, y_train, model, Adam(lr=0.00001), epochs=3, batch_size=200)
    model = train_model(x_train, y_train, model, Adam(lr=0.00001), epochs=2, batch_size=128)
    
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
