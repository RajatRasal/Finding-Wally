import os
from collections import Counter

import cloudpickle as pickle
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score as f1_score_sk
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD

from model import f1_score, build_model


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
    history = model.fit(X, y, batch_size=batch_size, shuffle=True,
                        steps_per_epoch=y_train.shape[0] / batch_size,
                        class_weight={0: zeros, 1: ones}, epochs=epochs,
                        use_multiprocessing=True, workers=-1, verbose=1)

    return model

def distributed_train_model(X):
    pass

def predict(X, model, threshold=0.6):
    pred = model.predict(X)
    pred[pred >= threshold] = 1
    pred[pred < threshold] = 0
    return pred


if __name__ == '__main__':
    data = pd.read_csv('./data/data.csv')
    print(f'SHAPE: {data.shape} ------------------------')

    X = np.zeros((data.shape[0], 224, 224, 3))
    y = [np.zeros((data.shape[0], 4)), np.zeros((data.shape[0], 1))]

    print('Training Model')

    train_set = int(0.8 * data.shape[0])

    with open('./x_train.pkl', 'rb') as f:
        x_train = pickle.load(f)

    with open('./x_test.pkl', 'rb') as f:
        x_test = pickle.load(f)

    with open('./y_train.pkl', 'rb') as f:
        y_train = pickle.load(f)

    with open('./y_test.pkl', 'rb') as f:
        y_test = pickle.load(f)

    y_train = y_train[1]
    y_test = y_test[1]

    print(y_test[:10])
    print('Y train counter:', Counter(list(y_train.flatten())))
    print('Y test counter:', Counter(list(y_test.flatten())))
    model = build_model()
    model = train_model(x_train, y_train, model, Adam(lr=0.00001), epochs=200, batch_size=200)
    result_in_sample = predict(x_train, model, threshold=0.6)
    result_out_sample = predict(x_test, model, threshold=0.6)

    print(f1_score_sk(y_train.astype('int32').flatten(), result_in_sample.astype('int32').flatten()))
    print(classification_report(y_train.astype('int32').flatten(), result_in_sample.astype('int32').flatten()))
    print(confusion_matrix(y_train.astype('int32').flatten(), result_in_sample.astype('int32').flatten()))

    print(f1_score_sk(y_test.astype('int32').flatten(), result_out_sample.astype('int32').flatten()))
    print(classification_report(y_test.astype('int32').flatten(), result_out_sample.astype('int32').flatten()))
    print(confusion_matrix(y_test.astype('int32').flatten(), result_out_sample.astype('int32').flatten()))

    with open('./result.pkl', 'wb') as f:
        pickle.dump(result_out_sample, f)
