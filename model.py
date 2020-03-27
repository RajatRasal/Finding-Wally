import os

import cv2 as cv
import cloudpickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import keras.backend as kb
from keras.layers import Dense
from keras import Model, losses
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16

from selective_search import scale_image_in_aspect_ratio


vgg = VGG16(weights='imagenet', include_top=True)

for layers in (vgg.layers):
    layers.trainable = False

# VGG Backbone - output from convolutional layers
backbone = vgg.layers[-4].output
# Attach my own FC layers on top of VGG backbone
fc = Dense(500, activation='relu', name='my_fc1')(backbone)
fc = Dense(128, activation='relu', name='my_fc2')(fc)

# Loss function for bounding box regression
# reg = Dense(4, activation='linear', name='bbox_reg')(fc)
# Loss function for foreground/background classification
cls = Dense(1, activation='sigmoid', name='fg_cls')(fc)

# Combine backbone and my outputs to form the NN pipeline
model = Model(inputs=vgg.input, outputs=[cls])
# print(model.summary())

def multitask_reg_cls_loss(y_true, y_pred):
    """
    reg_pred = y_pred[0]
    cls_pred = y_pred[1]
    reg_true = y_true[0]
    cls_true = y_true[1]

    loss = kb.mean(
        losses.binary_crossentropy(
            kb.reshape(cls_true, (-1, 1)),
            kb.reshape(cls_pred, (-1, 1))) + \
        (cls_true * (reg_true - reg_pred))
    )
    print(loss)
    """
    cls_pred = kb.clip(y_pred[0], kb.epsilon(), 1-kb.epsilon())
    cls_true = kb.clip(y_true[0], kb.epsilon(), 1-kb.epsilon())
    logloss = -(0.8 * cls_true * kb.log(cls_pred) + \
                0.2 * (1 - cls_true) * kb.log(1 - cls_pred))
    return kb.mean(logloss)
    # losses.binary_crossentropy(cls_true, cls_pred)

def recall_m(y_true, y_pred):
    true_positives = kb.sum(kb.round(kb.clip(y_true * y_pred, 0, 1)))
    possible_positives = kb.sum(kb.round(kb.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + kb.epsilon())
    return recall

optimizer = SGD(lr=0.0001)
model.compile(loss=multitask_reg_cls_loss, optimizer=optimizer,
              metrics=["accuracy", recall_m])


if __name__ == '__main__':
    new_width = 1000

    data = pd.read_csv('./data/data.csv')
    # data = data.sample(frac=1).reset_index(drop=True)

    X = np.zeros((data.shape[0], 224, 224, 3))
    y = [np.zeros((data.shape[0], 4)), np.zeros((data.shape[0], 1))]

    if os.path.lexists('./x.pkl'):
        print('Loading Data')
        with open('./x.pkl', 'rb') as f:
            X = pickle.load(f)
        with open('./y.pkl', 'rb') as f:
            y = pickle.load(f)
    else:
        print('Generating Data')
        for i, row in data.iterrows():
            if i % 100 == 0:
                print(i)
            full_img = cv.imread(f'./data/original-images/{int(row.actual)}.jpg')
            full_img_scaled = scale_image_in_aspect_ratio(full_img, 1000)
            proposal = full_img_scaled[int(row.y):int(row.y+row.h),
                                       int(row.x):int(row.x+row.w)]
            proposal = cv.resize(proposal, (224, 224))
            X[i] = proposal

            y[0][i, 0] = row.t_x
            y[0][i, 1] = row.t_y
            y[0][i, 2] = row.t_w
            y[0][i, 3] = row.t_h
            y[1][i, 0] = row.fg

        with open('./x.pkl', 'wb') as f:
            pickle.dump(X, f)

        with open('./y.pkl', 'wb') as f:
            pickle.dump(y, f)

    from collections import Counter

    train_set = 2000  # int(data.shape[0] * 0.9)

    print('Train:', Counter(y[1][:train_set].flatten()))
    _true = y[1][train_set:]
    _true[_true >= .5] = 1
    _true[_true < .5] = 0
    print('Test:', Counter(list(_true.flatten())))

    # history = model.fit(x=X[:train_set], y=[y[0][:train_set], y[1][:train_set]],
    history = model.fit(x=X[:train_set], y=[y[1][:train_set]],
                        epochs=3, shuffle=True, use_multiprocessing=True,
                        workers=5, verbose=1, validation_split=0.1, batch_size=100)

    """
    # result = model.predict(X[train_set:])
    print('-------------------------')
    print('PRED')
    test_set = 200
    """
    result = model.predict(X[train_set:])
    with open('./result.pkl', 'wb') as f:
        pickle.dump(result, f)
    result[result >= .5] = 1
    result[result < .5] = 0

    from sklearn.metrics import classification_report
    print(Counter(list(result.flatten())))
    print(classification_report(_true.flatten(), result.astype('int32').flatten()))
