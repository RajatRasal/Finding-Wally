import os

import cv2 as cv
import cloudpickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as kb
from tensorflow.keras import Model, losses
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l1
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score as f1_score_sk

from selective_search import scale_image_in_aspect_ratio


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

def build_model():
    vgg = VGG16(weights='imagenet', include_top=True)
    
    for layers in (vgg.layers): layers.trainable = False
    
    # VGG Backbone - output from convolutional layers
    backbone = vgg.layers[-4].output
    # Attach my own FC layers on top of VGG backbone
    # 300
    fc = Dense(300, activation='relu', name='my_fc1', 
               activity_regularizer=l1(0.001))(backbone)
    fc = Dense(100, activation='relu', name='my_fc2',
               activity_regularizer=l1(0.001))(fc)
    
    # Loss function for bounding box regression
    # reg = Dense(4, activation='linear', name='bbox_reg')(fc)
    # Loss function for foreground/background classification
    cls = Dense(1, activation='tanh', name='fg_cls')(fc)
    
    # Combine backbone and my outputs to form the NN pipeline
    # model = Model(inputs=vgg.input, outputs=[cls])
    model = Model(inputs=vgg.input, outputs=cls)
    # print(model.summary())
    return model

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

def recall(y_true, y_pred):
    true_positives = kb.sum(kb.round(kb.clip(y_true * y_pred, 0, 1)))
    possible_positives = kb.sum(kb.round(kb.clip(y_true, 0, 1)))
    _recall = true_positives / (possible_positives + kb.epsilon())
    return _recall

def precision(y_true, y_pred):
    true_positives = kb.sum(kb.round(kb.clip(y_true * y_pred, 0, 1)))
    predicted_positives = kb.sum(kb.round(kb.clip(y_pred, 0, 1)))
    _precision = true_positives / (predicted_positives + kb.epsilon())
    return _precision

def f1_score(y_true, y_pred):
    _precision = precision(y_true, y_pred)
    _recall = recall(y_true, y_pred)
    return 2 * ((_precision * _recall) / (_precision + _recall + kb.epsilon()))

"""
model.compile(loss=multitask_reg_cls_loss, optimizer=optimizer,
              metrics=["accuracy", recall_m])
"""


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

    # train_set = int(0.8 * data.shape[0])

    # print('Train:', Counter(y[1][:train_set].flatten()))
    # _true = y[1][train_set:]
    # print('Test:', Counter(list(_true.flatten())))

    # history = model.fit(x=X[:train_set], y=[y[0][:train_set], y[1][:train_set]],
    es = EarlyStopping(monitor='val_f1_score', min_delta=0.1, patience=10, mode='max',
                       restore_best_weights=True)
    optimizer = Adam(lr=0.00001)

    datagen = ImageDataGenerator()

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cvscores = []
    for train, test in kfold.split(X, y[1]):
        # Compile the model
        model = build_model()
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[f1_score])
        # Preprocess Input
        x_train = X[train]
        y_train = y[1][train]
        datagen.fit(x_train)
        x_test = X[test]
        y_test = y[1][test]
        # Fit the model
        class_weight = Counter(y_train.flatten())
        zeros = class_weight[1] / y_train.shape[0]
        ones = class_weight[0] / y_train.shape[0]
        history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=50),
                steps_per_epoch=y_train.shape[0] / 50,
                            class_weight={0: zeros, 1: ones}, epochs=5, shuffle=True, 
                            use_multiprocessing=True, workers=-1, verbose=1) 
        # validation_split=0.1, callbacks=[es])
        # Evaluate the model
        # scores = model.evaluate(datagen.flow(x_train, y_train, batch_size=y_train.shape[0]), verbose=0)
        pred = model.predict_generator(datagen.flow(x_test))  # y[1][test], verbose=0)
        cls_threshold = 0.5
        pred[pred >= cls_threshold] = 1
        pred[pred < cls_threshold] = 0
        print(confusion_matrix(y[1][test].flatten(), pred.flatten()))
        score = f1_score_sk(y[1][test].flatten(), pred.flatten())
        print("%s: %.2f%%" % ('f1 score', score * 100))
        cvscores.append(score * 100)
        print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

    print(cvscores)
    # result = model.predict(X[train_set:])
    """
    print('-------------------------')
    print('PRED')
    test_set = 200
    result = model.predict(X[train_set:])
    with open('./result.pkl', 'wb') as f:
        pickle.dump(result, f)
    cls_threshold = 0.5
    result[result >= cls_threshold] = 1
    result[result < cls_threshold] = 0

    from sklearn.metrics import classification_report, confusion_matrix
    print(Counter(list(result.flatten())))
    print(classification_report(_true.flatten(), result.astype('int32').flatten()))
    print(confusion_matrix(_true.flatten(), result.astype('int32').flatten()))
    """
