import cv2 as cv
import cloudpickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import keras.backend as kb
from keras.layers import Dense
from keras import Model, losses
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16

from selective_search import scale_image_in_aspect_ratio


vgg = VGG16(weights='imagenet', include_top=True)

for layers in (vgg.layers):
    layers.trainable = False

# VGG Backbone - output from convolutional layers
backbone = vgg.layers[-4].output
# Attach my own FC layers on top of VGG backbone
fc = Dense(4096, activation='relu', name='my_fc1')(backbone)
fc = Dense(128, activation='relu', name='my_fc2')(fc)

# Loss function for bounding box regression
reg = Dense(4, activation='linear', name='bbox_reg')(fc)
# Loss function for foreground/background classification
cls = Dense(1, activation='sigmoid', name='fg_cls')(fc)

# Combine backbone and my outputs to form the NN pipeline
model = Model(inputs=vgg.input, outputs=[reg, cls])
# print(model.summary())

def multitask_reg_cls_loss(y_true, y_pred):
    reg_pred = y_pred[0]
    cls_pred = y_pred[1]
    reg_true = y_true[0]
    cls_true = y_true[1]

    loss = kb.mean(losses.binary_crossentropy(cls_true, cls_pred) + \
                   cls_true * losses.mse(reg_true, reg_pred))
    return loss

optimizer = Adam(lr=0.0001)
model.compile(loss=multitask_reg_cls_loss, optimizer=optimizer, metrics=["accuracy"])

if __name__ == '__main__':
    new_width = 1000

    data = pd.read_csv('./data/data.csv')
    # data = data.sample(frac=1).reset_index(drop=True)

    X = np.zeros((data.shape[0], 224, 224, 3))
    y = [np.zeros((data.shape[0], 4)), np.zeros((data.shape[0], 1))]

    for i, row in data.iterrows():
        if i % 100 == 0:
            print(i)
        full_img = cv.imread(f'./data/original-images/{int(row.actual)}.jpg')
        full_img_scaled = scale_image_in_aspect_ratio(full_img, 1000)
        proposal = full_img_scaled[int(row.y):int(row.y+row.h), int(row.x):int(row.x+row.w)]
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

    train_set = int(data.shape[0] * 0.9)

    history = model.fit(x=X[:train_set], y=[y[0][:train_set], y[1][:train_set]], epoch=1,
                        shuffle=True, use_multiprocessing=True, workers=7, verbose=1)

    result = model.predict(x[train_set:])
    with open('./result.pkl', 'wb') as f:
        pickle.dump(result, f)

    from sklearn.metrics import classification_report

    print(classification_report(y[1][train_set:], result[1]))
