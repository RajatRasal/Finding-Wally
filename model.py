import cloudpickle as pickle
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as kb
from tensorflow.keras import Model, losses
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l1
from tensorflow.keras.applications.vgg16 import VGG16


def preprocess_data(x_train, y_train, x_test, y_test, processors=2, batch_size_per_processor=64):

    def scale(image, label):
        image = tf.cast(image, tf.float16)
        label = tf.cast(label, tf.float16)
        # image /= 255
        return image, label

    def normalize(image, label):
        # image = tf.image.per_image_standardization(image)
        return image, label

    batch_size = processors * batch_size_per_processor 

    tf.random.set_seed(42)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
        .map(scale) \
        .map(normalize) \
        .shuffle(x_train.shape[0]) \
        .batch(batch_size, drop_remainder=True) \
        .cache()
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)) \
        .map(scale) \
        .map(normalize) \
        .shuffle(x_test.shape[0]) \
        .batch(batch_size, drop_remainder=True) \
        .cache()

    return train_dataset, test_dataset

def build_model():
    vgg = VGG16(weights='imagenet', include_top=True)
    
    for layers in (vgg.layers): layers.trainable = False
    
    # VGG Backbone - output from convolutional layers
    backbone = vgg.layers[-4].output
    # Attach my own FC layers on top of VGG backbone
    # 300
    fc1 = Dense(300, activation='relu', name='my_fc1', 
                activity_regularizer=l1(0.001))(backbone)
    do1 = Dropout(0.5, seed=41)(fc1)
    fc2 = Dense(100, activation='relu', name='my_fc2',
                activity_regularizer=l1(0.001))(do1)
    do1 = Dropout(0.5, seed=41)(fc2)
    
    # Loss function for bounding box regression
    # reg = Dense(4, activation='linear', name='bbox_reg')(fc)
    # Loss function for foreground/background classification
    cls = Dense(1, activation='tanh', name='fg_cls')(do1)
    
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
