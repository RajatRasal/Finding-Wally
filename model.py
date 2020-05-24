import cloudpickle as pickle
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as kb
from tensorflow.keras import Model, losses
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.applications import (VGG16, VGG19, ResNet50, 
    ResNet152, ResNet101, InceptionResNetV2, Xception, DenseNet121,
    DenseNet169, DenseNet201
)
from tensorflow.keras.applications import vgg16  # import preprocess_input


def preprocess_data(x_train, y_train, x_test, y_test, processors=2,
    batch_size_per_processor=64, cache=True, seed=42
):
    tf.random.set_seed(seed)

    def convert_types(image, label):
        image = tf.cast(image, tf.float16)
        label = tf.cast(label, tf.float16)
        return image, label

    def random_flip(image, label):
        image = tf.image.random_flip_up_down(image, seed=seed)
        return image, label

    def vgg_preprocess(image, label):
        image = vgg16.preprocess_input(image)
        return image, label

    batch_size = processors * batch_size_per_processor 

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
        .map(vgg_preprocess) \
        .shuffle(x_train.shape[0]) \
        .batch(batch_size, drop_remainder=True)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)) \
        .map(vgg_preprocess) \
        .shuffle(x_test.shape[0]) \
        .batch(batch_size, drop_remainder=True)

    if cache:
        train_dataset = train_dataset.cache()
        test_dataset = test_dataset.cache()

    return train_dataset, test_dataset

def resnet152_backbone():
    base_model = ResNet152(weights='imagenet', include_top=False)

    for layers in base_model.layers:
        layers.trainable = False

    backbone = base_model.layers[-1].output
    backbone = GlobalAveragePooling2D()(backbone)
    return (base_model.input, backbone)

def resnet50_backbone():
    base_model = ResNet50(weights='imagenet')

    for layers in base_model.layers:
        layers.trainable = False

    backbone = base_model.layers[-2].output
    return (base_model.input, backbone)

def inception_resnet_backbone():
    base_model = InceptionResNetV2(weights='imagenet')

    for layers in base_model.layers:
        layers.trainable = False

    backbone = base_model.layers[-2].output
    return (base_model.input, backbone)

def densenet121_backbone():
    base_model = DenseNet121(weights='imagenet')

    for layers in base_model.layers:
        layers.trainable = False

    backbone = base_model.layers[-2].output
    return (base_model.input, backbone)

def vgg16_backbone():
    base_model = VGG16(weights='imagenet')

    for layers in base_model.layers:
        layers.trainable = False

    backbone = base_model.layers[-4].output
    return (base_model.input, backbone)

def build_model():
    # https://keras.io/api/applications/
    # VGG16 - f1-scroe: 0.68, loss: 0.3, Adam(lr=0.00001)
    # VGG19 - f1-scroe: 0.55, loss: 0.46
    # Resnet152 - f1-score 0.40, loss: 0.33
    # DenseNet121 - f1-score: 0.4, loss: 1.1

    # DenseNet121, DenseNet169, DenseNet201, ResNet50, ResNet152, ResNet101
    base_input, backbone = vgg16_backbone()

    # Combine backbone and my outputs to form the NN pipeline
    fc1 = Dense(300, activation='relu', name='additional_fc1',
        activity_regularizer=l1(0.001))(backbone)
    do1 = Dropout(0.5, seed=41)(fc1)
    fc2 = Dense(100, activation='relu', name='additional_fc2',
        activity_regularizer=l1(0.001))(do1)
    do1 = Dropout(0.5, seed=41)(fc2)
    
    # Loss function for foreground/background classification
    cls = Dense(1, activation='sigmoid', name='classifier')(do1)
    # Loss function for bounding box regression
    reg = Dense(4, activation='linear', name='regressor',
        activity_regularizer=l2(0.1))(do1)
    
    model = Model(inputs=base_input, outputs=[reg, cls])
    return model

def rcnn_reg_loss(y_true, y_pred):
    reg_pred = y_pred
    reg_true = y_true[:, :4]
    cls_true = y_true[:, 4:]
    # print(f'first row true: {cls_true[0]}')
    # print(f'Pred shape: {cls_pred.shape}, Actual shape: {cls_true.shape}')
    foreground_loss = tf.multiply(cls_true, tf.square(reg_true - reg_pred))
    sse_loss = tf.reduce_sum(foreground_loss)
    return sse_loss

def rcnn_cls_loss(y_true, y_pred):
    _cls_pred = y_pred
    _cls_true = y_true[:, 4:]
    # print(f'Pred shape: {_cls_pred.shape}, Actual shape: {_cls_true.shape}')
    cls_pred = kb.clip(_cls_pred, kb.epsilon(), 1 - kb.epsilon())
    cls_true = kb.clip(_cls_true, kb.epsilon(), 1 - kb.epsilon())
    # bce_loss = losses.binary_crossentropy(cls_true, cls_pred)
    weighted_bce_loss = -(0.2 * cls_true * kb.log(cls_true) + \
        0.8 * (1 - cls_true) * kb.log(1 - cls_true)
    )
    return weighted_bce_loss

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

def rcnn_cls_f1_score(y_true, y_pred):
    return f1_score(y_true[:, 4:], tf.math.round(y_pred))

def rcnn_reg_mse(y_true, y_pred):
    return losses.MSE(y_true[:, :4], y_pred)
