import cv2 as cv
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
import pandas as pd

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
    print(y_true, y_pred)
    return 1

optimizer = Adam(lr=0.0001)
# model.compile(loss=multitask_reg_cls_loss, optimizer=optimizer, metrics=["accuracy"])

if __name__ == '__main__':
    from selective_search import BBox

    new_width = 1000

    data = pd.read_csv('./data/data.csv', index_col=0)
    # data.astype('int32')
    sample = data.head(1)

    # print(sample)

    full_img = cv.imread(f'./data/original-images/12.jpg')
    full_img_scaled = scale_image_in_aspect_ratio(full_img, 1000)
    # print(sample.y, sample.y+sample.h, sample.x, sample.x+sample.w)
    proposal = full_img_scaled[int(sample.y):int(sample.y+sample.h),
                               int(sample.x):int(sample.x+sample.w)]
    proposal = cv.resize(proposal, (224, 224))
    x = model.predict(proposal.reshape((1, *proposal.shape)))
    print(x)
    # print(sample.dtypes)

    # print(data.head())
    # model.predict(
