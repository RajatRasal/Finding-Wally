from keras.layers import Dense
from keras import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16


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

# Combine backbone and my outputs to form the NN pipeline.
model = Model(inputs=vgg.input, outputs=[reg, cls])
print(model.summary())

for layer in model.layers:
    print(layer.name, layer.trainable)

def multitask_reg_cls_loss(y_true, y_pred):
    print(y_true, y_pred)
    return 1

optimizer = Adam(lr=0.0001)
# model.compile(loss=multitask_reg_cls_loss, optimizer=optimizer, metrics=["accuracy"])
