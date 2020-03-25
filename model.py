from keras.layers import Dense
from keras import Model
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16


vgg = VGG16(weights='imagenet', include_top=True)

for layers in (vgg.layers)[:15]:
    layers.trainable = False

X = vgg.layers[-2].output
predictions = Dense(2, activation="softmax")(X)
