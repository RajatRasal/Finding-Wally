from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os
import json
import socket
import time

from tensorflow import keras
import resnet


worker = ["146.169.53.220:2223", "146.169.53.219:2222"]
ip = socket.gethostbyname(socket.gethostname())
index = list(map(lambda x: x.split(':')[0], worker)).index(ip)
# model = distributed_train_model(x_train/255, y_train, build_model, workers, index, Adam(lr=0.00001))

os.environ['TF_CONFIG'] = json.dumps({"cluster": {"worker": worker}, "task": {"index": index, "type": "worker"}})
# strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
strategy = tf.compat.v1.distribute.experimental.MultiWorkerMirroredStrategy()

NUM_GPUS = 2
BS_PER_GPU = 128 
NUM_EPOCHS = 2

HEIGHT = 32
WIDTH = 32
NUM_CHANNELS = 3
NUM_CLASSES = 10
NUM_TRAIN_SAMPLES = 30000

BASE_LEARNING_RATE = 0.1
LR_SCHEDULE = [(0.1, 30), (0.01, 45)]


def normalize(x, y):
    x = tf.image.per_image_standardization(x)
    return x, y


def augmentation(x, y):
    x = tf.image.resize_with_crop_or_pad(x, HEIGHT + 8, WIDTH + 8)
    x = tf.image.random_crop(x, [HEIGHT, WIDTH, NUM_CHANNELS])
    x = tf.image.random_flip_left_right(x)
    return x, y 


def schedule(epoch):
    initial_learning_rate = BASE_LEARNING_RATE * BS_PER_GPU / 128
    learning_rate = initial_learning_rate
    for mult, start_epoch in LR_SCHEDULE:
        if epoch >= start_epoch:
            learning_rate = initial_learning_rate * mult
        else:
            break
    tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
    return learning_rate


(x, y), (x_test, y_test) = keras.datasets.cifar10.load_data()

print('++++++++++++++++++ DATASET +++++++++++++++++++++')
print(x.shape)
print(BS_PER_GPU * NUM_GPUS)
print('Remainder:', x.shape[0] % (BS_PER_GPU * NUM_GPUS))
rem = x.shape[0] % (BS_PER_GPU * NUM_GPUS)
print('Elems after repeat:', (x.shape[0] - rem) * NUM_EPOCHS)
batch_in_1_epoch = x.shape[0] // (BS_PER_GPU * NUM_GPUS)
print('Batches in 1 epoch:', batch_in_1_epoch)
# print('Steps per epoch:', ((x.shape[0] - rem) * NUM_EPOCHS) / )

with strategy.scope():
    start = time.time()
    train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    
    tf.random.set_seed(22)
    train_dataset = train_dataset.map(augmentation) \
            .map(normalize) \
            .shuffle(x.shape[0]) \
            .batch(BS_PER_GPU * NUM_GPUS, drop_remainder=True) \
            .repeat(NUM_EPOCHS) \
            .cache() \
            .prefetch(5 * BS_PER_GPU * NUM_GPUS)
    test_dataset = test_dataset.map(normalize) \
            .batch(BS_PER_GPU * NUM_GPUS, drop_remainder=True)
    
    input_shape = (HEIGHT, WIDTH, NUM_CHANNELS)
    img_input = tf.keras.layers.Input(shape=input_shape)
    opt = keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
    
    # model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
    model = resnet.resnet56(img_input=img_input, classes=NUM_CLASSES)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy']) 
    model.fit(train_dataset, steps_per_epoch=x.shape[0] // (BS_PER_GPU * NUM_GPUS), epochs=NUM_EPOCHS)
    print('Time Taken:', time.time() - start)
