import cloudpickle as pickle
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as kb
from tensorflow.keras import Model, losses
from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications import (VGG16, ResNet50, ResNet152,
    InceptionResNetV2, DenseNet121
)
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l1, l2


###############################################################################
# Preprocessing
###############################################################################
"""
def preprocess_data():
    # TODO: Abstract out preprocessing capabilities from selective_search.py
    pass

def train_test_split(x_train, y_train, x_test, y_test, processors=2,
    batch_size_per_processor=64, cache=True, seed=42
):
    # TODO: Replace preprocess_data with this function. Replace body of this function
    #  with preprocess data
    # should call 
    pass
"""

def load_csv_dataset(csv_file_path, test_images, reader='pd',
    image_col='image_no',
):
    if reader == 'pd':
        # Keep index column 
        df = pd.read_csv(csv_file_path, index_col=0)
        train, test = __split_df_data_by_images(df, test_images, image_col)
    elif reader == 'tf':
        COL_TYPES = [tf.string, tf.int32, tf.float32, tf.float32, \
            tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, \
            tf.float32, tf.int32
        ]
        tf_data = tf.data.experimental.CsvDataset(
            filenames=csv_file_path,
            record_defaults=COL_TYPES,
            select_cols=range(1, len(COL_TYPES)+1),  # Drop index column
            header=True
        )
        _test_images = tf.constant(test_images)
        train, test = __split_tf_data_by_images(tf_data, _test_images, image_col)
    else:
        assert NotImplementedError, 'This reader does not exist yet'
    return train, test

def __split_df_data_by_images(data, test_images, images_col='image_no'):
    test_mask = data[images_col].isin(test_images)
    test_data = data[test_mask]
    train_data = data[~test_mask]

    assert train_data.shape[0] + test_data.shape[0] == data.shape[0]
    assert train_data.shape[0] > test_data.shape[0] * 3
    print(train_data.shape[0], test_data.shape[0], data.shape[0])

    return train_data, test_data

def __split_tf_data_by_images(data, test_images, image_col='image_no'):
    test = data.filter(
        lambda *row: tf.reduce_any(tf.equal(row[1], test_images))
    )
    train = data.filter(
        lambda *row: tf.reduce_all(tf.not_equal(row[1], test_images))
    )
    return train, test

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
    parallel_calls = tf.data.experimental.AUTOTUNE
    SHUFFLE_BUFFER = batch_size_per_processor

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
        .map(vgg_preprocess, num_parallel_calls=parallel_calls) \
        .batch(batch_size, drop_remainder=True) \
        .cache() \
        .shuffle(SHUFFLE_BUFFER)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)) \
        .map(vgg_preprocess, num_parallel_calls=parallel_calls) \
        .batch(batch_size, drop_remainder=True) \
        .cache() \
        .shuffle(SHUFFLE_BUFFER)

    return train_dataset, test_dataset

def preprocess_data2(train):
    BATCH_SIZE = 128
    BUFFER_SIZE = BATCH_SIZE * 5
    SEED = 42
    TP_BATCH_FACTOR = 0.25
    TP_BATCH_SIZE = int(BATCH_SIZE * TP_BATCH_FACTOR)

    LABEL_COL = -1
    WALLY = 1 
    BACKGROUND = 0

    tf.random.set_seed(SEED)

    train = train.shuffle(BUFFER_SIZE)

    # In each batch there should be 25% 1s and 75% 0s.
    train_positive = train.filter(
        lambda *row: tf.math.equal(row[LABEL_COL], WALLY)
    )
    train_negative = train.filter(
        lambda *row: tf.math.equal(row[LABEL_COL], BACKGROUND)
    )
    # There are far fewer positive (Wally) images, so we can cycle
    #  through all of them twice.
    train_positive_batch = train_positive.batch(TP_BATCH_SIZE)  # .repeat(2)
    train_negative_batch = train_negative.batch(BATCH_SIZE - TP_BATCH_SIZE)

    # Get a single batch from train_positive and train_negative and
    #   concatenated into a single batch.
    batch_zip = (train_positive_batch, train_negative_batch)

    @tf.function
    def merge_csv_batches(batch_1, batch_2):
        cols = []
        for col_1, col_2 in zip(batch_1, batch_2):
            cols.append(tf.concat([col_1, col_2], 0))
        return cols

    @tf.function
    def load_region_proposal(image_data):
        image_file, x1, y1, w, h = image_data
        image_string = tf.io.read_file(image_file)
        image = tf.image.decode_jpeg(image_string)
        image = tf.image.crop_to_bounding_box(image, y1, x1, h, w)
        image = tf.image.resize(image, [224, 224])
        return image

    @tf.function
    def load_images(*batch):
        image_files = batch[0]
        x1s = tf.cast(batch[2], tf.int32)
        y1s = tf.cast(batch[3], tf.int32)
        ws = tf.cast(batch[4], tf.int32)
        hs = tf.cast(batch[5], tf.int32)
        images = tf.map_fn(
            load_region_proposal,
            (image_files, x1s, y1s, ws, hs),
            dtype=tf.float32
        )
        return (images, *batch[1:])

    @tf.function
    def preprocess_batch(*batch):
        return (vgg16.preprocess_input(batch[0]), *batch[1:])

    @tf.function
    def xy_split(image, no, x, y, w, h, x_t, y_t, w_t, h_t, fg):
        # find center coordinates of bounding boxes
        rp_center_x, rp_center_y = (x + w / 2, y + h / 2)
        gt_center_x, gt_center_y = (x_t + w_t / 2, y_t + h_t / 2)

        # calculate offset that we want to predict
        offset_x = (gt_center_x - rp_center_x) / w
        offset_y = (gt_center_y - rp_center_y) / h 
        offset_w = tf.math.log(w_t / w)
        offset_h = tf.math.log(h_t / h)

        X = image
        y = ((offset_x, offset_y, offset_w, offset_h), fg)
        return X, y

    # bboxes.append([full_img_path, no, x, y, w, h, x_t, y_t, w_t, h_t, fg])
    train = tf.data.experimental.CsvDataset.zip(batch_zip) \
        .map(merge_csv_batches, tf.data.experimental.AUTOTUNE) \
        .map(load_images, tf.data.experimental.AUTOTUNE) \
        .cache() \
        .map(preprocess_batch, tf.data.experimental.AUTOTUNE) \
        .map(xy_split, tf.data.experimental.AUTOTUNE) \
        .prefetch(tf.data.experimental.AUTOTUNE)

    return train

def load_and_preprocess_csv_data(csv_file_path):
    pass

###############################################################################
# Backbone Wrappers
###############################################################################

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

###############################################################################
# Model Definition
###############################################################################

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
    weighted_bce_loss = -(0.3 * cls_true * kb.log(cls_true) + \
        0.7 * (1 - cls_true) * kb.log(1 - cls_true)
    )
    return weighted_bce_loss

###############################################################################
# Training Loop Metrics
###############################################################################

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

###############################################################################
# Build and Compile
###############################################################################

def build_and_compile_model():
    model = build_model()
    multitask_loss = [rcnn_reg_loss, rcnn_cls_loss]
    metrics = [[rcnn_reg_mse], [rcnn_cls_f1_score]]
    model.compile(loss=multitask_loss,
        loss_weights=[0.5, 1],
        metrics=metrics,
        # optimizer=Adam(lr=0.00001),
        optimizer=SGD(lr=0.00001),
    )
    return model

def build_and_compile_distributed_model(strategy, batch_size):
    with strategy.scope():
        return build_and_compile_model()

###############################################################################
# Inference
###############################################################################

def predict(X, model, threshold=0.6):
    pred = model.predict(X)
    pred[pred >= threshold] = 1
    pred[pred < threshold] = 0
    return pred

def save_model(model, saved_model_path):
    model.save(saved_model_path)

def load_model(saved_model_path):
    custom_objects = {
        'rcnn_reg_loss': rcnn_reg_loss,
        'rcnn_cls_loss': rcnn_cls_loss,
        'rcnn_reg_mse': rcnn_reg_mse,
        'rcnn_cls_f1_score': rcnn_cls_f1_score
    }
    # TODO: Change this to load_weights
    model = tf.keras.models.load_model(saved_model_path,
        custom_objects=custom_objects,
        compile=True,
    )
    return model


def preprocess_dataset(dataset):
    BATCH_SIZE = 128
    BUFFER_SIZE = BATCH_SIZE * 5
    SEED = 42
    TP_BATCH_FACTOR = 0.25
    TP_BATCH_SIZE = int(BATCH_SIZE * TP_BATCH_FACTOR)

    LABEL_COL = -1
    WALLY = 1 
    BACKGROUND = 0

    tf.random.set_seed(SEED)

    dataset = dataset.shuffle(BUFFER_SIZE)

    # In each batch there should be 25% 1s and 75% 0s.
    dataset_positive = dataset.filter(
        lambda *row: tf.math.equal(row[LABEL_COL], WALLY)
    )
    dataset_negative = dataset.filter(
        lambda *row: tf.math.equal(row[LABEL_COL], BACKGROUND)
    )
    # There are far fewer positive (Wally) images, so we can cycle
    #  through all of them twice.
    dataset_positive_batch = dataset_positive.batch(TP_BATCH_SIZE)  # .repeat(2)
    dataset_negative_batch = dataset_negative.batch(BATCH_SIZE - TP_BATCH_SIZE)

    # Get a single batch from dataset_positive and dataset_negative and
    #   concatenated into a single batch.
    batch_zip = (dataset_positive_batch, dataset_negative_batch)

    @tf.function
    def merge_csv_batches(batch_1, batch_2):
        cols = []
        for col_1, col_2 in zip(batch_1, batch_2):
            cols.append(tf.concat([col_1, col_2], 0))
        return cols

    @tf.function
    def load_region_proposal(image_data):
        image_file, x1, y1, w, h = image_data
        image_string = tf.io.read_file(image_file)
        image = tf.image.decode_jpeg(image_string)
        image = tf.image.crop_to_bounding_box(image, y1, x1, h, w)
        image = tf.image.resize(image, [224, 224])
        return image

    @tf.function
    def load_images(*batch):
        image_files = batch[0]
        x1s = tf.cast(batch[2], tf.int32)
        y1s = tf.cast(batch[3], tf.int32)
        ws = tf.cast(batch[4], tf.int32)
        hs = tf.cast(batch[5], tf.int32)
        images = tf.map_fn(
            load_region_proposal,
            (image_files, x1s, y1s, ws, hs),
            dtype=tf.float32
        )
        return (images, *batch[1:])

    @tf.function
    def preprocess_batch(*batch):
        return (vgg16.preprocess_input(batch[0]), *batch[1:])

    @tf.function
    def xy_split(image, no, x, y, w, h, x_t, y_t, w_t, h_t, fg):
        # find center coordinates of bounding boxes
        rp_center_x, rp_center_y = (x + w / 2, y + h / 2)
        gt_center_x, gt_center_y = (x_t + w_t / 2, y_t + h_t / 2)

        # calculate offset that we want to predict
        offset_x = (gt_center_x - rp_center_x) / w
        offset_y = (gt_center_y - rp_center_y) / h 
        offset_w = tf.math.log(w_t / w)
        offset_h = tf.math.log(h_t / h)

        X = image
        y = (offset_x, offset_y, offset_w, offset_h, fg)
        return X, y

    @tf.function
    def col_to_row(image, labels):
        *offsets, fg = labels
        fg = tf.cast(fg, tf.float32)
        labels = tf.convert_to_tensor((*offsets, fg))
        labels = tf.transpose(labels)
        return image, labels

    dataset = tf.data.experimental.CsvDataset.zip(batch_zip) \
        .map(merge_csv_batches, tf.data.experimental.AUTOTUNE) \
        .map(load_images, tf.data.experimental.AUTOTUNE) \
        .cache() \
        .map(preprocess_batch, tf.data.experimental.AUTOTUNE) \
        .map(xy_split, tf.data.experimental.AUTOTUNE) \
        .map(col_to_row, tf.data.experimental.AUTOTUNE) \
        .prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


if __name__ == '__main__':
    CSV_FILE_PATH = './data/data.csv'
    TEST_IMAGES = [19, 31, 49, 20, 56, 21]

    train, test = load_csv_dataset(CSV_FILE_PATH, TEST_IMAGES, reader='tf')

    # t1 = len(list(test.as_numpy_iterator()))
    # print(t1)
    # t2 = len(list(train.as_numpy_iterator()))
    # print(t2)
    # assert t1 == 160312 
    # assert t2 == 1391841
    train = preprocess_dataset(train)

    for i, (X, y) in enumerate(train):
        print(f'----------------- {i} ------------------')
        print(y)
