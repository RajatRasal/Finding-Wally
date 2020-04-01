import os
from collections import Counter

import cloudpickle as pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score as f1_score_sk
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD

from model import f1_score, build_model


def k_fold_cv(X, y, model_builder, optimizer, folds=5, random_state=42, cls_threshold=0.6, 
              metric=f1_score_sk, keras_metrics=[f1_score], epochs=20, batch_size=32):
    # datagen = ImageDataGenerator(validation_split=0.1)

    kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    cvscores = []
    for train, test in kfold.split(X, y):
        # Compile the model
        model = model_builder()
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=keras_metrics)
        # Preprocess Input
        x_train = X[train]
        y_train = y[train]
        # datagen.fit(x_train)
        x_test = X[test]
        y_test = y[test]
        # Fit the model
        class_weight = Counter(y_train.flatten())
        zeros = class_weight[1] / y_train.shape[0]
        ones = class_weight[0] / y_train.shape[0]
        history = model.fit(x_train, y_train, batch_size=batch_size, shuffle=True,
                            steps_per_epoch=y_train.shape[0] / batch_size,
                            class_weight={0: zeros, 1: ones}, epochs=epochs,
                            use_multiprocessing=True, workers=-1, verbose=1) 
        # validation_split=0.1, callbacks=[es])
        # Evaluate the model
        # scores = model.evaluate(datagen.flow(x_train, y_train, batch_size=y_train.shape[0]), verbose=0)
        pred = model.predict(x_test)  # y[test], verbose=0)
        # TODO: Store all the images which are predicted 1s
        # TODO: Store all the 1s which are predicted 0s - FN
        pred[pred >= cls_threshold] = 1
        pred[pred < cls_threshold] = 0
        print(confusion_matrix(y[test].flatten(), pred.flatten()))
        score = f1_score_sk(y[test].flatten(), pred.flatten())
        print("%s: %.2f%%" % ('f1 score', score * 100))
        cvscores.append(score * 100)

    return cvscores

def train_model(X, y, model, optimizer, random_state=42,
                keras_metrics=[f1_score], epochs=100, batch_size=32):
    class_weight = Counter(y_train.flatten())
    print(class_weight)
    print(class_weight[0])
    print(class_weight[1])
    zeros = class_weight[1] / y_train.shape[0]
    ones = class_weight[0] / y_train.shape[0]

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=keras_metrics)
    history = model.fit(X, y, batch_size=batch_size, shuffle=True,
                        steps_per_epoch=y_train.shape[0] / batch_size,
                        class_weight={0: zeros, 1: ones}, epochs=epochs,
                        use_multiprocessing=True, workers=-1, verbose=1)

    return model

def predict(X, model, threshold=0.6):
    pred = model.predict(X)
    pred[pred >= threshold] = 1
    pred[pred < threshold] = 0
    return pred


if __name__ == '__main__':
    data = pd.read_csv('./data/data.csv')
    print(f'SHAPE: {data.shape} ------------------------')

    X = np.zeros((data.shape[0], 224, 224, 3))
    y = [np.zeros((data.shape[0], 4)), np.zeros((data.shape[0], 1))]

    """
    if os.path.lexists('./x.pkl'):
        print('Loading Data')
        with open('./x.pkl', 'rb') as f:
            X = pickle.load(f)
        with open('./y.pkl', 'rb') as f:
            y = pickle.load(f)
    else:
        raise FileNotFoundError('Dataset not created yet')
    """

    print('Training Model')
    # cv_scores = k_fold_cv(X, y[1], build_model, Adam(lr=0.00001))
    # print(cv_scores)
    # print("%.2f%% (+/- %.2f%%)" % (np.mean(cv_scores), np.std(cv_scores)))

    train_set = int(0.8 * data.shape[0])

    with open('./x_train.pkl', 'rb') as f:
        x_train = pickle.load(f)

    with open('./x_test.pkl', 'rb') as f:
        x_test = pickle.load(f)

    with open('./y_train.pkl', 'rb') as f:
        y_train = pickle.load(f)

    with open('./y_test.pkl', 'rb') as f:
        y_test = pickle.load(f)

    y_train = y_train[1]
    y_test = y_test[1]

    print(y_test[:10])
    print('Y train counter:', Counter(list(y_train.flatten())))
    print('Y test counter:', Counter(list(y_test.flatten())))
    model = build_model()
    model = train_model(x_train, y_train, model, Adam(lr=0.00001), epochs=200, batch_size=200)
    result_in_sample = predict(x_train, model, threshold=0.6)
    result_out_sample = predict(x_test, model, threshold=0.6)

    print(f1_score_sk(y_train.astype('int32').flatten(), result_in_sample.astype('int32').flatten()))
    print(classification_report(y_train.astype('int32').flatten(), result_in_sample.astype('int32').flatten()))
    print(confusion_matrix(y_train.astype('int32').flatten(), result_in_sample.astype('int32').flatten()))

    print(f1_score_sk(y_test.astype('int32').flatten(), result_out_sample.astype('int32').flatten()))
    print(classification_report(y_test.astype('int32').flatten(), result_out_sample.astype('int32').flatten()))
    print(confusion_matrix(y_test.astype('int32').flatten(), result_out_sample.astype('int32').flatten()))

    with open('./result.pkl', 'wb') as f:
        pickle.dump(result_out_sample, f)

    """
    print('Train:', Counter(y[1][:train_set].flatten()))
    _true = y[1][train_set:]
    print('Test:', Counter(list(_true.flatten())))

    print(data.shape)
    print(data.fg.sum())
    print(data.loc[data.actual.isin([19, 31, 49, 20])].shape)
    print(data.loc[data.actual.isin([19, 31, 49, 20])].fg.sum())
    """

    # history = model.fit(x=X[:train_set], y=[y[0][:train_set], y[1][:train_set]],
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
