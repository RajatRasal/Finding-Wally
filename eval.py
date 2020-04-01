import os
from collections import Counter

import cloudpickle as pickle
import numpy as np
import pandas as pd
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, \
        precision_recall_curve, roc_curve
from sklearn.metrics import f1_score as f1_score_sk

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


data = pd.read_csv('./data/data.csv')
print(f'SHAPE: {data.shape} ------------------------')

if os.path.lexists('./x.pkl'):
    print('Loading Data')
    with open('./x.pkl', 'rb') as f:
        X = pickle.load(f)

if os.path.lexists('./y.pkl'):
    with open('./y.pkl', 'rb') as f:
        y = pickle.load(f)
else:
    raise FileNotFoundError('Dataset not created yet')

cv_scores = k_fold_cv(X, y[1], build_model, Adam(lr=0.00001), epochs=10)
print(cv_scores)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cv_scores), np.std(cv_scores)))

"""
with open('./y.pkl', 'rb') as f:
    y = pickle.load(f)

_true = y[1][train_set:]

print('Test:', Counter(list(_true.flatten())))

with open('./result.pkl', 'rb') as f:
    result = pickle.load(f)

print(result.mean())
cls_threshold = 0.1
result[result >= cls_threshold] = 1
result[result < cls_threshold] = 0
print(Counter(list(result.flatten())))

print(classification_report(_true.flatten(), result.astype('int32').flatten()))
precision, recall, thresholds = precision_recall_curve(_true.flatten(),
                                                       result.astype('int32').flatten())
fscore = (2 * precision * recall) / (precision + recall)
fpr, tpr, thresholds2 = roc_curve(_true.flatten(), result.astype('int32').flatten())
# calculate the g-mean for each threshold
gmeans = np.sqrt(tpr * (1-fpr))
ix = np.argmax(fscore)
print('Best Threshold=%f, G-Mean=%.3f' % (thresholds2[ix], gmeans[ix]))
print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))
"""
