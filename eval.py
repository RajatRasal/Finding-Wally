from collections import Counter

import cloudpickle as pickle
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve, roc_curve

data = pd.read_csv('./data/data.csv')
train_set = int(0.8 * data.shape[0])

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
