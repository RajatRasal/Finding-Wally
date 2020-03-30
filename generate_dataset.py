import argparse

import pandas as pd
import numpy as np
import cloudpickle as pickle
import cv2 as cv

from selective_search import scale_image_in_aspect_ratio


def produce_dataset(data, x_name='./x.pkl', y_name='./y.pkl'):
    X = np.zeros((data.shape[0], 224, 224, 3))
    y = [np.zeros((data.shape[0], 4)), np.zeros((data.shape[0], 1))]
    
    for i, row in data.iterrows():
        full_img = cv.imread(f'./data/original-images/{int(row.actual)}.jpg')
        full_img_scaled = scale_image_in_aspect_ratio(full_img, 1000)
        proposal = full_img_scaled[int(row.y):int(row.y+row.h),
                                   int(row.x):int(row.x+row.w)]
        proposal = cv.resize(proposal, (224, 224))
        X[i] = proposal
    
        y[0][i, 0] = row.t_x
        y[0][i, 1] = row.t_y
        y[0][i, 2] = row.t_w
        y[0][i, 3] = row.t_h
        y[1][i, 0] = row.fg
    
    with open(x_name, 'wb') as f:
        pickle.dump(X, f)
    
    with open(y_name, 'wb') as f:
        pickle.dump(y, f)


description = 'Decide on how to split image data produced by selective_search.py'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('file', help='File with CSV data from selective_search.py')
parser.add_argument('--test-images', '-t', help='Test image numbers', type=int, nargs='+')
args = parser.parse_args()

data = pd.read_csv(args.file) 
data = data.astype({'actual': 'int32'})

unique_nums = np.unique(data.actual)

if args.test_images and all(map(lambda x: x in unique_nums, args.test_images)):
    print('Train-Test Split')
    train = data.loc[~data.actual.isin(args.test_images)].reset_index()
    test = data.loc[data.actual.isin(args.test_images)].reset_index()
    print('Writing Training Data')
    produce_dataset(train, './x_train.pkl', './y_train.pkl')
    print('Writing Test Data')
    produce_dataset(test, './x_test.pkl', './y_test.pkl')
else:
    print('Full Dataset')
    produce_dataset(data)
