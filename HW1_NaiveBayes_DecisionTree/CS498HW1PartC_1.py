#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 01:19:11 2019

@author: xiaosundi
"""

from mnist import MNIST
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

mndata = MNIST('')
mndata.gz = True
training_images, training_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()

test_images_array = np.asarray(test_images)
test_labels_array = np.asarray(test_labels)

training_images_array = np.asarray(training_images)
training_labels_array = np.asarray(training_labels)
training_labels_array_new = training_labels_array.reshape((60000,1))

training_data_arr = np.concatenate((training_images_array, training_labels_array_new), axis=1)


accuracy = 0
# split train and validation
np.random.shuffle(training_data_arr)


# reshaping and stretching, bounding
training_data_arr = training_data_arr.astype(np.uint8)
X_train_val = training_data_arr[:,:784]
X_train_val = X_train_val.reshape((60000,28,28))
X_train_val_bound = []

for i in range(len(X_train_val)):
    row_sum = X_train_val[i].sum(axis=1)
    row_index = []
    for each_zeros_row in range(len(row_sum)):
        if(row_sum[each_zeros_row] != 0):
            row_index.append(each_zeros_row)
    row_top = row_index[0]
    row_bottom = row_index[-1]
    col_sum = X_train_val[i].sum(axis=0)
    col_index = []
    for each_zeros_col in range(len(col_sum)):
        if(col_sum[each_zeros_col] != 0):
            col_index.append(each_zeros_col)
    col_top = col_index[0]
    col_bottom = col_index[-1]
    X_train_val_bound.append(X_train_val[i,row_top:row_bottom+1,col_top:col_bottom+1])
    

X_train_val_bound = np.asarray(X_train_val_bound)
X_train_val_bound = X_train_val_bound.astype(np.ndarray)
X_train_bound = []
for each_image in range(len(X_train_val_bound)):
    X_train_val_bound[each_image] = cv2.resize(X_train_val_bound[each_image],(20,20))
    image_shape = X_train_val_bound[each_image].shape
    X_train_val_bound[each_image] = X_train_val_bound[each_image].reshape((image_shape[0]*image_shape[1]))
    
    
X_train = X_train_val_bound[:48000][:]
X_validation = X_train_val_bound[48000:][:]
Y_train = training_data_arr[:48000, -1]
Y_validation = training_data_arr[48000:, -1]




clf = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=0)
clf.fit(X_train,Y_train)
prediction = clf.predict(X_validation)
result = (prediction == Y_validation).sum()
train_accuracy = result/len(Y_validation)


print("train_accuracy = ", train_accuracy )



accuracy = 0

# reshaping and stretching, bounding
test_images = np.asarray(test_images)
test_images = test_images.astype(np.uint8)
test_images = test_images.reshape((10000,28,28))
training_images = np.asarray(training_images)
training_images = training_images.astype(np.uint8)
training_images = training_images.reshape((60000,28,28))
X_train = []
X_test = []

for i in range(len(training_images)):
    row_sum = training_images[i].sum(axis=1)
    row_index = []
    for each_zeros_row in range(len(row_sum)):
        if(row_sum[each_zeros_row] != 0):
            row_index.append(each_zeros_row)
    row_top = row_index[0]
    row_bottom = row_index[-1]
    col_sum = training_images[i].sum(axis=0)
    col_index = []
    for each_zeros_col in range(len(col_sum)):
        if(col_sum[each_zeros_col] != 0):
            col_index.append(each_zeros_col)
    col_top = col_index[0]
    col_bottom = col_index[-1]
    X_train.append(training_images[i,row_top:row_bottom+1,col_top:col_bottom+1])
    
for i in range(len(test_images)):
    row_sum = test_images[i].sum(axis=1)
    row_index = []
    for each_zeros_row in range(len(row_sum)):
        if(row_sum[each_zeros_row] != 0):
            row_index.append(each_zeros_row)
    row_top = row_index[0]
    row_bottom = row_index[-1]
    col_sum = test_images[i].sum(axis=0)
    col_index = []
    for each_zeros_col in range(len(col_sum)):
        if(col_sum[each_zeros_col] != 0):
            col_index.append(each_zeros_col)
    col_top = col_index[0]
    col_bottom = col_index[-1]
    X_test.append(test_images[i,row_top:row_bottom+1,col_top:col_bottom+1])
    
    

X_train = np.asarray(X_train)
X_train = X_train.astype(np.ndarray)
X_test = np.asarray(X_test)
X_test = X_test.astype(np.ndarray)

for each_image in range(len(X_train)):
    X_train[each_image] = cv2.resize(X_train[each_image],(20,20))
    image_shape = X_train[each_image].shape
    X_train[each_image] = X_train[each_image].reshape((image_shape[0]*image_shape[1]))
    
for each_image in range(len(X_test)):
    X_test[each_image] = cv2.resize(X_test[each_image],(20,20))
    image_shape = X_test[each_image].shape
    X_test[each_image] = X_test[each_image].reshape((image_shape[0]*image_shape[1]))
    
    
    
training_images = X_train[:60000][:]
test_images = X_test[:10000][:]


clf = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=0)
clf.fit(training_images,training_labels)
prediction = clf.predict(test_images)
test_accuracy = (prediction == test_labels).sum()/len(test_labels)


print("test_accuracy = ", test_accuracy )






