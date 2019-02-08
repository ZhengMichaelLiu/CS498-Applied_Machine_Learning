#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 23:20:40 2019

@author: xiaosundi
"""

from mnist import MNIST
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split

mndata = MNIST('')
mndata.gz = True
training_images, training_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()
 


############################# Bernoulli Distribution, untouched#####################################
############train##########


accuracy = 0
# split train and validation
np.random.shuffle(training_data_arr)
training_data_arr = training_data_arr.astype(np.uint8)
X_train_val = training_data_arr[:,:784]
X_train = X_train_val[:48000][:]
X_validation = X_train_val[48000:][:]
Y_train = training_data_arr[:48000, -1]
Y_validation = training_data_arr[48000:, -1]


for i in range(len(X_train)):
    for j in range(len(X_train[0])):
        if X_train[i][j] > 128:
            X_train[i][j] = 1
        else: 
            X_train[i][j] = 0
            
for i in range(len(X_validation)):
    for j in range(len(X_validation[0])):
        if X_validation[i][j] > 128:
            X_validation[i][j] = 1
        else: 
            X_validation[i][j] = 0

# Calculate class probability
set_0 = []
set_1 = []
set_2 = []
set_3 = []
set_4 = []
set_5 = []
set_6 = []
set_7 = []
set_8 = []
set_9 = []

for i in range(len(Y_train)):
    if Y_train[i] == 0:
        set_0.append(X_train[i])
    if Y_train[i] == 1:
        set_1.append(X_train[i])
    if Y_train[i] == 2:
        set_2.append(X_train[i])
    if Y_train[i] == 3:
        set_3.append(X_train[i])
    if Y_train[i] == 4:
        set_4.append(X_train[i])
    if Y_train[i] == 5:
        set_5.append(X_train[i])
    if Y_train[i] == 6:
        set_6.append(X_train[i])
    if Y_train[i] == 7:
        set_7.append(X_train[i])
    if Y_train[i] == 8:
        set_8.append(X_train[i])
    if Y_train[i] == 9:
        set_9.append(X_train[i])
        

p0 = len(set_0) / len(Y_train)
p1 = len(set_1) / len(Y_train)
p2 = len(set_2) / len(Y_train)
p3 = len(set_3) / len(Y_train)
p4 = len(set_4) / len(Y_train)
p5 = len(set_5) / len(Y_train)
p6 = len(set_6) / len(Y_train)
p7 = len(set_7) / len(Y_train)
p8 = len(set_8) / len(Y_train)
p9 = len(set_9) / len(Y_train)

set_0 = np.asarray(set_0)
set_1 = np.asarray(set_1)
set_2 = np.asarray(set_2)
set_3 = np.asarray(set_3)
set_4 = np.asarray(set_4)
set_5 = np.asarray(set_5)
set_6 = np.asarray(set_6)
set_7 = np.asarray(set_7)
set_8 = np.asarray(set_8)
set_9 = np.asarray(set_9)


prob0 = (set_0.sum(axis=0)+10)/len(set_0)
prob1 = (set_1.sum(axis=0)+10)/len(set_1)
prob2 = (set_2.sum(axis=0)+10)/len(set_2)
prob3 = (set_3.sum(axis=0)+10)/len(set_3)
prob4 = (set_4.sum(axis=0)+10)/len(set_4)
prob5 = (set_5.sum(axis=0)+10)/len(set_5)
prob6 = (set_6.sum(axis=0)+10)/len(set_6)
prob7 = (set_7.sum(axis=0)+10)/len(set_7)
prob8 = (set_8.sum(axis=0)+10)/len(set_8)
prob9 = (set_9.sum(axis=0)+10)/len(set_9)

prob0 = prob0.reshape((784,1))
prob1 = prob1.reshape((784,1))
prob2 = prob2.reshape((784,1))
prob3 = prob3.reshape((784,1))
prob4 = prob4.reshape((784,1))
prob5 = prob5.reshape((784,1))
prob6 = prob6.reshape((784,1))
prob7 = prob7.reshape((784,1))
prob8 = prob8.reshape((784,1))
prob9 = prob9.reshape((784,1))


correct = 0
false = 0 
# Predict
for index in range(len(X_validation)):
    # print(index)
    probability = np.zeros(10)
    for i in range(784):
        if prob0[i] + prob1[i]+prob2[i]+prob3[i]+prob4[i]+prob5[i]+prob6[i]+prob7[i]+prob8[i]+prob9[i] != 0 :
            probability[0] += X_validation[index][i]*np.log(prob0[i]) + np.log(1-prob0[i])*(1-X_validation[index][i])
            probability[1] += X_validation[index][i]*np.log(prob1[i]) + np.log(1-prob1[i])*(1-X_validation[index][i])
            probability[2] += X_validation[index][i]*np.log(prob2[i]) + np.log(1-prob2[i])*(1-X_validation[index][i])
            probability[3] += X_validation[index][i]*np.log(prob3[i]) + np.log(1-prob3[i])*(1-X_validation[index][i])
            probability[4] += X_validation[index][i]*np.log(prob4[i]) + np.log(1-prob4[i])*(1-X_validation[index][i])
            probability[5] += X_validation[index][i]*np.log(prob5[i]) + np.log(1-prob5[i])*(1-X_validation[index][i])
            probability[6] += X_validation[index][i]*np.log(prob6[i]) + np.log(1-prob6[i])*(1-X_validation[index][i])
            probability[7] += X_validation[index][i]*np.log(prob7[i]) + np.log(1-prob7[i])*(1-X_validation[index][i])
            probability[8] += X_validation[index][i]*np.log(prob8[i]) + np.log(1-prob8[i])*(1-X_validation[index][i])
            probability[9] += X_validation[index][i]*np.log(prob9[i]) + np.log(1-prob9[i])*(1-X_validation[index][i])

    probability[0] += np.log(p0)
    probability[1] += np.log(p1)
    probability[2] += np.log(p2)
    probability[3] += np.log(p3)
    probability[4] += np.log(p4)
    probability[5] += np.log(p5)
    probability[6] += np.log(p6)
    probability[7] += np.log(p7)
    probability[8] += np.log(p8)
    probability[9] += np.log(p9)

    # print(probability)

    max_index = np.argmax(probability)

    if max_index == Y_validation[index]:
        correct += 1
    else:
        false += 1
    
accuracy += correct / (correct + false)

print("Train_acc: ", accuracy)


######################test#############################
accuracy = 0

test_images = np.asarray(test_images)
test_images = test_images.astype(np.uint8)

training_images = np.asarray(training_images)
training_images = training_images.astype(np.uint8)


for i in range(len(test_images)):
    for j in range(len(test_images[0])):
        if test_images[i][j] > 128:
            test_images[i][j] = 1
        else: 
            test_images[i][j] = 0
            
for i in range(len(training_images)):
    for j in range(len(training_images[0])):
        if training_images[i][j] > 128:
            training_images[i][j] = 1
        else: 
            training_images[i][j] = 0


# Calculate class probability
set_0 = []
set_1 = []
set_2 = []
set_3 = []
set_4 = []
set_5 = []
set_6 = []
set_7 = []
set_8 = []
set_9 = []

for i in range(len(training_labels)):
    if training_labels[i] == 0:
        set_0.append(training_images[i])
    if training_labels[i] == 1:
        set_1.append(training_images[i])
    if training_labels[i] == 2:
        set_2.append(training_images[i])
    if training_labels[i] == 3:
        set_3.append(training_images[i])
    if training_labels[i] == 4:
        set_4.append(training_images[i])
    if training_labels[i] == 5:
        set_5.append(training_images[i])
    if training_labels[i] == 6:
        set_6.append(training_images[i])
    if training_labels[i] == 7:
        set_7.append(training_images[i])
    if training_labels[i] == 8:
        set_8.append(training_images[i])
    if training_labels[i] == 9:
        set_9.append(training_images[i])
        

p0 = len(set_0) / len(training_labels)
p1 = len(set_1) / len(training_labels)
p2 = len(set_2) / len(training_labels)
p3 = len(set_3) / len(training_labels)
p4 = len(set_4) / len(training_labels)
p5 = len(set_5) / len(training_labels)
p6 = len(set_6) / len(training_labels)
p7 = len(set_7) / len(training_labels)
p8 = len(set_8) / len(training_labels)
p9 = len(set_9) / len(training_labels)

set_0 = np.asarray(set_0)
set_1 = np.asarray(set_1)
set_2 = np.asarray(set_2)
set_3 = np.asarray(set_3)
set_4 = np.asarray(set_4)
set_5 = np.asarray(set_5)
set_6 = np.asarray(set_6)
set_7 = np.asarray(set_7)
set_8 = np.asarray(set_8)
set_9 = np.asarray(set_9)


prob0 = (set_0.sum(axis=0)+10)/len(set_0)
prob1 = (set_1.sum(axis=0)+10)/len(set_1)
prob2 = (set_2.sum(axis=0)+10)/len(set_2)
prob3 = (set_3.sum(axis=0)+10)/len(set_3)
prob4 = (set_4.sum(axis=0)+10)/len(set_4)
prob5 = (set_5.sum(axis=0)+10)/len(set_5)
prob6 = (set_6.sum(axis=0)+10)/len(set_6)
prob7 = (set_7.sum(axis=0)+10)/len(set_7)
prob8 = (set_8.sum(axis=0)+10)/len(set_8)
prob9 = (set_9.sum(axis=0)+10)/len(set_9)

prob0 = prob0.reshape((784,1))
prob1 = prob1.reshape((784,1))
prob2 = prob2.reshape((784,1))
prob3 = prob3.reshape((784,1))
prob4 = prob4.reshape((784,1))
prob5 = prob5.reshape((784,1))
prob6 = prob6.reshape((784,1))
prob7 = prob7.reshape((784,1))
prob8 = prob8.reshape((784,1))
prob9 = prob9.reshape((784,1))


correct = 0
false = 0 
# Predict
for index in range(len(test_images)):
    # print(index)
    probability = np.zeros(10)
    for i in range(784):
        if prob0[i] + prob1[i]+prob2[i]+prob3[i]+prob4[i]+prob5[i]+prob6[i]+prob7[i]+prob8[i]+prob9[i] != 0 :
            probability[0] += test_images[index][i]*np.log(prob0[i]) + np.log(1-prob0[i])*(1-test_images[index][i])
            probability[1] += test_images[index][i]*np.log(prob1[i]) + np.log(1-prob1[i])*(1-test_images[index][i])
            probability[2] += test_images[index][i]*np.log(prob2[i]) + np.log(1-prob2[i])*(1-test_images[index][i])
            probability[3] += test_images[index][i]*np.log(prob3[i]) + np.log(1-prob3[i])*(1-test_images[index][i])
            probability[4] += test_images[index][i]*np.log(prob4[i]) + np.log(1-prob4[i])*(1-test_images[index][i])
            probability[5] += test_images[index][i]*np.log(prob5[i]) + np.log(1-prob5[i])*(1-test_images[index][i])
            probability[6] += test_images[index][i]*np.log(prob6[i]) + np.log(1-prob6[i])*(1-test_images[index][i])
            probability[7] += test_images[index][i]*np.log(prob7[i]) + np.log(1-prob7[i])*(1-test_images[index][i])
            probability[8] += test_images[index][i]*np.log(prob8[i]) + np.log(1-prob8[i])*(1-test_images[index][i])
            probability[9] += test_images[index][i]*np.log(prob9[i]) + np.log(1-prob9[i])*(1-test_images[index][i])

    probability[0] += np.log(p0)
    probability[1] += np.log(p1)
    probability[2] += np.log(p2)
    probability[3] += np.log(p3)
    probability[4] += np.log(p4)
    probability[5] += np.log(p5)
    probability[6] += np.log(p6)
    probability[7] += np.log(p7)
    probability[8] += np.log(p8)
    probability[9] += np.log(p9)

    # print(probability)

    max_index = np.argmax(probability)

    if max_index == test_labels[index]:
        correct += 1
    else:
        false += 1
    
accuracy += correct / (correct + false)

print("Train_acc: ", accuracy)















