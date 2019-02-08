from mnist import MNIST
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

mndata = MNIST('F:\CS498 - Applied Machine Learning\CS498HW1PartB\CS498HW1PartB')
mndata.gz = True
training_images, training_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()

for i in range(len(training_images)):
    for j in range(len(training_images[0])):
        if training_images[i][j] > 128:
            training_images[i][j] = 255
        else: 
            training_images[i][j] = 0
            
for i in range(len(test_images)):
    for j in range(len(test_images[0])):
        if test_images[i][j] > 128:
            test_images[i][j] = 255
        else: 
            test_images[i][j] = 0
#print("threshold")            

test_images_array = np.asarray(test_images)
test_labels_array = np.asarray(test_labels)

training_images_array = np.asarray(training_images)
training_labels_array = np.asarray(training_labels)
training_labels_array_new = training_labels_array.reshape((60000,1))

training_data_arr = np.concatenate((training_images_array, training_labels_array_new), axis=1)

############################### Normal Distribution, untouched ###############################
############################### Train ###############################
accuracy = 0
# split train and validation
np.random.shuffle(training_data_arr)

X_train = training_data_arr[:48000, :784]
X_validation = training_data_arr[48000:, :784]

Y_train = training_data_arr[:48000, -1]
Y_validation = training_data_arr[48000:, -1]

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

# Calculate each feature mean
mean0 = np.mean(set_0, axis = 0)
mean1 = np.mean(set_1, axis = 0)
mean2 = np.mean(set_2, axis = 0)
mean3 = np.mean(set_3, axis = 0)
mean4 = np.mean(set_4, axis = 0)
mean5 = np.mean(set_5, axis = 0)
mean6 = np.mean(set_6, axis = 0)
mean7 = np.mean(set_7, axis = 0)
mean8 = np.mean(set_8, axis = 0)
mean9 = np.mean(set_9, axis = 0)

# Calculate each feature std
std0 = np.std(set_0, axis = 0)
std1 = np.std(set_1, axis = 0)
std2 = np.std(set_2, axis = 0)
std3 = np.std(set_3, axis = 0)
std4 = np.std(set_4, axis = 0)
std5 = np.std(set_5, axis = 0)
std6 = np.std(set_6, axis = 0)
std7 = np.std(set_7, axis = 0)
std8 = np.std(set_8, axis = 0)
std9 = np.std(set_9, axis = 0)

# smoothing
smoothing_val = 15
std0 += smoothing_val
std1 += smoothing_val
std2 += smoothing_val
std3 += smoothing_val
std4 += smoothing_val
std5 += smoothing_val
std6 += smoothing_val
std7 += smoothing_val
std8 += smoothing_val
std9 += smoothing_val

correct = 0
false = 0 
# Predict
for index in range(len(X_validation)):
    # print(index)
    probability = np.zeros(10)
    for i in range(784):
        probability[0] += np.log(1 / np.sqrt(2 * np.pi * std0[i] * std0[i])) - ((X_validation[index][i] - mean0[i]) * (X_validation[index][i] - mean0[i]) / (2 * std0[i] * std0[i]))
        probability[1] += np.log(1 / np.sqrt(2 * np.pi * std1[i] * std1[i])) - ((X_validation[index][i] - mean1[i]) * (X_validation[index][i] - mean1[i]) / (2 * std1[i] * std1[i]))
        probability[2] += np.log(1 / np.sqrt(2 * np.pi * std2[i] * std2[i])) - ((X_validation[index][i] - mean2[i]) * (X_validation[index][i] - mean2[i]) / (2 * std2[i] * std2[i]))
        probability[3] += np.log(1 / np.sqrt(2 * np.pi * std3[i] * std3[i])) - ((X_validation[index][i] - mean3[i]) * (X_validation[index][i] - mean3[i]) / (2 * std3[i] * std3[i]))
        probability[4] += np.log(1 / np.sqrt(2 * np.pi * std4[i] * std4[i])) - ((X_validation[index][i] - mean4[i]) * (X_validation[index][i] - mean4[i]) / (2 * std4[i] * std4[i]))
        probability[5] += np.log(1 / np.sqrt(2 * np.pi * std5[i] * std5[i])) - ((X_validation[index][i] - mean5[i]) * (X_validation[index][i] - mean5[i]) / (2 * std5[i] * std5[i]))
        probability[6] += np.log(1 / np.sqrt(2 * np.pi * std6[i] * std6[i])) - ((X_validation[index][i] - mean6[i]) * (X_validation[index][i] - mean6[i]) / (2 * std6[i] * std6[i]))
        probability[7] += np.log(1 / np.sqrt(2 * np.pi * std7[i] * std7[i])) - ((X_validation[index][i] - mean7[i]) * (X_validation[index][i] - mean7[i]) / (2 * std7[i] * std7[i]))
        probability[8] += np.log(1 / np.sqrt(2 * np.pi * std8[i] * std8[i])) - ((X_validation[index][i] - mean8[i]) * (X_validation[index][i] - mean8[i]) / (2 * std8[i] * std8[i]))
        probability[9] += np.log(1 / np.sqrt(2 * np.pi * std9[i] * std9[i])) - ((X_validation[index][i] - mean9[i]) * (X_validation[index][i] - mean9[i]) / (2 * std9[i] * std9[i]))
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



############################### Test ###############################

accuracy = 0

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

# Calculate each feature mean
mean0 = np.mean(set_0, axis = 0)
mean1 = np.mean(set_1, axis = 0)
mean2 = np.mean(set_2, axis = 0)
mean3 = np.mean(set_3, axis = 0)
mean4 = np.mean(set_4, axis = 0)
mean5 = np.mean(set_5, axis = 0)
mean6 = np.mean(set_6, axis = 0)
mean7 = np.mean(set_7, axis = 0)
mean8 = np.mean(set_8, axis = 0)
mean9 = np.mean(set_9, axis = 0)

# Calculate each feature std
std0 = np.std(set_0, axis = 0)
std1 = np.std(set_1, axis = 0)
std2 = np.std(set_2, axis = 0)
std3 = np.std(set_3, axis = 0)
std4 = np.std(set_4, axis = 0)
std5 = np.std(set_5, axis = 0)
std6 = np.std(set_6, axis = 0)
std7 = np.std(set_7, axis = 0)
std8 = np.std(set_8, axis = 0)
std9 = np.std(set_9, axis = 0)

# smoothing
smoothing_val = 15
std0 += smoothing_val
std1 += smoothing_val
std2 += smoothing_val
std3 += smoothing_val
std4 += smoothing_val
std5 += smoothing_val
std6 += smoothing_val
std7 += smoothing_val
std8 += smoothing_val
std9 += smoothing_val

correct = 0
false = 0 
# Predict
for index in range(len(test_images)):
    # print(index)
    probability = np.zeros(10)
    for i in range(784):
        probability[0] += np.log(1 / np.sqrt(2 * np.pi * std0[i] * std0[i])) - ((test_images[index][i] - mean0[i]) * (test_images[index][i] - mean0[i]) / (2 * std0[i] * std0[i]))
        probability[1] += np.log(1 / np.sqrt(2 * np.pi * std1[i] * std1[i])) - ((test_images[index][i] - mean1[i]) * (test_images[index][i] - mean1[i]) / (2 * std1[i] * std1[i]))
        probability[2] += np.log(1 / np.sqrt(2 * np.pi * std2[i] * std2[i])) - ((test_images[index][i] - mean2[i]) * (test_images[index][i] - mean2[i]) / (2 * std2[i] * std2[i]))
        probability[3] += np.log(1 / np.sqrt(2 * np.pi * std3[i] * std3[i])) - ((test_images[index][i] - mean3[i]) * (test_images[index][i] - mean3[i]) / (2 * std3[i] * std3[i]))
        probability[4] += np.log(1 / np.sqrt(2 * np.pi * std4[i] * std4[i])) - ((test_images[index][i] - mean4[i]) * (test_images[index][i] - mean4[i]) / (2 * std4[i] * std4[i]))
        probability[5] += np.log(1 / np.sqrt(2 * np.pi * std5[i] * std5[i])) - ((test_images[index][i] - mean5[i]) * (test_images[index][i] - mean5[i]) / (2 * std5[i] * std5[i]))
        probability[6] += np.log(1 / np.sqrt(2 * np.pi * std6[i] * std6[i])) - ((test_images[index][i] - mean6[i]) * (test_images[index][i] - mean6[i]) / (2 * std6[i] * std6[i]))
        probability[7] += np.log(1 / np.sqrt(2 * np.pi * std7[i] * std7[i])) - ((test_images[index][i] - mean7[i]) * (test_images[index][i] - mean7[i]) / (2 * std7[i] * std7[i]))
        probability[8] += np.log(1 / np.sqrt(2 * np.pi * std8[i] * std8[i])) - ((test_images[index][i] - mean8[i]) * (test_images[index][i] - mean8[i]) / (2 * std8[i] * std8[i]))
        probability[9] += np.log(1 / np.sqrt(2 * np.pi * std9[i] * std9[i])) - ((test_images[index][i] - mean9[i]) * (test_images[index][i] - mean9[i]) / (2 * std9[i] * std9[i]))
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

print("Test_acc: ", accuracy)


