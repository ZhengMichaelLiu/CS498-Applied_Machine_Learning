import pickle
import numpy as np
from numpy import linalg as npla
import matplotlib.pyplot as plt
from scipy.spatial import distance

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

dict1 = unpickle('data_batch_1')
dict2 = unpickle('data_batch_2')
dict3 = unpickle('data_batch_3')
dict4 = unpickle('data_batch_4')
dict5 = unpickle('data_batch_5')
dict_test = unpickle('test_batch')

dicts = [dict1, dict2, dict3, dict4, dict5, dict_test]
# print(dict_test[b'data'].shape)
# (10000, 3072) 

# separate to 10 classes
classes =[[],[],[],[],[],[],[],[],[],[]]

for each_dict in dicts:
    for i in range((dict_test[b'data'].shape)[0]):
        classes[each_dict[b'labels'][i]].append(each_dict[b'data'][i])

# each class 6000 images
class0 = np.asarray(classes[0], dtype=np.float)
class1 = np.asarray(classes[1], dtype=np.float)
class2 = np.asarray(classes[2], dtype=np.float)
class3 = np.asarray(classes[3], dtype=np.float)
class4 = np.asarray(classes[4], dtype=np.float)
class5 = np.asarray(classes[5], dtype=np.float)
class6 = np.asarray(classes[6], dtype=np.float)
class7 = np.asarray(classes[7], dtype=np.float)
class8 = np.asarray(classes[8], dtype=np.float)
class9 = np.asarray(classes[9], dtype=np.float)

# For each class, find the mean image
class0_mean = np.mean(class0, axis=0)
class1_mean = np.mean(class1, axis=0)
class2_mean = np.mean(class2, axis=0)
class3_mean = np.mean(class3, axis=0)
class4_mean = np.mean(class4, axis=0)
class5_mean = np.mean(class5, axis=0)
class6_mean = np.mean(class6, axis=0)
class7_mean = np.mean(class7, axis=0)
class8_mean = np.mean(class8, axis=0)
class9_mean = np.mean(class9, axis=0)

classes_mean = [class0_mean, class1_mean, class2_mean, class3_mean, class4_mean, class5_mean, class6_mean, class7_mean, class8_mean, class9_mean]

# All Correct!
result = np.zeros((10, 10))
for i in range(10):
    for j in range(10):
        result[i, j] = distance.euclidean(classes_mean[i], classes_mean[j]) ** 2

np.savetxt("partb_distances.csv", result, delimiter=",")

#A = np.identity(5) - (1 / 5) * np.dot(np.ones(5)[:,None],(np.ones(5).T)[None,:])
# MDS
A = np.identity(10) - (1 / 10) * np.dot(np.ones(10)[:,None],(np.ones(10).T)[None,:])

W = (-1/2) * np.dot(np.dot(A, result), A.T)

eigenValues, eigenVectors = npla.eig(W)
idx = np.argsort(eigenValues)
idx = np.flip(idx)
eigenValues = eigenValues[idx]
eigenVectors = eigenVectors[:, idx]

s = 4
Sigma_s = np.zeros((s,s))
Sigma_s[0, 0] = eigenValues[0]
Sigma_s[1, 1] = eigenValues[1]

Sigma_s_sqrt = np.sqrt(Sigma_s)

U_s = eigenVectors[:, :s]

Y = np.dot(U_s, Sigma_s_sqrt)

plot_x = Y[:, 0]
plot_y = Y[:, 1]
plt.scatter(plot_x, plot_y)
plt.show()