import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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

# find first 20 principle component
pca0 = PCA(n_components = 20)
class0_transformed = pca0.fit_transform(class0)
class0_inversetransformed = pca0.inverse_transform(class0_transformed)

pca1 = PCA(n_components = 20)
class1_transformed = pca1.fit_transform(class1)
class1_inversetransformed = pca1.inverse_transform(class1_transformed)

pca2 = PCA(n_components = 20)
class2_transformed = pca2.fit_transform(class2)
class2_inversetransformed = pca2.inverse_transform(class2_transformed)

pca3 = PCA(n_components = 20)
class3_transformed = pca3.fit_transform(class3)
class3_inversetransformed = pca3.inverse_transform(class3_transformed)

pca4 = PCA(n_components = 20)
class4_transformed = pca4.fit_transform(class4)
class4_inversetransformed = pca4.inverse_transform(class4_transformed)

pca5 = PCA(n_components = 20)
class5_transformed = pca5.fit_transform(class5)
class5_inversetransformed = pca5.inverse_transform(class5_transformed)

pca6 = PCA(n_components = 20)
class6_transformed = pca6.fit_transform(class6)
class6_inversetransformed = pca6.inverse_transform(class6_transformed)

pca7 = PCA(n_components = 20)
class7_transformed = pca7.fit_transform(class7)
class7_inversetransformed = pca7.inverse_transform(class7_transformed)

pca8 = PCA(n_components = 20)
class8_transformed = pca8.fit_transform(class8)
class8_inversetransformed = pca8.inverse_transform(class8_transformed)

pca9 = PCA(n_components = 20)
class9_transformed = pca9.fit_transform(class9)
class9_inversetransformed = pca9.inverse_transform(class9_transformed)

class0_error = 0
for i in range(6000):
    for j in range(3072):
        class0_error += (class0[i, j] - class0_inversetransformed[i, j]) ** 2
class0_error /= 6000

class1_error = 0
for i in range(6000):
    for j in range(3072):
        class1_error += (class1[i, j] - class1_inversetransformed[i, j]) ** 2
class1_error /= 6000

class2_error = 0
for i in range(6000):
    for j in range(3072):
        class2_error += (class2[i, j] - class2_inversetransformed[i, j]) ** 2
class2_error /= 6000

class3_error = 0
for i in range(6000):
    for j in range(3072):
        class3_error += (class3[i, j] - class3_inversetransformed[i, j]) ** 2
class3_error /= 6000

class4_error = 0
for i in range(6000):
    for j in range(3072):
        class4_error += (class4[i, j] - class4_inversetransformed[i, j]) ** 2
class4_error/= 6000

class5_error = 0
for i in range(6000):
    for j in range(3072):
        class5_error += (class5[i, j] - class5_inversetransformed[i, j]) ** 2
class5_error /= 6000

class6_error = 0
for i in range(6000):
    for j in range(3072):
        class6_error += (class6[i, j] - class6_inversetransformed[i, j]) ** 2
class6_error /= 6000

class7_error = 0
for i in range(6000):
    for j in range(3072):
        class7_error += (class7[i, j] - class7_inversetransformed[i, j]) ** 2
class7_error /= 6000

class8_error = 0
for i in range(6000):
    for j in range(3072):
        class8_error += (class8[i, j] - class8_inversetransformed[i, j]) ** 2
class8_error /= 6000

class9_error = 0
for i in range(6000):
    for j in range(3072):
        class9_error += (class9[i, j] - class9_inversetransformed[i, j]) ** 2
class9_error /= 6000

X = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Y = [class0_error, class1_error, class2_error, class3_error, class4_error, class5_error, class6_error, class7_error, class8_error, class9_error]
plt.bar(X, Y)
plt.xlabel('Class Number')
plt.ylabel('Averaged Squared Error')
plt.title('Averaged Squared Error for each class')
plt.show()