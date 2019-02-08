import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def calculate_MSN(dataset1, dataset2):
    sum = 0
    for i in range(len(dataset1)):
        for j in range(4):
            sum += (dataset1[i, j] - dataset2[i, j]) ** 2

    return sum / len(dataset1)

dataset1 = pd.read_csv('dataI.csv')
dataset2 = pd.read_csv('dataII.csv')
dataset3 = pd.read_csv('dataIII.csv')
dataset4 = pd.read_csv('dataIV.csv')
dataset5 = pd.read_csv('dataV.csv')
dataset6 = pd.read_csv('iris.csv')

dataset1_new = dataset1.values
dataset2_new = dataset2.values
dataset3_new = dataset3.values
dataset4_new = dataset4.values
dataset5_new = dataset5.values
dataset6_new = dataset6.values

dataset = [dataset1_new, dataset2_new, dataset3_new, dataset4_new, dataset5_new, dataset6_new]

# use noiseless data to fit
msn_noiseless = []
for i in range(5):
    msn_row = []
    for j in range(5):
        pca = PCA(n_components = j)
        pca.fit(dataset[5])
        dataset_transformed_noiseless = pca.transform(dataset[i])
        dataset_inversetransformed = pca.inverse_transform(dataset_transformed_noiseless)
        msn_row.append(calculate_MSN(dataset_inversetransformed, dataset[5]))
    msn_noiseless.append(msn_row)

# use noise data to fit
msn_noise = []
for i in range(5):
    msn_row = []
    for j in range(5):
        pca = PCA(n_components = j)
        pca.fit(dataset[i])
        dataset_transformed_noise = pca.transform(dataset[i])
        dataset_inversetransformed = pca.inverse_transform(dataset_transformed_noise)
        msn_row.append(calculate_MSN(dataset_inversetransformed, dataset[5]))
    msn_noise.append(msn_row)

msn_total = np.concatenate((msn_noiseless, msn_noise),axis=1)
np.savetxt("zliu93.csv", msn_total, delimiter=",")

pca = PCA(n_components = 2)
pca.fit(dataset[0])
dataset_transformed_noise = pca.transform(dataset[0])
dataset_inversetransformed = pca.inverse_transform(dataset_transformed_noise)
np.savetxt("recon.csv", dataset_inversetransformed, delimiter=",")