import numpy as np
import pandas as pd
import typing as ty
from tqdm import tqdm
from dataclasses import dataclass

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture as GMM
from numpy.random import randint

from scipy.spatial.distance import mahalanobis
import numpy.linalg as lnalg


def impute_missing_values(data):
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    columns = data.columns
    data = pd.DataFrame(imputer.fit_transform(data))
    data.columns = columns
    return data


def encode_text_labels(data, feats_to_encode):
    for k in feats_to_encode:
        range_labname = list(set(data[k].values))
        for i in range(len(range_labname)):
            data.loc[data[k] == range_labname[i], k] = i
    return data


def drop_features(data, feats_to_drop):
    for k in feats_to_drop:
        data = data.drop(k, 1)
    return data


def min_max_scale(data, feats_to_scale):
    for k in feats_to_scale:
        mms = MinMaxScaler()
        data[k] = mms.fit_transform(data[k].values.reshape(-1, 1))
    return data


def augmentation(trianX, trainY, n_components=1, augmentation_size=256):
    print('Starting augmentation...')
    trainY = trainY.reshape(-1)
    arr_classes = np.unique(trainY)

    print('Training GMMs...')
    # now we train a GMM for each class
    gmms = dict()
    for cls in arr_classes:  # 2 classes
        gmm = GMM(n_components=n_components, covariance_type='full')
        gmm.fit(trianX[trainY == cls])
        gmms[cls] = (gmm.means_, gmm.covariances_)

    print('Calculating M distances...')
    # we now calculate the M distance for each class
    # we will use this for saliency check
    for cls in arr_classes:
        sub_sample = trianX[trainY == cls][np.random.choice(trianX[trainY == cls].shape[0], 500, replace=True)]  # (num_samples, num_feats)
        mh = np.empty((sub_sample.shape[0]))
        mu, sig = gmms[cls]  # mu: (num_comp, num_feats), sig: (num_comp, num_feats, num_feats)
        # isig = lnalg.inv(sig)
        for i in tqdm(range(sub_sample.shape[0])):
            # mh[i] = mahalanobis(sub_sample[i].reshape(-1), mu.reshape(-1), isig)
            mh[i] = min([mahalanobis(sub_sample[i].reshape(-1), mu[j], lnalg.inv(sig[j])) for j in range(n_components)])
        print('class', cls, 'mahalanobis mean', mh.mean())

    print('MD-based sanity check test...')
    # MD-based sanity check test
    inv_sig = dict()
    mh = np.zeros((arr_classes.shape[0]))
    x = trianX[:200]
    labels = trainY[:200]
    results = np.zeros(x.shape[0])

    for cls in arr_classes:
        mu, sig = gmms[cls]
        # isig = lnalg.inv(sig)
        # inv_sig[cls] = mu, isig
        inv_sig[cls] = mu, [lnalg.inv(s) for s in sig]

    for i in range(x.shape[0]):
        for cls in arr_classes:
            mu, isig = inv_sig[cls]
            # mh[cls] = mahalanobis(x[i].reshape(-1), mu.reshape(-1), isig)
            mh[cls] = min([mahalanobis(x[i].reshape(-1), mu[j], isig[j]) for j in range(n_components)])
        if np.argmin(mh) == labels[i]:
            results[i] = 1

    acc = results.sum() / results.shape[0]
    print('MD-based sanity check test accuracy:', acc)
    if acc > .9: print('Doing well')

    print('Augmenting data...')
    # now we augment the data
    new_class_arrays = []
    for cls in arr_classes:
        dlt = augmentation_size  # for each class we will newly create N=256 samples
        data = trianX[trainY == cls]
        data_mean = data.mean(axis=0)
        data_std = data.std(axis=0)
        sub_arr = np.empty((dlt, data.shape[1]))
        for col in range(data.shape[1]):
            sub_arr[:, col] = np.random.normal(data_mean[col], data_std[col], dlt)
        new_class_arrays.append(sub_arr)

    # new_sample = np.concatenate(new_class_arrays, axis=0)
    #
    # col_counter = 0
    # for _ in new_sample.T:
    #     s = new_sample[:, col_counter]
    #     s[s < .5] = 0
    #     s[s >= .5] = 1
    #     new_sample[:, col_counter] = s
    #     col_counter += 1
    #
    # new_sample_labels = np.concatenate((np.zeros(256), np.ones(256)), axis=0)

    # print(new_sample_labels)
    # print('0 values', (new_sample_labels == 0).sum())
    # print('1 values', (new_sample_labels == 1).sum())
    #
    # # now we assign class labels to the new samples
    # for i in range(new_sample.shape[0]):
    #     for cls in arr_classes:
    #         mu, isig = inv_sig[cls]
    #         mh[cls] = mahalanobis(new_sample[i].reshape(-1), mu.reshape(-1), isig)
    #     if np.argmin(mh) != new_sample_labels[i]:
    #         new_sample_labels[i] = np.argmin(mh)
    #
    # print(new_sample_labels)
    # print('0 values', (new_sample_labels == 0).sum())
    # print('1 values', (new_sample_labels == 1).sum())

    new_sample = np.concatenate(new_class_arrays, axis=0)
    new_sample_labels = np.concatenate([np.full((augmentation_size,), cls) for cls in arr_classes])

    print(new_sample_labels)
    print('Before class assignment')
    print('0 values', (new_sample_labels == 0).sum())
    print('1 values', (new_sample_labels == 1).sum())

    # assign class labels to the new samples
    for i in range(new_sample.shape[0]):
        mh = np.zeros((arr_classes.shape[0]))
        for cls in arr_classes:
            mu, isig = inv_sig[cls]
            # mh[cls] = mahalanobis(new_sample[i].reshape(-1), mu.reshape(-1), isig)
            mh[cls] = min([mahalanobis(new_sample[i].reshape(-1), mu[j], isig[j]) for j in range(n_components)])
        new_sample_labels[i] = np.argmin(mh)

    print(new_sample_labels)
    print('After class assignment')
    print('0 values', (new_sample_labels == 0).sum())
    print('1 values', (new_sample_labels == 1).sum())

    return np.concatenate((trianX, new_sample), axis=0).astype('float32'), np.concatenate((trainY, new_sample_labels), axis=0).reshape(-1, 1).astype('float32')


class DataContainer:

    def __init__(self, data_name, data_file, train_ratio, test_ratio, label_feat, feats_to_encode, feats_to_drop,
                 feats_to_scale, seed):

        self.name = data_name
        self.data_file = data_file
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.label_feat = label_feat
        self.feats_to_encode = feats_to_encode
        self.feats_to_drop = feats_to_drop
        self.feats_to_scale = feats_to_scale
        self.seed = seed

    def clean(self):
        data = self._read()
        data = self._encode(data)
        data = self._drop(data)
        if self.name == 'eicu':
            data = self._impute(data)
        data = self._scale(data)
        return data

    def _read(self):
        data = pd.read_csv(self.data_file)
        self.rawData = data
        self.feature_names = data.columns
        self.num_features = len(data.columns)

        if self.name != 'eicu':
            data = impute_missing_values(data)

        if self.name == 'mimic':
            data[['bmi_group']] = data[['bmi_group']] - 1
        elif self.name == 'framingham':
            data = impute_missing_values(data)
            data['age'] = pd.cut(data['age'], 5, labels=[1, 2, 3, 4, 5])
            data['cigsPerDay'] = pd.cut(data['cigsPerDay'], 6, labels=[1, 2, 3, 4, 5, 6])
            data['totChol'] = pd.cut(data['totChol'], 13, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
            data['sysBP'] = pd.cut(data['sysBP'], 10, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            data['diaBP'] = pd.cut(data['diaBP'], 8, labels=[1, 2, 3, 4, 5, 6, 7, 8])
            data['BMI'] = pd.cut(data['BMI'], 10, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            data['heartRate'] = pd.cut(data['heartRate'], 8, labels=[1, 2, 3, 4, 5, 6, 7, 8])
            data['glucose'] = pd.cut(data['glucose'], 15, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

        return data

    def _encode(self, data):
        return encode_text_labels(data, self.feats_to_encode)

    def _drop(self, data):
        data = drop_features(data, self.feats_to_drop)
        self.feature_names = data.columns[:-1]
        self.num_features = len(data.columns)
        return data

    def _impute(self, data):
        return impute_missing_values(data)

    def _scale(self, data):
        if self.name == 'framingham':
            data['age'] = data['age'].apply(lambda x: round(x * 0.2, 2))
            data['education'] = data['education'].apply(lambda x: round(x * 0.25, 2))
            data['cigsPerDay'] = data['cigsPerDay'].apply(lambda x: round(x * 0.16, 2))
            data['totChol'] = data['totChol'].apply(lambda x: round(x * 0.077, 2))
            data['sysBP'] = data['sysBP'].apply(lambda x: round(x * 0.1, 2))
            data['diaBP'] = data['diaBP'].apply(lambda x: round(x * 0.125, 2))
            data['BMI'] = data['BMI'].apply(lambda x: round(x * 0.1, 2))
            data['heartRate'] = data['heartRate'].apply(lambda x: round(x * 0.125, 2))
            data['glucose'] = data['glucose'].apply(lambda x: round(x * 0.067, 2))
            return data
        else:
            return min_max_scale(data, self.feats_to_scale)

    def split(self, data):
        np.random.seed(self.seed)
        hd1 = data[data[self.label_feat] == 1.0]
        hd0 = data[data[self.label_feat] == 0.0]
        hd1x = hd1.iloc[:, 0:-1].values.astype('float32')
        hd1y = hd1.iloc[:, -1].values.astype('float32')
        hd0x = hd0.iloc[:, 0:-1].values.astype('float32')
        hd0y = hd0.iloc[:, -1].values.astype('float32')
        hd1x_train, hd1x_test, hd1y_train, hd1y_test = \
            train_test_split(hd1x, hd1y, test_size=self.test_ratio, train_size=self.train_ratio)
        hd0x_train, hd0x_test, hd0y_train, hd0y_test = \
            train_test_split(hd0x, hd0y, test_size=self.test_ratio, train_size=self.train_ratio)
        trainX = np.vstack((hd1x_train, hd0x_train))
        trainY = np.vstack((hd1y_train.reshape([-1, 1]), hd0y_train.reshape([-1, 1]))).astype('int')
        testX = np.vstack((hd1x_test, hd0x_test))
        testY = np.vstack((hd1y_test.reshape([-1, 1]), hd0y_test.reshape([-1, 1]))).astype('int')
        return trainX, trainY, testX, testY, \
               hd0x_train, hd1x_train, hd0x_test, hd1x_test, hd0y_train, hd1y_train, hd0y_test, hd1y_test

