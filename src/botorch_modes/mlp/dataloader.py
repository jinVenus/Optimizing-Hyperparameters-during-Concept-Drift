# bj mlp test
import pandas as pd
import os
import numpy as np
from torchvision.datasets import mnist


def load_data(datasets_num, type):
    datasets = ['static', 'dynamic']

    dataset = datasets[datasets_num]

    if dataset == 'dynamic':
        data1 = pd.read_csv('E:\\ML\\Hiwi\\data\\MNIST\\raw\\mnist_dynamic.csv', sep=',', header=None)

        X_train1 = data1.iloc[:, 1:]
        y_train1 = data1.iloc[:, 0]
        X_train1 = np.array(X_train1)
        y_train1 = np.array(y_train1)

        if type == 'eval':
            X_train1 = X_train1.reshape(len(data1), -1)

            X_test = X_train1[50000:]
            X_val = X_train1[40000:50000]
            X_train = X_train1[:40000]
            y_test = y_train1[50000:]
            y_val = y_train1[40000:50000]
            y_train = y_train1[:40000]

        if type == 'test':
            X_train1 = X_train1.reshape(len(data1), -1)
            X_val = X_train1[40000:50000]
            X_test = X_train1[50000:]
            X_train = X_train1[:40000]
            y_val = y_train1[40000:50000]
            y_test = y_train1[50000:]
            y_train = y_train1[:40000]

        print(X_train.shape)

    elif dataset == 'static':
        data1 = pd.read_csv('E:\\ML\\Hiwi\\data\\MNIST\\raw\\mnist_cd_fr15.csv', sep=',', header=None)
        data2 = pd.read_csv('E:\\ML\\Hiwi\\data\\MNIST\\raw\\mnist_cd_fr0.csv', sep=',', header=None)

        X_train1 = data1.iloc[:, 1:]
        y_train1 = data1.iloc[:, 0]
        X_train1 = np.array(X_train1)
        y_train1 = np.array(y_train1)

        X_train2 = data2.iloc[:, 1:]
        y_train2 = data2.iloc[:, 0]
        X_train2 = np.array(X_train2)
        y_train2 = np.array(y_train2)

        if type == 'eval':
            X_train1 = X_train1.reshape(len(data1), -1)
            X_train2 = X_train2.reshape(len(data2), -1)
            X_test = X_train1[50000:]
            X_val = X_train1[40000:50000]
            X_train = X_train1[:40000]
            y_test = y_train1[50000:]
            y_val = y_train1[40000:50000]
            y_train = y_train1[:40000]

        if type == 'test':
            X_train1 = X_train1.reshape(len(data1), -1)
            X_train2 = X_train2.reshape(len(data2), -1)
            X_val = X_train1[40000:50000]
            X_test = X_train1[50000:]
            X_train = X_train1[:40000]
            y_val = y_train1[40000:50000]
            y_test = y_train1[50000:]
            y_train = y_train1[:40000]

    return X_train, y_train, X_val, y_val, X_test, y_test
