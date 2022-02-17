# bj mlp test
import numpy as np
import os
import pandas as pd
import torch
from skimage import io
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import torchvision
import numpy as np
from torch.autograd import Variable

# def convert(imgf, labelf, outf, n):
#     f = open(imgf, "rb")
#     o = open(outf, "w")
#     l = open(labelf, "rb")
#
#     f.read(16)
#     l.read(8)
#     images = []
#
#     for i in range(n):
#         image = [ord(l.read(1))]
#         for j in range(28 * 28):
#             image.append(ord(f.read(1)))
#         images.append(image)
#
#     for image in images:
#         o.write(",".join(str(pix) for pix in image) + "\n")
#     f.close()
#     o.close()
#     l.close()
#
#
# convert("E:\\ML\\Hiwi\\data\\MNIST\\raw\\train-images-idx3-ubyte", "E:\\ML\\Hiwi\\data\\MNIST\\raw\\train-labels-idx1-ubyte",
#         "E:\\ML\\Hiwi\\data\\MNIST\\raw\\mnist_train.csv", 60000)
# convert("E:\\ML\\Hiwi\\data\\MNIST\\raw\\t10k-images-idx3-ubyte", "E:\\ML\\Hiwi\\data\\MNIST\\raw\\t10k-labels-idx1-ubyte",
#         "E:\\ML\\Hiwi\\data\\MNIST\\raw\\mnist_test.csv", 10000)
#
# print("Convert Finished!")


# transform = transforms.Compose([transforms.ToTensor(),
#                                 transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
# X_train = datasets.MNIST(root="E:\\ML\\Hiwi\\data\\",
#                          transform=transform,
#                          train=True,
#                          download=True)
#
# X_test = datasets.MNIST(root="E:\\ML\\Hiwi\\data\\",
#                         transform=transform,
#                         train=False)
#
# data_loader_train = torch.utils.data.DataLoader(dataset=X_train,
#                                                 batch_size=32,
#                                                 shuffle=False)
# data_loader_test = torch.utils.data.DataLoader(dataset=X_test,
#                                                batch_size=32,
#                                                shuffle=False)
from torchvision.datasets import mnist

datasets = ['alldata', 'nddata', 'ovdata', 'unddata']
file_path = 'E:\\ML\\Hiwi\\data\\MNIST\\raw'


# def convert_to_img(train=True):
#     if (train):
#         f = open(file_path + 'train.txt', 'w')
#         data_path = file_path + '/train/'
#         if (not os.path.exists(data_path)):
#             os.makedirs(data_path)
#         for i, (img, label) in enumerate(zip(train_set[0], train_set[1])):
#             img_path = data_path + str(i) + '.jpg'
#             io.imsave(img_path, img.numpy())
#             f.write(img_path + ' ' + str(label) + '\n')
#         f.close()
#     else:
#         f = open(file_path + 'test.txt', 'w')
#         data_path = file_path + '/test/'
#         if (not os.path.exists(data_path)):
#             os.makedirs(data_path)
#         for i, (img, label) in enumerate(zip(test_set[0], test_set[1])):
#             img_path = data_path + str(i) + '.jpg'
#             io.imsave(img_path, img.numpy())
#             f.write(img_path + ' ' + str(label) + '\n')
#         f.close()
#
#
# convert_to_img(True)  # 转换训练集
# convert_to_img(False)  # 转换测试集


def load_data(datasets_num, type):

    # datasets = ['alldata', 'nddata', 'ovdata', 'unddata']
    datasets = ['MNIST', 'MNIST_cd', 'MNIST_mix', 'MNIST_28', 'MNIST_cd_82', 'MNIST_2882', 'MNIST_cd_fr15', 'MNIST_fr15_mix', 'MNIST_cd_fr5', "MNIST_fr5_mix"]

    dataset = datasets[datasets_num]


    file_path = 'E:\\ML\\Hiwi\\data\\MNIST\\raw'

    if dataset == 'MNIST':
        train_set = (
            mnist.read_image_file(os.path.join(file_path, 'train-images-idx3-ubyte')),
            mnist.read_label_file(os.path.join(file_path, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            mnist.read_image_file(os.path.join(file_path, 't10k-images-idx3-ubyte')),
            mnist.read_label_file(os.path.join(file_path, 't10k-labels-idx1-ubyte'))
        )

        X_train = train_set[0]
        y_train = train_set[1]

        if type == 'eval':

            X_train = X_train.reshape(60000, -1)
            X_val = X_train[50000:]
            X_train = X_train[:40000]
            y_val = y_train[50000:]
            y_train = y_train[:40000]

        if type == 'test':
            X_train = X_train.reshape(60000, -1)
            X_val = X_train[40000:50000]
            X_train = X_train[:40000]
            y_val = y_train[40000:50000]
            y_train = y_train[:40000]
        print(X_train.shape)

    elif dataset == 'MNIST_cd':
        data = pd.read_csv('E:\\ML\\Hiwi\\data\\MNIST\\raw\\mnist_train_cd.csv', sep=',', header=None)
        X_train = data.iloc[:, 1:]
        y_train = data.iloc[:, 0]
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        if type == 'eval':

            X_train = X_train.reshape(60000, -1)
            X_val = X_train[50000:]
            X_train = X_train[:40000]
            y_val = y_train[50000:]
            y_train = y_train[:40000]

        if type == 'test':
            X_train = X_train.reshape(60000, -1)
            X_val = X_train[40000:50000]
            X_train = X_train[:40000]
            y_val = y_train[40000:50000]
            y_train = y_train[:40000]
        print(X_train.shape)

    elif dataset == 'MNIST_mix':
        data1 = (
            mnist.read_image_file(os.path.join(file_path, 'train-images-idx3-ubyte')),
            mnist.read_label_file(os.path.join(file_path, 'train-labels-idx1-ubyte'))
        )

        X_train1 = data1[0]
        y_train1 = data1[1]

        data2 = pd.read_csv('E:\\ML\\Hiwi\\data\\MNIST\\raw\\mnist_train_cd.csv', sep=',', header=None)

        X_train2 = data2.iloc[:, 1:]
        y_train2 = data2.iloc[:, 0]
        X_train2 = np.array(X_train2)
        y_train2 = np.array(y_train2)

        if type == 'eval':
            X_train1 = X_train1.reshape(60000, -1)
            X_train2 = X_train2.reshape(60000, -1)
            X_val = X_train2[50000:]
            X_train = X_train1[:40000]
            y_val = y_train2[50000:]
            y_train = y_train1[:40000]

        if type == 'test':
            X_train1 = X_train1.reshape(60000, -1)
            X_train2 = X_train2.reshape(60000, -1)
            X_val = X_train2[40000:50000]
            X_train = X_train1[:40000]
            y_val = y_train2[40000:50000]
            y_train = y_train1[:40000]

    elif dataset == 'MNIST_28':
        data = pd.read_csv('E:\\ML\\Hiwi\\data\\MNIST\\raw\\mnist_cd_28.csv', sep=',', header=None)
        X_train = data.iloc[:, 1:]
        y_train = data.iloc[:, 0]
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        if type == 'eval':
            X_train = X_train.reshape(len(data), -1)
            X_val = X_train[50000:]
            X_train = X_train[:40000]
            y_val = y_train[50000:]
            y_train = y_train[:40000]

        if type == 'test':
            X_train = X_train.reshape(len(data), -1)
            X_val = X_train[40000:50000]
            X_train = X_train[:40000]
            y_val = y_train[40000:50000]
            y_train = y_train[:40000]
        print(X_train.shape)

    elif dataset == 'MNIST_cd_82':
        data = pd.read_csv('E:\\ML\\Hiwi\\data\\MNIST\\raw\\mnist_cd_82.csv', sep=',', header=None)
        X_train = data.iloc[:, 1:]
        y_train = data.iloc[:, 0]
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        if type == 'eval':
            X_train = X_train.reshape(len(data), -1)
            X_val = X_train[50000:]
            X_train = X_train[:40000]
            y_val = y_train[50000:]
            y_train = y_train[:40000]

        if type == 'test':
            X_train = X_train.reshape(len(data), -1)
            X_val = X_train[40000:50000]
            X_train = X_train[:40000]
            y_val = y_train[40000:50000]
            y_train = y_train[:40000]
        print(X_train.shape)

    elif dataset == 'MNIST_2882':
        data1 = pd.read_csv('E:\\ML\\Hiwi\\data\\MNIST\\raw\\mnist_cd_28.csv', sep=',', header=None)

        data2 = pd.read_csv('E:\\ML\\Hiwi\\data\\MNIST\\raw\\mnist_cd_82.csv', sep=',', header=None)

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
            X_val = X_train2[50000:]
            X_train = X_train1[:40000]
            y_val = y_train2[50000:]
            y_train = y_train1[:40000]

        if type == 'test':
            X_train1 = X_train1.reshape(len(data1), -1)
            X_train2 = X_train2.reshape(len(data2), -1)
            X_val = X_train2[40000:50000]
            X_train = X_train1[:40000]
            y_val = y_train2[40000:50000]
            y_train = y_train1[:40000]

    elif dataset == 'MNIST_cd_fr15':
        data = pd.read_csv('E:\\ML\\Hiwi\\data\\MNIST\\raw\\mnist_cd_fr15.csv', sep=',', header=None)
        X_train = data.iloc[:, 1:]
        y_train = data.iloc[:, 0]
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        if type == 'eval':
            X_train = X_train.reshape(60000, -1)
            X_val = X_train[50000:]
            X_train = X_train[:40000]
            y_val = y_train[50000:]
            y_train = y_train[:40000]

        if type == 'test':
            X_train = X_train.reshape(60000, -1)
            X_val = X_train[40000:50000]
            X_train = X_train[:40000]
            y_val = y_train[40000:50000]
            y_train = y_train[:40000]
        print(X_train.shape)

    elif dataset == 'MNIST_fr15_mix':
        data1 = (
            mnist.read_image_file(os.path.join(file_path, 'train-images-idx3-ubyte')),
            mnist.read_label_file(os.path.join(file_path, 'train-labels-idx1-ubyte'))
        )

        X_train1 = data1[0]
        y_train1 = data1[1]

        data2 = pd.read_csv('E:\\ML\\Hiwi\\data\\MNIST\\raw\\mnist_cd_fr15.csv', sep=',', header=None)

        X_train2 = data2.iloc[:, 1:]
        y_train2 = data2.iloc[:, 0]
        X_train2 = np.array(X_train2)
        y_train2 = np.array(y_train2)

        if type == 'eval':
            X_train1 = X_train1.reshape(60000, -1)
            X_train2 = X_train2.reshape(60000, -1)
            X_val = X_train2[50000:]
            X_train = X_train1[:40000]
            y_val = y_train2[50000:]
            y_train = y_train1[:40000]

        if type == 'test':
            X_train1 = X_train1.reshape(60000, -1)
            X_train2 = X_train2.reshape(60000, -1)
            X_val = X_train2[40000:50000]
            X_train = X_train1[:40000]
            y_val = y_train2[40000:50000]
            y_train = y_train1[:40000]

    elif dataset == 'MNIST_cd_fr5':
        data = pd.read_csv('E:\\ML\\Hiwi\\data\\MNIST\\raw\\mnist_cd_fr5.csv', sep=',', header=None)
        X_train = data.iloc[:, 1:]
        y_train = data.iloc[:, 0]
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        if type == 'eval':
            X_train = X_train.reshape(60000, -1)
            X_val = X_train[50000:]
            X_train = X_train[:40000]
            y_val = y_train[50000:]
            y_train = y_train[:40000]

        if type == 'test':
            X_train = X_train.reshape(60000, -1)
            X_val = X_train[40000:50000]
            X_train = X_train[:40000]
            y_val = y_train[40000:50000]
            y_train = y_train[:40000]
        print(X_train.shape)

    elif dataset == 'MNIST_fr5_mix':
        data1 = (
            mnist.read_image_file(os.path.join(file_path, 'train-images-idx3-ubyte')),
            mnist.read_label_file(os.path.join(file_path, 'train-labels-idx1-ubyte'))
        )

        X_train1 = data1[0]
        y_train1 = data1[1]

        data2 = pd.read_csv('E:\\ML\\Hiwi\\data\\MNIST\\raw\\mnist_cd_fr5.csv', sep=',', header=None)

        X_train2 = data2.iloc[:, 1:]
        y_train2 = data2.iloc[:, 0]
        X_train2 = np.array(X_train2)
        y_train2 = np.array(y_train2)

        if type == 'eval':
            X_train1 = X_train1.reshape(60000, -1)
            X_train2 = X_train2.reshape(60000, -1)
            X_val = X_train2[50000:]
            X_train = X_train1[:40000]
            y_val = y_train2[50000:]
            y_train = y_train1[:40000]

        if type == 'test':
            X_train1 = X_train1.reshape(60000, -1)
            X_train2 = X_train2.reshape(60000, -1)
            X_val = X_train2[40000:50000]
            X_train = X_train1[:40000]
            y_val = y_train2[40000:50000]
            y_train = y_train1[:40000]


    # elif dataset == 'MNIST_cd_dist':
    #     data = pd.read_csv('E:\\ML\\Hiwi\\data\\MNIST\\raw\\mnist_train_cd_DIST.csv', sep=',', header=None)
    #     X_train = data.iloc[:, 1:]
    #     y_train = data.iloc[:, 0]
    #     X_train = np.array(X_train)
    #     y_train = np.array(y_train)
    #
    #     if type == 'eval':
    #         X_train = X_train.reshape(60000, -1)
    #         X_val = X_train[50000:]
    #         X_train = X_train[:40000]
    #         y_val = y_train[50000:]
    #         y_train = y_train[:40000]
    #
    #     if type == 'test':
    #         X_train = X_train.reshape(60000, -1)
    #         X_val = X_train[40000:50000]
    #         X_train = X_train[:40000]
    #         y_val = y_train[40000:50000]
    #         y_train = y_train[:40000]
    #
    # elif dataset == 'MNIST_mix_dist':
    #     data1 = (
    #         mnist.read_image_file(os.path.join(file_path, 'train-images-idx3-ubyte')),
    #         mnist.read_label_file(os.path.join(file_path, 'train-labels-idx1-ubyte'))
    #     )
    #
    #     X_train1 = data1[0]
    #     y_train1 = data1[1]
    #
    #     data2 = pd.read_csv('E:\\ML\\Hiwi\\data\\MNIST\\raw\\mnist_train_cd_DIST.csv', sep=',', header=None)
    #
    #     X_train2 = data2.iloc[:, 1:]
    #     y_train2 = data2.iloc[:, 0]
    #     X_train2 = np.array(X_train2)
    #     y_train2 = np.array(y_train2)
    #
    #     if type == 'eval':
    #         X_train1 = X_train1.reshape(60000, -1)
    #         X_train2 = X_train2.reshape(60000, -1)
    #         X_val = X_train2[50000:]
    #         X_train = X_train1[:40000]
    #         y_val = y_train2[50000:]
    #         y_train = y_train1[:40000]
    #
    #     if type == 'test':
    #         X_train1 = X_train1.reshape(60000, -1)
    #         X_train2 = X_train2.reshape(60000, -1)
    #         X_val = X_train2[40000:50000]
    #         X_train = X_train1[:40000]
    #         y_val = y_train2[40000:50000]
    #         y_train = y_train1[:40000]
    #
    # elif dataset == 'MNIST_cd_tdist':
    #     data = pd.read_csv('E:\\ML\\Hiwi\\data\\MNIST\\raw\\mnist_train_cd_tDIST.csv', sep=',', header=None)
    #     X_train = data.iloc[:, 1:]
    #     y_train = data.iloc[:, 0]
    #     X_train = np.array(X_train)
    #     y_train = np.array(y_train)
    #
    #     if type == 'eval':
    #         X_train = X_train.reshape(60000, -1)
    #         X_val = X_train[50000:]
    #         X_train = X_train[:40000]
    #         y_val = y_train[50000:]
    #         y_train = y_train[:40000]
    #
    #     if type == 'test':
    #         X_train = X_train.reshape(60000, -1)
    #         X_val = X_train[40000:50000]
    #         X_train = X_train[:40000]
    #         y_val = y_train[40000:50000]
    #         y_train = y_train[:40000]
    #
    # elif dataset == 'MNIST_mix_tdist':
    #     data1 = (
    #         mnist.read_image_file(os.path.join(file_path, 'train-images-idx3-ubyte')),
    #         mnist.read_label_file(os.path.join(file_path, 'train-labels-idx1-ubyte'))
    #     )
    #
    #     X_train1 = data1[0]
    #     y_train1 = data1[1]
    #
    #     data2 = pd.read_csv('E:\\ML\\Hiwi\\data\\MNIST\\raw\\mnist_train_cd_tDIST.csv', sep=',', header=None)
    #
    #     X_train2 = data2.iloc[:, 1:]
    #     y_train2 = data2.iloc[:, 0]
    #     X_train2 = np.array(X_train2)
    #     y_train2 = np.array(y_train2)
    #
    #     if type == 'eval':
    #         X_train1 = X_train1.reshape(60000, -1)
    #         X_train2 = X_train2.reshape(60000, -1)
    #         X_val = X_train2[50000:]
    #         X_train = X_train1[:40000]
    #         y_val = y_train2[50000:]
    #         y_train = y_train1[:40000]
    #
    #     if type == 'test':
    #         X_train1 = X_train1.reshape(60000, -1)
    #         X_train2 = X_train2.reshape(60000, -1)
    #         X_val = X_train2[40000:50000]
    #         X_train = X_train1[:40000]
    #         y_val = y_train2[40000:50000]
    #         y_train = y_train1[:40000]
    #
    # elif dataset == 'MNIST_cd_dist_n0':
    #     data = pd.read_csv('E:\\ML\\Hiwi\\data\\MNIST\\raw\\mnist_train_cd_DIST_n0.csv', sep=',', header=None)
    #     X_train = data.iloc[:, 1:]
    #     y_train = data.iloc[:, 0]
    #     X_train = np.array(X_train)
    #     y_train = np.array(y_train)
    #
    #     if type == 'eval':
    #         X_train = X_train.reshape(60000, -1)
    #         X_val = X_train[50000:]
    #         X_train = X_train[:40000]
    #         y_val = y_train[50000:]
    #         y_train = y_train[:40000]
    #
    #     if type == 'test':
    #         X_train = X_train.reshape(60000, -1)
    #         X_val = X_train[40000:50000]
    #         X_train = X_train[:40000]
    #         y_val = y_train[40000:50000]
    #         y_train = y_train[:40000]
    #
    # elif dataset == 'MNIST_mix_dist_n0':
    #     data1 = (
    #         mnist.read_image_file(os.path.join(file_path, 'train-images-idx3-ubyte')),
    #         mnist.read_label_file(os.path.join(file_path, 'train-labels-idx1-ubyte'))
    #     )
    #
    #     X_train1 = data1[0]
    #     y_train1 = data1[1]
    #
    #     data2 = pd.read_csv('E:\\ML\\Hiwi\\data\\MNIST\\raw\\mnist_train_cd_DIST_n0.csv', sep=',', header=None)
    #
    #     X_train2 = data2.iloc[:, 1:]
    #     y_train2 = data2.iloc[:, 0]
    #     X_train2 = np.array(X_train2)
    #     y_train2 = np.array(y_train2)
    #
    #     if type == 'eval':
    #         X_train1 = X_train1.reshape(60000, -1)
    #         X_train2 = X_train2.reshape(60000, -1)
    #         X_val = X_train2[50000:]
    #         X_train = X_train1[:40000]
    #         y_val = y_train2[50000:]
    #         y_train = y_train1[:40000]
    #
    #     if type == 'test':
    #         X_train1 = X_train1.reshape(60000, -1)
    #         X_train2 = X_train2.reshape(60000, -1)
    #         X_val = X_train2[40000:50000]
    #         X_train = X_train1[:40000]
    #         y_val = y_train2[40000:50000]
    #         y_train = y_train1[:40000]
    #
    # elif dataset == 'MNIST_cd_dist_wo0':
    #     data = pd.read_csv('E:\\ML\\Hiwi\\data\\MNIST\\raw\\mnist_train_cd_DIST_wo0.csv', sep=',', header=None)
    #     X_train = data.iloc[:, 1:]
    #     y_train = data.iloc[:, 0]
    #     X_train = np.array(X_train)
    #     y_train = np.array(y_train)
    #
    #     if type == 'eval':
    #         X_train = X_train.reshape(60000, -1)
    #         X_val = X_train[50000:]
    #         X_train = X_train[:40000]
    #         y_val = y_train[50000:]
    #         y_train = y_train[:40000]
    #
    #     if type == 'test':
    #         X_train = X_train.reshape(60000, -1)
    #         X_val = X_train[40000:50000]
    #         X_train = X_train[:40000]
    #         y_val = y_train[40000:50000]
    #         y_train = y_train[:40000]
    #
    # elif dataset == 'MNIST_mix_dist_wo0':
    #     data1 = (
    #         mnist.read_image_file(os.path.join(file_path, 'train-images-idx3-ubyte')),
    #         mnist.read_label_file(os.path.join(file_path, 'train-labels-idx1-ubyte'))
    #     )
    #
    #     X_train1 = data1[0]
    #     y_train1 = data1[1]
    #
    #     data2 = pd.read_csv('E:\\ML\\Hiwi\\data\\MNIST\\raw\\mnist_train_cd_DIST_wo0.csv', sep=',', header=None)
    #
    #     X_train2 = data2.iloc[:, 1:]
    #     y_train2 = data2.iloc[:, 0]
    #     X_train2 = np.array(X_train2)
    #     y_train2 = np.array(y_train2)
    #
    #     if type == 'eval':
    #         X_train1 = X_train1.reshape(60000, -1)
    #         X_train2 = X_train2.reshape(60000, -1)
    #         X_val = X_train2[50000:]
    #         X_train = X_train1[:40000]
    #         y_val = y_train2[50000:]
    #         y_train = y_train1[:40000]
    #
    #     if type == 'test':
    #         X_train1 = X_train1.reshape(60000, -1)
    #         X_train2 = X_train2.reshape(60000, -1)
    #         X_val = X_train2[40000:50000]
    #         X_train = X_train1[:40000]
    #         y_val = y_train2[40000:50000]
    #         y_train = y_train1[:40000]
    # print(X_train.shape)



    # if machine_num == 0:
    #     # Load data for training
    #     X_train = np.load(os.path.join(file_path + datasets[0], 'train.npy'), allow_pickle=True)
    #     y_train = np.load(os.path.join(file_path + datasets[0], 'train_label.npy'), allow_pickle=True)
    #     # Load data for testing
    #     X_test = np.load(os.path.join(file_path + datasets[0], 'test.npy'), allow_pickle=True)
    #     y_test = np.load(os.path.join(file_path + datasets[0], 'test_label.npy'), allow_pickle=True)
    #     # Load data for validating
    #     X_val = np.load(os.path.join(file_path + datasets[0], 'val.npy'), allow_pickle=True)
    #     y_val = np.load(os.path.join(file_path + datasets[0], 'val_label.npy'), allow_pickle=True)
    #
    #     return X_train, y_train, X_val, y_val, X_test, y_test
    #
    # # Load data for training
    # X_train = np.load(
    #     os.path.join(file_path + datasets[datasets_num], datasets[datasets_num] + str(machine_num) + '.npy'),
    #     allow_pickle=True)
    # y_train = np.load(
    #     os.path.join(file_path + datasets[datasets_num], datasets[datasets_num] + '_label' + str(machine_num) + '.npy'),
    #     allow_pickle=True)
    #
    # # Load data for validating
    # X_val = np.load(os.path.join(file_path + datasets[0], 'val.npy'), allow_pickle=True)
    # y_val = np.load(os.path.join(file_path + datasets[0], 'val_label.npy'), allow_pickle=True)

    return X_train, y_train, X_val, y_val


def load_data_eval(machine_num, datasets_num):
    datasets = ['alldata', 'nddata', 'ovdata', 'unddata']
    file_path = 'E:\\ML\\Hiwi\\data\\'

    if machine_num == 0:
        # Load data for training
        X_train = np.load(os.path.join(file_path + datasets[0], 'train.npy'), allow_pickle=True)
        y_train = np.load(os.path.join(file_path + datasets[0], 'train_label.npy'), allow_pickle=True)
        # Load data for testing
        X_test = np.load(os.path.join(file_path + datasets[0], 'test.npy'), allow_pickle=True)
        y_test = np.load(os.path.join(file_path + datasets[0], 'test_label.npy'), allow_pickle=True)
        # Load data for validating
        X_val = np.load(os.path.join(file_path + datasets[0], 'val.npy'), allow_pickle=True)
        y_val = np.load(os.path.join(file_path + datasets[0], 'val_label.npy'), allow_pickle=True)

        # Load data for testing
        # X_test_s = np.load(os.path.join(file_path+datasets[0],'test_s.npy'),allow_pickle=True)
        # y_test_s = np.load(os.path.join(file_path+datasets[0],'test_label_s.npy'),allow_pickle=True)

        # Load data for all
        # X = np.load(os.path.join(file_path+datasets[0],'all.npy'),allow_pickle=True)
        # y = np.load(os.path.join(file_path+datasets[0],'all_label.npy'),allow_pickle=True)

        return X_train, y_train, X_val, y_val, X_test, y_test

    # Load data for training
    X_train = np.load(
        os.path.join(file_path + datasets[datasets_num], datasets[datasets_num] + str(machine_num) + '.npy'),
        allow_pickle=True)
    y_train = np.load(os.path.join(file_path + datasets[datasets_num],
                                   datasets[datasets_num] + '_label' + str(machine_num) + '.npy'), allow_pickle=True)

    # Load data for testing
    X_test = np.load(os.path.join(file_path + datasets[0], 'test.npy'), allow_pickle=True)
    y_test = np.load(os.path.join(file_path + datasets[0], 'test_label.npy'), allow_pickle=True)

    # Load data for validating
    X_val = np.load(os.path.join(file_path + datasets[0], 'val.npy'), allow_pickle=True)
    y_val = np.load(os.path.join(file_path + datasets[0], 'val_label.npy'), allow_pickle=True)

    # Load data for testing
    # X_test_s = np.load(os.path.join(file_path+datasets[0],'test_s.npy'),allow_pickle=True)
    # y_test_s = np.load(os.path.join(file_path+datasets[0],'test_label_s.npy'),allow_pickle=True)

    # Load data for all
    # X = np.load(os.path.join(file_path+datasets[0],'all.npy'),allow_pickle=True)
    # y = np.load(os.path.join(file_path+datasets[0],'all_label.npy'),allow_pickle=True)

    return X_train, y_train, X_val, y_val, X_test, y_test
