import pandas as pd


def dynamic_generator_r(rate, s):
    """
    Generate different dataset with fault rate.
    :params i: fault rate.
    :param s: The step of concept drift for fault rate.
    """
    # fr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    # dynamic windows for different
    fr = [a for a in range(0, 16, s)]

    # path
    file_path = 'E:\\ML\\Hiwi\\data\\MNIST\\raw\\'

    file_name = 'E:\\ML\\Hiwi\\data\\MNIST\\raw\\mnist_dynamic.csv'

    # new data in the dynamic dataset
    data = pd.read_csv(file_path + 'mnist_cd_fr' + str(fr[rate]) + '.csv', sep=',', header=None)
    data.to_csv(file_name, index=False, header=False)