# Tis file is used to train the single node using the local data
# The result is used as the base line
# Each node will have its own hyper-parameter.

import os
import subprocess
import csv
import time
import sys
import getopt
from Hiwi.src.algorithms.botorch_modes.rf.rf_func import rf_model

# 40000 + 中间10000eval

def main(argv):

    num_params = 7
    datasets = 0
    # parsed by string
    # params by nodes
    # in the end is the weights (number equals to the num_nodes)
    # different parameters are seperated by _
    hyper_parameters = ''

    try:
        opts, args = getopt.getopt(argv, "hd:i:o:s:", ["nodes=", "hyperp", "csvfname", "datasets"])
    except getopt.GetoptError:
        print (
            'main.py -d <id of nodes> -i <string for all the given params, including weights> -o <csv file name> -s <training datasets>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print (
                'call_ml_b.py -d <id of nodes> -i <string for all the given params, including weights> -o <csv file name> -s <training datasets>')
            sys.exit()
        # elif opt in ("-d", "--nodes"):
        #     machine_num = int(arg)
        elif opt in ("-i", "--hyperp"):
            print(str(arg))
            hyper_parameters = str(arg)
        elif opt in ("-o", "--csvfname"):
            print(str(arg))
            csv_file_name = str(arg)
        elif opt in ("-s", "--datasets"):
            print(str(arg))
            datasets_number = int(arg)

    hp_params = []

    print('start callmlidv')
    # decompose the received hyper parameters
    temp_hp = hyper_parameters.split("_")
    for i in range(0, num_params):
        hp_params.append(int(float(temp_hp[i])))

    # train_time, val_acc = mlp_model(hypers, machine_num, datasets_number)
    train_time, val_acc = rf_model(hp_params, datasets_number)

    with open(csv_file_name, 'a+') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow([str(val_acc)])
    csvFile.close()

    print(train_time, val_acc)


if __name__ == "__main__":
    main(sys.argv[1:])
