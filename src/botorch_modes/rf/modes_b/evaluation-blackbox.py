# Tis file is used to train the single node using the local data
# The result is used as the base line
# Each node will have its own hyper-parameter.

import os
import subprocess
import csv
import time
import sys
import getopt

from Hiwi.src.algorithms.dynamic_dataloader import dynamic_generator


def main(argv):

    for i in range(0, 4):
        dynamic_generator(0, 0, 0, i)
        folder_name = 'rf_mnist_cd_' + str(i)
        method_name = 'gp_ei_rf'
        num_nodes = 1
        win_num = 0
        fr = 15
        num_params = 7
        datasets_number = 13
        # parsed by string
        # params by nodes
        # in the end is the weights (number equals to the num_nodes)
        # different parameters are seperated by _
        hyper_parameters = ''

        # 40000 + 最后10000test

        # try:
        #     opts, args = getopt.getopt(argv, "hd:i:o:s:", ["nodes=", "hyperp", "csvfname", "datasets"])
        # except getopt.GetoptError:
        #     print(
        #         'main.py -d <id of nodes> -i <string for all the given params, including weights> -o <csv file name> -s <training datasets>')
        #     sys.exit(2)
        # for opt, arg in opts:
        #     if opt == '-h':
        #         print(
        #             'call-ml-b.py -d <id of nodes> -i <string for all the given params, including weights> -o <csv file name> -s <training datasets>')
        #         sys.exit()
        #     # elif opt in ("-d", "--nodes"):
        #     #     machine_num = int(arg)
        #     elif opt in ("-i", "--hyperp"):
        #         hyper_parameters = str(arg)
        #     elif opt in ("-o", "--csvfname"):
        #         method_name = str(arg)
        #     elif opt in ("-s", "--datasets"):
        #         datasets_number = int(arg)

        hp_params = []

        # result_name = 'E:\\ML\\Hiwi\\results\\' + folder_name + '_dataset_' + str(datasets_number) + '_win' + str(win_num) + '.csv'
        result_name = 'E:\\ML\\Hiwi\\results\\' + folder_name + '_dataset_' + str(datasets_number) + '.csv'

        with open(result_name, 'w') as csvFile:
            writer = csv.writer(csvFile)
        csvFile.close()

        for i in range(1, 2):
            # hp_file_name = 'E:\\ML\\Hiwi\\results\\' + 'hp_' + method_name + '_dataset_' + str(
            #     13) + '_trail' + str(i) + '_win' + str(win_num) + '.csv'


            hp_file_name = 'E:\\ML\\Hiwi\\results\\' + 'hp_' + method_name + '_dataset_' + str(
                13) + '_trail' + str(i) + '_cd' + str(i) + '.csv'
            with open(hp_file_name, 'r') as csvfile:
                hp_temp = [row for row in csv.reader(csvfile)]
            csvfile.close()

            predicted_temp = hp_temp[0][0].split()
            if (predicted_temp[0] == '[['):
                predicted_temp.pop(0)
            else:
                predicted_temp[0] = predicted_temp[0].replace('[', '0')
            predicted_temp[-1] = predicted_temp[-1].replace(']', '0')

            observed_temp = hp_temp[2][0].split()
            if (observed_temp[0] == '[['):
                observed_temp.pop(0)
            else:
                observed_temp[0] = observed_temp[0].replace('[', '0')

            observed_temp[-1] = observed_temp[-1].replace(']', '0')

            # read the weights
            # predicted_weights = []
            # observed_weights = []
            # for nd in range (num_params, num_params+num_nodes):
            #     predicted_weights.append(float(predicted_temp[nd]))
            #     observed_weights.append(float(observed_temp[nd]))

            predicted_hp = ''
            observed_hp = ''

            current_time = time.time()

            remote_path = '/home/jjshi/modes/botorch/'
            local_path = 'E:\\ML\\Hiwi\\results\\'

            # csv_predicted_name = remote_path + folder_name + '/modes-b/predicted_results_' + str(method_name) + '_trial_' + str(i) + '_' + str(current_time) + '.csv'
            csv_predicted_name_local = local_path + folder_name + '_predicted_results_' + 'trial_' + str(i) + '_' + str(
                current_time) + '.csv'

            # csv_observed_name = remote_path + folder_name + '/modes-b/observed_results_' + str(
            #     method_name) + '_trial_' + str(i) + '_' + str(current_time) + '.csv'
            csv_observed_name_local = local_path + folder_name + '_observed_results_' + 'trial_' + str(i) + '_' + str(
                current_time) + '.csv'

            with open(csv_predicted_name_local, 'w') as csvFile:
                writer = csv.writer(csvFile)
            csvFile.close()
            with open(csv_observed_name_local, 'w') as csvFile:
                writer = csv.writer(csvFile)
            csvFile.close()

            for j in range(0, num_params):
                predicted_hp = predicted_hp + str(predicted_temp[j])
                observed_hp = observed_hp + str(observed_temp[j])
                if (j < num_params - 1):
                    predicted_hp = predicted_hp + '_'
                    observed_hp = observed_hp + '_'

            commands_predicted = 'python E:\\ML\\Hiwi\\src\\algorithms\\botorch_modes\\rf\\eval_ml_idv.py' + ' -i ' + str(
                predicted_hp) + ' -o ' + str(csv_predicted_name_local) + ' -s ' + str(datasets_number) + ' &'
            commands_observed = 'python E:\\ML\\Hiwi\\src\\algorithms\\botorch_modes\\rf\\eval_ml_idv.py' + ' -i ' + str(
                observed_hp) + ' -o ' + str(csv_observed_name_local) + ' -s ' + str(datasets_number) + ' &'

            os.system(commands_predicted)
            os.system(commands_observed)

            finished = 0
            finished_2 = 0
            while (finished == 0):
                with open(csv_predicted_name_local, 'r') as csvfile:
                    data = [row for row in csv.reader(csvfile)]
                csvfile.close()
                # print('current finished set: ', len(data))
                if (len(data) < (num_nodes)):
                    time.sleep(10)
                else:
                    finished = 1

            while (finished_2 == 0):
                with open(csv_observed_name_local, 'r') as csvfile:
                    data = [row for row in csv.reader(csvfile)]
                csvfile.close()
                # print('current finished set: ', len(data))
                if (len(data) < (num_nodes)):
                    time.sleep(10)
                else:
                    finished_2 = 1

            with open(csv_predicted_name_local, 'r') as csvfile:
                predicted_results_temp = [row for row in csv.reader(csvfile)]
            csvfile.close()

            with open(csv_observed_name_local, 'r') as csvfile:
                observed_results_temp = [row for row in csv.reader(csvfile)]
            csvfile.close()

            temp_results_1 = 0
            temp_results_2 = 0

            # for k in range(0, num_nodes):
            #     temp_results_1 = float(predicted_results_temp[k][0]) * predicted_weights[int(predicted_results_temp[k][1])] + temp_results_1
            #     temp_results_2 = float(observed_results_temp[k][0]) * observed_weights[int(observed_results_temp[k][1])] + temp_results_2

            with open(result_name, 'a+') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow([str(float(predicted_results_temp[0][0])), str(float(observed_results_temp[0][0]))])
            csvFile.close()



if __name__ == "__main__":
    # main(sys.argv[1:])
    main(1)
