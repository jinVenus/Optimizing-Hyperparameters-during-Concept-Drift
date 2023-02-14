# Tis file is used to train the single node using the local data
# The result is used as the base line
# Each node will have its own hyper-parameter.

import os
import subprocess
import csv
import time
import sys
import getopt

def main(argv):
    folder_name = 'mlp_mnist_continue'
    method_name = 'ngp-nei'
    num_nodes = 4
    num_params = 5
    node_id = 0
    datasets_number = 1
    # parsed by string
    # params by nodes
    # in the end is the weights (number equals to the num_nodes)
    # different parameters are seperated by _
    hyper_parameters = ''

    try:
        opts, args = getopt.getopt(argv, "hd:i:o:s:n:", ["nodes=", "hyperp", "csvfname", "datasets", "node"])
    except getopt.GetoptError:
        print (
            'main.py -d <id of nodes> -i <string for all the given params, including weights> -o <csv file name> -s <training datasets>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print (
                'call-ml-b.py -d <id of nodes> -i <string for all the given params, including weights> -o <csv file name> -s <training datasets>')
            sys.exit()
        elif opt in ("-d", "--nodes"):
            machine_num = int(arg)
        elif opt in ("-i", "--hyperp"):
            hyper_parameters = str(arg)
        elif opt in ("-o", "--csvfname"):
            method_name = str(arg)
        elif opt in ("-s", "--datasets"):
            datasets_number = int(arg)
        elif opt in ("-n", "--node"):
            node_id = int(arg)

    hp_params = []

    result_name = '/home/junjie/modes/botorch/' + folder_name + '/base-line/results-' + method_name + '-dataset-' + str(datasets_number) + '-node-' + str(node_id) + '.csv'

    with open(result_name, 'w') as csvFile:
        writer = csv.writer(csvFile)
    csvFile.close()

    for i in range (1, 11):
        hp_file_name = 'hp-' + method_name + '-dataset-' + str(datasets_number) + '-node-' + str(node_id) + '-trail' + str(i) + '.csv'
        # print (hp_file_name)
        with open(hp_file_name, 'r') as csvfile:
            hp_temp = [row for row in csv.reader(csvfile)]
        csvfile.close()

        predicted_temp = hp_temp[0][0].split()
        if (predicted_temp[0] == '[['):
            predicted_temp.pop(0)
        else:
            predicted_temp[0] = predicted_temp[0].replace('[', '0')
        predicted_temp[-1] = predicted_temp[-1].replace(']', '0')


        observed_temp = hp_temp[1][0].split()
        if (observed_temp[0] == '[['):
            observed_temp.pop(0)
        else:
            observed_temp[0] = observed_temp[0].replace('[', '0')

        observed_temp[-1] = observed_temp[-1].replace(']', '0')

        predicted_hp = ''
        observed_hp = ''

        current_time = time.time()

        remote_path = '/home/junjie/modes/botorch/'
        local_path = '/home/junjie/modes/botorch/'

        csv_predicted_name = remote_path + folder_name + '/base-line/predicted_results_' + str(method_name) + '-node-' + str(node_id) + '_trial_' + str(i) + '_' + str(current_time) + '.csv'
        csv_predicted_name_local = local_path + folder_name + '/base-line/predicted_results_' + str(method_name) + '-node-' + str(node_id) + '_trial_' + str(i) + '_' + str(current_time) + '.csv'

        csv_observed_name = remote_path + folder_name + '/base-line/observed_results_' + str(method_name) + '-node-' + str(node_id) + '_trial_' + str(i) + '_' + str(current_time) + '.csv'
        csv_observed_name_local = local_path + folder_name + '/base-line/observed_results_' + str(method_name) + '-node-' + str(node_id) + '_trial_' + str(i) + '_' + str(current_time) + '.csv'

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



        commands_predicted = 'ssh junjie@slave1 python3 /home/junjie/modes/botorch/' + folder_name + '/eval-ml-idv.py' + ' -i ' + str(predicted_hp) + ' -o ' + str(csv_predicted_name) + ' -d ' + str(node_id) + ' -s ' + str(datasets_number) + ' &'
        commands_observed = 'ssh junjie@slave1 python3 /home/junjie/modes/botorch/' + folder_name + '/eval-ml-idv.py' + ' -i ' + str(observed_hp) + ' -o ' + str(csv_observed_name) + ' -d ' + str(node_id) + ' -s ' + str(datasets_number) + ' &'

        os.system(commands_predicted)
        os.system(commands_observed)

        finished = 0
        finished_2 = 0
        while (finished == 0):
            with open(csv_predicted_name_local, 'r') as csvfile:
                data = [row for row in csv.reader(csvfile)]
            csvfile.close()
            # print('current finished set 1: ', len(data))
            if (len(data) < 1):
                time.sleep(10)
            else:
                finished = 1


        while (finished_2 == 0):
            with open(csv_observed_name_local, 'r') as csvfile:
                data = [row for row in csv.reader(csvfile)]
            csvfile.close()
            # print('current finished set 2: ', len(data))
            if (len(data) < 1):
                time.sleep(10)
            else:
                finished_2 = 1


        with open(csv_predicted_name_local, 'r') as csvfile:
            predicted_results_temp = [row for row in csv.reader(csvfile)]
        csvfile.close()

        with open(csv_observed_name_local, 'r') as csvfile:
            observed_results_temp = [row for row in csv.reader(csvfile)]
        csvfile.close()

        temp_results_1 = float(predicted_results_temp[0][0])
        temp_results_2 = float(observed_results_temp[0][0])

        with open(result_name, 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow([str(float(temp_results_1)), str(float(temp_results_2))])
        csvFile.close()


if __name__ == "__main__":
    main(sys.argv[1:])
