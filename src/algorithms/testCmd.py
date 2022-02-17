import os
import time

# hp = [int(61.03794141), int(1.66359048), int(16.621841), int(14.05320706), int(13.98009677), int(1.00152747),
#    int(0.96022582)]
#
# local_path = 'E:\\ML\\Hiwi\\results\\'
# folder_name = 'rf_mnist_continue'
# current_time = time.time()
#
# hp_param = ''
# for j in range(0, 7):
#     hp_param = hp_param + str(hp[j])
#     if (j < 7 - 1):
#         hp_param = hp_param + '_'
# print(hp_param)
#
# csv_predicted_name_local = local_path + folder_name + '_predicted_results_' + 'trial_' + str(1) + '_' + str(
#             current_time) + '.csv'
#
# commands_predicted = 'python E:\\ML\\Hiwi\\src\\algorithms\\botorch_modes\\rf\\eval_ml_idv.py' + ' -i ' + str(
#                 hp_param) + ' -o ' + str(csv_predicted_name_local) + ' -s ' + str(1) + ' &'
#
# result = os.system(commands_predicted)
# print(result)

commands_predicted = 'python E:\\ML\\Hiwi\\src\\algorithms\\botorch_modes\\rf\\modes_b\\pure_gp_ei.py' + ' -d ' + str(5
                 ) + ' &'

result = os.system(commands_predicted)
print(result)