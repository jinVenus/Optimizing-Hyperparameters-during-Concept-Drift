import os
import time

commands_predicted = 'python E:\\ML\\Hiwi\\src\\algorithms\\botorch_modes\\mlp\\modes_b\\pure_gp_ei_adwin_r.py' + ' -d '\
                     + str(2) + ' -n ' + str(100) + ' -p ' + str(0) + ' -m ' + 'random' + ' -s ' + str(1) + ' &'

result = os.system(commands_predicted)
print(result)