# bj mlp test
import numpy as np
import os

def load_data(machine_num, datasets_num):

	datasets = ['alldata', 'nddata', 'ovdata', 'unddata']
	file_path = '/home/jjshi/datasets/mnist/'

	if machine_num == 0:
		# Load data for training
		X_train = np.load(os.path.join(file_path + datasets[0],'train.npy'),allow_pickle=True)
		y_train = np.load(os.path.join(file_path + datasets[0],'train_label.npy'),allow_pickle=True)
		# Load data for testing
		X_test = np.load(os.path.join(file_path + datasets[0],'test.npy'),allow_pickle=True)
		y_test = np.load(os.path.join(file_path + datasets[0],'test_label.npy'),allow_pickle=True)
		# Load data for validating
		X_val = np.load(os.path.join(file_path + datasets[0],'val.npy'),allow_pickle=True)
		y_val = np.load(os.path.join(file_path + datasets[0],'val_label.npy'),allow_pickle=True)
		
		return X_train, y_train, X_val, y_val, X_test, y_test

	# Load data for training
	X_train = np.load(os.path.join(file_path + datasets[datasets_num],datasets[datasets_num]+str(machine_num)+'.npy'),allow_pickle=True)
	y_train = np.load(os.path.join(file_path + datasets[datasets_num],datasets[datasets_num]+'_label'+str(machine_num)+'.npy'),allow_pickle=True)

	# Load data for validating
	X_val = np.load(os.path.join(file_path + datasets[0],'val.npy'),allow_pickle=True)
	y_val = np.load(os.path.join(file_path + datasets[0],'val_label.npy'),allow_pickle=True)

	return X_train, y_train, X_val, y_val


def load_data_eval(machine_num, datasets_num):
	datasets = ['alldata', 'nddata', 'ovdata', 'unddata']
	file_path = '/home/jjshi/datasets/mnist/'

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

