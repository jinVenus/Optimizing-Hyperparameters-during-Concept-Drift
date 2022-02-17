# bj mlp test
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import timeit
from Hiwi.src.algorithms.botorch_modes.rf.dataloader import load_data
#from dataloader import load_data

def mlp_model_test(hyper_vector,datasets_num):
	# pool of hypers
	layer = hyper_vector[0] + 1
	unit = hyper_vector[1]
	lay_unit = (unit,) * layer
	act = ['identity', 'logistic', 'tanh', 'relu']
	alh = hyper_vector[3]
	seed = 0
	ini = hyper_vector[4]

	# Build model 
	mlp = MLPClassifier(hidden_layer_sizes=lay_unit,
						max_iter=60, 
						alpha=alh, 
						activation = act[hyper_vector[2]],
                    	solver ='adam', 
                    	tol=1e-4, 
                    	learning_rate_init=ini,
                    	random_state=0,
                    	verbose=False)

	# Training and Validating
	X_train, y_train, X_val, y_val = load_data(0, 'eval')

	start = timeit.default_timer()
	mlp.fit(X_train, y_train)
	score_val = mlp.score(X_val, y_val)
	stop = timeit.default_timer()
	time = stop - start

	# Results
	#print('Training Time: ',stop - start)
	#print("Training set score: %f" % mlp.score(X_train, y_train))
	#print("Test set score: %f" % mlp.score(X_test, y_test))

	return time, score_val
