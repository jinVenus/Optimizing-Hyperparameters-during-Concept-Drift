# bj mlp test
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import timeit
from Hiwi.src.algorithms.botorch_modes.boost.dataloader import load_data


# from dataloader import load_data

def adaboost_model_test(hyper_vector, datanum):
    # pool of hypers
    clf = ["DT", "mlp"]

    trees = hyper_vector[0] + 1
    split = hyper_vector[1]
    if (hyper_vector[2] == 0):
        depth = None
    else:
        depth = hyper_vector[2]
    leaf = hyper_vector[3]
    alg = ['SAMME', 'SAMME.R']
    lr = hyper_vector[5]

    # if hyper_vector[0] == 0:
    model = DecisionTreeClassifier(max_depth=depth, min_samples_split=split, min_samples_leaf=leaf)


    # Build model
    ada = AdaBoostClassifier(base_estimator=model,
                             n_estimators=trees,
                             algorithm=alg[hyper_vector[4]],
                             learning_rate=lr)

    # Training and Validating
    X_train, y_train, X_val, y_val = load_data(datanum, 'eval')
    start = timeit.default_timer()

    ada.fit(X_train, y_train)
    score_val = ada.score(X_val, y_val)
    stop = timeit.default_timer()
    time = stop - start

    # Results
    # print('Training Time: ',stop - start)
    # print("Training set score: %f" % mlp.score(X_train, y_train))
    # print("Test set score: %f" % mlp.score(X_test, y_test))

    return time, score_val
