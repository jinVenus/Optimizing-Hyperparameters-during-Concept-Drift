import torch
import os
import csv
import sys
from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.exceptions import BadInitialCandidatesWarning
import time
import getopt

from botorch.models import SingleTaskGP, FixedNoiseGP, ModelListGP
from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.optim import optimize_acqf

import random

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
dtype = torch.double
folder_name = 'rf_mnist_continue'

def call_external(X, num_node, num_param, dataset):
    dataset = int(dataset)
    hp = X.cpu().numpy()
    current_time = time.time()
    csv_file_name = '/home/jjshi/modes/botorch/' + folder_name + '/modes-i/dataset' + str(dataset) + '_resluts_' + str(
        current_time) + '.csv'
    csv_file_name_2 = '/home/junjie/modes/botorch/' + folder_name + '/modes-i/dataset' + str(dataset) + '_resluts_' + str(
        current_time) + '.csv'

    with open(csv_file_name_2, 'w') as csvFile:
        writer = csv.writer(csvFile)
    csvFile.close()

    for i in range(0, num_node):
        node_id = int(i % 4)
        hp_param = ''
        for j in range(0, num_param):
            hp_param = hp_param + str(int(hp[i][j]))
            if (j < num_param - 1):
                hp_param = hp_param + '_'
        #print (hp_param)
        commands = 'ssh jjshi@ls12-srv0 python3 /home/jjshi/modes/botorch/' + folder_name + '/call-ml-idv.py' + ' -i ' + str(
            hp_param) + ' -o ' + str(csv_file_name) + ' -d ' + str(node_id) + ' -s ' + str(dataset) + ' &'
        os.system(commands)

    finished = 0

    while (finished == 0):
        with open(csv_file_name_2, 'r') as csvfile:
            data = [row for row in csv.reader(csvfile)]
        csvfile.close()
        #print('current finished set: ', len(data))
        if (len(data) < (num_node)):
            time.sleep(10)
        else:
            finished = 1

    with open(csv_file_name_2, 'r') as csvfile:
        data_result = [row for row in csv.reader(csvfile)]
    csvfile.close()

    sorted_results = sorted(data_result, key=lambda x: x[1])

    array_results = []
    for k in range(0, num_node):
        array_results.append(float(sorted_results[k][0]))

    tensor_results = torch.tensor(array_results, device=device, dtype=dtype)

    return tensor_results


def generate_initial_data(dataset):
    # generate training data
    train_x = generate_rand(4, 7)
    exact_obj = call_external(train_x, 4, 7, dataset)
    for i in range(0, 3):
        train_temp = generate_rand(4, 7)
        obj_temp = call_external(train_temp, 4, 7, dataset)
        train_x = torch.cat((train_x, train_temp), 0)
        exact_obj = torch.cat((exact_obj, obj_temp), 0)

    train_obj = exact_obj.unsqueeze(-1)
    best_tensor_value, indices = torch.max(exact_obj, 0)
    best_observed_value = best_tensor_value.item()
    current_best_config = exact_obj[indices].item()

    return train_x, train_obj, best_observed_value, current_best_config

    
def initialize_model(train_x, train_obj, state_dict=None):
    model = SingleTaskGP(train_x, train_obj)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    # load state dict if it is passed
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return mll, model

BATCH_SIZE = 4
bounds = torch.tensor([[0, 0, 0, 2, 1, 0, 0], [149.99, 2.99, 40.99, 20.99, 20.99, 1.99, 1.99]], device=device, dtype=dtype)


def generate_rand (sets, size):
    random_set = []

    for i in range(0, sets):
        data = []
        data.append(random.randint(0, 149))
        data.append(random.randint(0, 2))
        data.append(random.randint(0, 40))
        data.append(random.randint(2, 20))
        data.append(random.randint(1, 20))
        data.append(random.randint(0, 1))
        data.append(random.randint(0, 1))
        random_set.append(data)

    return torch.tensor(random_set, device=device, dtype=dtype)




def optimize_acqf_and_get_observation(acq_func, dataset):
    """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=BATCH_SIZE,
        num_restarts=10,
        raw_samples=512,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
    )
    # observe new values
    new_x = candidates.detach()
    exact_obj = call_external(new_x, 4, 7, dataset).unsqueeze(-1)  # add output dimension
    new_obj = exact_obj
    return new_x, new_obj


N_TRIALS = 10
N_BATCH = 25
MC_SAMPLES = 256
def main(argv):
    dataset = 1

    try:
        opts, args = getopt.getopt(argv, "hd:", ["dataset="])
    except getopt.GetoptError:
        print ('random parallel with input dataset')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('random parallel with input dataset')
            sys.exit()
        elif opt in ("-d", "--dataset"):
            dataset = int(arg)
    # average over multiple trials
    for trial in range(1, N_TRIALS + 1):

        print(f"\nTrial {trial:>2} of {N_TRIALS} ", end="")
        best_observed_ei, best_observed_nei = [], []

        # call helper functions to generate initial training data and initialize model
        train_x_ei, train_obj_ei, best_observed_value_ei, current_best_config = generate_initial_data(dataset)
        mll_ei, model_ei = initialize_model(train_x_ei, train_obj_ei)

        best_observed_ei.append(best_observed_value_ei)

        # run N_BATCH rounds of BayesOpt after the initial random batch
        for iteration in range(1, N_BATCH + 1):

            # fit the models
            fit_gpytorch_model(mll_ei)

            # define the qEI and qNEI acquisition modules using a QMC sampler
            qmc_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES)

            # for best_f, we use the best observed noisy values as an approximation
            qEI = qExpectedImprovement(
                model=model_ei,
                best_f=train_obj_ei.max(),
                sampler=qmc_sampler,
            )
            # optimize and get new observation
            new_x_ei, new_obj_ei = optimize_acqf_and_get_observation(qEI, dataset)

            # update training points
            train_x_ei = torch.cat([train_x_ei, new_x_ei])
            train_obj_ei = torch.cat([train_obj_ei, new_obj_ei])

            # update progress

            best_value_ei = train_obj_ei.max().item()
            best_observed_ei.append(best_value_ei)

            # reinitialize the models so they are ready for fitting on next iteration
            # use the current state dict to speed up fitting
            mll_ei, model_ei = initialize_model(
                train_x_ei,
                train_obj_ei,
                model_ei.state_dict(),
            )


        # return the best configuration
        best_tensor_ei, indices_ei = torch.max(train_obj_ei, 0)
        train_best_x_ei = train_x_ei[indices_ei].cpu().numpy()


        from botorch.acquisition import PosteriorMean

        argmax_pmean_ei, max_pmean_ei = optimize_acqf(
            acq_function=PosteriorMean(model_ei),
            bounds=bounds,
            q=1,
            num_restarts=20,
            raw_samples=2048,
        )

        csv_file_name = '/home/junjie/modes/botorch/' + folder_name + '/modes-i/hp-gp-qei-dataset-' + str(
            dataset) + '-trail' + str(trial) + '.csv'

        with open(csv_file_name, 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow([str(argmax_pmean_ei.cpu().numpy()), str(max_pmean_ei.cpu().numpy())]) # ei prediction
            writer.writerow([str(train_best_x_ei), str(best_tensor_ei.cpu().numpy())]) # ei observation

        csvFile.close()

if __name__ == "__main__":
    main(sys.argv[1:])


