import torch
import os
import csv
import sys
from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.acquisition.analytic import ExpectedImprovement, NoisyExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.exceptions import BadInitialCandidatesWarning
import time
import getopt

from botorch.models import SingleTaskGP, FixedNoiseGP, ModelListGP
from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.optim import optimize_acqf

import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
folder_name = 'rf_mnist_continue'

def call_external(X, num_trial, num_param, dataset, node_id):
    num_node = 1
    dataset = int(dataset)
    array_results = []
    hp = X.cpu().numpy()
    for i in range (0, num_trial):
        current_time = time.time()
        csv_file_name = '/home/jjshi/modes/botorch/' + folder_name + '/base-line/dataset' + str(dataset) + '_resluts_' + str(
        current_time) + '.csv'
        csv_file_name_2 = '/home/jjshi/modes/botorch/' + folder_name + '/base-line/dataset' + str(dataset) + '_resluts_' + str(
        current_time) + '.csv'

        with open(csv_file_name_2, 'w') as csvFile:
            writer = csv.writer(csvFile)
        csvFile.close()

        hp_param = ''
        for j in range(0, num_param):
            hp_param = hp_param + str(hp[i][j])
            if (j < num_param - 1):
                hp_param = hp_param + '_'
        #print (hp_param)
        commands = 'python3 /home/jjshi/modes/botorch/' + folder_name + '/call-ml-idv.py' + ' -i ' + str(
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

        result = data_result[0][0]

        array_results.append(float(result))

    tensor_results = torch.tensor(array_results, device=device, dtype=dtype)

    return tensor_results

NOISE_SE = 0.1
train_yvar = torch.tensor(NOISE_SE**2, device=device, dtype=dtype)

def generate_initial_data(dataset, node_id):
    # generate training data
    train_x = generate_rand(20, 7)
    exact_obj = call_external(train_x, 20, 7, dataset, node_id).unsqueeze(-1)
    train_obj = exact_obj + NOISE_SE * torch.randn_like(exact_obj)

    best_tensor_value, indices = torch.max(exact_obj, 0)
    best_observed_value = best_tensor_value.item()

    current_best_config = exact_obj[indices].item()

    return train_x, train_obj, best_observed_value, current_best_config


def initialize_model(train_x, train_obj, state_dict=None):
    model = FixedNoiseGP(train_x, train_obj, train_yvar.expand_as(train_obj)).to(train_x)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    # load state dict if it is passed
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return mll, model

BATCH_SIZE = 1
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



def optimize_acqf_and_get_observation(acq_func, dataset, node_id):
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
    exact_obj = call_external(new_x, BATCH_SIZE, 7, dataset, node_id).unsqueeze(-1)  # add output dimension
    new_obj = exact_obj + NOISE_SE * torch.randn_like(exact_obj)
    return new_x, new_obj

warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

N_TRIALS = 10
N_BATCH = 100
MC_SAMPLES = 256

def main(argv):
    dataset = 1
    node_id = 1
    try:
        opts, args = getopt.getopt(argv, "hd:n:", ["dataset=", "nodeid="])
    except getopt.GetoptError:
        print ('random parallel with input dataset')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('random parallel with input dataset')
            sys.exit()
        elif opt in ("-d", "--dataset"):
            dataset = int(arg)
        elif opt in ("-n", "--nodeid"):
            node_id = int(arg)
# average over multiple trials
    for trial in range(1, N_TRIALS + 1):

        print(f"\nTrial {trial:>2} of {N_TRIALS} ", end="")
        best_observed_ei, best_observed_nei = [], []

        # call helper functions to generate initial training data and initialize model
        train_x_ei, train_obj_ei, best_observed_value_ei, current_best_config = generate_initial_data(dataset, node_id)

        train_x_nei, train_obj_nei = train_x_ei, train_obj_ei
        best_observed_value_nei = best_observed_value_ei
        mll_nei, model_nei = initialize_model(train_x_nei, train_obj_nei)

        best_observed_nei.append(best_observed_value_nei)

        # run N_BATCH rounds of BayesOpt after the initial random batch
        for iteration in range(1, N_BATCH + 1):

            # fit the models
            fit_gpytorch_model(mll_nei)

            # define the qEI and qNEI acquisition modules using a QMC sampler
            qmc_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES)

            # for best_f, we use the best observed noisy values as an approximation

            NEI = NoisyExpectedImprovement(
                model=model_nei,
                X_observed=train_x_nei,
            )

            # optimize and get new observation
            new_x_nei, new_obj_nei = optimize_acqf_and_get_observation(NEI, dataset, node_id)

            # update training points

            train_x_nei = torch.cat([train_x_nei, new_x_nei])
            train_obj_nei = torch.cat([train_obj_nei, new_obj_nei])

            # update progress

            best_value_nei = train_obj_nei.max().item()
            best_observed_nei.append(best_value_nei)

            # reinitialize the models so they are ready for fitting on next iteration
            # use the current state dict to speed up fitting
            mll_nei, model_nei = initialize_model(
                train_x_nei,
                train_obj_nei,
                model_nei.state_dict(),
            )

        # return the best configuration

        best_tensor_nei, indices_nei = torch.max(train_obj_nei, 0)
        train_best_x_nei = train_x_nei[indices_nei].cpu().numpy()


        from botorch.acquisition import PosteriorMean


        argmax_pmean_nei, max_pmean_nei = optimize_acqf(
            acq_function=PosteriorMean(model_nei),
            bounds=bounds,
            q=1,
            num_restarts=20,
            raw_samples=2048,
        )

        csv_file_name = '/home/jjshi/modes/botorch/' + folder_name + '/base-line/hp-ngp-nei-dataset-' + str(
            dataset) + '-node-' + str(node_id) + '-trail' + str(trial) + '.csv'

        with open(csv_file_name, 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow([str(argmax_pmean_nei.cpu().numpy()), str(max_pmean_nei.cpu().numpy())])  # nei prediction
            writer.writerow([str(train_best_x_nei), str(best_tensor_nei.cpu().numpy())])  # nei observation


        csvFile.close()

if __name__ == "__main__":
    main(sys.argv[1:])
