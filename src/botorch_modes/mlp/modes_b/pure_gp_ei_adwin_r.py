import torch
import os
import csv
import sys
from botorch import fit_gpytorch_model
from botorch.acquisition.analytic import ExpectedImprovement, NoisyExpectedImprovement
import time
import getopt
import pandas as pd

from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

from .src.algorithms.botorch_modes.mlp.slide_training_win import slide
from .src.algorithms.Adjust_window_size_r import Aj_Win_r
from .src.algorithms.dynamic_dataloader_r import dynamic_generator_r
from .optim import optimize_acqf

import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
folder_name = 'mlp_mnist_continue'


def call_external(X, num_trial, num_param, dataset):
    num_node = 1
    dataset = int(dataset)
    array_results = []
    hp = X.cpu().numpy()
    for i in range(0, num_trial):

        csv_file_name_2 = 'E:\\ML\\Hiwi\\data\\' + folder_name + str(
            dataset) + '_resluts' + '.csv'

        with open(csv_file_name_2, 'w') as csvFile:
            writer = csv.writer(csvFile)
        csvFile.close()

        hp_param = ''
        for j in range(0, num_param):
            hp_param = hp_param + str(hp[i][j])
            if (j < num_param - 1):
                hp_param = hp_param + '_'
        print(hp_param)

        commands = 'python ..\\src\\botorch_modes\\mlp\\call_ml_idv_r.py' + ' -i ' + str(
            hp_param) + ' -o ' + str(csv_file_name_2) + ' -s ' + str(dataset) + ' &'
        os.system(commands)
        time.sleep(10)
        finished = 0

        while (finished == 0):
            with open(csv_file_name_2, 'r') as csvfile:
                data = [row for row in csv.reader(csvfile)]
            csvfile.close()
            # print('current finished set: ', len(data))
            if (len(data) < (num_node)):
                time.sleep(10)
            else:
                finished = 1

        with open(csv_file_name_2, 'r') as csvfile:
            data_result = [row for row in csv.reader(csvfile)]
        csvfile.close()
        sum_result = float(data_result[0][0])

        result = sum_result
        print("result:" + str(result))
        array_results.append(result)

    tensor_results = torch.tensor(array_results, device=device, dtype=dtype)
    return tensor_results


def generate_initial_data(dataset, n):
    # generate training data
    train_x = generate_rand(n)
    exact_obj = call_external(train_x, n, 5, dataset)
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


BATCH_SIZE = 1
bounds = torch.tensor([[0, 10, 0, 2, 1], [14.99, 150.99, 3.99, 5, 4]], device=device, dtype=dtype)


def generate_rand(sets):
    random_set = []

    for i in range(0, sets):
        data = []
        data.append(random.randint(0, 14))
        data.append(random.randint(10, 150))
        data.append(random.randint(0, 3))
        data.append(random.uniform(2, 5))
        data.append(random.uniform(1, 4))

        random_set.append(data)

    return torch.tensor(random_set, device=device, dtype=dtype)


def optimize_acqf_and_get_observation(acq_func, dataset):
    """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""
    # optimize
    new_x, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=BATCH_SIZE,
        num_restarts=10,
        raw_samples=512,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
    )
    # observe new values

    exact_obj = call_external(new_x, BATCH_SIZE, 5, dataset).unsqueeze(-1)  # add output dimension
    new_obj = exact_obj
    return new_x, new_obj


def optimize_acqf_and_get_observation_with_existingX(train_x_ei, dataset):
    """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""
    # observe new values

    exact_obj = call_external(train_x_ei, BATCH_SIZE, 5, dataset).unsqueeze(-1)  # add output dimension
    new_obj = exact_obj
    return train_x_ei, new_obj


# warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
# warnings.filterwarnings('ignore', category=RuntimeWarning)

N_TRIALS = 7
N_BATCH = 5000
MC_SAMPLES = 256
new = 0
pre_fr = 0
mode = ''
step = 0
dataset = 0


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "hd:n:p:m:s:", ["dataset=",  "new=", "prefr=", "mode=", "step="])
    except getopt.GetoptError:
        print('random parallel with input dataset')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('random parallel with input dataset')
            sys.exit()
        elif opt in ("-d", "--dataset"):
            dataset = int(arg)
            print("dataset: " + str(arg))
        elif opt in ("-n", "--new"):
            print("new: " + str(arg))
            new = int(arg)
        elif opt in ("-p", "--prefr"):
            print("prefr: " + str(arg))
            pre_fr = int(arg)
        elif opt in ("-m", "--mode"):
            print("mode: " + str(arg))
            mode = str(arg)
        elif opt in ("-s", "--step"):
            print("step: " + str(arg))
            step = int(arg)
    # average over multiple trials
    for trial in range(1, N_TRIALS + 1):
        dynamic_generator_r(0)
        print(f"\nTrial {trial:>2} of {N_TRIALS} ", end="")
        best_observed_ei, best_observed_nei = [], []

        # call helper functions to generate initial training data and initialize model
        train_x_ei, train_obj_ei, best_observed_value_ei, current_best_config = generate_initial_data(dataset, n=30)
        mll_ei, model_ei = initialize_model(train_x_ei, train_obj_ei)

        best_observed_ei.append(best_observed_value_ei)

        fault_rate = 0
        count = 0
        win = 0

        # run N_BATCH rounds of BayesOpt after the initial random batch
        for iteration in range(1, N_BATCH + 1):
            # fit the models

            fit_gpytorch_model(mll_ei)

            # for best_f, we use the best observed noisy values as an approximation
            EI = ExpectedImprovement(
                model=model_ei,
                best_f=train_obj_ei.max(),
            )
            # optimize and get new observation
            new_x_ei, new_obj_ei = optimize_acqf_and_get_observation(EI, dataset)

            count += 1

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

            # When the number of newly found points reaches "new", the result is recorded
            # and the sliding window is started and concept drift occurs.
            if count == new:

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


                csv_file_name = '..\\results\\' + 'hp_gp_ei_mlp_dataset_' + str(
                    dataset) + '_trail' + str(trial) + '_' + str(mode) + 's' + str(step) + '_' + str(win) + 'test.csv'

                with open(csv_file_name, 'w') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerow(
                        [str(argmax_pmean_ei.cpu().numpy()), str(max_pmean_ei.cpu().numpy())])  # ei prediction
                    writer.writerow([str(train_best_x_ei), str(best_tensor_ei.cpu().numpy())])  # ei observation

                csvFile.close()

                # Update datasets' fault_rate and win_number
                fault_rate += step
                win += 1
                print("i=" + str(fault_rate))

                # Change dataset(Concept Drift occurs)
                if fault_rate <= 15:
                    dynamic_generator_r(fault_rate)
                else:
                    break

                # data = pd.read_csv('/home/jin/Hiwi/data/mnist_dynamic.csv', sep=',', header=None)
                data = pd.read_csv('E:\\ML\\Hiwi\\data\\MNIST\\raw\\mnist_dynamic.csv', sep=',', header=None)
                old, new, pfr = Aj_Win_r(pre_fr, data.iloc[:, 1:])
                pre_fr = pfr

                # Depending on the settings, select reused points and dropped or retrained points
                retrain_x, train_x_ei, train_obj_ei = slide(old, train_x_ei, train_obj_ei)

                # retain Hyperparameters with new dataset
                if mode == "retrain":
                    for X in retrain_x:
                        new_x, new_obj = optimize_acqf_and_get_observation_with_existingX(X, dataset)
                        train_x_ei = torch.cat([train_x_ei, new_x])
                        train_obj_ei = torch.cat([train_obj_ei, new_obj])

                        mll_ei, model_ei = initialize_model(
                            train_x_ei,
                            train_obj_ei,
                            model_ei.state_dict(),
                        )

                        fit_gpytorch_model(mll_ei)

                # Found new random points
                else:
                    for i in range(30 - old):
                        # for best_f, we use the best observed noisy values as an approximation
                        EI = ExpectedImprovement(
                            model=model_ei,
                            best_f=train_obj_ei.max(),
                        )
                        # optimize and get new observation
                        new_x_ei, new_obj_ei = optimize_acqf_and_get_observation(EI, dataset)

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

                        fit_gpytorch_model(mll_ei)

                print(train_x_ei)
                print(train_obj_ei)

                # reset
                count = 0


if __name__ == "__main__":
    main(sys.argv[1:])
