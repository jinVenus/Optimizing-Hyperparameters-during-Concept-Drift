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
from Hiwi.src.algorithms.dynamic_dataloader import dynamic_generator
from botorch.optim import optimize_acqf

import random

from gpytorch.utils import warnings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
folder_name = 'rf_mnist_continue'


def call_external(X, num_trial, num_param, dataset):
    num_node = 1
    dataset = int(dataset)
    array_results = []
    hp = X.cpu().numpy()
    for i in range(0, num_trial):
        current_time = time.time()
        # csv_file_name = 'E:\\ML\\Hiwi\\data\\' + folder_name + str(
        #     dataset) + '_resluts_' + str(
        #     current_time) + '.csv'
        csv_file_name_2 = 'E:\\ML\\Hiwi\\data\\' + folder_name + str(
            dataset) + '_resluts_' + str(
            current_time) + '.csv'

        with open(csv_file_name_2, 'w') as csvFile:
            writer = csv.writer(csvFile)
        csvFile.close()

        hp_param = ''
        for j in range(0, num_param):
            hp_param = hp_param + str(hp[i][j])
            if (j < num_param - 1):
                hp_param = hp_param + '_'
        print(hp_param)

        commands = 'python E:\\ML\\Hiwi\\src\\algorithms\\botorch_modes\\rf\\call_ml_idv.py' + ' -i ' + str(
            hp_param) + ' -o ' + str(csv_file_name_2) + ' -s ' + str(dataset) + ' &'
        os.system(commands)
        time.sleep(2)
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

        # sorted_results = sorted(data_result, key=lambda x: x[1])
        #sorted_results = sorted(data_result)

        # # weights
        # weights = []
        # for j in range(num_param, num_param + num_node):
        #     weights.append(hp[i][j])

        # for j in range(0, num_node):
        #     # print ('weights: ', weights)
        #     # print ('sorted results: ', sorted_results)
        sum_result = float(data_result[0][0])

        os.remove(csv_file_name_2)

        result = sum_result

        array_results.append(result)

    tensor_results = torch.tensor(array_results, device=device, dtype=dtype)

    return tensor_results


def generate_initial_data(dataset):
    # generate training data
    train_x = generate_rand(30)
    exact_obj = call_external(train_x, 30, 7, dataset)
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
bounds = torch.tensor(
    [[0, 0, 0, 2, 1, 0, 0], [149.99, 2.99, 40.99, 20.99, 20.99, 1.99, 1.99]],
    device=device, dtype=dtype)


def generate_rand(sets):
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
    new_x, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=BATCH_SIZE,
        num_restarts=10,
        raw_samples=512,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
    )
    # observe new values

    exact_obj = call_external(new_x, BATCH_SIZE, 7, dataset).unsqueeze(-1)  # add output dimension
    new_obj = exact_obj
    return new_x, new_obj


# warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
# warnings.filterwarnings('ignore', category=RuntimeWarning)

N_TRIALS = 1
N_BATCH = 320
MC_SAMPLES = 256
slide = 20


def main(argv):
    dataset = 3
    try:
        opts, args = getopt.getopt(argv, "hd:", ["dataset="])
    except getopt.GetoptError:
        print('random parallel with input dataset')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('random parallel with input dataset')
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

        # print(train_x_ei)
        # print(train_obj_ei)

        i = 0
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

            # 每次算完HP变数据集
            # 画折线图（横轴 datasets）来评价

            if count == 20:

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

                csv_file_name = 'E:\\ML\\Hiwi\\results\\' + 'hp_gp_ei_rf_dataset_' + str(
                    dataset) + '_trail' + str(trial) + '_win' + str(win) + '.csv'

                with open(csv_file_name, 'w') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerow(
                        [str(argmax_pmean_ei.cpu().numpy()), str(max_pmean_ei.cpu().numpy())])  # ei prediction
                    writer.writerow([str(train_best_x_ei), str(best_tensor_ei.cpu().numpy())])  # ei observation

                csvFile.close()

                i += 1
                win += 1


                train_x_ei = train_x_ei[slide:]
                train_obj_ei = train_obj_ei[slide:]

                dynamic_generator(0, 0, 0, i)

                print(train_x_ei)
                print(train_obj_ei)

                count = 0

        # # return the best configuration
        # best_tensor_ei, indices_ei = torch.max(train_obj_ei, 0)
        # train_best_x_ei = train_x_ei[indices_ei].cpu().numpy()
        #
        # from botorch.acquisition import PosteriorMean
        #
        # argmax_pmean_ei, max_pmean_ei = optimize_acqf(
        #     acq_function=PosteriorMean(model_ei),
        #     bounds=bounds,
        #     q=1,
        #     num_restarts=20,
        #     raw_samples=2048,
        # )
        #
        # csv_file_name = 'E:\\ML\\Hiwi\\results\\' + 'hp_gp_ei_dataset_' + str(
        #     dataset) + '_trail' + str(trial) + '.csv'
        #
        # with open(csv_file_name, 'w') as csvFile:
        #     writer = csv.writer(csvFile)
        #     writer.writerow([str(argmax_pmean_ei.cpu().numpy()), str(max_pmean_ei.cpu().numpy())])  # ei prediction
        #     writer.writerow([str(train_best_x_ei), str(best_tensor_ei.cpu().numpy())])  # ei observation
        #
        # csvFile.close()


if __name__ == "__main__":
    #main(1)
    main(sys.argv[1:])

