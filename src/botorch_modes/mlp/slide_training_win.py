import torch


def slide(old, train_x_ei, train_obj_ei):
    """
    Update train_x_ei and train_obj_ei by using old.
    :param old: Number of points that need to be reused
    :param train_x_ei: Combination of hyperparameters found by MBO.
    :param train_obj_ei: The model results corresponding to each hyperparameter combination.
    :return retrain_x: The data to be retrained in "retrain" mode.
    :return train_x_ei: Retained data from train_x_ei.
    :return train_obj_ei: Retained data from train_obj_ei
    """
    if not old == 0:
        tend, sorted_indices = torch.sort(train_obj_ei, 0)
        train_x_ei = train_x_ei[sorted_indices]
        train_obj_ei = train_obj_ei[sorted_indices]
        tend = train_x_ei
        train_x_ei = train_x_ei[(-old):]
        train_obj_ei = train_obj_ei[(-old):]

        train_x_ei = torch.reshape(train_x_ei, (old, 5))
        train_obj_ei = torch.reshape(train_obj_ei, (old, 1))

        retrain_x = tend[-30:(-old)]

    else:
        tend, sorted_indices = torch.sort(train_obj_ei, 0)
        train_x_ei = train_x_ei[sorted_indices]
        train_obj_ei = train_obj_ei[sorted_indices]
        tend = train_x_ei
        train_x_ei = train_x_ei[-1:]
        train_obj_ei = train_obj_ei[-1:]

        train_x_ei = torch.reshape(train_x_ei, (1, 5))
        train_obj_ei = torch.reshape(train_obj_ei, (1, 1))

        retrain_x = tend[-30:]

    return retrain_x, train_x_ei, train_obj_ei
