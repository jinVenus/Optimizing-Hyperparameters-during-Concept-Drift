import numpy as np
import pickle


def Aj_Win_r(pre_fr, data):
    """
    Set the number of reused HP point for the dataset after concept drift.
    :param pre_fr: The fault rate before concept drift occurs.
    :param data: The data after concept drift occurs and the new fault rate need to be detected.
    :return: The number of reused HP points, the number of new HP points that needs to be mined and the
    current fault rate.
    """

    # load the fault rate detection model(in "rb" type)
    f = open('E:\\ML\\Hiwi\\models\\MLP\\detectfr.pkl', 'rb')
    s = f.read()
    model = pickle.loads(s)
    res = model.predict(data.iloc[50000:, :])

    # Calculate the detected fault rate
    curr_fr = int(round(np.mean(res)))
    print("pre_fr:" + str(pre_fr))
    print("predict_fr:" + str(curr_fr))

    # Allocated amount
    if abs(curr_fr - pre_fr) == 1:
        old = 25
        new = 100

    elif abs(curr_fr - pre_fr) == 2:
        old = 17
        new = 100

    elif abs(curr_fr - pre_fr) == 3:
        old = 10
        new = 100

    elif abs(curr_fr - pre_fr) == 4:
        old = 7
        new = 100

    elif abs(curr_fr - pre_fr) > 4:
        old = 5
        new = 100

    else:
        old = 29
        new = 1

    return old, new, curr_fr
