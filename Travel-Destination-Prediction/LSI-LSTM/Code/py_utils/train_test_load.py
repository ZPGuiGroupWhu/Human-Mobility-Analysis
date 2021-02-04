import numpy as np
import json
from sklearn.model_selection import train_test_split, KFold
import random


def train_test_load(file_path, Cut_set=0.03333, K_fold=False):
    file = open(file_path, 'r')
    lines = file.readlines()
    # the key of line_dict
    traj_key = ["lngs", "lats", "travel_dis", "spd", "azimuth", "sem_pt"]
    attr_key = ["weekday", "destination", "dis_total", "start_time", "sem_O", "norm_dict"]
    dataset = []
    for line in lines:
        raw_dict = json.loads(line)
        raw_dict["cur_pt"] = [raw_dict["lngs"][-1], raw_dict["lats"][-1]]
        dataset.append(raw_dict)
    file.close()

    if not K_fold:
        dataset = trajectory_cut(dataset, attr_key, traj_key, Cut_set=Cut_set)
        train_set, eval_set = train_test_split(dataset, test_size=0.2, random_state=1)

        result = [train_set, eval_set]
    else:
        result = trajectory_cut(dataset, attr_key, traj_key, Cut_set=Cut_set)
        np.random.shuffle(result)

    return np.array(result), len(result[0]) + len(result[1])


def trajectory_cut(data, attr_key, traj_key, Cut_set):
    dataset = []
    # sub-trajectory product
    for raw_dict in data:

        # cut by seq_length #
        if 0 < Cut_set < 1:
            step = 1
            while step * Cut_set < 1:
                baby_dict = {}
                cut_length = int(len(raw_dict["lngs"]) * step * Cut_set)
                if cut_length == 0:
                    cut_length = 1
                for key in attr_key:
                    baby_dict[key] = raw_dict[key]
                for key in traj_key:
                    baby_dict[key] = raw_dict[key][:cut_length]
                baby_dict["cur_pt"] = [baby_dict["lngs"][-1], baby_dict["lats"][-1]]
                dataset.append(baby_dict)
                step += 1
        else:
            print("Error number of Cut_Proportion!")
    random.shuffle(dataset)

    return dataset
