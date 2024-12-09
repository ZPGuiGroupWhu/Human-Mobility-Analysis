import os
import pandas as pd
import numpy as np
from model.utils import constants
from model.utils.io import read, write
from model.trait.scoring_parameters import calculate_scoring_parameters

REVERSE_SCORING_LIST = [
    'ratio_stay_time_in_home',
    "rg4/rg_quantity",
    "speed_mean",
    "speed_max",
    "over_speed_ratio",
    "over_speed_quantity",
    "junction_over_speed",
    "junction_speed_mean",
    "day_entropy",
    "datetime_entropy"]


def _logistic_score(data, reverse_scoring=False, b=0.0, a=1.0):
    if reverse_scoring:
        result = data.apply(lambda x: 1 / (1 + np.exp(a * x - a * b)))
    else:
        result = data.apply(lambda x: 1 / (1 + np.exp(a * b - a * x)))
    return result


def logistic_scores(features, parameter):
    features.set_index("user_id", inplace=True)

    items_scores = pd.DataFrame(index=features.index)
    for column, data in features.items():
        if column in REVERSE_SCORING_LIST:
            bi = parameter[column][0]
            ai = 1 / parameter[column][1]
            items_scores[column] = _logistic_score(data, reverse_scoring=True, b=bi, a=ai)
        else:
            bi = parameter[column][0]
            ai = 1 / parameter[column][1]
            items_scores[column] = _logistic_score(data, reverse_scoring=False, b=bi, a=ai)
    items_scores = items_scores.reset_index()
    items_scores = items_scores[constants.SCALE_ORDER]
    return items_scores


def trait_scores(items_scores, method='sum'):
    ocean_score = pd.DataFrame(columns=["user_id", "extroversion", "openness", "neuroticism", "conscientiousness"])
    list_extroversion = ["trips_per_month",
                         'ratio_stay_time_in_home',
                         "trip_length",
                         "rg_time",
                         "rg_quantity",
                         'shopping',
                         'recreation',
                         'restaurant'
                         ]
    list_openness = ['ratio_of_uninterested_trips',
                     'rg4/rg_quantity',
                     'k_quantity',
                     'random_entropy',
                     'location_entropy',
                     'OD_entropy',
                     'sequence_entropy',
                     'distance_from_home_entropy']
    list_neuroticism = ["speed_std_mean",
                        'speed_mean_std',
                        "speed_std_max",
                        "acceleration_std_max",
                        "harsh_shift_ratio_std",
                        "harsh_steering_ratio_std",
                        "harsh_shift_ratio_mean",
                        "harsh_steering_ratio_mean"
                        ]
    list_conscientiousness = ["speed_mean",
                              "speed_max",
                              "over_speed_ratio",
                              "over_speed_quantity",
                              "junction_over_speed",
                              "junction_speed_mean",
                              "day_entropy",
                              "datetime_entropy"]
    if method == 'sum':
        ocean_score["extroversion"] = items_scores.loc[:, list_extroversion].sum(axis=1)
        ocean_score["openness"] = items_scores.loc[:, list_openness].sum(axis=1)
        ocean_score["neuroticism"] = items_scores.loc[:, list_neuroticism].sum(axis=1)
        ocean_score["conscientiousness"] = items_scores.loc[:, list_conscientiousness].sum(axis=1)
        ocean_score["user_id"] = items_scores["user_id"]
    elif method == 'mean':
        ocean_score["extroversion"] = items_scores.loc[:, list_extroversion].sum(axis=1) / len(list_extroversion)
        ocean_score["openness"] = items_scores.loc[:, list_openness].sum(axis=1) / len(list_openness)
        ocean_score["neuroticism"] = items_scores.loc[:, list_neuroticism].sum(axis=1) / len(list_neuroticism)
        ocean_score["conscientiousness"] = items_scores.loc[:, list_conscientiousness].sum(axis=1) / len(
            list_conscientiousness)
        ocean_score["user_id"] = items_scores["user_id"]
    return ocean_score


def scorer_group(item_score, method='predefined_parameters'):
    if method == 'predefined_parameters':
        parameters = read(r'./model/trait/auxiliary_data/label_parameters.csv')
        data = item_score.copy()
        data.set_index("user_id", inplace=True)
        for index, column in data.items():
            column_name = str(index) + "_label"
            quantile0 = parameters[index][0]
            quantile27 = parameters[index][1]
            quantile73 = parameters[index][2]
            quantile100 = parameters[index][3]
            a = [quantile0, quantile27, quantile73, quantile100]
            data[column_name] = pd.cut(column, bins=a, labels=['low', 'medium', 'high'])
        data = data[[x for x in data.columns if '_label' in x]]
        data = data.reset_index()
        return data

    if method == 'itself':
        data = item_score.copy()
        if len(item_score) < 2:
            print(
                "The subjects in this dataset is less than 2, so cannot divide them into three groups by "
                "itself. Please try setting method='predefined_parameters' to use predefined parameters.")
            return pd.DataFrame()
        data.set_index("user_id", inplace=True)
        for index, column in data.items():
            column_name = str(index) + "_label"
            quantile0 = column.quantile(0) - 0.01
            quantile27 = column.quantile(0.27)
            quantile73 = column.quantile(0.73)
            quantile100 = column.quantile(1) + 0.01
            a = [quantile0, quantile27, quantile73, quantile100]
            data[column_name] = pd.cut(column, bins=a, labels=['low', 'medium', 'high'])
        data = data[[x for x in data.columns if '_label' in x]]
        data = data.reset_index()
        return data


def logistic_trait_scores_and_label(features, scoring_parameters, grouping_method):
    items_scores = logistic_scores(features, scoring_parameters)
    sum_ocean_score = trait_scores(items_scores, method='sum')
    item_and_ocean_score = pd.merge(items_scores, sum_ocean_score, on='user_id')
    item_and_ocean_label = scorer_group(item_and_ocean_score, grouping_method)
    return item_and_ocean_score, item_and_ocean_label


def logistic_trait_scores(features, scoring_parameters):
    items_scores = logistic_scores(features, scoring_parameters)
    sum_ocean_score = trait_scores(items_scores, method='sum')
    item_and_ocean_score = pd.merge(items_scores, sum_ocean_score, on='user_id')
    return item_and_ocean_score


def scoring(INPUT_PATH=r'./result/trajectory_profiles', method='predefined_parameters'):
    """
    calculate item scores and trait scores, and divide high, medium, and low scorers according to trait scores.
    :param INPUT_PATH: str, optional
        The folder where the calculated features in TTS are stored.
    :param method: str, optional
        method='itself'. The median and IQR used for scoring, and the quantile used for dividing high, medium and low scorers are computed by the input dataset.
        method='predefined_parameters'. The median and IQR used for scoring, and the quantile used for dividing high, medium and low scorers are computed by the dataset of our study.
    """
    features_df = read(os.path.join(INPUT_PATH, 'features_group.csv'))
    parameters = None
    if method == 'itself':
        if len(features_df) < 2:
            print(
                "The subjects in this dataset is less than 2, so cannot calculate the scoring parameters "
                "by itself. Please try setting method='predefined_parameters' to use predefined parameters.")
            return False
        calculate_scoring_parameters(INPUT_PATH)
        parameters = read(os.path.join(INPUT_PATH, 'scoring_parameters.csv'))

    if method == 'predefined_parameters':
        parameters = read(r'./model/trait/auxiliary_data/scoring_parameters.csv')

    score, label = logistic_trait_scores_and_label(features_df, parameters, method)
    write(os.path.join(INPUT_PATH, 'item_and_trait_scores.csv'), score)
    write(os.path.join(INPUT_PATH, 'item_and_trait_labels.csv'),
          label)


def scoring_sampling_half(INPUT_PATH=r'./result/data_split_half_reliability/split_data_features_and_trait_scores',
                          method='predefined_parameters'):
    """
    calculate item scores and trait scores when trajectories are split half, used in calculate data split half reliability.
    :param INPUT_PATH: str, optional
        The folder where the calculated features in TTS are stored.
    :param method: str, optional
        method='itself'. The median and IQR used for scoring, and the quantile used for dividing high, medium and low scorers are computed by the input dataset.
        method='predefined_parameters'. The median and IQR used for scoring, and the quantile used for dividing high, medium and low scorers are computed by the dataset of our study.
    """
    parameters = None
    if method == 'itself':
        calculate_scoring_parameters(INPUT_PATH)
        parameters = read(os.path.join(INPUT_PATH, 'scoring_parameters.csv'))

    if method == 'predefined_parameters':
        parameters = read(r'./model/trait/auxiliary_data/scoring_parameters.csv')

    for root, dirs, files in os.walk(INPUT_PATH):
        if files:
            # features_df_train = read(os.path.join(root, root.split('\\')[-1] + '_train_features_group.csv'))
            # score_train = logistic_trait_scores(features_df_train, parameters)
            # write(os.path.join(root, root.split('\\')[-1] + '_train_item_and_trait_scores.csv'), score_train)

            features_df_test = read(os.path.join(root, root.split('\\')[-1] + '_test_features_group.csv'))
            score_test = logistic_trait_scores(features_df_test, parameters)
            write(os.path.join(root, root.split('\\')[-1] + '_test_item_and_trait_scores.csv'), score_test)

            features_df_train = read(os.path.join(root, root.split('\\')[-1] + '_train_features_group.csv'))
            score_train = logistic_trait_scores(features_df_train, parameters)
            write(os.path.join(root, root.split('\\')[-1] + '_train_item_and_trait_scores.csv'), score_train)


def scoring_sampling(INPUT_PATH=r'./result/trajectory_profiles', method='predefined_parameters'):
    parameters = None
    if method == 'itself':
        calculate_scoring_parameters(INPUT_PATH)
        parameters = read(os.path.join(INPUT_PATH, 'scoring_parameters.csv'))

    if method == 'predefined_parameters':
        parameters = read(r'./model/trait/auxiliary_data/scoring_parameters.csv')

    for root, dirs, files in os.walk(INPUT_PATH):
        if files:
            features_df = read(os.path.join(root, root.split('\\')[-1] + '_features_group.csv'))
            score = logistic_trait_scores(features_df, parameters)
            write(os.path.join(root, root.split('\\')[-1] + '_item_and_trait_scores.csv'), score)


if __name__ == '__main__':
    input_path = r"../../result/trajectory_profiles"
    scoring(input_path)
