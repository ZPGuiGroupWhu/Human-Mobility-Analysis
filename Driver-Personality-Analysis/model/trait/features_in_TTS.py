import os
import shutil

from model.feature.feature import read_L, time_of_day_entropy, datetime_entropy, radius_of_gyration, \
    ratio_of_k_radius_of_gyration, K, random_entropy, location_entropy, sequence_entropy, od_entropy, trips_per_month, \
    trip_length, ratio_of_uninterested_trips, distance_from_home_entropy, ratio_stay_home_time, POI_features, \
    driving_behavior_features, sliding_window_entropy
import pandas as pd
from model.utils import constants
from model.utils.io import write
import geopandas as gpd


def calculate_features_individual(L, MAP_PATH):
    """
    calculate the features used as items in Trajectory Trait Scale of a single individual given their L.
    :param L: dataframe
        The dataframe of stay points sequence.
    :param MAP_PATH: str
        the dir storing the semantic map.
    :return: dict
        the features used as items in trajectory trait scale of the individual.
    """

    '''
    extroversion
    trips_per_month,trip_length,rg_time,rg_quantity,
    shopping_nbh,recreation_nbh,restaurant_nbh,
    ratio_stay_time_in_home
    '''
    userid = L[constants.USER][0]
    # trips per month
    average_trip_count = trips_per_month(L)
    # trip_length
    average_trip_length = trip_length(L)
    # radius of gyration
    rg_quantity = radius_of_gyration(L)
    rg_time = radius_of_gyration(L, method='time')
    # POI features

    semantic_map = gpd.read_file(os.path.join(MAP_PATH, 'fishnet_500.geojson'))
    semantics = POI_features(L, semantic_map)
    time_in_home = ratio_stay_home_time(L, start_night='01:00', end_night='06:00')
    '''
    openness
    ratio_of_uninterested_trips, rg4/rg_quantity, k_quantity,
    random_entropy, location_entropy, OD_entropy,sequence_entropy, distance_from_home_entropy
    '''
    # ratio of uninterested trips
    uninterested_trips_ratio = ratio_of_uninterested_trips(L, min_count=3)
    rg4_rg = ratio_of_k_radius_of_gyration(L, k=4)
    k_quantity = K(L)
    s = 90
    length = 7
    re = sliding_window_entropy(L, constants.CHECK_IN_TIME, s, length, random_entropy)
    le = sliding_window_entropy(L, constants.CHECK_IN_TIME, s, length, location_entropy)
    se = sliding_window_entropy(L, constants.CHECK_IN_TIME, s, length, sequence_entropy)
    hde = sliding_window_entropy(L, constants.CHECK_IN_TIME, s, length, distance_from_home_entropy, start_night='01:00',
                                 end_night='06:00')
    ode = sliding_window_entropy(L, constants.CHECK_IN_TIME, s, length, od_entropy)

    '''
    neuroticism
    speed_std_mean, speed_mean_std, 
    speed_std_max,acceleration_std_max,
    harsh_shift_ratio_std, harsh_steering_ratio_std, 
    harsh_shift_ratio_mean, harsh_steering_ratio_mean
    '''
    # driving behaviors
    user_opra = driving_behavior_features(L)

    '''
    conscientiousness
    speed_mean, speed_max,
    over_speed_ratio, over_speed_quantity,
    junction_over_speed, junction_speed_mean,
    day_entropy, datetime_entropy
    '''
    te = sliding_window_entropy(L, constants.CHECK_IN_TIME, s, length, time_of_day_entropy)
    dte = sliding_window_entropy(L, constants.CHECK_IN_TIME, s, length, datetime_entropy)

    scale_individual = {'user_id': userid,

                        'trips_per_month': average_trip_count,
                        'ratio_stay_time_in_home': time_in_home,
                        'trip_length': average_trip_length,
                        'rg_time': rg_time,
                        'rg_quantity': rg_quantity,
                        'shopping': semantics[0],
                        'recreation': semantics[1],
                        'restaurant': semantics[2],

                        'ratio_of_uninterested_trips': uninterested_trips_ratio,
                        'rg4/rg_quantity': rg4_rg,
                        'k_quantity': k_quantity,
                        'random_entropy': re,
                        'location_entropy': le,
                        'OD_entropy': ode,
                        'sequence_entropy': se,
                        'distance_from_home_entropy': hde,

                        'speed_std_mean': user_opra[0],
                        'speed_mean_std': user_opra[1],
                        'speed_std_max': user_opra[2],
                        'acceleration_std_max': user_opra[3],
                        'harsh_shift_ratio_std': user_opra[4],
                        'harsh_steering_ratio_std': user_opra[5],
                        'harsh_shift_ratio_mean': user_opra[6],
                        'harsh_steering_ratio_mean': user_opra[7],

                        'speed_mean': user_opra[8],
                        'speed_max': user_opra[9],
                        'over_speed_ratio': user_opra[10],
                        'over_speed_quantity': user_opra[11],
                        'junction_over_speed': user_opra[12],
                        'junction_speed_mean': user_opra[13],
                        'day_entropy': te,
                        'datetime_entropy': dte}
    print('the features of an individual are ok')
    return scale_individual


def features_in_TTS_sampling_half(INPUT_PATH=r"./result/L_with_driving_behavior",
                                  MAP_PATH='./model/feature/auxiliary_data', OUTPUT_PATH=r'./result/trajectory_profiles'):
    """
     calculate the features used as items in Trajectory Trait Scale of group given their sampled L (sampling ratio is 0.5).
    :param INPUT_PATH: str
        the dir path including two dir (train and test). in each dir, the files (.csv) of sampled stay points sequence (sub-L). Each file records the sub-L of an individual.
    :param MAP_PATH: str
        the dir storing the semantic map.
    :param OUTPUT_PATH: str
        the dir path used for output the features file (.csv).
    """
    for root, dirs, files in os.walk(INPUT_PATH):
        if files:
            output_dir = OUTPUT_PATH + './' + root.split('\\')[-2]
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            features = pd.DataFrame(columns=constants.SCALE_ORDER)
            for fname in files:
                print(fname)
                filename_input = os.path.join(root, fname)
                sp_df = read_L(filename_input)
                scale_individual = calculate_features_individual(sp_df, MAP_PATH)
                features = features.append(scale_individual, ignore_index=True)
            features_filename_output = root.split('\\')[-2] + '_' + root.split('\\')[-1] + '_features_group.csv'
            write(os.path.join(output_dir, features_filename_output), features)


def features_in_TTS_sampling(INPUT_PATH=r"./result/L_with_driving_behavior",
                             MAP_PATH='./model/feature/auxiliary_data', OUTPUT_PATH=r'./result/trajectory_profiles'):
    """
     calculate the features used as items in Trajectory Trait Scale of group given their sampled L.
    :param INPUT_PATH: str
        the dir path including two dir (train and test). in each dir, the files (.csv) of sampled stay points sequence (sub-L). Each file records the sub-L of an individual.
    :param MAP_PATH: str
        the dir storing the semantic map.
    :param OUTPUT_PATH: str
        the dir path used for output the features file (.csv).
    """
    for root, dirs, files in os.walk(INPUT_PATH):
        if files:
            output_root = OUTPUT_PATH + './' + 'random_sampling_result_' + root.split('\\')[-1] + '_100'

            output_path = output_root + './' + 'random_sampling_' + root.split('\\')[-1] + '_' + \
                          root.split('\\')[-2].split('_')[-1]
            print(output_path)
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            features = pd.DataFrame(columns=constants.SCALE_ORDER)
            for fname in files:
                print(fname)
                filename_input = os.path.join(root, fname)
                sp_df = read_L(filename_input)
                scale_individual = calculate_features_individual(sp_df, MAP_PATH)
                features = features.append(scale_individual, ignore_index=True)
            features_filename_output = 'random_sampling_' + root.split('\\')[-1] + '_' + \
                                       root.split('\\')[-2].split('_')[
                                           -1] + '_features_group.csv'
            print(features_filename_output)
            write(os.path.join(output_path, features_filename_output), features)


def features_in_TTS(INPUT_PATH=r"./result/L_with_driving_behavior", MAP_PATH='./model/feature/auxiliary_data',
                    OUTPUT_PATH=r'./result/trajectory_profiles'):
    """
     Calculate the features used as items in Trajectory Trait Scale of group given their sampled L.
    :param INPUT_PATH: str
    :param MAP_PATH: str
        The dir storing the semantic map.
    :param OUTPUT_PATH: str
        The dir path used for output the features file (.csv).
    """
    if os.path.exists(OUTPUT_PATH):
        shutil.rmtree(OUTPUT_PATH)
    os.makedirs(OUTPUT_PATH)

    for root, dirs, files in os.walk(INPUT_PATH):
        if files:
            features = pd.DataFrame(columns=constants.SCALE_ORDER)
            for fname in files:
                print(fname)
                filename_input = os.path.join(root, fname)
                sp_df = read_L(filename_input)
                features_individual = calculate_features_individual(sp_df, MAP_PATH)
                features = features.append(features_individual, ignore_index=True)
            features_filename_output = 'features_group.csv'
            write(os.path.join(OUTPUT_PATH, features_filename_output), features)


if __name__ == '__main__':
    input_path = r"../../result/L_with_driving_behavior"
    output_path = r'../../result/trajectory_profiles'
    map_path = r'../feature/auxiliary_data'
    features_in_TTS(input_path, map_path, output_path)
