from model.preprocessing.trips import generate_trips
from model.preprocessing.stay_points_sequence import generate_L
from model.feature.driving_behavior import driving_behavior
from model.trait.features_in_TTS import features_in_TTS
from model.trait.scoring import scoring
from model.evaluation.item_discrimination import item_discrimination
from model.evaluation.internal_consistency import internal_consistency
from model.evaluation.data_split_half_reliability import data_split_half_reliability

# the result will be stored in './result'.
if __name__ == '__main__':
    # build trajectory profiles using the dataset in data/sample_trajectory
    # trajectory preprocessing
    # concat trajectory into trips
    generate_trips()
    print('concat trajectory into trips is done. \n ____________________________________')
    # cluster flameout points into stay points and form the stay point sequence L
    generate_L()
    print('detecting stay points is done. \n ____________________________________')

    # extract driving behavior features and merge them with L
    driving_behavior()
    print('extracting driving behavior features is done. \n ____________________________________')

    # calculate the features in Trajectory Trait Scale. If you need to calculate a single feature or features not
    # included in TTS, please call the function in feature.py.
    features_in_TTS()
    print('calculating the features in Trajectory Trait Scale (TTS) is done. \n ____________________________________')
    # calculate item scores and trait scores, and divide high, medium, and low scorers according to trait scores
    scoring(method='predefined_parameters')
    # method='itself'. The median and IQR used for scoring, and the quantile used for dividing high, medium and low scorers are computed by the input dataset
    # method='predefined_parameters'. The median and IQR used for scoring, and the quantile used for dividing high, medium and low scorers are computed by the dataset of our study
    print('calculating item scores and trait scores is done. \n ____________________________________')

    # evaluate the item discrimination and reliability of TTS
    # item discrimination. If you want to use the trajectory profiles in our paper for item discrimination analysis, please set the parameter INPUT_PATH='./data/trajectory_profiles'
    item_discrimination(INPUT_PATH='./data/trajectory_profiles/trajectory dataset D1')
    print('evaluating the item discrimination is done. \n ____________________________________')

    # internal consistency. If you want to use the trajectory profiles in our paper for calculating internal consistency, please set the parameter INPUT_PATH='./data/trajectory_profiles'
    internal_consistency(INPUT_PATH='./data/trajectory_profiles/trajectory dataset D1')
    print('evaluating the internal consistency is done. \n ____________________________________')

    # data split_half_reliability
    data_split_half_reliability(sampling_times=2)
    print('evaluating the internal consistency is done. \n ____________________________________')

