import os
from model.utils.io import read, write


def median_IQR(features):
    quantile = features.quantile([0.25, 0.5, 0.75])
    print(quantile)
    result = quantile.T
    result['IQR'] = result[0.75] - result[0.25]
    result = result[[0.5, 'IQR']]
    result = result.T
    print(result)
    return result


def calculate_scoring_parameters(INPUT_PATH):
    feature_values = read(os.path.join(INPUT_PATH, 'features_group.csv'))
    m_IQR = median_IQR(feature_values)
    write(os.path.join(INPUT_PATH, 'scoring_parameters.csv'), m_IQR)


if __name__ == '__main__':
    input_path = r'../../result/trajectory_profiles'
    calculate_scoring_parameters(input_path)
