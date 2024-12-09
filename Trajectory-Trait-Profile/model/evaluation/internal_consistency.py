import shutil

import pandas as pd
import os
import pingouin as pg
from model.utils.io import read, write


def a(data):
    """
    :param data: dataframe
    :return: alpha
    pg.cronbach_alpha(data=df, ci=.99)
    (0.7734375, array([0.062, 0.962]))(alpha,(confidence interval))
    """
    alpha = pg.cronbach_alpha(data)
    return [str(float('{:.3f}'.format(i))) for i in [alpha[0], alpha[1][0], alpha[1][1]]]


def _is_competent(data):
    if len(data) > 1:
        return True
    print('Calculating internal_consistency requests '
          'subjects must be more than 1.')
    return False


def internal_consistency(INPUT_PATH=r'./result/trajectory_profiles', OUTPUT_PATH=r'./result/internal_consistency'):
    if os.path.exists(OUTPUT_PATH):
        shutil.rmtree(OUTPUT_PATH)
    os.makedirs(OUTPUT_PATH)

    items_score = read(os.path.join(INPUT_PATH, 'item_and_trait_scores.csv'))

    if _is_competent(items_score):
        e = a(items_score.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8]].copy())
        o = a(items_score.iloc[:, [9, 10, 11, 12, 13, 14, 15, 16]].copy())
        n = a(items_score.iloc[:, [17, 18, 19, 20, 21, 22, 23, 24]].copy())
        c = a(items_score.iloc[:, [25, 26, 27, 28, 29, 30, 31, 32]].copy())
        a_result = pd.DataFrame(columns=['name', 'a', 'Lower limit', 'Upper limit'])
        a_result.loc[len(a_result)] = ['Extroversion'] + e
        a_result.loc[len(a_result)] = ['Openness'] + o
        a_result.loc[len(a_result)] = ['Neuroticism'] + n
        a_result.loc[len(a_result)] = ['Conscientiousness'] + c
        write(os.path.join(OUTPUT_PATH, 'a.csv'), a_result)


if __name__ == '__main__':
    input_path = r'../../data/trajectory_profiles'
    output_path = r'../../result/internal_consistency'
    # input_path = r'../../result/trajectory_profiles'
    internal_consistency(input_path, output_path)
