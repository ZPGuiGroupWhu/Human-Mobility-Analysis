import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp
import math
from model.utils.io import read, write
from model.preprocessing.split_dataset import split_random
from model.trait.features_in_TTS import features_in_TTS_sampling_half
from model.trait.scoring import scoring_sampling_half

plt.rcParams.update({'font.sans-serif': 'Arial'})
plt.rcParams.update({'font.size': 7})


def concat_experiment_data(input_dir, output_dir):
    score_train_all_experiment = pd.DataFrame()
    score_test_all_experiment = pd.DataFrame()
    abs_dif_all_experiment = pd.DataFrame()
    for root, dirs, files in os.walk(input_dir):
        if files:
            experiment_name = root.split('\\')[-1]
            print(experiment_name)
            train = read(os.path.join(root, root.split('\\')[-1] + '_train_item_and_trait_scores.csv'))
            test = read(os.path.join(root, root.split('\\')[-1] + '_test_item_and_trait_scores.csv'))
            abs_dif = _calculate_abs_difference_all(train, test)

            train.insert(1, 'experiment_name', experiment_name)
            test.insert(1, 'experiment_name', experiment_name)
            score_train_all_experiment = score_train_all_experiment.append(train)
            score_test_all_experiment = score_test_all_experiment.append(test)
            abs_dif.insert(1, 'experiment_name', experiment_name)
            abs_dif_all_experiment = abs_dif_all_experiment.append(abs_dif)
    # Output the merged train, test and dif
    score_train_all_experiment.sort_values(by=['user_id', 'experiment_name'], inplace=True)
    score_test_all_experiment.sort_values(by=['user_id', 'experiment_name'], inplace=True)
    write(os.path.join(output_dir, 'train_scores_all_experiment.csv'), score_train_all_experiment)
    write(os.path.join(output_dir, 'test_scores_all_experiment.csv'), score_test_all_experiment)

    write(os.path.join(output_dir, 'scores_abs_difference.csv'), abs_dif_all_experiment)


def _calculate_absolute_difference(item1, item2):
    return (item1 - item2).abs().tolist()


def _calculate_abs_difference_all(scale1, scale2):
    scale1_scale2 = pd.merge(scale1, scale2, on='user_id')
    dif_columns_name = scale1.columns.tolist()
    dif_column_num = len(dif_columns_name) - 1
    dif_all = pd.DataFrame(columns=dif_columns_name)
    dif_all['user_id'] = scale1_scale2['user_id']
    scale1_scale2.set_index("user_id", inplace=True)
    for i in range(dif_column_num):
        dif_all.iloc[:, i + 1] = _calculate_absolute_difference(scale1_scale2.iloc[:, i],
                                                                scale1_scale2.iloc[:, i + dif_column_num])
    return dif_all


def _merge_data_and_label(data, label):
    sub_data = data[['user_id', 'experiment_name', 'extroversion', 'openness', 'neuroticism', 'conscientiousness']]
    sub_label = label[
        ['user_id', 'extroversion_label', 'openness_label', 'neuroticism_label', 'conscientiousness_label']]
    sub_data_label = pd.merge(sub_data, sub_label, on='user_id', how='left')
    return sub_data_label


def merge_data_and_label(label_dir, output_dir):
    label = read(os.path.join(label_dir, 'item_and_trait_labels.csv'))

    dif = read(os.path.join(output_dir, 'scores_abs_difference.csv'))
    ocean_dif_label = _merge_data_and_label(dif, label)
    write(os.path.join(output_dir, 'difference_label.csv'), ocean_dif_label)

    train = read(os.path.join(output_dir, 'train_scores_all_experiment.csv'))
    train_label = _merge_data_and_label(train, label)
    write(os.path.join(output_dir, 'train_label.csv'), train_label)

    test = read(os.path.join(output_dir, 'test_scores_all_experiment.csv'))
    test_label = _merge_data_and_label(test, label)
    write(os.path.join(output_dir, 'test_label.csv'), test_label)


def scatter_score(output_dir):
    # data preparation
    x_score = read(os.path.join(output_dir, 'train_label.csv'))
    y_score = read(os.path.join(output_dir, 'test_label.csv'))
    x_name_list = ['Extroversion(train)', 'Openness(train)', 'Neuroticism(train)', 'Conscientiousness(train)']
    y_name_list = ['Extroversion(test)', 'Openness(test)', 'Neuroticism(test)', 'Conscientiousness(test)']
    label_name_list = ['Extroversion label', 'Openness label', 'Neuroticism label', 'Conscientiousness label']
    x_score = x_score.rename(columns={'extroversion': x_name_list[0], 'openness': x_name_list[1],
                                      'neuroticism': x_name_list[2], 'conscientiousness': x_name_list[3]})
    x_score = x_score.rename(columns={'extroversion_label': label_name_list[0], 'openness_label': label_name_list[1],
                                      'neuroticism_label': label_name_list[2],
                                      'conscientiousness_label': label_name_list[3]})
    y_score = y_score.rename(columns={'extroversion': y_name_list[0], 'openness': y_name_list[1],
                                      'neuroticism': y_name_list[2], 'conscientiousness': y_name_list[3]})
    data = pd.merge(x_score, y_score, on=['user_id', 'experiment_name'])

    # plot scatter
    if len(data) < 200:
        dot_size = 5
        dot_alpha = 0.5
    else:
        dot_size = 1
        dot_alpha = 0.1

    fig, axs = plt.subplots(1, 4, figsize=(8, 2.1))
    for i in range(4):
        axs[i].set_xlim(left=0, right=8)
        axs[i].set_ylim(bottom=0, top=8)
        axs[i].set_xticks(np.arange(0, 8.1))
        axs[i].set_yticks(np.arange(0, 8.1))
        axs[i].tick_params(width=0.5)
        for _, s in axs[i].spines.items():
            s.set_linewidth(0.5)
        sns.scatterplot(data=data, x=x_name_list[i], y=y_name_list[i], hue=label_name_list[i], ax=axs[i],
                        palette=['tab:red', 'tab:orange', 'tab:blue'],
                        hue_order=['high', 'medium', 'low'],
                        legend=False,
                        s=dot_size,
                        linewidth=0,
                        alpha=dot_alpha
                        )
        axs[i].grid(alpha=0.5, linewidth=0.5)
        # axs[i].legend(loc='upper left', title=label_name_list[i])
        plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatter.tiff'), dpi=300)


def kdeplot_difference(output_dir):
    def _kdeplot_difference_trait(ocean_dif_label, trait, ax):
        origin_trait = trait
        trait = "Absolute difference in " + origin_trait
        ocean_dif_label = ocean_dif_label.rename(columns={origin_trait: trait})
        for _, s in ax.spines.items():
            s.set_linewidth(0.5)
        ax.set_xlabel("Absolute difference in\n " + origin_trait)
        ax.tick_params(width=0.5)
        sns.despine()
        sns.kdeplot(data=ocean_dif_label, x=trait, label='all', color='tab:gray', shade=True, ax=ax, cut=0, linewidth=1)
        sns.kdeplot(data=ocean_dif_label[ocean_dif_label[origin_trait + '_label'] == 'high'], x=trait, label='high',
                    color='tab:red', ax=ax, cut=0, linewidth=1)
        sns.kdeplot(data=ocean_dif_label[ocean_dif_label[origin_trait + '_label'] == 'medium'], x=trait,
                    label='medium', color='tab:orange', ax=ax, cut=0, linewidth=1)
        sns.kdeplot(data=ocean_dif_label[ocean_dif_label[origin_trait + '_label'] == 'low'], x=trait, label='low',
                    color='tab:blue', ax=ax, cut=0, linewidth=1)
        plt.tight_layout()
        # ax.legend(labels=['high', 'medium', 'low', 'all'])
        return True

    difference_label = read(os.path.join(output_dir, 'difference_label.csv'))
    if len(difference_label) < 2:
        print(
            'The experimental data is less than 2, cannot draw the density distribution of the absolute difference between two sets of scores. Please try increasing subjects or sampling times.')
        return False
    fig, axs = plt.subplots(1, 4, figsize=(8, 2), sharex=True)
    _kdeplot_difference_trait(difference_label, 'extroversion', axs[0])
    _kdeplot_difference_trait(difference_label, 'openness', axs[1])
    _kdeplot_difference_trait(difference_label, 'neuroticism', axs[2])
    _kdeplot_difference_trait(difference_label, 'conscientiousness', axs[3])
    # plt.show()
    plt.savefig(os.path.join(output_dir, 'density_distribution.tiff'), dpi=300)


def _std(data):
    return data.std()


def _ttest_zero(difference):
    return ttest_1samp(difference.tolist(), 0)
    # return ztest(difference.tolist(), value=0)


def _calculate_consistency(difference):
    return difference.abs().mean()


def _calculate_by_label(data, method):
    res_by_label = pd.DataFrame(index=['high', 'medium', 'low'])
    res_by_label['extroversion'] = data.groupby('extroversion_label')['extroversion'].apply(
        method)
    res_by_label['openness'] = data.groupby('openness_label')['openness'].apply(
        method)
    res_by_label['neuroticism'] = data.groupby('neuroticism_label')['neuroticism'].apply(
        method)
    res_by_label['conscientiousness'] = data.groupby('conscientiousness_label')['conscientiousness'].apply(
        method)
    res_by_label = res_by_label.reindex(res_by_label.index.rename('label'))

    res_by_label.loc['all'] = [method(data['extroversion']),
                               method(data['openness']),
                               method(data['neuroticism']),
                               method(data['conscientiousness'])]
    res_by_label.reset_index(inplace=True)
    res_by_label.fillna('no data', inplace=True)
    return res_by_label


def calculate_consistence(output_dir):
    dif = read(os.path.join(output_dir, 'difference_label.csv'))
    if len(dif) < 2:
        print(
            'The experimental data is less than 2, cannot calculate the standardized β. Please try increasing subjects or sampling times.')
        return False
    train = read(os.path.join(output_dir, 'train_label.csv'))
    test = read(os.path.join(output_dir, 'test_label.csv'))
    ttest = _calculate_by_label(dif, _ttest_zero)
    ttest_t = ttest.applymap(lambda x: 'no data' if x == 'no data' else x[0])
    ttest_p = ttest.applymap(lambda x: 'no data' if x == 'no data' else x[1])
    ttest_t['label'] = ttest['label']
    ttest_p['label'] = ttest['label']
    cons = _calculate_by_label(dif, _calculate_consistency)
    std_train = _calculate_by_label(train, _std)
    std_test = _calculate_by_label(test, _std)
    standa_cons = pd.DataFrame(columns=cons.columns, index=[0, 1, 2, 3])
    for i in range(cons.shape[1]):
        for j in range(cons.shape[0]):
            if i == 0:
                standa_cons.iat[j, i] = cons.iat[j, i]
            else:
                if cons.iat[j, i] == 'no data':
                    standa_cons.iat[j, i] = 'no data'
                else:
                    standa_cons.iat[j, i] = cons.iat[j, i] / ((std_train.iat[j, i] + std_test.iat[j, i]) / 2)

    df1 = pd.merge(ttest_t, ttest_p, on='label', suffixes=("_t", "_p"))
    df2 = pd.merge(df1, cons, on='label', suffixes=("_t", "_cons"))
    df3 = pd.merge(df2, standa_cons, on='label', suffixes=("", "_stand_cons"))

    def format_series(x):
        for i, v in x.iteritems():
            if isinstance(v, str):
                x[i] = x[i]
            else:
                x[i] = '{:.3f}'.format(x[i])
        return x

    df3 = df3.apply(lambda x: format_series(x))
    write(os.path.join(output_dir, 'consistency_result.csv'), df3)


def data_split_half_reliability(INPUT_PATH=r"./result/L_with_driving_behavior",
                                OUTPUT_PATH=r'./result/data_split_half_reliability',
                                LABEL_PATH=r'./result/trajectory_profiles',
                                **kwargs):
    if os.path.exists(OUTPUT_PATH):
        shutil.rmtree(OUTPUT_PATH)
    os.makedirs(OUTPUT_PATH)

    # split_data
    split_data_path = os.path.join(OUTPUT_PATH, 'split_data')
    if not os.path.exists(split_data_path):
        os.makedirs(split_data_path)
    split_random(INPUT_PATH, split_data_path, **kwargs)

    # calculate features in TTS
    features_path = os.path.join(OUTPUT_PATH, 'split_data_features_and_trait_scores')
    if not os.path.exists(features_path):
        os.makedirs(features_path)
    features_in_TTS_sampling_half(INPUT_PATH=split_data_path, MAP_PATH=r'./model/feature/auxiliary_data',
                                  OUTPUT_PATH=features_path)

    # scoring
    scoring_sampling_half(INPUT_PATH=features_path)

    # data_split_half_reliability
    concat_experiment_data(input_dir=features_path, output_dir=OUTPUT_PATH)
    merge_data_and_label(label_dir=LABEL_PATH, output_dir=OUTPUT_PATH)
    # The relation between the two sets of scores in four trajectory_profiles
    scatter_score(output_dir=OUTPUT_PATH)
    # The density distribution of the absolute difference between two sets of scores in four trajectory_profiles for different scorer groups
    kdeplot_difference(output_dir=OUTPUT_PATH)
    # calculate β coefficient and standardized β
    calculate_consistence(output_dir=OUTPUT_PATH)


if __name__ == '__main__':
    input_path = r"../../result/L_with_driving_behavior"
    output_path = r'../../result/data_split_half_reliability'
    label_path = r'../../result/trajectory_profiles'
    data_split_half_reliability(INPUT_PATH=input_path, OUTPUT_PATH=output_path, LABEL_PATH=label_path,
                                sampling_times=10)
