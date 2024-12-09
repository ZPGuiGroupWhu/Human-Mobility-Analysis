import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
from scipy import stats
from model.utils import constants
from model.utils.io import read, write

plt.rcParams.update({'font.sans-serif': 'Arial'})
plt.rcParams.update({'font.size': 8})


def _plot_discrimination(data, title, color, output_path):
    fig, axs = plt.subplots(8, 1, figsize=(2.8, 10))
    for i in range(8):
        subdata = data.iloc[:, [i, -1]]
        subtitle = title[title['name'] == subdata.columns.values[0]].values.tolist()[0]
        subdata.columns = [subdata.columns.values[0], '']
        axs[i].set_xlim(left=0, right=1)
        axs[i].set_ylim(bottom=0, top=1)
        axs[i].set_xticks(np.arange(0, 1, 0.1))
        axs[i].set_yticks(np.arange(0, 1, 0.1))
        axs[i].xaxis.set_ticklabels([])

        if subtitle[2] < 0.001:
            p_str = r"$p<0.001$  "
        else:
            p_str = r"$p=%.3f$  " % subtitle[2]

        axs[i].set_title(r"$t=%.3f$  " % subtitle[1] + p_str + r"$d_s=%.3f$ " % subtitle[3],
                         fontsize=8)
        vp = sns.violinplot(y=subdata.columns.values[1], x=subdata.columns.values[0], order=["high", "low"],
                            data=subdata, ax=axs[i],
                            inner=None, saturation=0.5)
        vp.collections[0].set_edgecolor(color)
        vp.collections[0].set_facecolor(color)
        vp.collections[0].set_alpha(0.7)
        vp.collections[1].set_edgecolor(color)
        vp.collections[1].set_facecolor(color)
        vp.collections[1].set_alpha(0.3)
        bp = sns.boxplot(y=subdata.columns.values[1], x=subdata.columns.values[0], data=subdata, ax=axs[i],
                         width=0.05,
                         palette=[color, color],
                         fliersize=0,
                         linewidth=0.3,
                         order=["high", "low"]
                         )
        for line in bp.get_lines()[0:12]:
            line.set_color(color)
    fig.tight_layout()
    plt.savefig(os.path.join(output_path, data.columns.values[-1] + '.tiff'), dpi=300)


def _statistic_discrimination(data, output_path):
    ttest_result = pd.DataFrame(
        columns=['name', 'label', 'F', 'pvalue', 't', 'dof', 'tail', 'pvalue', 'CI95%', 'cohen-d'])
    simple_result = pd.DataFrame(
        columns=['name', 't', 'pvalue', 'cohen-d'])
    for i in range(8):
        subdata = data.iloc[:, [i, -1]]
        # 方差是否相等
        column_data = subdata.columns.values[0]
        column_label = subdata.columns.values[1]
        high = subdata[subdata[column_label] == 'high'][column_data]
        low = subdata[subdata[column_label] == 'low'][column_data]
        levene = stats.levene(high, low)
        ttest_equal_v = pg.ttest(high, low, correction=False).values.tolist()
        ttest_unequal_v = pg.ttest(high, low, correction=True).values.tolist()
        equal_v = [column_data, 'Equal variances assumed', levene[0], levene[1], ttest_equal_v[0][0],
                   ttest_equal_v[0][1], ttest_equal_v[0][2],
                   ttest_equal_v[0][3], ttest_equal_v[0][4], ttest_equal_v[0][5]]
        unequal_v = [column_data, 'Equal variances not assumed', '', '', ttest_unequal_v[0][0], ttest_unequal_v[0][1],
                     ttest_unequal_v[0][2], ttest_unequal_v[0][3],
                     ttest_unequal_v[0][4], ttest_unequal_v[0][5]]
        ttest_result.loc[len(ttest_result)] = equal_v
        ttest_result.loc[len(ttest_result)] = unequal_v

        ttest_simple = [column_data, ttest_unequal_v[0][0], ttest_unequal_v[0][3], ttest_unequal_v[0][5]]
        simple_result.loc[len(simple_result)] = ttest_simple
    # simple_result = simplt_result.T.reset_index()
    write(os.path.join(output_path, data.columns.values[-1] + '_simple.csv'), simple_result)
    write(os.path.join(output_path, data.columns.values[-1] + '.csv'), ttest_result)
    return simple_result


def item_discrimination_trait(data, trait_name, output_path):
    data.drop(data[data.iloc[:, -1] == 'medium'].index, inplace=True)
    stat_result = _statistic_discrimination(data, output_path)
    c1 = '#D55066'
    c2 = '#C39B09'
    c3 = '#6FB4C0'
    c4 = '#977CA7'
    colors = {'extroversion': c1, 'openness': c2, 'neuroticism': c3, 'conscientiousness': c4}
    _plot_discrimination(data, stat_result, colors[trait_name], output_path)


def _is_competent(label):
    if (len(label[label == 'high']) > 1) & (len(label[label == 'low']) > 1):
        return True
    print(label.name.split('_')[0],
          ': calculating item discrimination requests '
          'subjects both in high and low scorers must be more than 1')
    return False


def item_discrimination(INPUT_PATH=r'./result/trajectory_profiles', OUTPUT_PATH=r'./result/item_discrimination'):
    measure_group = read(os.path.join(INPUT_PATH, 'item_and_trait_scores.csv'))
    label_group = read(os.path.join(INPUT_PATH, 'item_and_trait_labels.csv'))

    if os.path.exists(OUTPUT_PATH):
        shutil.rmtree(OUTPUT_PATH)
    os.makedirs(OUTPUT_PATH)

    measure_label = measure_group.merge(label_group.iloc[:, [0, -4, -3, -2, -1]], on='user_id')
    # change column name to be the same as in the paper
    measure_label.rename(columns=constants.TRAIT_NAMES, inplace=True)
    # item discrimination analyze
    if _is_competent(label_group['extroversion_label']):
        item_discrimination_trait(measure_label.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, -4]].copy(), 'extroversion',
                                  OUTPUT_PATH)

    if _is_competent(label_group['openness_label']):
        item_discrimination_trait(measure_label.iloc[:, [9, 10, 11, 12, 13, 14, 15, 16, -3]].copy(), 'openness',
                                  OUTPUT_PATH)
    if _is_competent(label_group['neuroticism_label']):
        item_discrimination_trait(measure_label.iloc[:, [17, 18, 19, 20, 21, 22, 23, 24, -2]].copy(), 'neuroticism',
                                  OUTPUT_PATH)
    if _is_competent(label_group['conscientiousness_label']):
        item_discrimination_trait(measure_label.iloc[:, [25, 26, 27, 28, 29, 30, 31, 32, -1]].copy(),
                                  'conscientiousness',
                                  OUTPUT_PATH)


if __name__ == '__main__':
    # input_path = r'../../result/trajectory_profiles'
    output_path = r'../../result/item_discrimination'
    input_path = r'../../result/trajectory_profiles'
    item_discrimination(input_path, output_path)
