from datetime import datetime
import os
import pandas as pd
import shutil
from model.utils.io import read, write


def _split_random_individual(data, sampling_ratio=0.5):
    """
    随机切分
    :param data: 需要切分的数据
    :param sampling_ratio: 训练集的数量比例
    :return: 训练集和测试集
    """
    data = data.sample(frac=1.0)
    cut_idx = int(round(sampling_ratio * data.shape[0]))
    train_data, test_data = data.iloc[:cut_idx], data.iloc[cut_idx:]
    train_data = train_data.sort_values('checkin_time')
    test_data = test_data.sort_values('checkin_time')
    return train_data, test_data


def split_random(INPUT_PATH, OUTPUT_PATH, sampling_ratio=0.5, sampling_times=100):
    for i in range(sampling_times):
        print('sampling_times: ', i + 1)
        root_path = OUTPUT_PATH + './random_sampling_' + str(sampling_ratio) + '_' + str(i + 1)
        if sampling_ratio == 0.5:
            train_output_path = OUTPUT_PATH + './random_sampling_' + str(sampling_ratio) + '_' + str(i + 1) + './train'
            test_output_path = OUTPUT_PATH + './random_sampling_' + str(sampling_ratio) + '_' + str(i + 1) + './test'
        else:
            train_output_path = OUTPUT_PATH + './random_sampling_' + str(sampling_ratio) + '_' + str(i + 1) + './' + str(
                sampling_ratio)
            test_output_path = OUTPUT_PATH + './random_sampling_' + str(sampling_ratio) + '_' + str(i + 1) + './' + str(
                1 - sampling_ratio)
        if os.path.exists(root_path):
            shutil.rmtree(root_path)
        os.makedirs(root_path)
        if os.path.exists(train_output_path):
            shutil.rmtree(train_output_path)
        os.makedirs(train_output_path)
        if os.path.exists(test_output_path):
            shutil.rmtree(test_output_path)
        os.makedirs(test_output_path)
        for filename in os.listdir(INPUT_PATH):
            print(filename)
            file_path = os.path.join(INPUT_PATH, filename)
            data = read(file_path)
            data_train, data_test = _split_random_individual(data, sampling_ratio)
            if sampling_ratio == 0.5:
                filename_train = filename.split("_")[0] + '_random_sampling_' + str(sampling_ratio) + '_' + str(
                    i + 1) + "_train.csv"
                filename_test = filename.split("_")[0] + '_random_sampling_' + str(sampling_ratio) + '_' + str(
                    i + 1) + "_test.csv"
            else:
                filename_train = filename.split("_")[0] + '_random_sampling_' + str(sampling_ratio) + '_' + str(
                    i + 1) + '.csv'
                filename_test = filename.split("_")[0] + '_random_sampling_' + str(1 - sampling_ratio) + '_' + str(
                    i + 1) + '.csv'
            write(os.path.join(train_output_path, filename_train), data_train)
            write(os.path.join(test_output_path, filename_test), data_test)


if __name__ == '__main__':
    input_path = r'../../result/L_with_driving_behavior'
    output_path = r'../../result/split_random_0.5'
    split_random(input_path, output_path, sampling_times=10)
   