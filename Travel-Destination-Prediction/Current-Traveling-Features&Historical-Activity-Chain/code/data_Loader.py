#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2022/1/28 21:29
# @Author: zhangxiaotong
# @File  : data_Loader.py
import json
import os
import time
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from matplotlib.ticker import MultipleLocator
from sklearn.model_selection import train_test_split
from torch import optim, float32

# transform tensor to Variable
from torch.utils.data import Dataset, DataLoader, Sampler


def to_var(var, device):
    if torch.is_tensor(var):
        var = var.to(device)
        return var
    if isinstance(var, int) or isinstance(var, float):  # isinstance() 函数来判断一个对象是否是一个已知的类型，类似 type()
        return var
    if isinstance(var, dict):
        for key in var:
            var[key] = to_var(var[key], device)
        return var
    if isinstance(var, list):
        var = list(map(lambda x: to_var(x, device), var))
        return var


def write2file(config, path, info):
    f = open(path + str(config['ID']) + '-log_latest_model.txt', 'a')
    f.write(info)
    f.close()


def write_each_trajectory(config, path, traj_result, augType, expMode):
    if augType == 'Original':
        if expMode == "grids":
            f = open(path + str(config['ID']) + '-each_trajectory-Grids.txt', 'w')
        elif expMode == "cp_lens":
            f = open(path + str(config['ID']) + '-each_trajectory-Cp_lens.txt', 'w')
        else:
            f = open(path + str(config['ID']) + '-each_trajectory-Original.txt', 'w')
    elif augType == 'Duplicated_Edges':
        f = open(path + str(config['ID']) + '-each_trajectory-Duplicated_Edges.txt', 'a')
    else:
        f = open(path + str(config['ID']) + '-each_trajectory-Plus.txt', 'a')
    for i in range(len(traj_result)):
        json_str = json.dumps(traj_result[i])
        f.write(json_str)
        f.write('\n')


def write_self_attention(config, path, self_attn_sum, traj_id, Kfold):
    df = pd.DataFrame(self_attn_sum.cpu().numpy())
    print(traj_id)
    print(df)
    if Kfold:
        df.to_csv(path + str(config['ID']) + '-self_attn-KFold.csv')
    else:
        df.to_csv(path + str(config['ID']) + '-self_attn-NotKFold.csv')


def write2file_loss(config, data):
    path = config['logfile_path']
    f_c = open(path + str(config['ID']) + '-loss.csv', 'w')
    f_c.close()
    f = open(path + str(config['ID']) + '-loss.csv', 'a')
    for key in data:
        f.write(key + ':' + ','.join(data[key]) + '\n')
    #   plt_line(config, data['train_MAE'], data['eval_MAE'], 'MAE')
    #   plt_line(config, data['train_MRE'], data['eval_MRE'], 'MRE')
    f.close()


def plt_line(config, train_res, test_res, ylabel, epoch=0):
    # plot config
    ax = plt.subplot(111)
    xmajorLocator = MultipleLocator(5)  # 将x主刻度标签设置为5的倍数
    ax.xaxis.set_major_locator(xmajorLocator)
    xminorLocator = MultipleLocator(1)  # 将x轴次刻度标签设置为1的倍数
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.xaxis.grid(True, which='major')  # x坐标轴的网格使用主刻度
    ax.yaxis.grid(True, which='major')  # y坐标轴的网格使用主刻度

    if epoch == 0:
        _e = config['epochs']
        x = list(range(1, config['epochs'] + 1))
    else:
        _e = epoch
        x = list(range(1, epoch + 1))

    plt.plot(x, list(map(lambda data: round(float(data), 3), train_res)), c='green', label='训练集')
    plt.plot(x, list(map(lambda data: round(float(data), 3), test_res)), c='red', label='验证集')
    plt.xlabel('迭代次数')
    plt.ylabel(ylabel)
    plt.legend(loc='upper left')
    plt.title('编号: ' + str(config['ID']))
    if not os.path.exists(config['logplot_path']):
        os.makedirs(config['logplot_path'])
    plt.savefig(
        config['logplot_path'] + str(config['ID']) + '_' + ylabel + '_e' + str(_e) + '_lr' + str(config['lr']) + '.png')
    # plt.show()
    plt.close()


def train(model, train_loader, eval_loader, config, K_fold=False, result=None):
    device = config['device']  # 设备
    lr = config['lr']  # 学习率
    epochs = config['epochs']  # 迭代数
    model = model.to(device)
    model_save_path = config['model_save_path']
    #   save_every = config['save_every']

    dis_loss = {"train_MAE": [], "train_MRE": [], "train_RMSE": [], "train_ACC": [],
                "eval_MAE": [], "eval_MRE": [], "eval_RMSE": [], "eval_ACC": []}
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        # L2 regularization --weight_decay  权重衰减（weight decay）与L2正则化
        print(lr)
        optimiter = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-8)
        step = 1
        MAE, MRE, RMSE, Accuracy = [], [], [], []
        for attr, traj, semantic, history in train_loader:
            attr, traj, semantic, history = to_var(attr, device), to_var(traj, device), to_var(
                semantic, device), to_var(history, device)
            # 增加一个承接each_traj_result的参数，无实义
            Loss, MRE_loss, MAE_loss, RMSE_loss, semantic_loss, accuracy, _, self_attn = model.eval_on_batch(attr, traj,
                                                                                                             semantic,
                                                                                                             history)
            # update the model
            optimiter.zero_grad()
            Loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            '''
            if config['clip_gradient'] is not None:
                total_norm = nn.utils.clip_grad_norm_(model.parameters(), config['clip_gradient'])  # 梯度裁剪
                if total_norm > config['clip_gradient']:
                    pass
            '''
            # print("clipping gradient: {} with coef {}".format(total_norm, config['clip_gradient'] /
            # total_norm))
            optimiter.step()

            MAE.append(float(MAE_loss.data))
            RMSE.append(float(RMSE_loss.data))
            MRE.append(float(MRE_loss.data))
            Accuracy.append(float(semantic_loss.data))
            # Accuracy.append(float(accuracy))
            # print('Training batch:', step, " loss_mean(meters): ", float(Loss.data), float(MRE_loss.data))
            step += 1

        dis_loss["train_MAE"].append(str(np.mean(MAE)))
        dis_loss["train_RMSE"].append(str(np.sqrt(np.mean(RMSE))))
        dis_loss["train_MRE"].append(str(np.mean(MRE)))
        # 每次都把训练得到的平均值存入dis_loss数组

        train_info = 'Training: Epoch {epoch} of {epochs} ....\nTime:{time}, Time_consuming:{time_consume}s\n' \
                     'Train_mean_Loss(MAE): {train_MAE}\nTrain_mean_Loss(RMSE):{train_RMSE}\n' \
                     "Train_mean_Loss(MRE): {train_MRE}\n".format(epoch=epoch, epochs=epochs,
                                                                  train_MAE=dis_loss["train_MAE"][-1],
                                                                  train_RMSE=dis_loss["train_RMSE"][-1],
                                                                  train_MRE=dis_loss["train_MRE"][-1],
                                                                  time=datetime.datetime.now(),
                                                                  time_consume=time.time() - start_time)

        if not K_fold:  # 非K折实验时，将训练日志进行记录
            write2file(config, config['logfile_path'], train_info)
        if K_fold:  # 交叉验证时，将训练日志进行记录
            write2file(config, config['logfile_path'], train_info)
        if epoch % 10 == 0:
            print(train_info)

        with torch.no_grad():
            model.eval()  # 将模型转换为测试模式
            step = 1
            MAE, MRE, RMSE, Accuracy = [], [], [], []
            total_each_traj_result = []
            self_attn_sum = None
            for attr, traj, semantic, history in eval_loader:
                attr, traj, semantic, history = to_var(attr, device), to_var(traj, device), to_var(
                    semantic, device), to_var(history, device)
                Loss, MRE_loss, MAE_loss, RMSE_loss, semantic_loss, accuracy, each_traj_result, self_attn = model. \
                    eval_on_batch(attr, traj, semantic, history)

                # history["history_weekday_sin"].shape = torch.Size([32, 101])
                history_weekday_sin = torch.index_select(history["history_weekday_sin"], 1, torch.tensor([0]).cuda())
                # print(attr["id"].shape) # torch.Size([32])
                traj_id = attr["id"].cpu().numpy()
                # print(history_weekday_sin.shape)        # torch.Size([32, 1])
                # 这里需要在隐藏层特征维度上面sum一下，得到每个batch中近期出行链的权重大小，dataLoader的大小为数据总量除以batch_size
                # self_attn_sum.shape = torch.Size([20, 101])
                '''self_attn_sum = torch.index_select(self_attn, 2, torch.tensor([0]).cuda())
                self_attn_sum = self_attn_sum.squeeze()'''
                # print(self_attn_sum.shape)              # torch.Size([32, 20, 1])

                MAE.append(float(MAE_loss.data))
                RMSE.append(float(RMSE_loss.data))
                MRE.append(float(MRE_loss.data))
                Accuracy.append(float(semantic_loss.data))
                # 达到迭代次数时再装填数组，不到迭代次数不写文件
                if epoch == epochs:
                    for i in range(len(each_traj_result)):
                        total_each_traj_result.append(each_traj_result[i])
                step += 1
            dis_loss["eval_MAE"].append(str(np.mean(MAE)))
            dis_loss["eval_MRE"].append(str(np.mean(MRE)))
            dis_loss["eval_RMSE"].append(str(np.sqrt(np.mean(RMSE))))

            # 在这里更改学习率
            # 是否更改学习率需要设定好一个阈值，MAE设置为30比较好
            # dis_loss["eval_MAE"][-1] 是最后一次的预测结果
            # dis_loss["eval_MAE"][-2] 是上一次的预测结果

            if len(dis_loss["eval_MAE"]) > 2:  # 避免数组越界
                if float(dis_loss["eval_MAE"][-1]) > float(dis_loss["eval_MAE"][-2]) - 30:
                    lr = 0.0002
                    # 将学习率改小
                else:
                    lr = lr
                    # 学习率不变
            else:
                lr = lr
                # 学习率不变
            eval_info = 'Evaluating: Epoch {epoch} of {epochs} ....\nTime:{time}, Time_consuming:{time_consume}s\n' \
                        'Evaluating_mean_Loss(MAE): {eval_MAE}\nEvaluating_mean_Loss(RMSE):{eval_RMSE}\n' \
                        'Evaluating_mean_Loss(MRE): {eval_MRE}\n'.format(epoch=epoch, epochs=epochs,
                                                                         eval_MAE=dis_loss["eval_MAE"][-1],
                                                                         eval_MRE=dis_loss["eval_MRE"][-1],
                                                                         eval_RMSE=dis_loss["eval_RMSE"][-1],
                                                                         time=datetime.datetime.now(),
                                                                         time_consume=time.time() - start_time)
            if result is not None and epoch == epochs:
                # result["MAE"].append(float(dis_loss["eval_MAE"][-1]))
                result["MAE"].append(MAE)
                # result["MRE"].append(float(dis_loss["eval_MRE"][-1]))
                result["MRE"].append(MRE)
                # result["RMSE"].append(float(dis_loss["eval_RMSE"][-1]))
                result["RMSE"].append(RMSE)

                result["MSemanticE"].append(Accuracy)

            # 输出验证精度等信息

            if epoch % 10 == 0:
                print(eval_info)

            if not K_fold:
                write2file(config, config['logfile_path'], eval_info)

            if K_fold:
                write2file(config, config['logfile_path'], eval_info)
                # 这里要传入所有的轨迹精度信息
                # 测试一遍没问题就屏蔽这个接口，达到迭代次数再输出
                # write_each_trajectory(config, config['logfile_path'], total_each_traj_result)

            if epoch >= epochs - 5 and not K_fold:
                # save checkpoint every epoch_gap
                model_set = {'user_id': config['ID'],
                             'epochs': config['epochs'],
                             'lr': config['lr'],
                             'hidden_size': config['hidden_size'],
                             'total_traj': config['total_traj'],
                             'output_size': config['output_size'],
                             'optimizer_state_dict': optimiter.state_dict(),
                             'state_dict': model.state_dict()}
                if not os.path.exists(model_save_path):
                    os.makedirs(model_save_path)
                torch.save(model_set, model_save_path + config['ID'] + '_DeepSTC_' + str(epoch) + '.pth')
            if epoch == epochs:  # 达到迭代次数后绘图
                plt_line(config, dis_loss['train_MAE'], dis_loss['eval_MAE'], 'MAE', epoch)
                plt_line(config, dis_loss['train_MRE'], dis_loss['eval_MRE'], 'MRE', epoch)
                plt_line(config, dis_loss['train_RMSE'], dis_loss['eval_RMSE'], 'RMSE', epoch)
                # 要实现一个把所有轨迹信息预测精度信息写入文件的功能
                write_each_trajectory(config,
                                      config['logfile_path'],
                                      total_each_traj_result,
                                      augType='Original',
                                      expMode="grids")
                # 要实现一个把自注意力权重写入文件的功能
                '''write_self_attention(config=config, path=config['logfile_path'],
                                     self_attn_sum=self_attn_sum, traj_id=traj_id, Kfold=K_fold)'''

            model.train()  # 更换回训练模式

    if not K_fold:
        write2file_loss(config, dis_loss)


def test(model, model_save_dict, eval_loader, config, result=None):
    device = config['device']  # 设备
    lr = config['lr']  # 学习率
    epochs = config['epochs']  # 迭代数
    model = model.to(device)
    model_save_path = config['model_save_path']
    state_dict = torch.load(model_save_path + model_save_dict)['state_dict']
    model.load_state_dict(state_dict)
    dis_loss = {"train_MAE": [], "train_MRE": [], "train_RMSE": [], "train_ACC": [],
                "eval_MAE": [], "eval_MRE": [], "eval_RMSE": [], "eval_ACC": []}
    with torch.no_grad():
        model.eval()  # 将模型转换为测试模式
        step = 1
        MAE, MRE, RMSE, Accuracy = [], [], [], []
        total_each_traj_result = []
        for attr, traj, semantic, history in eval_loader:
            attr, traj, semantic, history = to_var(attr, device), to_var(traj, device), to_var(
                semantic, device), to_var(history, device)
            Loss, MRE_loss, MAE_loss, RMSE_loss, semantic_loss, accuracy, each_traj_result, self_attn = model. \
                eval_on_batch(attr, traj, semantic, history)
            # Append arrays
            MAE.append(float(MAE_loss.data))
            RMSE.append(float(RMSE_loss.data))
            MRE.append(float(MRE_loss.data))
            # 达到迭代次数时再装填数组，不到迭代次数不写文件
            for i in range(len(each_traj_result)):
                total_each_traj_result.append(each_traj_result[i])
            step += 1
        dis_loss["eval_MAE"].append(str(np.mean(MAE)))
        dis_loss["eval_MRE"].append(str(np.mean(MRE)))
        dis_loss["eval_RMSE"].append(str(np.sqrt(np.mean(RMSE))))

        if result is not None:
            result["MAE"].append(MAE)
            result["MRE"].append(MRE)
            result["RMSE"].append(RMSE)

        write_each_trajectory(config,
                              config['logfile_path'],
                              total_each_traj_result,
                              augType='Original',
                              expMode="grids")


# store all data with one class
class SuperDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.lengths = list(map(lambda x: len(x["lng"]), self.data))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class BatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        super(Sampler, self).__init__()
        self.lengths = dataset.lengths
        self.data_len = len(dataset)
        self.batch_size = batch_size
        self.indices = list(range(self.data_len))

        # turn the fn to generator
        """
        divide the data into chunks in size of batch_size*10
        every chunk has sorted by data_length to make training stability
        data: all the traj_dict(include raw traj_dict and cut_traj_dict)
    """

    def __iter__(self):
        # shuffle the data
        np.random.shuffle(self.indices)
        chunk_size = self.batch_size * 10
        chunks = (self.data_len + chunk_size - 1) // chunk_size  # " / "就表示 浮点数除法，返回浮点结果;" // "表示整数除法。
        for i in range(chunks):
            partial_indices = self.indices[i * chunk_size:(i + 1) * chunk_size]
            partial_indices.sort(key=lambda x: self.lengths[x], reverse=True)
            self.indices[i * chunk_size:(i + 1) * chunk_size] = partial_indices
        self.batches = (self.data_len + self.batch_size - 1) // self.batch_size
        for i in range(self.batches):
            yield self.indices[i * self.batch_size:(i + 1) * self.batch_size]

    def __len__(self):
        return self.batches


def collate_fn(batch_data):
    """
    :param batch_data: every batch data
    :return: attr,traj  --Tensor(data) and normalization the sequences of 'lng lat dis'
    """
    traj_key = ["lng", "lat", "travel_dis", "spd", "v_mean", "v_std", "angle", "time_consume", "start_time_s",
                "time_stamp_s", "weekday_s", "start_time_c", "time_stamp_c", "weekday_c", "c_entropy", "d_entropy"]
    attr_key = ["cur_pt", "id", "cluster_id", "destination", "dis_total", "normlization_list", "normlization_list_xy"]

    semantic_key = ["lng", "lat", "semantic_0", "semantic_1", "semantic_2", "semantic_3", "semantic_4", "dest_semantic"]

    for i in range(3):
        semantic_key.append("word_vec_" + str(i) + "")
    # 补充一个history_key
    history_key = ["history_lng", "history_lat", "history_semantic_0", "history_semantic_1", "history_semantic_2",
                   "history_semantic_3", "history_semantic_4", "history_weekday_sin", "history_weekday_cos",
                   "history_start_time_sin",
                   "history_start_time_cos"]
    attr = {}
    traj = {}
    semantic = {}
    history = {}
    # print(batch_data[7]['id'])   # 输出的数据很大，所有的数据，以字典存储   # 这里可以显示出来ID
    # 目前可以知道每个batch中随机选择的id
    # 然后根据id之间的相似性将相似的轨迹相加
    # 如果没有相似的轨迹就不相加
    # 相加之后的结果是什么，其实应该做加权平均，将两者的表征按照权重结合起来
    # 根据id的值对历史出行活动进行切分，保留当前id之前的

    for key in attr_key:
        attr[key] = torch.Tensor([item[key] for item in batch_data])

    for key in traj_key:
        seqs = np.asarray([item[key] for item in batch_data])
        padded_seqs = numpy_fillna(seqs)
        if key == "sem_pt":
            traj[key] = torch.from_numpy(padded_seqs).long()
        else:
            traj[key] = torch.from_numpy(padded_seqs).float()

    for key in semantic_key:
        seqs = np.asarray([item[key] for item in batch_data])
        padded_seqs = numpy_fillna(seqs)
        semantic[key] = torch.from_numpy(padded_seqs).float()
    for key in history_key:
        seqs = np.asarray([item[key] for item in batch_data])
        padded_seqs = numpy_fillna(seqs)
        history[key] = torch.from_numpy(padded_seqs).float()

    # 当前轨迹序列长度
    lens = [len(item["lng"]) for item in batch_data]
    traj["lens"] = torch.LongTensor(lens)
    # 历史轨迹序列长度
    his_lens = [len(item["history_lng"]) for item in batch_data]
    traj["his_lens"] = torch.LongTensor(his_lens)

    return attr, traj, semantic, history


# normalize data by fill 0
def numpy_fillna(data):
    lens = np.asarray([len(item) for item in data])
    mask = np.arange(lens.max()) < lens[:, None]
    out = np.zeros(mask.shape, dtype=np.float32)
    out[mask] = np.concatenate(
        data)  # numpy提供了numpy.concatenate((a1,a2,...), axis=0)函数。能够一次完成多个数组的拼接。其中a1,a2,...是数组类型的参数
    return out


def trajectory_cut(data, attr_key, traj_key, semantic_key, history_key, Cut_set, steering_angle_threshold=0.2,
                   need_cut_by_steer=True):
    dataset = []
    # data product
    for raw_dict in data:
        # cut by seq_length #
        if 0 < Cut_set < 1:
            step = 1
            while step * Cut_set < 1:
                baby_dict = {}
                cut_length = int(len(raw_dict["lng"]) * step * Cut_set)
                # print(cut_length)  # 输出子轨迹长度
                if cut_length == 0:
                    cut_length = 1
                # 历史出行活动链不必切分
                for key in history_key:
                    baby_dict[key] = raw_dict[key]
                # 特征不用切分
                for key in attr_key:
                    baby_dict[key] = raw_dict[key]
                # 当前轨迹需要切分，traj+gat+semantic
                for key in traj_key:
                    baby_dict[key] = raw_dict[key][:cut_length]
                # 增加语义信息
                for key in semantic_key:
                    baby_dict[key] = raw_dict[key][:cut_length]
                baby_dict["cur_pt"] = [baby_dict["lng"][-1], baby_dict["lat"][-1]]
                baby_dict["traj_id"] = raw_dict["id"]
                baby_dict["cut_length"] = step
                dataset.append(baby_dict)
                step += 1
        else:
            print("Error number of Cut_Proportion!")
    np.random.shuffle(dataset)
    # return value is a list
    return dataset


def train_test_split_byLength(dataset):
    traj_total_length = []
    for i in range(len(dataset)):
        traj_total_length.append(len(dataset[i]["lng"]))
    percentile_1 = np.percentile(traj_total_length, 70)
    percentile_2 = np.percentile(traj_total_length, 90)
    train_set = []
    eval_set = []
    for traj in dataset:
        if len(traj["lng"]) <= percentile_1:
            train_set.append(traj)
        elif percentile_1 < len(traj["lng"]) <= percentile_2:
            eval_set.append(traj)
        else:
            continue
    return train_set, eval_set


def train_test_split_byCuttingRatio(dataset):
    train_set = []
    eval_set = []
    for traj in dataset:
        if traj["traj_id"] % 5 != 0 and traj["traj_id"] % 5 != 1 and traj["dis_total"] > 0:
            train_set.append(traj)
        elif traj["cut_length"] < 18 and (traj["traj_id"] % 5 == 0 or traj["traj_id"] % 5 == 1) and traj["dis_total"] > 0:
            train_set.append(traj)
        elif 18 <= traj["cut_length"] < 30 and (traj["traj_id"] % 5 == 0 or traj["traj_id"] % 5 == 1) and traj["dis_total"] > 0:
            eval_set.append(traj)
        else:
            continue
    return train_set, eval_set


def get_dataloader(data, batch_size):
    dataset = SuperDataset(data=data)
    batch_sampler = BatchSampler(dataset, batch_size)
    data_loader = DataLoader(dataset=dataset,
                             collate_fn=lambda x: collate_fn(x),
                             batch_sampler=batch_sampler,
                             pin_memory=False)
    # DataLoader在数据集上提供单进程或多进程的迭代器
    # 几个关键的参数意思：
    # - shuffle：设置为True的时候，每个世代都会打乱数据集
    # - collate_fn：如何取样本的，我们可以定义自己的函数来准确地实现想要的功能
    # - drop_last：告诉如何处理数据集长度除于batch_size余下的数据。True就抛弃，否则保留
    return data_loader


def re_sampler(data_list, level):
    return data_list[::level]


def train_test_load(file_path, Cut_set=0.03333, resample_level=1, cut_frist=False, normalization=False, K_fold=False):
    file = open(file_path, 'r')
    lines = file.readlines()
    # the key of line_dict
    traj_key = ["lng", "lat", "travel_dis", "spd", "v_mean", "v_std", "angle", "time_consume", "start_time_s",
                "time_stamp_s", "weekday_s", "start_time_c", "time_stamp_c", "weekday_c", "c_entropy", "d_entropy"]
    # attr_key增加了轨迹编号和类簇编号两个字段
    attr_key = ["id", "cluster_id", "destination", "dis_total", "normlization_list", "normlization_list_xy"]
    semantic_key = ["lng", "lat", "semantic_0", "semantic_1", "semantic_2", "semantic_3", "semantic_4", "dest_semantic"]
    for i in range(3):
        semantic_key.append("word_vec_" + str(i) + "")
    history_key = ["history_lng", "history_lat", "history_semantic_0", "history_semantic_1", "history_semantic_2",
                   "history_semantic_3", "history_semantic_4", "history_weekday_sin", "history_weekday_cos",
                   "history_start_time_sin",
                   "history_start_time_cos"]
    dataset = []
    for line in lines:
        raw_dict = json.loads(line)
        # record the current pt
        for key in traj_key:
            raw_dict[key] = re_sampler(raw_dict[key], level=resample_level)
        for key in semantic_key:
            raw_dict[key] = re_sampler(raw_dict[key], level=resample_level)
        for key in history_key:
            raw_dict[key] = re_sampler(raw_dict[key], level=resample_level)

        raw_dict["cur_pt"] = [raw_dict["lng"][-1], raw_dict["lat"][-1]]
        if len(raw_dict["history_lng"]) >= 20:
            dataset.append(raw_dict)
    file.close()
    # 在构建训练集和验证集时增加语义信息，调用trajectory_cut函数，参数列表已更改
    if not K_fold:
        if cut_frist:
            dataset_cut = trajectory_cut(dataset, attr_key, traj_key, semantic_key, history_key, Cut_set=Cut_set)
            # train_set, eval_set = train_test_split_byLength(dataset=dataset_cut)
            train_set, eval_set = train_test_split_byCuttingRatio(dataset=dataset_cut)
        else:
            # 需要自己定义一个训练集-测试集划分的函数
            # 根据轨迹的切分比例来划分训练集-测试集，如果把长度低于某一阈值的放入训练集，其余放入测试集
            train_set, eval_set = train_test_split(dataset, test_size=0.2,
                                                   random_state=1)
            # train_test_split()函数是用来随机划分样本数据为训练集和测试集的，当然也可以人为的切片划分。
            train_set = trajectory_cut(train_set, attr_key, traj_key, semantic_key, history_key,
                                       Cut_set=Cut_set)
            eval_set = trajectory_cut(eval_set, attr_key, traj_key, semantic_key, history_key, Cut_set=Cut_set)
        result = [train_set, eval_set]
    else:
        result = trajectory_cut(dataset, attr_key, traj_key, semantic_key, history_key, Cut_set=Cut_set)
        np.random.shuffle(result)  # 随机打乱数据
    return np.array(result), len(result[0]) + len(result[1])
