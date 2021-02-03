from hyper_params_config import config
from py_utils import train_test_load, get_dataloader, train
from model import LSI_LSTM
import numpy as np
from sklearn.model_selection import KFold
import math

if __name__ == '__main__':
    # parameters config
    filepath = config['data_path']
    BATCH_SIZE = config['BATCH_SIZE']
    EVAL_BATCH_SIZE = config['EVAL_BATCH_SIZE']
    _device = config['device']
    hidden_size = config['hidden_size']

    # if need cross validation?
    k_fold = False

    # one test
    if not k_fold:
        lsi_lstm = LSI_LSTM(hidden_size=hidden_size)
        [train_part, eval_part], data_length = train_test_load(filepath, K_fold=False)
        train_loader = get_dataloader(data=train_part, batch_size=BATCH_SIZE)
        eval_loader = get_dataloader(data=eval_part, batch_size=EVAL_BATCH_SIZE)

        config['total_traj'] = data_length
        tips1 = 'Total number of trajectory：{:}'.format(config['total_traj']) + '\n'
        tips2 = 'Train number of data(batch_num x batch_size):{batch_num} x {batch_size}\n'.format(
            batch_num=int(math.ceil(len(train_part) / BATCH_SIZE)),
            batch_size=BATCH_SIZE)
        tips3 = 'Eval number of data(batch_num x batch_size):{batch_num} x {batch_size}'.format(
            batch_num=int(math.ceil(len(eval_part) / EVAL_BATCH_SIZE)),
            batch_size=EVAL_BATCH_SIZE)
        print('\n-----------------------------------------------------------\n')
        print(tips1 + tips2 + tips3)

        train(model=lsi_lstm, train_loader=train_loader, eval_loader=eval_loader, config=config,
              K_fold=False)

    # k fold cross validation
    else:
        print("Five cross validation:\n")
        result = {"MAE": [], "RMSE": [], "MRE": []}
        data_set, data_length = train_test_load(filepath, K_fold=True)
        for train_index, test_index in KFold(n_splits=5).split(data_set):
            lsi_lstm = LSI_LSTM(hidden_size=hidden_size)
            train_part, eval_part = data_set[train_index], data_set[test_index]
            train_loader = get_dataloader(data=train_part, batch_size=BATCH_SIZE)
            eval_loader = get_dataloader(data=eval_part, batch_size=EVAL_BATCH_SIZE)
            train(model=lsi_lstm, train_loader=train_loader, eval_loader=eval_loader,
                  config=config, K_fold=True, result=result)
        info = '\nThe result of five cross validation:\n' \
               'MAE: ' + str(np.mean(result["MAE"])) + '(' + str(np.std(result["MAE"])) + ')\n' \
                                                                                          'RMSE: ' + str(
            np.mean(result["RMSE"])) + '(' + str(np.std(result["RMSE"])) + ')\n' \
                                                                           'MRE: ' + str(
            np.mean(result["MRE"])) + '(' + str(np.std(result["MRE"])) + ')\n'
        print(info)