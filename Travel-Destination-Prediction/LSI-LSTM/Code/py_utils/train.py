import torch
import torch.nn as nn
from torch import optim
from .to_var import to_var
import time
import datetime
import os
import numpy as np


def train(model, train_loader, eval_loader, config, K_fold=False, result=None):
    device = config['device']
    lr = config['lr']
    epochs = config['epochs']
    model = model.to(device)
    model_save_path = config['model_save_path']

    optimiter = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-8)
    dis_loss = {"train_MAE": [], "train_MRE": [], "train_RMSE": [], "train_ACC": [],
                "eval_MAE": [], "eval_MRE": [], "eval_RMSE": [], "eval_ACC": []}
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        step = 1
        MAE, MRE, RMSE, Accuracy = [], [], [], []
        for attr, traj in train_loader:
            attr, traj = to_var(attr, device), to_var(traj, device)
            Loss, MRE_loss, MAE_loss, RMSE_loss, accuracy = model.eval_on_batch(attr, traj)

            # update the model
            optimiter.zero_grad()
            Loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            if config['clip_gradient'] is not None:
                total_norm = nn.utils.clip_grad_norm_(model.parameters(), config['clip_gradient'])
                if total_norm > config['clip_gradient']:
                    pass

            optimiter.step()

            MAE.append(float(MAE_loss.data))
            RMSE.append(float(RMSE_loss.data))
            MRE.append(float(MRE_loss.data))
            step += 1

        dis_loss["train_MAE"].append(str(np.mean(MAE)))
        dis_loss["train_RMSE"].append(str(np.sqrt(np.mean(RMSE))))
        dis_loss["train_MRE"].append(str(np.mean(MRE)))

        train_info = 'Training: Epoch {epoch} of {epochs} ....\nTime:{time}, Time_consuming:{time_consume}s\n' \
                     'Train_mean_Loss(MAE): {train_MAE}\nTrain_mean_Loss(RMSE):{train_RMSE}\n' \
                     "Train_mean_Loss(MRE): {train_MRE}\n".format(epoch=epoch, epochs=epochs,
                                                                  train_MAE=dis_loss["train_MAE"][-1],
                                                                  train_RMSE=dis_loss["train_RMSE"][-1],
                                                                  train_MRE=dis_loss["train_MRE"][-1],
                                                                  time=datetime.datetime.now(),
                                                                  time_consume=time.time() - start_time)

        if not K_fold or epoch == epochs:
            print(train_info)

        with torch.no_grad():
            model.eval()
            step = 1
            MAE, MRE, RMSE, Accuracy = [], [], [], []
            for attr, traj in eval_loader:
                attr, traj = to_var(attr, device), to_var(traj, device)
                Loss, MRE_loss, MAE_loss, RMSE_loss, accuracy = model.eval_on_batch(attr, traj)

                MAE.append(float(MAE_loss.data))
                RMSE.append(float(RMSE_loss.data))
                MRE.append(float(MRE_loss.data))
                step += 1
            dis_loss["eval_MAE"].append(str(np.mean(MAE)))
            dis_loss["eval_MRE"].append(str(np.mean(MRE)))
            dis_loss["eval_RMSE"].append(str(np.sqrt(np.mean(RMSE))))
            eval_info = 'Evaluating: Epoch {epoch} of {epochs} ....\nTime:{time}, Time_consuming:{time_consume}s\n' \
                        'Evaluating_mean_Loss(MAE): {eval_MAE}\nEvaluating_mean_Loss(RMSE):{eval_RMSE}\n' \
                        'Evaluating_mean_Loss(MRE): {eval_MRE}\n'.format(epoch=epoch, epochs=epochs,
                                                                         eval_MAE=dis_loss["eval_MAE"][-1],
                                                                         eval_MRE=dis_loss["eval_MRE"][-1],
                                                                         eval_RMSE=dis_loss["eval_RMSE"][-1],
                                                                         time=datetime.datetime.now(),
                                                                         time_consume=time.time() - start_time)
            if result is not None and epoch == epochs:
                result["MAE"].append(float(dis_loss["eval_MAE"][-1]))
                result["MRE"].append(float(dis_loss["eval_MRE"][-1]))
                result["RMSE"].append(float(dis_loss["eval_RMSE"][-1]))

            if not K_fold or epoch == epochs:
                print(eval_info)

            if epoch == epochs and not K_fold:
                # save checkpoint every epoch_gap
                model_set = {'epochs': config['epochs'],
                             'lr': config['lr'],
                             'hidden_size': config['hidden_size'],
                             'total_traj': config['total_traj'],
                             'optimizer_state_dict': optimiter.state_dict(),
                             'state_dict': model.state_dict()}
                if not os.path.exists(model_save_path):
                    os.makedirs(model_save_path)
                torch.save(model_set, model_save_path + '_cp(' +
                           str(round(float(np.mean(MAE)), 1)) + ', ' + str(round(float(np.mean(MRE)), 3))
                           + ')LSI_LSTM_' + str(epoch) + '.pth')

        model.train()


