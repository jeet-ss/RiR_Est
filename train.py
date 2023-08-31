import torch
import os
import numpy as np
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.model import Geometry_estimator, MLP_reflectionCoeff, Link_model
from utils.dataLoader import ISMDataset
from utils.trainer import Trainer

def train(args):
    # MOdel
    model_num = 1
    # hyperparams
    lr = 0.001          # org 0.001
    early_stopping = 100
    batch_size = 50
    args.fp = "./rirData/ism_400k.npy"
    epochStart = 0      # org 2000
    epochEnd = 10
    name_variant = ''
    retrain_model_epoch = 59

    # load data
    dd = np.load(args.fp)
    #dd = np.concatenate((dd[:2450,:] , dd[2452:,:]))
    data_pd = pd.DataFrame(dd)
    print(f"Data shape: {data_pd.shape}, Epochs from {epochStart} to {epochEnd}")

    # divide data
    #train_data, rest_data = train_test_split(data_pd, train_size=0.7, random_state=1)
    #val_data, test_data = train_test_split(rest_data, test_size=0.333333, random_state=1)
    n = len(data_pd)
    train_head = data_pd.index[int(0.7*n)]  # 7:2:1
    val_head = data_pd.index[int(0.9*n)]
    train_data = data_pd.loc[:train_head-1, :]
    val_data = data_pd.loc[train_head:val_head-1, :]
    test_data = data_pd.loc[val_head:, :]

    train_batches = torch.utils.data.DataLoader(ISMDataset(data=train_data), batch_size=batch_size,  num_workers=4)
    val_batches = torch.utils.data.DataLoader(ISMDataset(data=val_data), batch_size=batch_size,  num_workers=4)
    test_batches = torch.utils.data.DataLoader(ISMDataset(data=test_data), batch_size=batch_size,  num_workers=4)
    print(f"batches: train-{len(train_batches)}, val-{len(val_batches)}, test-{len(test_batches)}")
    
    # loss funciton
    loss_function = torch.nn.MSELoss()

    trainer = Trainer(model_no=model_num, lr=lr, criterion=loss_function, train_loader=train_batches, val_loader=val_batches, test_loader=test_batches, early_stopping_patience=early_stopping, name_variant=name_variant)
    # load model for retraining
    #trainer.restore_checkpoint(epoch_n=retrain_model_epoch)
    loss = trainer.fit(epochs_start=epochStart, epochs_end=epochEnd)
    print("Loss", loss)
    np.save(str(model_num)+ '_'+ name_variant+'_lossData.npy', loss)

    # scoring
    #scores = trainer.test_model(retrain_model_epoch)
    #print("scores:", scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training of Room geometry estimator')
    parser.add_argument('--fp', '-filepath', type=str, default="./rirData/ism_100.npy", help='file path of data')
    args = parser.parse_args()

    train(args)
