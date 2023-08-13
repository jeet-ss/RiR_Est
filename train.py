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
    model_num = 3
    # hyperparams
    lr = 0.001          # org 0.001
    early_stopping = 100
    batch_size = 50
    args.fp = "./rirData/ism_400.npy"
    epochs = 3       # org 2000

    # load data
    dd = np.load(args.fp)
    #dd = np.concatenate((dd[:2450,:] , dd[2452:,:]))
    data_pd = pd.DataFrame(dd)
    print("DAta shape", data_pd.shape)

    # divide data
    train_data, rest_data = train_test_split(data_pd, train_size=0.7, random_state=1)
    val_data, test_data = train_test_split(rest_data, test_size=0.333333, random_state=1)

    train_batches = torch.utils.data.DataLoader(ISMDataset(data=train_data), batch_size=batch_size,  num_workers=4)
    val_batches = torch.utils.data.DataLoader(ISMDataset(data=val_data), batch_size=batch_size,  num_workers=4)
    test_batches = torch.utils.data.DataLoader(ISMDataset(data=test_data), batch_size=batch_size,  num_workers=4)
    print("batches:" , len(train_batches), len(val_batches), len(test_batches))
    
    # loss funciton
    loss_function = torch.nn.MSELoss()

    trainer = Trainer(model_no=model_num, lr=lr, criterion=loss_function, train_loader=train_batches, val_loader=val_batches, test_loader=test_batches, early_stopping_patience=early_stopping)

    loss = trainer.fit(epochs=epochs)
    print("Loss", loss)
    np.save(str(model_num)+ '_lossData.npy', loss)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training of Room geometry estimator')
    parser.add_argument('--fp', '-filepath', type=str, default="./rirData/ism_100.npy", help='file path of data')
    args = parser.parse_args()

    train(args)
