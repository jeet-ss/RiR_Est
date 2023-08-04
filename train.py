import torch
import numpy as np
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.model import Geometry_estimator
from utils.dataLoader import ISMDataset
from utils.trainer import Trainer


def train(args):
    # hyperparams
    lr = 0.001          # org 0.001
    early_stopping = 100
    batch_size = 2
    args.fp = "./rirData/ism_10k.npy"
    epochs = 1        # org 2000

    # load data
    dd = np.load(args.fp)
    dd = np.concatenate((dd[:2450,:] , dd[2452:,:]))
    data_pd = pd.DataFrame(dd)
    print("DAta shape", data_pd.shape)

    #
    batches = torch.utils.data.DataLoader(ISMDataset(data=data_pd), batch_size=batch_size,  num_workers=4)
    print("batches:" , len(batches))
    #
    geo_est = Geometry_estimator()
    optimizer = torch.optim.Adam(geo_est.parameters(), lr=lr)
    loss_function = torch.nn.MSELoss()

    trainer = Trainer(model=geo_est, optimizer=optimizer, criterion=loss_function, train_loader=batches, early_stopping_patience=early_stopping)

    loss, minLoss = trainer.fit(epochs=epochs)
    print("Loss", minLoss, loss)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training of Room geometry estimator')
    parser.add_argument('--fp', '-filepath', type=str, default="./rirData/ism_100.npy", help='file path of data')
    args = parser.parse_args()

    train(args)
