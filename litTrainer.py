import torch
import numpy as np
import lightning.pytorch as pl
from torch import Tensor
import pandas as pd

from utils.model import Geometry_estimator
from utils.dataLoader import ISMDataset


# Cuda variables
USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

class Lit_GeoEstimator(pl.LightningModule):
    def __init__(self,) :
        super().__init__()
        self._model = Geometry_estimator().type(dtype)
        self.lr = 0.001

    # def forward(self, inp):
    #     return self._model(inp)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def training_step(self, batch, batch_id):
        geo, coeff, rir = batch
        geo = geo.type(dtype)
        coeff = coeff.type(dtype)
        rir = rir.type(dtype)
        pred = self._model(rir.unsqueeze(axis=1))
        loss = torch.nn.functional.mse_loss(pred, geo)
        # logging
        self.log("trainLoss", loss)
        return loss
    
   


if __name__ == '__main__':
    modelTrainer = Lit_GeoEstimator()

<<<<<<< HEAD
    dd = np.load("./rirData/ism_400.npy")
=======
    dd = np.load("./rirData/ism_40k.npy")
>>>>>>> 9926d391994e6f2a1e4b1d81add599859eda7c92
    #dd = np.concatenate((dd[:2400,:] , dd[2500:,:]))
    data=pd.DataFrame(dd)
    trainLoader = torch.utils.data.DataLoader(ISMDataset(data=data), batch_size=50, shuffle=True, num_workers=4)

<<<<<<< HEAD
    trainer = pl.Trainer(max_epochs=1)
=======
    trainer = pl.Trainer(max_epochs=2000)
>>>>>>> 9926d391994e6f2a1e4b1d81add599859eda7c92
    trainer.fit(model=modelTrainer, train_dataloaders=trainLoader)
