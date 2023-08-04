import torch
import os
import tqdm
import numpy as np
from utils.logger import Logger

# Global variables
LOG_EVERY_N_STEPS = 100

# CUDA variables
USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
device = 'cuda' if USE_CUDA else 'cpu'

# logging
logger = Logger('./logs')

class Trainer:
    def __init__(self, 
                model,      # model to train
                optimizer,  # optimizer to use
                criterion,  # loss function to use
                train_loader,  # data loader for training
                #val_loader,   # data loader for validation
                early_stopping_patience, # early stopping

                ):
        self.model = model.type(dtype)
        self.optimizer = optimizer
        self.criterion = criterion.type(dtype)
        self.train_batches= train_loader
        #self.val_batches= val_loader

    def train_step(self, rir_features, geo_labels, batch_idx):
        # reset zero grad
        self.optimizer.zero_grad()
        #
        pred = self.model(rir_features.unsqueeze(axis=1))
        #print("in train", pred.size(), geo_labels.size() )
        # calculate loss
        loss = self.criterion(pred, geo_labels)
        #print("loss")
        #print("loss", batch_idx, loss)
        if torch.isnan(loss):
            print("is nan", batch_idx)
            return 0
        else:   
            #print("not nan loss: ", batch_idx, loss)
        #
            loss.backward()
        #
            self.optimizer.step()

            return loss.detach().item()

    def train_epoch(self):
        # set training mode
        self.model.train()
        # iterate through the training set
        loss = 0
        #for idx, batches in tqdm.tqdm(enumerate(self.train_batches), unit='batch', total=len(self.train_batches), desc='loading batches'):
        for idx, batches in enumerate(self.train_batches):
            geo_l, coeff_l, rir_f = batches
            geo_l = geo_l.type(dtype)
            coeff_l = coeff_l.type(dtype)
            rir_f = rir_f.type(dtype)
                
            # perform a training step
            loss_step = self.train_step(rir_f, geo_l, idx)
            loss += loss_step
        # calculate the average loss for the epoch and return it
        avg_loss = loss/self.train_batches.__len__()
        return avg_loss


    def val_epoch():
        pass

    def fit(self, epochs=-1):
        assert epochs > 0, 'Epochs > 0'
        #
        loss_train = np.array([])
        loss_val = np.array([])
        epoch_counter = 0
        min_loss = np.Inf
        #
        #while (True):
        for i in tqdm.trange(epochs):
            # stop by epoch number
            # if epoch_counter >= epochs:
            #     break
            # increment Counter
            #epoch_counter += 1
            train_loss = self.train_epoch()
            #val_loss = self.val_epoch()
            #
            loss_train = np.append(loss_train, train_loss)
            logger.scalar_summary("loss", train_loss, i)
            #loss_val = np.append(loss_val, val_loss)
            if train_loss < min_loss:
                min_loss = train_loss
            
        return train_loss, min_loss