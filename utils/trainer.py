import torch
import os
import tqdm
import numpy as np
from utils.logger import Logger
from sklearn.metrics import mean_absolute_error, mean_squared_error

from utils.model import Geometry_estimator, MLP_reflectionCoeff, Link_model

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
                #model,      # model to train
                model_no,    # model number for selection
                lr,         # learning rate
                #optimizer,  # optimizer to use
                criterion,  # loss function to use
                train_loader,  # data loader for training
                val_loader,   # data loader for validation
                early_stopping_patience, # early stopping

                ):
        #self._model = model.type(dtype)
        # 
        if model_no == 1:
            self._model= Geometry_estimator().type(dtype)
            self.train_step = self.train_step_Geo
            self.val_step = self.val_test_step_Geo
            self.save_model = self.save_checkpoint_Geo
            self.concat_batchLabels = self.concat_batchLables_Geo
            logger.update_dir('./logs_Geo')
        elif model_no == 2:
            self._model= MLP_reflectionCoeff().type(dtype)
            self.train_step = self.train_step_reCoeff
            self.val_step = self.val_test_step_reCoeff
            self.save_model = self.save_checkpoint_ReCoeff
            self.concat_batchLabels = self.concat_batchLables_ReCoeff
            logger.update_dir('./logs_reCoeff')
        elif model_no == 3:
            self._model= Link_model().type(dtype)
            self.train_step = self.train_step_Link
            self.val_step = self.val_test_step_Link
            self.save_model = self.save_checkpoint_Link
            self.concat_batchLabels = self.concat_batchLables_Link
            logger.update_dir('./logs_Link')
        else:
            raise NotImplementedError("invalid model number, should be between 1-3")
        # define optimizer here for modularity
        self.optimizer = torch.optim.Adam(self._model.parameters(), lr=0.001)
        self.criterion = criterion.type(dtype)
        self.train_batches= train_loader
        self.val_batches= val_loader
        self._early_stopping_patience = early_stopping_patience

    def save_checkpoint_Geo(self, epoch):
        torch.save({'state_dict': self._model.state_dict()}, 'Geo_checkpoints/checkpoint_{:03d}.ckp'.format(epoch))

    def save_checkpoint_ReCoeff(self, epoch):
        torch.save({'state_dict': self._model.state_dict()}, 'ReCoeff_checkpoints/checkpoint_{:03d}.ckp'.format(epoch))

    def save_checkpoint_Link(self, epoch):
        torch.save({'state_dict': self._model.state_dict()}, 'Link_checkpoints/checkpoint_{:03d}.ckp'.format(epoch))
    
    def restore_checkpoint(self, epoch_n):
        ckp = torch.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), device)
        self._model.load_state_dict(ckp['state_dict'])

    def train_step_Geo(self, rir_features, geo_labels, coeff_l, batch_idx):
        # reset zero grad
        self.optimizer.zero_grad()
        #
        pred = self._model(rir_features.unsqueeze(axis=1))
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

    def train_step_reCoeff(self, rir_features, geo_labels, coeff_l, batch_idx):
        # reset zero grad
        self.optimizer.zero_grad()
        #
        pred = self._model(rir_features)
        #print("in train", pred.squeeze(axis=1).size(), coeff_l.size() )
        # calculate loss
        loss = self.criterion(pred, coeff_l)
        #print("loss")
        #print("loss", batch_idx, loss)
        if torch.isnan(loss):
            print("is nan2", batch_idx)
            return 0
        else:   
            #print("not nan loss: ", batch_idx, loss)
        #
            loss.backward()
        #
            self.optimizer.step()

            return loss.detach().item()
        
    def train_step_Link(self, rir_features, geo_labels, coeff_l, batch_idx):
        batch_size = rir_features.size()[0]
        # reset zero grad
        self.optimizer.zero_grad()
        #
        pred = self._model(rir_features, geo_labels).reshape(batch_size,3,3)
        #print("in train", pred.squeeze(axis=1).size(),room_geo.unsqueeze(axis=1), coeff_l.reshape(3,2).size() )
        # calculate loss
        #labels = torch.cat((geo_labels.reshape(batch_size, 3, 1), coeff_l.reshape(batch_size,3,2)), dim=2)
        labels = self.reshape_lables3x3(batch_size, geo_labels, coeff_l)
        # loss
        loss = self.criterion(pred, labels)
        #print("loss", batch_idx, loss)
        if torch.isnan(loss):
            print("is nan2", batch_idx)
            return 0
        else:   
        #
            loss.backward()
        #
            self.optimizer.step()

            return loss.detach().item()
        
    def val_test_step_Geo(self, rir_features, geo_labels, coeff_l, batch_idx):
        # predict
        pred = self._model(rir_features.unsqueeze(axis=1))
        # loss
        loss = self.criterion(pred, geo_labels)
        # 
        if torch.isnan(loss):
            return 0
        else:
            return loss.detach().item(), pred.detach()
    
    def val_test_step_reCoeff(self, rir_features, geo_labels, coeff_l, batch_idx):
        # predict
        pred = self._model(rir_features)
        # loss
        loss = self.criterion(pred, coeff_l)
        # 
        if torch.isnan(loss):
            return 0
        else:
            return loss.detach().item(), pred.detach()
    
    def val_test_step_Link(self, rir_features, geo_labels, coeff_l, batch_idx):
        batch_size = rir_features.size()[0]
        # predict
        pred = self._model(rir_features, geo_labels).reshape(batch_size,3,3)
        # Label reshaping in 3x3
        #labels = torch.cat((geo_labels.reshape(batch_size, 3, 1), coeff_l.reshape(batch_size,3,2)), dim=2)
        labels = self.reshape_lables3x3(batch_size, geo_labels, coeff_l)
        # loss
        loss = self.criterion(pred, labels)
        # 
        if torch.isnan(loss):
            return 0
        else:
            return loss.detach().item(), pred.detach()



    def train_epoch(self):
        # set training mode
        self._model.train()
        # iterate through the training set
        loss = 0
        #for idx, batches in tqdm.tqdm(enumerate(self.train_batches), unit='batch', total=len(self.train_batches), desc='loading batches'):
        for idx, batches in enumerate(self.train_batches):
            geo_l, coeff_l, rir_f = batches
            geo_l = geo_l.type(dtype)
            coeff_l = coeff_l.type(dtype)
            rir_f = rir_f.type(dtype)
                
            # perform a training step
            loss_step = self.train_step(rir_f, geo_l, coeff_l, idx)
            #loss_step = self.train_step_reCoeff(rir_f, coeff_l, idx)
            #loss_step = self.train_step_Link(rir_f, geo_l, coeff_l, idx)
            loss += loss_step
        # calculate the average loss for the epoch and return it
        avg_loss = loss/self.train_batches.__len__()

        return avg_loss

    def val_epoch(self):
        # eval mode
        self._model.eval()
        # 
        loss = 0
        batch_pred = torch.empty(0)
        batch_labels = torch.empty(0)
        #
        with torch.no_grad():
            for idx, batches in enumerate(self.val_batches):
                geo_l, coeff_l, rir_f = batches
                geo_l = geo_l.type(dtype)
                coeff_l = coeff_l.type(dtype)
                rir_f = rir_f.type(dtype)
                # validation step and add loss
                loss_step, pred = self.val_step(rir_f, geo_l, coeff_l, idx)
                #loss_step, pred = self.val_test_step_reCoeff(rir_f, coeff_l, idx)
                #loss_step, pred = self.val_test_step_Link(rir_f, geo_l, coeff_l, idx)

                loss += loss_step
                # save prediciton
                batch_pred = torch.cat((batch_pred, pred.cpu()))
                #batch_labels = torch.cat((batch_labels, geo_l.cpu()))
                batch_labels = self.concat_batchLabels(batch_labels, geo_l.cpu(), coeff_l.cpu())
            # avg loss
            avg_loss = loss/self.val_batches.__len__()
            #print("inval", batch_labels.size(), batch_pred.size())
            # calculate score
            bias = self.Bias_Calc(batch_pred, batch_labels)
            print("sCore:", bias)
            # print score
            return avg_loss
        
    # fucntion to reshape labels for 3rd model
    def reshape_lables3x3(self, batch_size, geo_labels, coeff_l):
        return torch.cat((geo_labels.reshape(batch_size, 3, 1), coeff_l.reshape(batch_size,3,2)), dim=2)
    
    # functions required for validation / testing
    def concat_batchLables_Geo(self, batch_labels, geo_l, coeff_l):
        return torch.cat((batch_labels, geo_l))
    def concat_batchLables_ReCoeff(self, batch_labels, geo_l, coeff_l):
        return torch.cat((batch_labels, coeff_l))
    def concat_batchLables_Link(self, batch_labels, geo_l, coeff_l):
        return torch.cat((batch_labels, self.reshape_lables3x3(geo_l.size()[0], geo_l, coeff_l)))

    def score_func(self, batch_pred, batch_labels):
        # Bias measures the mean deviation of our estimates from the true value
        bias = np.mean(np.abs(batch_labels - batch_pred), axis=0)
        # rmse 
        rmse = np.sqrt(np.mean((np.square(batch_labels - batch_pred)), axis=0))
        # variance
        var = np.var(batch_pred, axis=0)
        #
        return bias, var, rmse

    def fit(self, epochs=-1):
        assert epochs > 0, 'Epochs > 0'
        #
        loss_train = np.array([])
        loss_val = np.array([])
        epoch_counter = 0
        min_loss = np.Inf
        criteria_counter = 0
        #
        #while (True):
        for i in tqdm.trange(epochs):
            # stop by epoch number
            # if epoch_counter >= epochs:
            #     break
            # increment Counter
            #epoch_counter += 1
            train_loss = self.train_epoch()
            val_loss = self.val_epoch()
            #
            loss_train = np.append(loss_train, train_loss)
            loss_val = np.append(loss_val, val_loss)
            #logging
            logger.scalar_summary("train_Loss", train_loss, i)
            logger.scalar_summary("val_Loss", val_loss, i)
            # save model if better
            if train_loss < min_loss:
                criteria_counter = 0
                min_loss = train_loss
                # save checkpoint
            
            else:
                criteria_counter += 1            
            # early stopping
            if criteria_counter > self._early_stopping_patience:
                print("Early Stopping Criteria activated")
                break

            
        return val_loss, min_loss