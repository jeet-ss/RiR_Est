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
                train_loader=None,  # data loader for training
                val_loader=None,   # data loader for validation
                test_loader=None,    # test data
                early_stopping_patience=100, # early stopping
                name_variant = '',          # string for differenctiating experiments
                ):
        #self._model = model.type(dtype)
        # 
        if model_no == 1:
            self._model= Geometry_estimator().type(dtype)
            self.train_step = self.train_step_Geo
            self.val_step = self.val_test_step_Geo
            self.concat_batchLabels = self.concat_batchLables_Geo
            self.root_path = 'Geo_checkpoints'
            logger.update_dir('./logs_Geo')
        elif model_no == 2:
            self._model= MLP_reflectionCoeff().type(dtype)
            self.train_step = self.train_step_reCoeff
            self.val_step = self.val_test_step_reCoeff
            self.concat_batchLabels = self.concat_batchLables_ReCoeff
            self.root_path = 'Ref_checkpoints'
            logger.update_dir('./logs_reCoeff')
        elif model_no == 3:
            self._model= Link_model().type(dtype)
            self.train_step = self.train_step_Link
            self.val_step = self.val_test_step_Link
            self.concat_batchLabels = self.concat_batchLables_Link
            self.root_path = 'Link_checkpoints'
            logger.update_dir('./logs_Link')
        else:
            raise ImportWarning("invalid model number, should be between 1-3")
        # define optimizer here for modularity
        self.optimizer = torch.optim.Adam(self._model.parameters(), lr=lr, betas=(0.9, 0.999))
        self.criterion = criterion.type(dtype)
        self.train_batches = train_loader
        self.val_batches = val_loader
        self.test_batches = test_loader
        self._early_stopping_patience = early_stopping_patience

    def save_checkpoint(self, epoch):
        if os.path.isdir(self.root_path) != True:
            os.makedirs(self.root_path)
        torch.save({'state_dict': self._model.state_dict()}, os.path.join(self.root_path,'checkpoint_{:03d}.ckp'.format(epoch)))

    def restore_checkpoint(self, epoch_n):
        ckp = torch.load(os.path.join(self.root_path, 'checkpoint_{:03d}.ckp'.format(epoch_n)), device)
        self._model.load_state_dict(ckp['state_dict'])

    def train_step_Geo(self, rir_features, geo_labels, coeff_l, batch_idx):
        # reset zero grad
        self.optimizer.zero_grad()
        #
        pred = self._model(rir_features.unsqueeze(axis=1))
        #print("in train", pred.size(), geo_labels.size() )
        # calculate loss
        loss_t = self.criterion(pred, geo_labels)
        #loss = torch.mean(torch.Tensor([torch.pow(torch.linalg.norm(i), 2) for i in loss_t]))
        #loss.requires_grad_()
        loss = torch.mean(loss_t)
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

            return loss.detach().item(), loss_t.detach(), pred.detach()

    def train_step_reCoeff(self, rir_features, geo_labels, coeff_l, batch_idx):
        # reset zero grad
        self.optimizer.zero_grad()
        #
        pred = self._model(rir_features)
        #print("in train", pred.squeeze(axis=1).size(), coeff_l.size() )
        # calculate loss
        loss_t = self.criterion(pred, coeff_l)
        loss = torch.mean(loss_t)
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

            return loss.detach().item(), loss_t.detach(), pred.detach()
        
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
        loss_t = self.criterion(pred, labels)
        loss = torch.mean(loss_t)
        #print("loss", batch_idx, loss)
        if torch.isnan(loss):
            print("is nan2", batch_idx)
            return 0
        else:   
        #
            loss.backward()
        #
            self.optimizer.step()

            return loss.detach().item(), loss_t.detach(), pred.detach()
        
    def val_test_step_Geo(self, rir_features, geo_labels, coeff_l, batch_idx):
        # predict
        pred = self._model(rir_features.unsqueeze(axis=1))
        # loss
        loss_t = self.criterion(pred, geo_labels)
        loss = torch.mean(loss_t)
        #print("in geo", loss, loss_t.size(), pred.size())
        # 
        if torch.isnan(loss):
            return 0
        else:
            return loss.detach().item(), loss_t.detach(), pred.detach()
    
    def val_test_step_reCoeff(self, rir_features, geo_labels, coeff_l, batch_idx):
        # predict
        pred = self._model(rir_features)
        # loss
        loss_t = self.criterion(pred, coeff_l)
        loss = torch.mean(loss_t)
        # 
        if torch.isnan(loss):
            return 0
        else:
            return loss.detach().item(), loss_t.detach(), pred.detach()
    
    def val_test_step_Link(self, rir_features, geo_labels, coeff_l, batch_idx):
        batch_size = rir_features.size()[0]
        # predict
        pred = self._model(rir_features, geo_labels).reshape(batch_size,3,3)
        # Label reshaping in 3x3
        #labels = torch.cat((geo_labels.reshape(batch_size, 3, 1), coeff_l.reshape(batch_size,3,2)), dim=2)
        labels = self.reshape_lables3x3(batch_size, geo_labels, coeff_l)
        # loss
        loss_t = self.criterion(pred, labels)
        loss = torch.mean(loss_t)
        # 
        if torch.isnan(loss):
            return 0
        else:
            return loss.detach().item(), loss_t.detach(), pred.detach()



    def train_epoch(self):
        # set training mode
        self._model.train()
        # iterate through the training set
        loss = 0
        batch_pred = torch.empty(0)
        batch_loss = torch.empty(0)
        batch_labels = torch.empty(0)
        #for idx, batches in tqdm.tqdm(enumerate(self.train_batches), unit='batch', total=len(self.train_batches), desc='loading batches'):
        for idx, batches in enumerate(self.train_batches):
            geo_l, coeff_l, rir_f = batches
            geo_l = geo_l.type(dtype)
            coeff_l = coeff_l.type(dtype)
            rir_f = rir_f.type(dtype)
                
            # perform a training step
            loss_step, loss_t, pred = self.train_step(rir_f, geo_l, coeff_l, idx)
            loss += loss_step
            batch_labels = self.concat_batchLabels(batch_labels, geo_l.cpu(), coeff_l.cpu())
            batch_pred = torch.cat((batch_pred, pred.cpu()))
            batch_loss = torch.cat((batch_loss, loss_t.cpu()))
        # calculate the average loss for the epoch and return it
        avg_loss = loss/self.train_batches.__len__()

        return avg_loss, batch_pred, batch_loss, batch_labels

    def val_epoch(self, batches, mode='val'):
        # eval mode
        self._model.eval()
        # 
        loss = 0
        batch_pred = torch.empty(0)
        batch_labels = torch.empty(0)
        batch_loss = torch.empty(0)
        #
        with torch.no_grad():
            for idx, batches in enumerate(batches):
               
                geo_l, coeff_l, rir_f = batches
                geo_l = geo_l.type(dtype)
                coeff_l = coeff_l.type(dtype)
                rir_f = rir_f.type(dtype)
                # validation step and add loss
                loss_step, loss_t, pred = self.val_step(rir_f, geo_l, coeff_l, idx)
                loss += loss_step
                # save prediciton
                batch_pred = torch.cat((batch_pred, pred.cpu()))
                #batch_labels = torch.cat((batch_labels, geo_l.cpu()))
                batch_labels = self.concat_batchLabels(batch_labels, geo_l.cpu(), coeff_l.cpu())
                batch_loss = torch.cat((batch_loss, loss_t.cpu()))
            # avg loss
            avg_loss = loss/self.val_batches.__len__()
            #print("inval", batch_labels.size(), batch_pred.size())
            if mode == 'test':
                # calculate score
                scores = self.score_func(batch_pred, batch_labels)
                # print score
                print("Scores: ", "\n", "1. RMSE: ", scores['rmse'], "\n", "2. Bias: ", scores['bias'], "\n", "3. Var: ", scores['var']  ) # rmse
                #print(scores)
                return avg_loss, scores, batch_pred, batch_loss, batch_labels
            else :
                return avg_loss, batch_pred, batch_loss, batch_labels
        
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
        bias = torch.mean(torch.abs(batch_labels - batch_pred), axis=0)
        # rmse 
        rmse = torch.sqrt(torch.mean((torch.square(batch_labels - batch_pred)), axis=0))
        # variance
        var = torch.var(batch_pred, axis=0)
        #
        return {"bias":bias,"var":var,"rmse":rmse}
    
    def test_model(self, epoch):
        '''
        epoch is checkpoint number to load mmodel
        '''
        # call to restore checkpoint
        self.restore_checkpoint(epoch)
        # run test batch
        test_loss, scores = self.val_epoch(self.test_batches, mode='test')

        return  {
            "test_loss" : test_loss,
            "scores" : scores
        }
    

    def fit(self, epochs_start=1, epochs_end=0):
        epochs = epochs_end - epochs_start
        assert epochs > 0, 'Epochs > 0'
        #
        loss_train = np.array([])
        pred_train = torch.empty(0)
        loss_train_t = torch.empty(0)
        label_train = torch.empty(0)

        loss_val = np.array([])
        pred_val = torch.empty(0)
        loss_val_t = torch.empty(0)
        label_val = torch.empty(0)
        min_loss = np.Inf
        criteria_counter = 0
        #
        #while (True):
        for i in range(epochs_start, epochs_end):
            
            train_loss, train_pred, train_loss_t, train_labels = self.train_epoch()
            val_loss, val_pred, val_loss_t, val_labels = self.val_epoch(self.val_batches)
            #
            loss_train = np.append(loss_train, train_loss)
            pred_train = torch.cat((pred_train, train_pred))
            loss_train_t = torch.cat((loss_train_t, train_loss_t))
            label_train = torch.cat((label_train, train_labels))
            loss_val = np.append(loss_val, val_loss)
            pred_val = torch.cat((pred_val, val_pred))
            loss_val_t = torch.cat((loss_val_t, val_loss_t))
            label_val = torch.cat((label_val, val_labels))
            #logging
            logger.scalar_summary("train_Loss", train_loss, i)
            logger.scalar_summary("val_Loss", val_loss, i)
            # save model if better
            if train_loss < min_loss:
                criteria_counter = 0
                min_loss = train_loss
                # save checkpoint
                self.save_checkpoint(i)
            
            else:
                criteria_counter += 1            
            # early stopping
            if criteria_counter > self._early_stopping_patience:
                print(f"Early Stopping Criteria activated at epoch {i} with criterion counter at {criteria_counter}")
                break
                criteria_counter = 0
            # print
            print(f'at epoch-{i}, trainLoss-{train_loss}, valLoss-{val_loss}')
        
        # run test batch
        test_loss, scores, test_pred, test_loss_t, test_labels = self.val_epoch(self.test_batches, mode='test')

            
        return  {
            "train_loss" : loss_train,
            "train_pred" : pred_train.numpy(),
            "train_loss_time" : loss_train_t.numpy(),
            "train_labels" : label_train.numpy(),
            "test_loss" : test_loss,
            "test_pred" : test_pred.numpy(),
            "test_loss_t" : test_loss_t.numpy(),
            "test_labels" : test_labels.numpy(),
            "val_loss" : loss_val,
            "val_pred" : pred_val.numpy(),
            "val_loss_time" : loss_val_t.numpy(),
            "val_labels" : label_val.numpy(),
            "min_loss" : min_loss,
            "scores" : scores
        }