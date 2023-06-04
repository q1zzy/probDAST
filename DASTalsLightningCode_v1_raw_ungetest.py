# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 15:22:54 2023

@author: mb
"""

import os
import math
import time
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.autograd import Variable
from torch.utils.data import TensorDataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from DAST_main import DAST
import matplotlib.pyplot as plt


data_path = './path/to/your/data'


class DASTLightning(pl.LightningModule):
    def __init__(self, config):
        super(DASTLightning, self).__init__()
        self.config = config
        self.model = DAST(                                              #Das wohl so umschreiben mit oben import, dass es DAST modell importiert ?!
            config['dim_val_s'], config['dim_attn_s'],
            config['dim_val_t'], config['dim_attn_t'],
            config['dim_val'], config['dim_attn'],
            config['time_step'], config['input_size'],
            config['dec_seq_len'], config['output_sequence_length'],
            config['n_decoder_layers'], config['n_encoder_layers'],
            config['n_heads']
        )
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.config['lr'])

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = torch.sqrt(self.criterion(out * self.config['max_rul'], y * self.config['max_rul']))
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        out[out < 0] = 0
        loss = torch.sqrt(self.criterion(out * self.config['max_rul'], y * self.config['max_rul']))
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def myScore(self, Target, Pred):
        tmp1 = 0
        tmp2 = 0
        for i in range(len(Target)):
            if Target[i] > Pred[i]:
                tmp1 = tmp1 + math.exp((-Pred[i] + Target[i]) / 13.0) - 1
            else:
                tmp2 = tmp2 + math.exp((Pred[i] - Target[i]) / 10.0) - 1
        tmp = tmp1 + tmp2
        return tmp

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('avg_val_loss', avg_loss, prog_bar=True, logger=True)

        y_true = torch.cat([x['y'] for x in outputs]).detach().numpy()
        y_pred = torch.cat([x['pred'] for x in outputs]).detach().numpy()
        test_score = self.myScore(y_true * self.config['max_rul'], y_pred * self.config['max_rul'])
        self.log('test_score', test_score, prog_bar=True, logger=True)

# Your data loading and preprocessing code here
# ...

# Create the PyTorch Lightning model
config = {
    'dim_val_s': 64,
    'dim_attn_s': 64,
    'dim_val_t': 64,
    'dim_attn_t': 64,
    'dim_val': 64,
    'dim_attn': 64,
    'time_step': 42,
    'input_size': 14,
    'dec_seq_len': 4,
    'output_sequence_length': 1,
    'n_decoder_layers': 1,
    'n_encoder_layers': 2,
    'n_heads': 4,
    'max_rul': 125,
    'lr': 0.001,
    'batch_size': 2048,
    'epochs': 80
}

model = DASTLightning(config)

# Create the PyTorch Lightning data module
class DASTDataModule(pl.LightningDataModule):
    def __init__(self, config, X_train, Y_train, X_test, Y_test):
        super().__init__()
        self.config = config
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        
    def prepare_data(self):
        # Load and process the data
        X_train = sio.loadmat(f'{self.data_path}/slide_window/F002_window_size_trainX_new.mat')['train1X_new']
        X_train = X_train.reshape(len(X_train), 42, 14)
        Y_train = sio.loadmat(f'{self.data_path}/slide_window/F002_window_size_trainY.mat')['train1Y'].transpose()
        X_test = sio.loadmat(f'{self.data_path}/slide_window/F002_window_size_testX_new.mat')['test1X_new']
        X_test = X_test.reshape(len(X_test), 42, 14)
        Y_test = sio.loadmat(f'{self.data_path}/slide_window/F002_window_size_testY.mat')['test1Y'].transpose()

        self.X_train = Variable(torch.Tensor(X_train).float())
        self.Y_train = Variable(torch.Tensor(Y_train).float())
        self.X_test = Variable(torch.Tensor(X_test).float())
        self.Y_test = Variable(torch.Tensor(Y_test).float())

    def train_dataloader(self):
        train_dataset = TensorDataset(self.X_train, self.Y_train)
        return Data.DataLoader(dataset=train_dataset, batch_size=self.config['batch_size'], shuffle=True)

    def val_dataloader(self):
        test_dataset = TensorDataset(self.X_test, self.Y_test)
        return Data.DataLoader(dataset=test_dataset, batch_size=self.config['batch_size'], shuffle=False)


data_module = DASTDataModule(config, data_path)


# Train the model
logger = TensorBoardLogger('tb_logs', name='DAST')
trainer = Trainer(max_epochs=config['epochs'], logger=logger)
trainer.fit(model, data_module)
