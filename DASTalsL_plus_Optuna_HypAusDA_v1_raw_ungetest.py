# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 15:27:42 2023

@author: mb
"""

import optuna
from optuna.integration import PyTorchLightningPruningCallback
import torch
import pytorch_lightning as pl
from pytorch_lightning import Callback
import math

data_path = './path/to/your/data'


class DASTLightning(pl.LightningModule):
    def __init__(self, config):
        super(DASTLightning, self).__init__()
        self.config = config
        self.model = DAST(
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


# Define Callback Class
class MetricsCallback(Callback):
    def __init__(self):
        super().__init__()
        self.train_loss = []
        self.train_acc = []
        self.val_acc = []
        self.val_loss = []

    def on_validation_end(self, trainer, pl_module):
        self.val_loss.append(trainer.callback_metrics['val_loss'].cpu().detach().numpy())
        self.val_acc.append(trainer.callback_metrics['val_acc'].cpu().detach().numpy())

    def on_train_epoch_end(self, trainer, pl_module):
        self.train_loss.append(trainer.callback_metrics['train_loss'].cpu().detach().numpy())
        self.train_acc.append(trainer.callback_metrics['train_acc'].cpu().detach().numpy())

# Objective function for optimization
def objective(trial, datamodule):
    # Define optimization parameters
    params = {
        'batch_size': trial.suggest_categorical('batch_size', [512, 1024, 1536, 2048]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.1, 0.01, 0.001]),
        'hidden_units': trial.suggest_int('hidden_units', 64, 256, step=64),
        'n_heads': trial.suggest_int('n_heads', 1, 16, step = 2),
        'n_encoder_layers': trial.suggest_int('n_encoder_layers', [1, 2, 3]),
        'n_decoder_layers': trial.suggest_int('n_decoder_layers', [1, 2, 3]),
        'dropout': trial.suggest_float('dropout', 0.0, 0.9, step=0.1)
    }

    # Create model instance with parameters
    model = DASTLightningModel(params=params)

    # Custom callback function
    metrics_cb = MetricsCallback()

    # Pruning callback for faster optimization
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_acc")

    # Trainer
    trainer = pl.Trainer(
        logger=True,
        log_every_n_steps=2,
        check_val_every_n_epoch=1,
        enable_checkpointing=False,
        max_epochs=80,
        gpus=[1],
        callbacks=[pruning_callback, metrics_cb],
        enable_model_summary=False,
        enable_progress_bar=False
    )

    trainer.logger.log_hyperparams(params)
    trainer.fit(model, datamodule=datamodule)

    # Target metric
    target_list = np.array(metrics_cb.val_acc)
    target = target_list.max()

    # Free GPU memory
    torch.cuda.empty_cache()

    # Return target metric
    return target.round(6)

# Load data
data_module = DASTDataModule(config, data_path)

# Define optimization task
obj_func = lambda trial: objective(trial, data_module)

# Define pruner
pruner = optuna.pruners.HyperbandPruner(
    min_resource=8,
    max_resource='auto',
    reduction_factor=3,
    bootstrap_count=0
)

# Define sampler (optimization strategy)
sampler = optuna.samplers.TPESampler()

# Define study
study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

# Run optimization task
study.optimize(obj_func, n_trials=400)
trial = study.best_trial.params
trial['objective'] = study.best_trial.values[0]

# Save results
with open(os.path.join(results_folder_path, 'HP_DAST_{}.json'.format(seq_len)), 'w') as fp:
    json.dump(trial, fp)
