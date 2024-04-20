import copy
import numpy as np
import pandas as pd

import torch
from torch import nn
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.regression import MeanAbsoluteError
from sklearn.preprocessing import StandardScaler

import optuna
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

from darts import TimeSeries
from darts.models import TCNModel
from darts.dataprocessing.transformers import Scaler


torch.set_float32_matmul_precision('high')
random_seed = 42
n_epochs = 150
BATCH_SIZE = 1024
learning_rate = 0.03
torch.cuda.manual_seed(random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)
torch.backends.cudnn.deterministic = True

scaler = StandardScaler()
transformer = Scaler(scaler)
df = pd.read_csv("../data/exchange_rate_imputed.csv")
df['date'] = pd.to_datetime(df['date'])
columns = ['0', '1', '2', '3', '4', '5', '6', 'OT']


train = TimeSeries.from_dataframe(df.iloc[:-700], "date", ['0', '1', '2', '3', '4', '5', '6', 'OT'])
val = TimeSeries.from_dataframe(df.iloc[-700:], "date", ['0', '1', '2', '3', '4', '5', '6', 'OT'])
to_forecast = TimeSeries.from_dataframe(df.iloc[-700:], "date", ['0', '1', '2', '3', '4', '5', '6', 'OT'])

transformer.fit(train)
train = transformer.transform(train)
val = transformer.transform(val)
to_forecast = transformer.transform(to_forecast)

y_true = df.iloc[-100:]['6'].values
y_var = transformer._fitted_params[0].var_[6]
y_mean =  transformer._fitted_params[0].mean_[6]



def objective(trial):
    
    # input/output sizes hyperparameter
    input_chunk_length = trial.suggest_categorical("input_chunk_length", [365, 512])
    output_chunk_length = trial.suggest_categorical("output_chunk_length", [10,20])
    
    # Define LSTM hyperparameters
    k = trial.suggest_categorical("kernel_size", [2,3, 4, 5])
    d = trial.suggest_categorical("dilation_base", [2,3,4,5])
    n = trial.suggest_int("num_layers", 1, 4)
    num_filters = trial.suggest_int("num_filters", 2, 5)
    dropout = trial.suggest_float("dropout", 0.0, 0.8, step=0.1)

    # stop training when validation loss does not decrease more than 0.0001 (`min_delta`) over
    # a period of 9 epochs (`patience`)
    my_stopper = EarlyStopping(
    monitor="val_loss",
    patience=9,
    min_delta=0.0001,
    mode='min',)
    pl_trainer_kwargs={"callbacks": [my_stopper]}
    
    # Define the criterion (MAE)
    criterion = MeanAbsoluteError()
    
    
    model = TCNModel(
    input_chunk_length=input_chunk_length,
    output_chunk_length=output_chunk_length,
    n_epochs=n_epochs,
    batch_size = BATCH_SIZE,
    random_state = random_seed,
    optimizer_kwargs = {'lr': learning_rate},
    lr_scheduler_kwargs = {"patience":4} ,
    pl_trainer_kwargs = pl_trainer_kwargs,
    lr_scheduler_cls = ReduceLROnPlateau,
    torch_metrics = criterion,
    kernel_size=k, num_filters=num_filters, num_layers=2, dilation_base=d, weight_norm=False, dropout=dropout,
    )

    model.fit(train, val_series = val)   
    pred = model.predict(n = 100, series = to_forecast)
    pred = (pred['6'].values() * np.sqrt(y_var)) + y_mean
    mae = mean_absolute_error(y_true, pred) 

    return mae

