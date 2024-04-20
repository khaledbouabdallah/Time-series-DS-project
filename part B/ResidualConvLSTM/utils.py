import copy
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

import optuna
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt

random_seed = 42
torch.cuda.manual_seed(42)
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
BATCH_SIZE = 2048

#torch.autograd.set_detect_anomaly(True)

df = pd.read_csv('exchange_rate_imputed.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# normalization X
columns_ = df.columns
normlizer = StandardScaler().fit(df)
df = pd.DataFrame(normlizer.transform(df),columns = df.columns)


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss, epoch):
        if validation_loss < self.min_validation_loss:
            if self.min_validation_loss - validation_loss >= self.min_delta:
                self.counter = 0
                print(f"New best validation loss epoch {epoch} = {self.min_validation_loss}")  
            self.min_validation_loss = validation_loss     
        else: 
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def plot_train_val(train_losses,validation_losses):
    # Plot the validation losses
    plt.plot(validation_losses, label='Validation Losses', color='orange')
    # Plot the train losses
    plt.plot(train_losses, label='Train Losses', color='blue')
    # Add labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    # Add legend
    plt.legend()
    # Show plot
    plt.show()

def plot_val(validation_losses):
    # Plot the validation losses
    plt.plot(validation_losses[-30:], label='Validation Losses', color='orange')
    # Add labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Last 30 Validation Losses')
    # Add legend
    plt.legend()
    # Show plot
    plt.show()

class ResidualConvLSTM(nn.Module):
    def __init__(self, input_size, hidden_size_cnn, hidden_size_lstm, num_layers, dropout):
        # Call the parent class's __init__ method
        nn.Module.__init__(self)
        
        # Initialize the attributes
        # self.hidden_size = hidden_size
        # self.num_layers = num_layers
        
        # Define 1D convolutional layers
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=hidden_size_cnn, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_size_cnn, out_channels=hidden_size_cnn*2, kernel_size=3, padding=1)
        
        # LSTM layers with residual connections
        self.lstm_blocks = nn.ModuleList([nn.LSTM(input_size=hidden_size_cnn*2 if i == 0 else hidden_size_lstm,
                                                  hidden_size=hidden_size_lstm,
                                                  num_layers=1,
                                                  batch_first=False) for i in range(num_layers)])
        
        # Fully connected layers with dropout
        self.fc_1 = nn.Linear(hidden_size_lstm, 2024)
        self.fc_2 = nn.Linear(2024, 1024)
        self.fc_3 = nn.Linear(1024, 100)   
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        # Convolutional layers
        x = x.permute(1, 2, 0)
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = out.permute(0, 2, 1)
        
        # LSTM layers with residual connections
        for i, lstm_block in enumerate(self.lstm_blocks): 
            if i != 0:
                residual = out
                out, _ = lstm_block(out)
                out = out + residual
            else:
                out, _ = lstm_block(out)
        
        # Extract the last output of the sequence
        out = out[:, -1, :]
        
        # Fully connected layers with dropout
        out = self.relu(self.fc_1(out))
        out = self.dropout(out)
        out = self.relu(self.fc_2(out))
        out = self.fc_3(out)
        # reverse normalization
        return out

# Define the LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.fc_1 = nn.Linear(hidden_size, 256)
        self.fc_2 = nn.Linear(256, 100) 
        self.relu = nn.ReLU()
    
    def forward(self, x):     
        _ , batch_size, features_size = x.shape    
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc_2(out)
        return out
    
def train(model, dataloader, x_val, y_val, criterion, optimizer, epochs=100):
    
    losses_val = []
    losses_train = []
    dataset_size = len(dataloader.dataset)
    best_val = float('inf')

    x_val = x_val.view((8, x_val.shape[1], x_val.shape[2]//8))

    early_stopper = EarlyStopper(patience=30, min_delta=0.000001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10)
    for epoch in range(epochs):
        model.train()
        for id_batch, (x_batch, y_batch) in enumerate(dataloader):
            x_batch = x_batch.unsqueeze(0)
            x_batch = x_batch.view((8, x_batch.shape[1], x_batch.shape[2]//8))
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)  
            loss.backward()
            optimizer.step()
        losses_train.append(loss.item())
        # validation
        model.eval()
        with torch.no_grad():
           output = model(x_val)
           output = (output * np.sqrt(normlizer.var_[6])) + normlizer.mean_[6]
           validation_loss =  criterion(output.squeeze().squeeze(), y_val)
           losses_val.append(validation_loss.item()) 
            
        scheduler.step(validation_loss)
            
                 
        if early_stopper.early_stop(validation_loss, epoch):
            print(f"early stopping in epoch = {epoch}")         
            break
            
        if best_val > validation_loss:
              validation_loss = best_val
              torch.save(model.state_dict(), 'model_weights.pth')
            
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{epochs} train Loss {loss.item()} lr: {optimizer.param_groups[0]["lr"]}')
           
    plot_train_val(losses_train,losses_val)
    plot_val(losses_val)
    
def predict(model, X):
    model.eval()
    with torch.no_grad():
        y_pred = model(X)
    return y_pred.cpu().numpy().flatten()

def create_lags(df, lags,columns=None):
    if columns is None:
        columns = df.columns
        
    df = df.copy()  
    for lag in range(lags):
        lag = lag + 1
        for column in columns:       
            df[f'{column}_lag_{lag}'] = df[column].shift(lag).copy()
    return df

def create_future(df, future, columns=None):
    if columns is None:
        columns = df.columns
        
    df = df.copy()
    for column in columns:
        for step in range(future):
            step = step + 1
            df[f'{column}_future_{step}'] = df[column].shift(-step).copy()
    return df

def create_data(df, lags, steps, x_columns=None, y_columns=None, test_size=100, verbose=False):
    
    cols = set(df.columns.to_list()).difference(set(y_columns))
    
    if x_columns is None:
        x_columns = df.columns
    
    df1 = create_lags(df, lags, x_columns) 
    df1 = create_future(df1, steps, columns=y_columns)
    df1.dropna(inplace=True)
    
    columns_x = [col for col in df1.columns if 'lag' in col]
    columns_y = [col for col in df1.columns if col not in columns_x] 
    columns_y = [col for col in columns_y if col not in cols]
    X = df1[columns_x]
    y = df1[columns_y]
        
    X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
    y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]
    
    X_train = torch.tensor(X_train.to_numpy(), dtype=torch.float32, device = device)
    X_test = torch.tensor(X_test.to_numpy(), dtype=torch.float32, device = device)
    y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32, device = device)
    y_test = torch.tensor(y_test.to_numpy(), dtype=torch.float32, device = device)
       
    if verbose:
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    return X_train, X_test, y_train, y_test

def objective(trial):
    
    # Lag hyperparameter
    lag = trial.suggest_categorical("lag", [180, 365, 545, 720, 900])
    
    # Define LSTM hyperparameters
    hidden_size_lstm = trial.suggest_int("hidden_size_lstm", 64, 512, step=64)
    hidden_size_cnn = trial.suggest_int("hidden_size_cnn", 16, 256, step=16)
    num_layers = trial.suggest_int("num_layers", 2, 6)
    dropout = trial.suggest_float("dropout", 0.1, 0.8, step=0.1)
    #learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-2, log=True)
    learning_rate = 0.0001
       
    # Get training and testing data

    X_train, X_val,  y_train, y_val = create_data(df, lags=lag, steps = 99, y_columns=['6'], test_size=100)

    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    
    # get validation set
    X_val = create_lags(df, lags=lag-1).iloc[-1]
    X_val = torch.tensor(np.expand_dims(X_val.to_numpy(), axis=0), dtype=torch.float32, device = device)
    
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
 
    # Create and train the model
    #model = LSTM(lag*8 , hidden_size, num_layers, dropout)
    model = ResidualConvLSTM(lag , hidden_size_cnn, hidden_size_lstm , num_layers, dropout)
    model = model.cuda()
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  
    train(model, dataloader, X_val.unsqueeze(0), torch.tensor(y_val , dtype=torch.float32, device = device), criterion, optimizer, epochs=500)
         
    # Evaluate the model
    model.load_state_dict(torch.load('model_weights.pth'))
    model.eval()

  
    X_val = X_val.unsqueeze(0)  
    X_val = X_val.view((8, X_val.shape[1], X_val.shape[2]//8))
    
    
    y_pred = predict(model, X_val) 
    y_pred = (y_pred * np.sqrt(normlizer.var_[6])) + normlizer.mean_[6]
    mae = mean_absolute_error(y_val, y_pred.flatten())
    
    return mae