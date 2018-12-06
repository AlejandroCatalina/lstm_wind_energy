import torch
import torch.nn as nn
import numpy as np
import os

from torch.utils.data import DataLoader

from model import LSTM, CLSTM
from data import Sotavento

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device: ', device)

# Network params
input_channels = 8
height = 15
width = 8
input_dim = input_channels * height * width
hidden_dim = 32
output_dim = 1
num_lstm_layers = 1
learning_rate = 1e-3
num_epochs = 10000
batch_size = 100

#####################
# Load data
#####################
path = os.path.expanduser('~/dev/lstm_wind_energy/datasets/')
stv_train = Sotavento(path)
stv_test = Sotavento(path, train=False)

#####################
# Build model
#####################
model = CLSTM(
    input_dim,
    hidden_dim,
    batch_size=batch_size,
    output_dim=output_dim,
    num_lstm_layers=num_lstm_layers,
    input_channels=input_channels,
    height=height,
    width=width)
model.to(device)

# model = LSTM(
#     input_dim,
#     hidden_dim,
#     batch_size=batch_size,
#     output_dim=output_dim,
#     num_layers=num_lstm_layers)
# model.to(device)

loss_fn = torch.nn.MSELoss(reduction='sum')
mae_fn = torch.nn.L1Loss(reduction='elementwise_mean')

optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

#####################
# Train model
#####################

hist = np.zeros(num_epochs)

for t in range(num_epochs):
    train_loader = DataLoader(stv_train, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(stv_test, batch_size=batch_size, shuffle=False)
    hist = []
    hist_mae = []
    for i, train in enumerate(train_loader, 0):
        X, y = train
        X, y = X.to(device), y.to(device)
        if len(X) != batch_size:
            pad_size = batch_size - len(X)
            padding_X = torch.zeros((pad_size, input_dim)).to(device)
            padding_y = torch.zeros((pad_size)).to(device)
            X = torch.cat((X, padding_X))
            y = torch.cat((y, padding_y))

        X.view(batch_size, -1, 1)
        # Initialise hidden state
        # Don't do this if you want your LSTM to be stateful
        model.hidden = model.init_hidden()

        # Forward pass
        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        hist.append(loss.item())

        mae_loss = mae_fn(y_pred, y)
        hist_mae.append(mae_loss.item())

        # Zero out gradient, else they will accumulate between epochs
        optimiser.zero_grad()

        # Backward pass
        loss.backward()

        # Update parameters
        optimiser.step()

    # after iteration compute train error
    if t % 100 == 0:
        print("Epoch ", t, "Train MSE: ",
              np.sum(hist), "Test MAE: ", np.mean(hist_mae))
        with torch.no_grad():
            # compute test error over mini-batches
            test_mse_loss = []
            test_mae_loss = []
            for j, test in enumerate(test_loader, 0):
                X_test, y_test = test
                X_test, y_test = X_test.to(device), y_test.to(device)
                test_size = batch_size
                if len(X_test) != batch_size:
                    test_size = len(X_test)
                    pad_size = batch_size - len(X_test)
                    padding_X = torch.zeros((pad_size,
                                             X_test.shape[1])).to(device)
                    padding_y = torch.zeros((pad_size)).to(device)
                    X_test = torch.cat((X_test, padding_X))
                    y_test = torch.cat((y_test, padding_y))

                y_pred_test = model(X_test)[:test_size]
                y_test_ = y_test[:test_size]
                test_mse_loss.append(loss_fn(y_pred_test, y_test_).item())
                test_mae_loss.append(mae_fn(y_pred_test, y_test_).item())
            print("Epoch ", t, "Test MSE: ",
                  np.sum(test_mse_loss), "Test MAE: ", np.mean(test_mae_loss))
