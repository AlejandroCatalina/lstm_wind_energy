import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_svmlight_file
from torch.utils.data import DataLoader

from model import LSTM, CLSTM
from data import Sotavento

# Network params
input_channels = 8
height = 15
width = 8
input_dim = input_channels * height * width
hidden_dim = 32
output_dim = 1
num_lstm_layers = 1
learning_rate = 1e-3
num_epochs = 1000
batch_size = 100
dtype = torch.float

#####################
# Load data
#####################
stv_train = Sotavento('/Users/alex/dev/lstm_wind_energy/datasets/')
stv_test = Sotavento('/Users/alex/dev/lstm_wind_energy/datasets/', train = False)

#####################
# Build model
#####################
model = LSTM(
    input_dim,
    hidden_dim,
    batch_size=batch_size,
    output_dim=output_dim,
    num_layers=num_lstm_layers)

loss_fn = torch.nn.MSELoss(size_average=False)

optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

#####################
# Train model
#####################

hist = np.zeros(num_epochs)

for t in range(num_epochs):
    train_loader = DataLoader(stv_train, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(stv_test, batch_size=batch_size, shuffle=False)
    hist = []
    for i, train in enumerate(train_loader, 0):
        X, y = train
        if len(X) != batch_size:
            pad_size = batch_size - len(X)
            padding_X = torch.zeros((pad_size, input_dim))
            padding_y = torch.zeros((pad_size))
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

        # Zero out gradient, else they will accumulate between epochs
        optimiser.zero_grad()

        # Backward pass
        loss.backward()

        # Update parameters
        optimiser.step()

    # after iteration compute train error
    if t % 100 == 0:
        print("Epoch ", t, "Train MSE: ", np.sum(hist))
        with torch.no_grad():
            # compute test error over mini-batches
            test_loss = []
            for j, test in enumerate(test_loader, 0):
                X_test, y_test = test
                test_size = batch_size
                if len(X_test) != batch_size:
                    test_size = len(X_test)
                    pad_size = batch_size - len(X_test)
                    padding_X = torch.zeros((pad_size, X_test.shape[1]))
                    padding_y = torch.zeros((pad_size))
                    X_test = torch.cat((X_test, padding_X))
                    y_test = torch.cat((y_test, padding_y))

                y_pred_test = model(X_test)[:test_size]
                y_test_ = y_test[:test_size]
                test_loss.append(loss_fn(y_pred_test, y_test_).item())
            print("Epoch ", t, "Test MSE: ", np.sum(test_loss))
