import torch
import torch.nn as nn
import numpy as np

class LSTM(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 batch_size,
                 output_dim=1,
                 num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (num_layers, batch_size, hidden_dim).
        lstm_out, self.hidden = self.lstm(
            input.view(-1, self.batch_size, self.input_dim))

        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
        return y_pred.view(-1)


class CLSTM(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 batch_size,
                 output_dim=1,
                 num_lstm_layers=2,
                 kernel_size=2,
                 input_channels=8,
                 height=15,
                 width=8):
        super(CLSTM, self).__init__()
        self.height = height
        self.width = width
        self.input_channels = input_channels
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_lstm_layers = num_lstm_layers
        self.kernel_size = kernel_size

        # Define the Conv layer
        self.conv = nn.Conv2d(self.input_channels, self.hidden_dim,
                              self.kernel_size)
        # output is (num_train, hidden_dim, out_1, out_2)
        self.relu = nn.ReLU()

        self.max_pool = nn.MaxPool2d(kernel_size=kernel_size)
        # output is (num_train,  hidden_dim, out_11, out_22)

        out_1 = int(
            np.floor((self.height - 1 * (self.kernel_size - 1) - 1) / 1 + 1))
        out_2 = int(
            np.floor((self.width - 1 * (self.kernel_size - 1) - 1) / 1 + 1))
        out_11 = int(
            np.floor((out_1 - 1 * (self.kernel_size - 1) - 1) / self.kernel_size + 1))
        out_22 = int(
            np.floor((out_2 - 1 * (self.kernel_size - 1) - 1) / self.kernel_size + 1))
        self.dim = self.hidden_dim * out_11 * out_22

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.dim, self.hidden_dim, self.num_lstm_layers)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_lstm_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_lstm_layers, self.batch_size, self.hidden_dim))

    def forward(self, input):
        input_view = input.view(self.batch_size, self.height, self.width,
                                self.input_channels)
        input_tr = input_view.permute([0, 3, 1, 2])
        conv_out = self.relu(self.max_pool(self.conv(input_tr)))

        # Forward pass through LSTM layer
        # shape input to LSTM must be (input_size, batch_size, dim)
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (num_lstm_layers, batch_size, hidden_dim).
        lstm_out, self.hidden = self.lstm(
            conv_out.view(-1, self.batch_size, self.dim))

        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
        return y_pred.view(-1)
