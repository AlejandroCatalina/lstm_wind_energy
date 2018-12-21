import numpy as np
import pyro
import pyro.distributions as dist
import pyro.infer
import pyro.optim
import torch
import torch.nn as nn
from torch.distributions import LogNormal, Normal


class BNN(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 batch_size,
                 output_dim=1,
                 num_dense_layers=1,
                 dropout=0.2):
        super(BNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.dropout = dropout
        self.num_dense_layers = num_dense_layers

        total_layers = self.num_dense_layers
        if isinstance(self.hidden_dim, list):
            if len(self.hidden_dim) != total_layers:
                pad = nn.ConstantPad1d((0, total_layers), self.hidden_dim[-1])
                self.hidden_dim = pad(torch.IntTensor(self.hidden_dim))
                self.hidden_dim = self.hidden_dim.numpy().tolist()
        else:
            self.hidden_dim = [self.hidden_dim] * total_layers

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(p=self.dropout)

        # define dense layers
        self.dense = []
        dim = input_dim
        for hidden_units in self.hidden_dim:
            self.dense.append(nn.Linear(dim, hidden_units).cuda())
            dim = hidden_units
        self.dense.append(nn.Linear(dim, output_dim).cuda())

        # Define the output layer
        # self.linear = nn.Linear(self.dim, output_dim)

    def forward(self, input):
        input_dense = input
        for linear in self.dense:
            input_dense = self.dropout(
                linear(input_dense.view(self.batch_size, -1)))
        y_pred = input_dense
        return y_pred.view(-1)

    def model(self, input, targets):

        priors = {}
        for i, dense in enumerate(self.dense):
            dw_prior = Normal(
                loc=torch.zeros_like(dense.weight),
                scale=torch.ones_like(dense.weight))
            db_prior = Normal(
                loc=torch.zeros_like(dense.bias),
                scale=torch.ones_like(dense.bias))
            priors['dense[{}].weight'.format(i)] = dw_prior
            priors['dense[{}].bias'.format(i)] = db_prior

        # set prior on scale
        scale = LogNormal(0.0, 1.0)

        # lift module parameters to random variables sampled from the priors
        lifted_module = pyro.random_module("bnn", self, priors)
        # sample a regressor (which also samples w and b)
        lifted_reg_model = lifted_module()

        pred = lifted_reg_model(input)

        pyro.sample("obs", Normal(loc=pred, scale=scale), obs=targets)
