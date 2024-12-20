# -*- coding: utf-8 -*-
"""

A type of RNN to generate abstract output (given size)
"""

import numpy as np
import torch
from torch import nn

TAU = 50
g = 1.5


class rRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(rRNNCell, self).__init__()

        self.w_ih = nn.Linear(input_size, hidden_size)
        self.w_hh = nn.Linear(hidden_size, hidden_size)
        self.w_ho = nn.Linear(hidden_size, output_size)

        # self.y_0 = nn.init.xavier_uniform_(torch.Tensor(1, output_size))
        # self.y_0 = nn.Parameter(self.y_0, requires_grad=True)

        nn.init.normal_(self.w_hh.weight, mean=0, std=g / np.sqrt(hidden_size))

        # only for sparse initialization
        # nn.init.sparse_(self.w_hh.weight, sparsity=0.1)

        nn.init.constant_(self.w_ih.weight, 0)
        nn.init.constant_(self.w_ho.weight, 0)

    def forward(self, input_now, hidden_last):
        # if output_last is None:
        #     output_last = self.y_0

        relu_filter = nn.ReLU()
        r_last = relu_filter(torch.tanh(hidden_last))

        # dynamic evolving rule
        h_now = ((1 - 1 / TAU) * hidden_last 
                 + (self.w_ih(input_now) + self.w_hh(r_last)) / TAU)

        r_now = relu_filter(torch.tanh(h_now))
        output_now = self.w_ho(r_now)

        return h_now, r_now, output_now


class rRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(rRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.cell = rRNNCell(input_size, hidden_size, output_size)

    def forward(self, input_series, initial_states):
        time_steps = input_series.size(1)
        # (batch, time, neuron)

        h = initial_states

        h_series = []
        r_series = []
        z_series = []

        for t in range(time_steps):
            h, r, z = self.cell(input_series[:, t, :], h)
            h_series.append(h)
            r_series.append(r)
            z_series.append(z)

        return (torch.stack(h_series, dim=1),
                torch.stack(r_series, dim=1),
                torch.stack(z_series, dim=1))
