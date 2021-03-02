import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn import GRU


def truncate_param(param, value):
    param_copy = param.clone()
    param_copy[torch.abs(param_copy) >= value] = 0.
    return param_copy


class Attention(nn.Module):
    def __init__(self, hidden_size, attention_size, bidirectional,
                 init_std=0.1, init_const=0.1):
        super(Attention, self).__init__()

        if bidirectional:
            hidden_size *= 2
        self.linear1 = nn.Linear(hidden_size, attention_size)
        nn.init.normal(self.linear1.weight, std=init_std)
        nn.init.constant(self.linear1.bias, init_const)
        self.linear2 = nn.Linear(attention_size, 1)
        nn.init.normal(self.linear2.weight, std=init_std)
        nn.init.constant(self.linear2.bias, init_const)

    def forward(self, x):
        v = torch.sigmoid(self.linear1(x))
        alphas = torch.softmax(self.linear2(v), dim=-2)
        output = torch.sum(x * alphas, dim=1)
        return output, alphas


class AttConvRNN(nn.Module):
    def __init__(self, C, H, W, EC, L1=128, L2=256,
                 gru_cell_units=128,
                 attention_size=1,
                 num_linear=768,
                 pool_stride_height=2,
                 pool_stride_width=4,
                 F1=64,
                 bidirectional=True,
                 dropout_keep_prob=1,
                 init_mean=0.,
                 init_std=0.01,
                 init_const=0.01):
        super(AttConvRNN, self).__init__()

        self.conv1 = nn.Conv2d(C, L1, (5, 3), padding=(2, 1))
        nn.init.normal(self.conv1.weight, mean=init_mean, std=init_std)
        self.conv1.weight.data = truncate_param(self.conv1.weight.data, init_mean + init_std * 2.)
        nn.init.constant(self.conv1.bias, init_const)
        self.max_pool1 = nn.MaxPool2d((pool_stride_height, pool_stride_width),
                                      stride=(pool_stride_height, pool_stride_width))

        self.conv2 = nn.Conv2d(L1, L2, (5, 3), padding=(2, 1))
        nn.init.normal(self.conv2.weight, mean=init_mean, std=init_std)
        self.conv2.weight.data = truncate_param(self.conv2.weight.data, init_mean + init_std * 2.)
        nn.init.constant(self.conv2.bias, init_const)
        self.conv3 = nn.Conv2d(L2, L2, (5, 3), padding=(2, 1))
        nn.init.normal(self.conv3.weight, mean=init_mean, std=init_std)
        self.conv3.weight.data = truncate_param(self.conv3.weight.data, init_mean + init_std * 2.)
        nn.init.constant(self.conv3.bias, init_const)
        self.conv4 = nn.Conv2d(L2, L2, (5, 3), padding=(2, 1))
        nn.init.normal(self.conv4.weight, mean=init_mean, std=init_std)
        self.conv4.weight.data = truncate_param(self.conv4.weight.data, init_mean + init_std * 2.)
        nn.init.constant(self.conv4.bias, init_const)
        self.conv5 = nn.Conv2d(L2, L2, (5, 3), padding=(2, 1))
        nn.init.normal(self.conv5.weight, mean=init_mean, std=init_std)
        self.conv5.weight.data = truncate_param(self.conv5.weight.data, init_mean + init_std * 2.)
        nn.init.constant(self.conv5.bias, init_const)
        self.conv6 = nn.Conv2d(L2, L2, (5, 3), padding=(2, 1))
        nn.init.normal(self.conv6.weight, mean=init_mean, std=init_std)
        self.conv6.weight.data = truncate_param(self.conv6.weight.data, init_mean + init_std * 2.)
        nn.init.constant(self.conv6.bias, init_const)

        self.linear1 = nn.Linear(L2 * W // pool_stride_width,
                                 num_linear)
        nn.init.normal(self.linear1.weight, mean=init_mean, std=init_std)
        self.linear1.weight.data = truncate_param(self.linear1.weight.data, init_mean + init_std * 2.)
        nn.init.constant(self.linear1.bias, init_const)

        self.gru = nn.GRU(num_linear, gru_cell_units, batch_first=True, bidirectional=bidirectional)
        bias_len = len(self.gru.bias_hh_l0)
        nn.init.constant(self.gru.bias_hh_l0[bias_len // 4:bias_len // 2], 1.)
        nn.init.constant(self.gru.bias_ih_l0[bias_len // 4:bias_len // 2], 1.)
        if bidirectional:
            nn.init.constant(self.gru.bias_hh_l0_reverse[bias_len // 4:bias_len // 2], 1.)
            nn.init.constant(self.gru.bias_ih_l0_reverse[bias_len // 4:bias_len // 2], 1.)
        self.attention = Attention(gru_cell_units,
                                   attention_size=attention_size,
                                   bidirectional=bidirectional,
                                   init_std=init_std, init_const=init_const)

        self.linear2 = nn.Linear(gru_cell_units * (2 if bidirectional else 1), F1)
        nn.init.normal(self.linear2.weight, mean=init_mean, std=init_std)
        self.linear2.weight.data = truncate_param(self.linear2.weight.data, init_mean + init_std * 2.)
        nn.init.constant(self.linear2.bias, init_const)
        self.linear3 = nn.Linear(F1, EC)
        nn.init.normal(self.linear3.weight, mean=init_mean, std=init_std)
        self.linear3.weight.data = truncate_param(self.linear3.weight.data, init_mean + init_std * 2.)
        nn.init.constant(self.linear3.bias, init_const)

        self.activation = nn.ELU()  # nn.LeakyReLU(1e-2)

    def forward(self, x):
        x_01 = self.activation(self.conv1(x))
        x_02 = self.max_pool1(x_01)
        x_03 = self.activation(self.conv2(x_02))
        x_04 = self.activation(self.conv3(x_03))
        x_05 = self.activation(self.conv4(x_04))
        x_06 = self.activation(self.conv5(x_05))
        x_07 = self.activation(self.conv6(x_06)).permute(0, 2, 1, 3)
        x_08 = self.activation(self.linear1(x_07.contiguous().view(x_07.shape[0], x_07.shape[1], -1)))
        x_09, _ = self.gru(x_08)
        x_10, alphas = self.attention(x_09)
        # x_10, alphas = self.attention(x_08)
        x_11 = self.activation(self.linear2(x_10))
        x_12 = self.linear3(x_11)

        # if torch.max(x_12[0] - x_12[1]) < 1e-3 and torch.min(x_12[0] - x_12[1]) < 1e-3:
        #     stop
        # var = x_11
        # print([torch.max(var), torch.min(var), torch.mean(var), torch.std(var)])
        # print([torch.max(torch.abs(var[0] - var[1])), torch.min(torch.abs(var[0] - var[1])),
        #        torch.mean(torch.abs(var[0] - var[1])), torch.std(torch.abs(var[0] - var[1]))])
        # torch.max(torch.abs(x_12[0] - x_12[1])) < 1e-5 and torch.min(torch.abs(x_12[0] - x_12[1])) < 1e-5

        return x_12
