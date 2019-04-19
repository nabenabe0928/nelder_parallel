import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import typing 
from numpy import ndarray
from typing import NamedTuple

class HyperParameters(
    NamedTuple( "_HyperParameters",
                [("batch_size", int),
                 ("lr", float),
                 ("momentum", float),
                 ("weight_decay", float),
                 ("width_coef1", int),
                 ("width_coef2", int),
                 ("width_coef3", int),
                 ("n_blocks1", int),
                 ("n_blocks2", int),
                 ("n_blocks3", int),
                 ("drop_rates1", float),
                 ("drop_rates2", float),
                 ("drop_rates3", float),
                 ("lr_decay", float)
                 ])):
    pass

def get_hyperparameters(hp_parser):
    type_hints = typing.get_type_hints(HyperParameters)
    var_names = list(type_hints.keys())
    hp = {var_name: getattr(hp_parser, var_name) for var_name in var_names}

    return HyperParameters(**hp)
    

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride = 1, drop_rate = 0.2, kernel_size = 3):
        super(BasicBlock, self).__init__()
        self.in_is_out = (in_ch == out_ch and stride == 1)
        self.drop_rate = drop_rate
        
        self.shortcut = nn.Sequential() if self.in_is_out else nn.Conv2d(in_ch, out_ch, 1, padding = 0, stride = stride, bias = True)
        self.bn1 = nn.BatchNorm2d(in_ch)        
        self.c1 = nn.Conv2d(in_ch, out_ch, kernel_size, stride = stride, padding = 1, bias = True)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.c2 = nn.Conv2d(out_ch, out_ch, kernel_size, padding = 1, bias = True)

    def forward(self, x):
        h = F.relu(self.bn1(x))
        h = self.c1(h)
        h = F.dropout2d(h, p = self.drop_rate)
        h = F.relu(self.bn2(h))
        h = self.c2(h)

        return h + self.shortcut(x)

class WideResNet(nn.Module):
    def __init__(self, hp_parser):
        super(WideResNet, self).__init__()
        
        self.hyperparameters = get_hyperparameters(hp_parser)

        self.batch_size = self.hyperparameters.batch_size
        self.weight_decay = self.hyperparameters.weight_decay
        self.lr = self.hyperparameters.lr
        self.lr_decay = self.hyperparameters.lr_decay
        self.momentum = self.hyperparameters.momentum

        self.n_blocks = [self.hyperparameters.n_blocks1, self.hyperparameters.n_blocks2, self.hyperparameters.n_blocks3]
        self.in_chs = [ 16, 16 * self.hyperparameters.width_coef1, 32 * self.hyperparameters.width_coef2, 64 * self.hyperparameters.width_coef3 ]
        self.epochs = 200
        lr_step = [0.3, 0.6, 0.8]
        self.lr_step = [int(self.epochs * ls) for ls in lr_step]
        
        self.conv1 = nn.Conv2d(3, self.in_chs[0], 3, padding = 1, bias = True)
        self.conv2 = self._add_groups(self.n_blocks[0], self.in_chs[0], self.in_chs[1], self.hyperparameters.drop_rates1)
        self.conv3 = self._add_groups(self.n_blocks[1], self.in_chs[1], self.in_chs[2], self.hyperparameters.drop_rates2, stride = 2)
        self.conv4 = self._add_groups(self.n_blocks[2], self.in_chs[2], self.in_chs[3], self.hyperparameters.drop_rates3, stride = 2)
        self.bn = nn.BatchNorm2d(self.in_chs[3])
        self.full_conn = nn.Linear(self.in_chs[3], 100)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        h = F.relu(self.bn(h))
        h = F.avg_pool2d(h, 8)
        h = h.view(-1, self.in_chs[3])
        h = self.full_conn(h)
        
        return F.log_softmax(h, dim = 1)

    def _add_groups(self, n_blocks, in_ch, out_ch, drop_rate, stride = 1):
        blocks = []

        for _ in range(int(n_blocks)):
            blocks.append(BasicBlock(in_ch, out_ch, stride = stride, drop_rate = drop_rate))
            
            in_ch, stride = out_ch, 1

        return nn.Sequential(*blocks)
