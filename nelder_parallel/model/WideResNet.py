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

    hp["batch_size"] = type_hints["batch_size"]((2 ** hp["batch_size"]))
    hp["lr"] = type_hints["lr"](10 ** hp["lr"])
    hp["weight_decay"] = type_hints["weight_decay"](10 ** hp["weight_decay"])
    hp["width_coef1"] = type_hints["width_coef1"](2 ** hp["width_coef1"])
    hp["width_coef2"] = type_hints["width_coef2"](2 ** hp["width_coef2"])
    hp["width_coef3"] = type_hints["width_coef3"](2 ** hp["width_coef3"])
    hp["lr_decay"] = type_hints["lr_decay"](10 ** hp["lr_decay"])

    return HyperParameters(**hp)
    

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride = 1, drop_rate = 0.2, kernel_size = 3):
        super(BasicBlock, self).__init__()
        self.in_is_out = (in_ch == out_ch)
        self.drop_rate = drop_rate
        self.shortcut = self.in_is_out and None or nn.Conv2d(in_ch, out_ch, 1, padding = 0, stride = stride, bias = False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.c1 = nn.Conv2d(in_ch, out_ch, kernel_size, padding = 1, stride = stride, bias = False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.c2 = nn.Conv2d(out_ch, out_ch, kernel_size, padding = 1, bias = False)

    def forward(self, x):
        h0 = F.relu(self.bn1(x))
        h = self.c1(h0)
        h = F.dropout2d(h, p = self.drop_rate)
        h = F.relu(self.bn2(h))
        h = self.c2(h)

        if self.in_is_out:
            return h + x
        else:
            return h + self.shortcut(h0)

class WideResNet(nn.Module):
    def __init__(self, hp_parser):
        super(WideResNet, self).__init__()
        
        self.hyperparameters = get_hyperparameters(hp_parser)

        # will add the way to convert later
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
        
        self.c1 = nn.Conv2d(3, self.in_chs[0], 3, padding = 1, bias = False)
        self.c2 = self._add_groups(self.n_blocks[0], self.in_chs[0], self.in_chs[1], self.hyperparameters.drop_rates1)
        self.c3 = self._add_groups(self.n_blocks[1], self.in_chs[1], self.in_chs[2], self.hyperparameters.drop_rates2, stride = 2)
        self.c4 = self._add_groups(self.n_blocks[2], self.in_chs[2], self.in_chs[3], self.hyperparameters.drop_rates3, stride = 2)
        self.bn = nn.BatchNorm2d(self.in_chs[3])
        self.full_conn = nn.Linear(self.in_chs[3], 100)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def convert_hyperparameters(self, hyperparameters):
        batch_size = 2 ** (hyperparameters.batch_size)
        weight_decay = 10 ** hyperparameters.weight_decay
        lr = 10 ** hyperparameters.lr
        lr_decay = 10 ** hyperparameters.lr_decay
        momentum = 1 - 10 ** hyperparameters.momentum

        return batch_size, weight_decay, lr, lr_decay, momentum

    def forward(self, x):
        h = self.c1(x)
        h = self.c2(h)
        h = self.c3(h)
        h = self.c4(h)
        h = F.avg_pool2d(h, 8)
        h = F.relu(self.bn(h))
        h = h.view(-1, self.in_chs[3])
        h = self.full_conn(h)
        
        return F.log_softmax(h, dim = 1)

    def _add_groups(self, n_blocks, in_ch, out_ch, drop_rate, stride = 1):
        blocks = []

        for _ in range(int(n_blocks)):
            blocks.append(BasicBlock(in_ch, out_ch, stride = stride, drop_rate = drop_rate))
            
            in_ch, stride = out_ch, 1

        return nn.Sequential(*blocks)