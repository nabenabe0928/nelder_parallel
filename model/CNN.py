import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy import ndarray
from typing import Callable
from typing import List
from typing import Optional
from typing import NamedTuple
from typing import Tuple
import math


class CNN(nn.Module):

    def __init__(self, hyperparameters):
        super(CNN, self).__init__()
        self.batch_size = hyperparameters.batch_size 
        self.epochs = 160
        self.lr = hyperparameters.lr
        self.momentum = hyperparameters.momentum
        self.weight_decay = hyperparameters.weight_decay
        self.nesterov = False
        self.drop_rate = hyperparameters.drop_rate
        #self.in_chs = []
        self.lr_decay = hyperparameters.lr_decay
        lr_step = [155./160., 158./160.]
        self.lr_step = [int(self.epochs * ls) for ls in lr_step]

        self.c1 = nn.Conv2d(3, 32, 5, padding=2)
        self.c2 = nn.Conv2d(32, 32, 5, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.c3 = nn.Conv2d(32, 64, 5, padding=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.full_conn1 = nn.Linear(576, 64)
        self.full_conn2 = nn.Linear(64, 100)

        # Initialize paramters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        h = self.c1(x)
        h = F.relu(F.max_pool2d(h, 3, stride=2))
        h = self.c2(h)
        h = F.relu(h)
        h = self.bn2(F.avg_pool2d(h, 3, stride=2))
        h = self.c3(h)
        h = F.relu(h)
        h = self.bn3(F.avg_pool2d(h, 3, stride=2))

        h = h.view(h.size(0), -1)
        h = self.full_conn1(h)
        h = F.dropout2d(h, p = self.drop_rate, training = self.training)
        h = self.full_conn2(h)
        return F.log_softmax(h, dim = 1)
