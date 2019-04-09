import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import typing
from numpy import ndarray
from typing import Callable
from typing import List
from typing import Optional
from typing import NamedTuple
from typing import Tuple

# ref https://github.com/liuzhuang13/DenseNet/tree/master/models

class DenseNetBCParameters(
	NamedTuple(
		"_DenseNetBCParameters",
		[("batch_size", int),
		 ("epochs", int),
		 ("lr", float),
		 ("momentum", float), # between 0 and 1
		 ("weight_decay", float), 
		 ("nesterov", Optional[bool]),
		 ("label", Optional[int]), # cifar-10 or cifar-100
		 ("n_layers", Callable[[int], ndarray]), # how many convolutional blocks we have in one group of blocks
		 ("compressions", Callable[[float], ndarray]),
		 ("growth_rates", Callable[[int], ndarray]), 
		 ("growth_coefs", Callable[[int], ndarray]), 
		 ("drop_rates_db", Callable[[float], ndarray]), # dropout rate in each groups: between 0 and 1
		 ("drop_rates_tl", Callable[[float], ndarray]), # dropout rate in each groups: between 0 and 1
		 ("lr_step", Callable[[float], ndarray]), # when we reduce the learning rate
		 ("lr_decay", float)
		 # ("n_groups", int), # how many groups of convolutional blocks we have in the model
		])):
	pass

class dnbc():
	def __init__(
			self,
			batch_size = 64,
			epochs = 300,
			lr = 0.1,
			momentum = 0.9,
			weight_decay = 1.0e-4,
			nesterov = True,
			label = 100,
			n_layers = [16, 16, 16],
			compressions = [0.5, 0.5],
			growth_rates = [24, 24, 24],
			growth_coefs = [2, 4, 4, 4],
			drop_rates_db = [[0.2, 0.2], [0.2, 0.2], [0.2, 0.2]],
			drop_rates_tl = [0.2, 0.2],
			lr_step = [ 1 / 2, 3 / 4, 1],
			lr_decay = 0.1
			):	

		self.hyperparameters = DenseNetBCParameters(
			batch_size, epochs, lr, momentum, weight_decay, nesterov, label, n_layers, compressions, growth_rates, growth_coefs, drop_rates_db, drop_rates_tl, lr_step, lr_decay)
		self.model = DenseNetBC(self.hyperparameters).cuda()

	

class TransionLayer(nn.Module):

	def __init__(self, in_ch, out_ch, drop_rate = 0.2):
		super(TransionLayer, self).__init__()
		self.bn = nn.BatchNorm2d(in_ch)
		self.conv = nn.Conv2d(in_ch, out_ch, 1, bias = False)
		self.drop_rate = drop_rate

	def forward(self, x):
		h = F.relu(self.bn(x))
		h = F.dropout2d(self.conv(h), p = self.drop_rate)
		h = F.avg_pool2d(h, 2)
		return h

class DenseBlock(nn.Module):

	def __init__(self, in_ch, growth_rate, growth_coef = 4, drop_rates = [0.2, 0.2]):
		super(DenseBlock, self).__init__()
		self.bn1 = nn.BatchNorm2d(in_ch)
		self.bottle = nn.Conv2d(in_ch, growth_rate * growth_coef, 1, padding = 0, bias = False)
		self.bn2 = nn.BatchNorm2d(growth_rate * growth_coef)
		self.conv = nn.Conv2d(growth_rate * growth_coef, growth_rate, 3, padding = 1, bias = False)
		self.drop_rates = drop_rates

	def forward(self, x):
		h = F.relu(self.bn1(x))
		h = self.bottle(x)
		h = F.dropout2d(h, p = self.drop_rates[0])
		h = F.relu(self.bn2(x))
		h = self.conv(x)
		h = F.dropout2d(h, p = self.drop_rates[1])
		return torch.cat((h, x), dim = 1)


class DenseNetBC(nn.Module):
	def __init__(self, hyperparameters):
		super(DenseNetBC, self).__init__()
		
		self.n_layers = hyperparameters.n_layers
		self.growth_rates = hyperparameters.growth_rates
		self.growth_coefs = hyperparameters.growth_coefs
		self.compressions = hyperparameters.compressions
		self.drop_rates_db = hyperparameters.drop_rates_db
		self.drop_rates_tl = hyperparameters.drop_rates_tl
		
		self.conv = nn.Conv2d(3, self.growth_coefs[0] * self.growth_rates[0], 3, padding = 1, bias = False)

		in_ch = self.growth_coefs[0] * self.growth_rates[0]
		self.db1 = self._add_DenseBlock(self.n_layers[0], in_ch, self.drop_rates_db[0], self.growth_rates[0], self.growth_coefs[1])
		in_ch += self.growth_rates[0] * self.n_layers[0]
		self.tl1 = TransionLayer(in_ch, math.floor(in_ch * self.compressions[0]), self.drop_rates_tl[0])
		in_ch = math.floor(in_ch * self.compressions[0])

		self.db2 = self._add_DenseBlock(self.n_layers[1], in_ch, self.drop_rates_db[1], self.growth_rates[1], self.growth_coefs[2])
		in_ch += self.growth_rates[1] * self.n_layers[1]
		self.tl2 = TransionLayer(in_ch, math.floor(in_ch * self.compressions[1]), self.drop_rates_tl[1])
		in_ch = math.floor(in_ch * self.compressions[1])

		self.db3 = self._add_DenseBlock(self.n_layers[2], in_ch, self.drop_rates_db[2], self.growth_rates[2], self.growth_coefs[3])
		in_ch += self.growth_rates[2] * self.n_layers[2]
		
		self.bn = nn.BatchNorm2d(in_ch)
		self.full_conn = nn.Linear(in_ch, hyperparameters.label)

	def forward(self, x):
		h = self.conv(x)
		h = self.db1(x)
		h = self.tl1(x)
		h = self.db2(x)
		h = self.tl2(x)
		h = self.db3(x)
		
		h = F.relu(self.bn(h))
		h = F.avg_pool2d(h, h.size(2))
		
		h = h.view(h.size(0), -1)
		h = self.full_conn(h)

		return F.log_softmax(h)

	def _add_DenseBlock(self, n_layers, in_ch, drop_rate, growth_rate, growth_coef):
		layers = []

		for i in range(int(n_layers)):
			layers.append(DenseBlock(in_ch + growth_rate * i, growth_rate, growth_coef = growth_coef, drop_rates = self.drop_rates))
		
		return nn.Sequential(*layers)