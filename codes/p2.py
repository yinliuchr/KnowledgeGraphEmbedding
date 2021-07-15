from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import random

import numpy as np
import torch

import torch.nn as nn

from torch.utils.data import DataLoader

from model import KGEModel

from dataloader import TrainDataset
from dataloader import BidirectionalOneShotIterator

x = torch.randn(2,2)
print(x)
y = x.view(-1)
print(y)

x = torch.tensor([[1.0,2,3],[4,5,6]])
y = torch.tensor([[1,3,5], [2,4,6]])
print(torch.stack([x,y], dim=2))

print(x.norm(dim=1))

print('mean: ', x.mean())

z = x.repeat(3,2)
print(z)

a = torch.tensor([1,2,3])
print(a.repeat(2,1))

b = y.T
print(b)
b[:,1] = a
print(b)


print('\n\n\n')

a = torch.nn.Embedding(3,5)
print('Emb: ', a.weight)
print(a.weight.detach().cpu().numpy())


print('\n\n\n')
a1, a2, a3 = torch.tensor([1,2,3]), torch.tensor([4,5,6]), torch.tensor([7,8,9])
a = torch.cat([a1,a2,a3], dim= 0)
print(a)

print('\n\n\n')

input = torch.randn((2,3), requires_grad=True)
# target = torch.empty((2,3)).random_(3)
target = torch.tensor([[.8,.4, 3.9], [-1.2, .3, 2.1]])
print(input)
print(target)
loss = nn.BCELoss()
m = nn.Sigmoid()
output  =  loss(m(input), target)
print('output', output)


# print('\n\n\n')
# e = nn.Embedding(2,3)
# print(e.weight)
#
#
# print('\n\n\n')
# input = torch.randn((2,3,3))
# print(torch.sigmoid(input))



print('\n\n\n')
# a = torch.tensor([[.8,.4, 3.9], [-1.2, .3, 2.1]])
# # print(a[1,[1,2]])
# target = torch.tensor([0.]).expand_as(a)
# target[:,0] = torch.ones(2).view(2,1)
# print(target)
# # print(torch.ones(2))
target = torch.tensor([1.0] + [0.] * 4)
target = target.repeat(3,1)
print(target)

print('\n\n\n')
a = torch.tensor([[3,2,1,4],[5,7,3,8]])
print(a)
print(torch.argsort(a, dim=1, descending=True))

print('\n\n\n')
a = torch.tensor([5,2,4,6,8,1,9,3,6])
b = 6
print((a == b).nonzero())

print('\n\n\n')
a = None
if not a: print('andioan')

