import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

def generate_data(up=True, length=16, resolution=5):
    seed = torch.empty(2).uniform_(0, 1)
    low = seed.min()
    high = seed.max()

    stds = torch.tensor([0.1])[:, None, None].expand(length, resolution, resolution)

    if up:
        means = torch.linspace(low, high, length)
    else:
        means = torch.linspace(high, low, length)
    means = means[:, None, None].expand(-1, resolution, resolution)

    seq = torch.normal(means, stds)
    return seq

print(generate_data(length=4))

# x_train = []
# y_train = []
# for i in range(20):
#     x_train.append(generate_data(True))
#     y_train.append(torch.tensor(1))
# for i in range(20):
#     x_train.append(generate_data(False))
#     y_train.append(torch.tensor(0))

# x_train = torch.stack(x_train, 0)
# y_train = torch.stack(y_train, 0)

# print(x_train.shape)
# print(y_train.shape)
