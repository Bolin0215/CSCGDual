import torch
import numpy as np
from torch.autograd import Variable
from scipy.spatial import distance
import torch.nn.functional as F

def loss_att(a, b, mask, length):
    epsilon = 1e-8
    a.data.masked_fill_(mask, -(1e+8))
    b.data.masked_fill_(mask, -(1e+8))
    a = F.softmax(a, 2) + epsilon
    b = F.softmax(b, 2) + epsilon
    print(a, b)
    x_a = a * torch.log(a / ((b + a) / 2))
    x_b = b * torch.log(b / ((b + a) / 2))
    x_a.data.masked_fill_(mask, 0)
    x_b.data.masked_fill_(mask, 0)
    x_a = torch.sum(x_a, 2)
    x_b = torch.sum(x_b, 2)
    # x_a = torch.sum(x_a, 1) / Variable(torch.FloatTensor(length))
    # x_b = torch.sum(x_b, 1) / Variable(torch.FloatTensor(length))
    kl_div = x_a + x_b
    kl_div = kl_div / 2
    print(kl_div)

def jsd(a, b):
    for batchid in range(len(a)):
        for rowid in range(len(a[batchid])):
            print(a[batchid][rowid][:-1], b[batchid][rowid][:-1])
            jsd = distance.jensenshannon(a[batchid][rowid][:-1], b[batchid][rowid][:-1])
            print(jsd**2)

if __name__ == '__main__':
    # a = np.array([[[0.1, 0.7, 0.2], [0.1, 0.8, 0.1]], [[0.3, 0.7], [0.4, 0.6]],[[0.5, 0.5], [0.1, 0.9]]])
    # b = np.array([[[0.2, 0.8], [0.3, 0.7]], [[0.5, 0.5], [0.4, 0.6]], [[0.2, 0.8], [0.1, 0.9]]])
    a = np.array([[[0.1, 0.9, 0.0], [0.2, 0.8, 0.0]]])
    b = np.array([[[0.3, 0.7, 0.0], [0.1, 0.9, 0.0]]])
    # a = np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    # b = np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    # mask = np.array([[[0,0],[0,0]],[[0,0],[0,0]],[[0,0],[0,0]]])
    mask = np.array([[[0, 0, 1], [0, 0, 1]]])
    # length = [2,2,2]
    jsd(a, b,)
    length = [2]
    a = Variable(torch.from_numpy(a).float())
    b = Variable(torch.from_numpy(b).float())
    mask = torch.ByteTensor(mask)
    loss_att(a, b, mask, length)
