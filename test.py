from random import uniform
from mygrad.engine import Tensor
from mygrad.nn import MLP
import numpy as np
import pdb

# a = Tensor([1.0, 2.0, 3.0])
# b = Tensor([9, 8, 7])

# print(a + b)
# print(a*b)

# u = Tensor([[1, 2], [3, 4]])
# v = Tensor([[5, 6], [7, 8]])

# print(u+v)
# print(u*v)

# w = Tensor([1, 1])
# x = u*w
# x.backward()
# print(u)

# how do we get all the way to MLP?
# an FC layer
#   a 2D weight tensor
#   a 1D bias tensor
#   an activation function
# a loss function


def MSELoss(ys, os):
    # pdb.set_trace()
    # we probably don't even need to zip these up, right?
    return ((ys - os) ** 2).sum()


def train(net, lossfunc, lrate, indata, truth, epochs):
    # batch size is here so we do SGD and not merely GD

    for _ in range(epochs):
        outs = net(indata)
        loss = lossfunc(truth, outs)

        net.zero_grad()

        loss.backward()
        for p in net.parameters():
            p.data -= lrate * p.grad
        


def eval(net, lossfunc, indata, truth):
    return lossfunc(truth, net(indata))


nin = 10
nouts = [6, nin]
xs = Tensor([uniform(-1, 1) for _ in range(nin)])
mlp = MLP(nin, nouts)
lr = 0.1

ys = Tensor([1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0])

train(mlp, MSELoss, lr, xs, ys, 100)
loss = eval(mlp, MSELoss, xs, ys)

print(f"loss: {loss}, final outs: {mlp(xs)}")
# print(f"all params: {mlp.parameters()}")