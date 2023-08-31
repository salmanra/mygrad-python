from random import uniform
from mygrad.nn import MLP


def MSELoss(ys, os):
    return sum((y - out) ** 2 for (y, out) in zip(ys, os))


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
nouts = [1, nin]
xs = [uniform(-1, 1) for _ in range(nin)]
mlp = MLP(nin, nouts)
lr = 0.1

ys = [1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0]

train(mlp, MSELoss, lr, xs, ys, 100)
loss = eval(mlp, MSELoss, xs, ys)

print(f"loss: {loss}, final outs: {mlp(xs)}")
# print(f"all params: {mlp.parameters()}")
