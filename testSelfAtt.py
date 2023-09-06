from random import uniform
from mygrad.engine import Tensor
from mygrad.nn import FullyConnectedLayer, SelfAttention
import numpy as np
import pdb

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

# what does a self attention layer do on its own???
# it forces each vector in the input sequence 
# to LOOK AT all the other vectors in the input
# sequence. something about global relationships
# is learned, if not something about global 
# structure. The input sequence does get reduced
# to an input set, so some structure is lost.

# transformers surround self att layers with 
# residuals, layernorms, MLPs, FCs for dim reduction...
# these all provide an architecture opportunites
# to learn a lot about a lot.

N = 10
K = 4

Xs = Tensor([np.random.default_rng().uniform(low=0, high=1, size=N*K).reshape(N, K)])
Ys = np.random.randint(low=0, high=2, size=N)

att = SelfAttention(k=K, heads=None)

# now an FC layer K -> 1, classify !!!
fc = FullyConnectedLayer(K, 1, act=False)

# don't have a good way to compose modules yet... could do a nn.Sequential type thing
# oh. and how does backprop work with composed modules?
# This here is why the backprop graph made with build_topo is just not enough...
# that graph is getting built every time a tensor is created...
# can we extend the graph to be a continuous thing that gets built
# every time a module is created?
# yes, but then something has to be aware of modules going together.

print(fc(att(Xs).T()))

# print(f"loss: {loss}, final outs: {mlp(xs)}")
# print(f"all params: {mlp.parameters()}")