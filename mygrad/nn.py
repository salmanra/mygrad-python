import numpy as np

from mygrad.engine import Tensor
import pdb

# want to build up to a MLP using Tensors as defined in engine.py


class Module:
    def zero_grad(self):
        for tensor in self.parameters():
            tensor.grad = np.zeros_like(tensor.grad)

    def parameters(self):
        return []


class FullyConnectedLayer(Module):
    # needs an in-dim, an out-dim, and an act function
    def __init__(self, indim, outdim, act='relu') -> None:
        # He initialization, the goat
        self.W = Tensor(np.random.default_rng().normal(0, 2/(indim*outdim), size=indim*outdim).reshape(outdim, indim))
        self.b = Tensor(np.zeros(shape=(outdim)))
        # gonna have to make it a "look it up" activation function
        self.act = act

    def __call__(self, other):
        # assume other is a Tensor
        return ((self.W @ other) + self.b).relu() if self.act else self.W @ other + self.b

    def parameters(self):
        # an array of two Tensors... is this what life is?
        # TODO: verify that this can work. mostly just wait until 
        # this causes a bug... nominative determinism
        return [self.W, self.b]

    def zero_grad(self):
        self.W.grad = np.zeros_like(self.W.grad)
        self.b.grad = np.zeros_like(self.b.grad)


class MLP(Module):
    def __init__(self, indim, outdims):
        all_layers = [indim] + outdims
        self.layers = [FullyConnectedLayer(all_layers[i], all_layers[i+1], i != len(outdims) - 1) for i in range(len(outdims))]

    def __call__(self, other):
        result = other
        for layer in self.layers:
            result = layer(result)
        return result
    
    def parameters(self):
        # an array of tensors...
        return [tensor for layer in self.layers for tensor in layer.parameters()]
    
    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()
