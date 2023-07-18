import random
import numpy as np

from mygrad.engine import Tensor


## want to build up to a MLP using Tensors as defined in engine.py
class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []
    
class FullyConnectedLayer(Module):
    # needs an in-dim, an out-dim, and an act function
    def __init__(self, indim, outdim, act='relu') -> None:
        # He initialization, the goat
        self.W = Tensor(np.random.default_rng().normal(0, 2/(indim*outdim), size=indim*outdim).reshape(outdim, indim))
        self.b = Tensor(np.zeros(shape=(outdim)))
        # there's got to be a better way to make the activation function then just "look it up"
        self.act = act

    def __call__(self, other):
        return ((self.W @ other) + self.b).relu()
        