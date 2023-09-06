from typing import Any
import numpy as np

from mygrad.engine import Tensor
import pdb

class Module:
    def zero_grad(self):
        for tensor in self.parameters():
            tensor.grad = np.zeros_like(tensor.grad)

    def parameters(self):
        return []

class FullyConnectedLayer(Module):
    # needs an in-dim, an out-dim, and an act function
    def __init__(self, indim, outdim, act=True) -> None:
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
        # yes. the whole point of having this is to let an optimizer 
        # use the .grad and .data attributes of each param.
        # this is perfect.
        return [self.W, self.b]

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
    
class SelfAttention(Module):
    # selfattention... 
    # the params are just the transformations from input to q, k, v
    # so everything funky just goes to the forward pass
    def __init__(self, k, heads) -> None:
        self.k = k # input dim
        self.make_query = Tensor([np.random.default_rng().normal(0, 2/(k**2), size=k**2).reshape(k, k)])
        self.make_key = Tensor([np.random.default_rng().normal(0, 2/(k**2), size=k**2).reshape(k, k)])
        self.make_value = Tensor([np.random.default_rng().normal(0, 2/(k**2), size=k**2).reshape(k, k)])

    
    def __call__(self, other: Tensor) -> Any:
        # forward pass

        # do I need tensor transpose here?
        queries = (other @ self.make_query) / (self.k ** 0.5) 
        keys = other @ self.make_key 
        values = other @ self.make_value

        # definitely need tensor transpose here
        Wp =  queries @ keys.T()
        Wf = Wp.softmax()

        outs = Wf @ values
        return outs
    
    def parameters(self):
        return [self.make_query, self.make_key, self.make_value]
