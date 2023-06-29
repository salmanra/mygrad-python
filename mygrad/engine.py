"""
    My recreation of Andrej Karpathy's micrograd Value
"""
from math import exp, log
import math
import numpy as np


class Tensor:
    # once data and grad are set up as Tensors, is it all too easy from there?
    # ie, does numpy make it all too easy from there?
    # can we get to an MLP with no big conceptual leaps?
    # what other architectures can we get to with no big conceptual leaps?

    def __init__(self, data, dtype=float, _children=(), _op="") -> None:
        self.data = np.array(data, dtype=dtype) # you'd better be a np.ndarray
        self.grad = np.zeros_like(self.data)
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __add__(self, other):
        out = Tensor(self.data + other.data, float, (self, other), "+")
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        out = Tensor(self.data.dot(other.data), float, (self, other), "*")
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def exp(self):
        # what's happening here?
        # we take the exponent of each element of the Tensor,
        # then we must backprop through each element of the Tensor
        out = Tensor([exp(elt) for elt in self.data])
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def log(self):
        return self

    def __pow__(self, other):
        return self

    def tanh(self):
        return self

    def relu(self):
        return self

    def backward(self):
        # topological sort
        # reverse topo, for each node: _backward

        topo = []
        visited = set()
        # it's just a post-order traversal my guy...

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * (other**-1)

    def __rsub__(self, other):
        return (-self) + other

    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"
