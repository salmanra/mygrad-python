"""
    My recreation of Andrej Karpathy's micrograd Value
"""
from math import exp, log
import math


class Value:
    """Baby blue"""

    def __init__(self, data, _children=(), _op="") -> None:
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        t = self.data + other.data
        out = Value(t, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        t = self.data * other.data
        out = Value(t, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += other.grad * out.grad

        out._backward = _backward
        return out

    def exp(self):
        t = exp(self.data)
        out = Value(t, (self,), "exp")

        def _backward():
            self.grad += t * out.grad

        out._backward = _backward
        return out

    def log(self):
        t = log(self.data)
        out = Value(t, (self,), "log")

        def _backward():
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        t = self.data**other.data
        out = Value(t, (self, other), "pow")
        def _backward():
            self.grad += other.data * (self.data ** (other.data - 1)) * out.grad
            if math.fabs(self.data) < 0.0001:
                return
            other.grad += (
                math.copysign(log(math.fabs(self.data)), self.data) * t * out.grad
            )

        out._backward = _backward
        return out

    def tanh(self):
        t = (exp(2 * self.data) + 1) / (exp(2 * self.data) - 1)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        t = self.data if self.data > 0 else 0.0
        out = Value(t, (self,), "ReLU")
        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

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
        return f"Value(data={self.data}, grad={self.grad})"
