"""
    My recreation of Andrej Karpathy's micrograd Value
"""
from math import exp, log
import math
import numpy as np
import itertools as itt


class Tensor:
    # once data and grad are set up as Tensors, is it all too easy from there?
    # ie, does numpy make it all too easy from there?
    # can we get to an MLP with no big conceptual leaps?
    # what other architectures can we get to with no big conceptual leaps?

    def __init__(self, data, dtype=float, _children=(), _op="") -> None:
        self.data = np.array(data, dtype=dtype).squeeze() # you can be zero-dim. I will allow it. in fact, I will squeeze you
        self.grad = np.zeros_like(self.data)
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __add__(self, other):
        # let's say out is constructed via broadcasting. 
        # let's say self is the smaller of the two.
        # there's no way to "unbroadcast" out.grad to self.data!
        # yuck
        # is this why we build a backwards graph? to store self somewhere as its big, broadcasted, backpropable self?
        # is this why requires_grad=False exists? Not just performance, but literally being unable to backprop?
        out = Tensor(self.data + other.data, float, (self, other), "+")
        def _backward():
            # if self.grad is smaller than out.grad, sum out.grad on each dimension 
            # that it is larger to make up the difference.
            # how do I write that in code?
            # let's look at just other.grad
            #   start at the end of both tuples (out.grad.shape and other.grad.shape)
            #   are they the same? are they different? is one of them 1?
            #   if they're different, 
            def undo_broadcast(weights):
                # let's assume that out was broadcasted correctly, and we never have to check if the shapes
                # are unbroadcastable.
                # should fillvalue be 1 here?
                shapearr = []
                for dim, (outsize, weightsize) in enumerate(itt.zip_longest(reversed(out.grad.shape), reversed(weights.grad.shape), fillvalue=0)):
                    if outsize > weightsize:
                        shapearr.append(len(out.grad.shape) - dim - 1)
                return shapearr

            self.grad += out.grad if out.grad.size == self.grad.size else out.grad.sum(axis=tuple(undo_broadcast(self)))
            other.grad += out.grad if out.grad.size == other.grad.size else out.grad.sum(axis=tuple(undo_broadcast(other)))
        out._backward = _backward

        return out

    def __mul__(self, other):
        '''
        multiply arguments element-wise
        '''
        out = Tensor(self.data.__mul__(other.data), float, (self, other), "*")
        def _backward():
            # TODO: verify this backward pass.
            # it is definitely wrong, as mat-vec elt-wise mul backward pass does not work
            #   something about broadcasting!
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad 
        out._backward = _backward

        return out
    
    def __matmul__(self, other):
        '''
        @ operator
        This is the only (?) place where two scalars of dim one form a scalar of dim zero.
        The most natural solution is to choke off the zero-dimness at init.
        '''
        out = Tensor(self.data.__matmul__(other.data), float, (self, other), '@')
        def isVector(tensor):
            return tensor.shape.count(1) == (len(tensor.shape) - 1)
        def _backward():
            # for now, break it down by tensor shape:
            #   1-D @ 1-D
            #   2-D @ 1-D
            #   2-D @ 2-D
            # if I am not mistaken, all matmuls are broadcast operations of these three 
            # basic matmuls. does numpy do kroenecker prod? pytorch?
            # is there a different operator for batch matmul?

            # we can determine the basic shape of the input tensors by the shape of the output tensors
            #   scalar 1d-1d
            #   vector 2d-1d
            #   matrix 2d-2d

            # the rule is always an "outer product" of out.grad and other.data
            # np.outer is specifically for two vectors, so we need to read the 
            # shape of each matrix 
            if isVector(other.data):
                self.grad += np.outer(out.grad, other.data).squeeze()
            else:
                self.grad += np.dot(out.grad, np.transpose(other.data)) 

            # since the grad of the B (in AB = C) is never an outer product, np.dot handles it nicely
            other.grad += np.dot(self.data.T, out.grad)
        out._backward = _backward
        return out
    
    def sum(self, axis=None):
        # need this to construct a loss fxn that yields a scalar
        # TODO: extend this to self of order greater than one
        out = Tensor(self.data.sum(axis=axis), float, (self,), 'sum')
        def _backward():
            # want self.grad to grow to size of self
            self.grad += out.grad * np.ones_like(self.data)

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

    # TODO: implement these buggers
    def log(self):
        return self

    def __pow__(self, other):
        return self

    def tanh(self):
        return self

    def relu(self):
        # mask self.data with self.data > 0
        mask = self.data > 0
        rel = self.data * mask
        out = Tensor(rel.tolist, self.data.dtype, (self,), 'relu')
        def _backward():
            self.grad += mask * out.grad
            
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
        self.grad = np.array(1.0)
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
        return f"Tensor(data={self.data}, grad={self.grad}, shape={self.data.shape})"
