"""
    My recreation of Andrej Karpathy's micrograd Value
"""
from math import exp, log
import pdb
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

    # need __iter__ and __next__ to make Tensors iterable, zippable
    # TODO: implement for 1D tensors. 
    # TODO: conceive how to extend to 2D, ND
    def __iter__(self):
        ''' 
        returns a Tensor iterator?
        what goes into the Tensor iterator type?
        where do I put the iter type?
        
        iter needs a reference to self.data (self being the Tensor)
        iter needs to be imported with Tensor. 
        '''
        self._index = 0
        return self
    
    def __next__(self):
        if self._index < len(self.data):
            value = self.data[self._index]
            self._index += 1
            return value
        else:
            raise StopIteration
    # if self.grad is smaller than out.grad, sum out.grad on each dimension
    # that it is larger to make up the difference.
    # how do I write that in code?
    # let's look at just other.grad
    #   start at the end of both tuples (out.grad.shape and other.grad.shape)
    #   are they the same? are they different? is one of them 1?
    #   if they're different, 
    @staticmethod
    def undo_broadcast(out, weights):
        '''
        out: ndarray that's just too big
        weights: ndarray that's just too small
        '''
        # let's assume that out was broadcasted correctly, and we never have to check if the shapes
        # are unbroadcastable.
        # should fillvalue be 1 here?
        shapearr = []
        for dim, (outsize, weightsize) in enumerate(itt.zip_longest(reversed(out.shape), reversed(weights.shape), fillvalue=0)):
            if outsize > weightsize:
                shapearr.append(len(out.shape) - dim - 1)
        return shapearr
    
    def __add__(self, other):
        # let's say out is constructed via broadcasting. 
        # let's say self is the smaller of the two.
        # there's no way to "unbroadcast" out.grad to self.data!
        # yuck
        # is this why we build a backwards graph? to store self somewhere as its big, broadcasted, backpropable self?
        # is this why requires_grad=False exists? Not just performance, but literally being unable to backprop?
        out = Tensor(self.data + other.data, float, (self, other), "+")

        def _backward():
            self.grad += out.grad if out.grad.size == self.grad.size else out.grad.sum(axis=tuple(Tensor.undo_broadcast(out.grad, self.grad)))
            other.grad += out.grad if out.grad.size == other.grad.size else out.grad.sum(axis=tuple(Tensor.undo_broadcast(out.grad, other.grad)))
        out._backward = _backward

        return out

    def __mul__(self, other):
        '''
        multiply arguments element-wise
        '''
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data.__mul__(other.data), float, (self, other), "*")
        def _backward():
            # TODO: verify this backward pass.
            # it is definitely wrong, as mat-vec elt-wise mul backward pass does not work
            #   something about broadcasting!
            prod1 = other.data * out.grad
            prod2 = self.data * out.grad
            self.grad += prod1 if prod1.size == self.grad.size else prod1.sum(axis=tuple(Tensor.undo_broadcast(prod1, self.grad)))
            other.grad += prod2 if prod2.size == other.grad.size else prod2.sum(axis=tuple(Tensor.undo_broadcast(prod2, other.grad)))
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
        # brotherman... why aren't you just using numpy exp???
        # is there anyway this backward pass is correct???

        out = Tensor(np.exp(self.data), dtype=self.data.dtype, _children=(self,), _op='exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    # TODO: implement these buggers
    def log(self):   
        # TODO: extrapolate to 2D, ND     
        out = Tensor([log(elt) for elt in self.data])
        def _backward():
            self.grad += out.grad / out.data
        out._backward = _backward

        return out
    
    def T(self):
        # absolutely no clue if this looks correct
        data = self.data.T
        out = Tensor(data, float, (self,), 'T')
        def _backward():
            self.grad += out.grad.T

        out._backward = _backward
        return out
    
    def __pow__(self, other):
        # assume other is just some int
        # TODO: extrapolate for self 2D, ND
        out = Tensor(np.power(self.data, other), dtype=self.data.dtype, _children=(self,), _op='pow')
        def _backward():
            self.grad += other * np.power(self.data, other-1) * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        return self

    def softmax(self, dim=-1):
        # now this... this is inefficient
        # and maybe even incorrect.
        # correctness rides on .sum()
        return self.exp()/self.exp().sum(axis=dim)

    def relu(self):
        # mask self.data with self.data > 0
        mask = self.data > 0
        rel = self.data * mask
        out = Tensor(rel, self.data.dtype, (self,), 'relu')

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
        return self * Tensor(-1)

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
