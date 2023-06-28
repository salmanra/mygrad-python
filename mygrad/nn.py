import random
from mygrad.engine import Value


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

            # I guesss any fuction is an abstract function in python

    def parameters(self):
        return []


class Neuron(Module):
    """a Neuron built on our Value!!!"""

    def __init__(self, nin, relu):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        self.relu = relu

    def __call__(self, xs):
        # self(xs)
        act = (
            sum([w * x for (w, x) in zip(self.w, xs)], self.b)
            if len(self.w) > 1
            else self.w[0] * xs + self.b
        )
        return act.relu() if self.relu else act

    def parameters(self):
        """returns a List of all nin+1 params of this neuron"""
        return self.w + [self.b]

    def __repr__(self):
        return f'{"ReLU" if self.relu else "Linear"} Neuron({len(self.w)})'


class Layer(Module):
    """a layer of Neurons!"""

    def __init__(self, nin, nout, act=True):
        self.neurons = [Neuron(nin, act) for _ in range(nout)]

    def __call__(self, xs):
        # just want to call each Neuron on the input return the output as an array
        arr = [n(xs) for n in self.neurons]
        return arr if len(arr) > 1 else arr[0]

    def parameters(self):
        # still just a list of parameters
        return [p for n in self.neurons for p in n.parameters()]


class MLP(Module):
    """a multilayer perceptron!!!"""

    def __init__(self, nin, nouts):
        # nin - int: input dimenstion
        # nouts List(int): hidden+output dimenstions
        sz = [nin] + nouts
        self.layers = [
            (Layer(sz[i], sz[i + 1], i != (len(nouts) - 1))) for i in range(len(nouts))
        ]

    def __call__(self, xs):
        out = xs
        for layer in self.layers:
            out = layer(out)
        return out

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]
