from random import uniform
from mygrad.engine import Tensor
import numpy as np

# a = Tensor([1.0, 2.0, 3.0])
# b = Tensor([9, 8, 7])

# print(a + b)
# print(a*b)

u = Tensor([[1, 2], [3, 4]])
v = Tensor([[5, 6], [7, 8]])

print(u+v)
print(u*v)

w = Tensor([1, 1])
x = u*w
x.backward()
print(u)

# how do we get all the way to MLP?