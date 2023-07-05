import torch
from mygrad.nn import Tensor

# this one is for aligning my definition of mygrad Tensor operators with those of Pytorch
# later, this will be used to construct the forward passes

W = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
x = torch.tensor([1.0, 2.0])

# these two are the same: elt-wise mul
print(W.__mul__(x))
print(W*x)

# this one gets broadcasted
print(W+x)

# these three are the same: matrix multiplication
print(W.__matmul__(x))
print(W.matmul(x))
print(W@x)

# now mygrad. my sense is that it already differs

V = Tensor([[1.0, 2.0], [3.0, 4.0]])
y = Tensor([1.0, 2.0])

# # these are obviouslty the same: __mul__ is the operator *
print(V.__mul__(y))
print(V*y)

# # matmul is undefined
print(V.__matmul__(y))
print(V@y)

# # add works like pytorch: our underlying numpy rep broadcasts this the same way
print(V.__add__(y))
print(V+y)