import torch
from mygrad.engine import Tensor
import numpy as np

# emacs is a wonderous thing. magit is a wonderous thing.
# autocomplete is a wonderous thing.


def compmul(arr1, arr2):
    w = torch.tensor(arr1, requires_grad=True)
    x = torch.tensor(arr2, requires_grad=True)
    b = w*x  # did we broadcast? did we drop it in dirt? no, grow up
    b.retain_grad()

    loss = b.sum()
    loss.backward()
    print(f'w: {w.data}')
    print(f'w.grad: {w.grad}')
    print(f'x.grad: {x.grad}')
    print(f'b: {b.data}')

    # now, do we align?
    mw = Tensor(arr1)
    mx = Tensor(arr2)
    mb = mw*mx
    print(f'mb: {mb}')
    # print(f'mx: {mx}')
    # print(f'mw: {mw}')

    mloss = mb.sum()
    mloss.backward()
    print(f'mw: {mw}')
    print(f'mx: {mx}')

# we want to verify that the addition backward pass can work even 
# when the result relies on broadcasting.
# how do we test that? We need out.grad to not be a scalar, and to 
# not be homogenous.


def compadd(arr1, arr2):
    w = torch.tensor(arr1, requires_grad=True)
    x = torch.tensor(arr2, requires_grad=True)
    b = w+x  # did we broadcast? did we drop it in dirt? no, grow up
    b.retain_grad()

    c = torch.arange(b.numel()).reshape(b.shape)
    d = b*c  # I guess we're testing multiplcation too!
    loss = d.sum() 
    loss.backward()
    print(f'w: {w.data}')
    print(f'w.grad: {w.grad}')
    print(f'x.grad: {x.grad}')
    print(f'b: {b.data}')

    # now, do we align?
    mw = Tensor(arr1)
    mx = Tensor(arr2)
    mb = mw+mx
    print(f'mb: {mb}')
    # print(f'mx: {mx}')
    # print(f'mw: {mw}')
    
    # let's make a vector to multiply with b
    mc = Tensor(np.arange(mb.data.size).reshape(mb.data.shape))
    md = mb * mc
    mloss = md.sum()
    mloss.backward()
    print(f'mw: {mw}')
    print(f'mx: {mx}')

# this one's for __matmul__ and sum


def comparemygrad(array1, array2):
    ''' 
    a comparison between pytorch's treatment of matmul between two matrices and mygrad
    '''
    w = torch.tensor(array1, requires_grad=True)
    x = torch.tensor(array2, requires_grad=True)
    b = w@x
    b.retain_grad()
    loss = b.sum() 
    loss.backward()
    print(f'w: {w.data}')
    print(f'w.grad: {w.grad}')
    print(f'x.grad: {x.grad}')
    # print(f'b: {b.data}')
    # print(f'b.grad: {b.grad}')
    # print(f'loss: {loss.data}')

    # now, do we align?
    mw = Tensor(array1)
    mx = Tensor(array2)
    mb = mw@mx
    # print(f'mb: {mb}')
    # print(f'mx: {mx}')
    # print(f'mw: {mw}')

    mloss = mb.sum()
    # print(f'mloss: {mloss}')
    mloss.backward()
    print(f'mw: {mw}')
    # print(f'mw.grad: {mw.grad}')
    print(f'mx: {mx}')
    # print(f'mb: {mb.data}')
    # print(f'mb.grad: {mb.grad}')
    # print(f'mloss.grad: {mloss.grad}')
    # print(f'mloss.data: {mloss.data}')


compmul([[1.0, 2], [3, 4]], [1.0, 2])

# ALIGNED PARADIGMS: aka, we beasted
# addition with broadcasting
# compadd([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [1.0, 2.0]) 
# addition without broadcasting
# compadd([[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]])
#
# square matmul
# comparemygrad([[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]])
#
# non-square matmul
# comparemygrad([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
# 
# mat-vec mul
# comparemygrad([[1.0, 2.0], [3.0, 4.0]], [1.0, 2.0])
# comparemygrad([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [1.0, 2.0, 3.0])
# 
# dot product
# comparemygrad([1.0, 2.0], [3.0, 4.0])
