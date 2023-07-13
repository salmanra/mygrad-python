import torch
from mygrad.engine import Tensor


def comparemygrad(array1, array2):
    ''' a comparison between pytorch's treatment of matmul between two matrices and mygrad
    TODO: right now, these are square matrices, make one for non-square matrices of valid size
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
    # we can't, we don't have .sum
    mw = Tensor(array1)
    mx = Tensor(array2)
    mb = mw@mx
    mloss = mb.sum()
    mloss.backward()
    print(f'mw: {mw.data}')
    print(f'mw.grad: {mw.grad}')
    print(f'mx.grad: {mx.grad}')
    # print(f'mb: {mb.data}')
    # print(f'mb.grad: {mb.grad}')
    # print(f'mloss.grad: {mloss.grad}')
    # print(f'mloss.data: {mloss.data}')


comparemygrad([[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]])
# comparemygrad([[1.0, 2.0], [3.0, 4.0]], [1.0, 2.0])
