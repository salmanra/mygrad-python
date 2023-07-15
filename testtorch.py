import torch
from mygrad.engine import Tensor


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



### ALIGNED PARADIGMS: aka, we beasted
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
