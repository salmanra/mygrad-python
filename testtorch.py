import torch

w = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
b = w@x
b.retain_grad()
loss = b.sum() 
loss.backward()
print(f'w: {w.data}')
print(f'w.grad: {w.grad}')
# print(f'x.grad: {x.grad}')
print(f'b: {b.data}')
# print(f'b.grad: {b.grad}')

# now, do we align?
# we can't, we don't have .sum