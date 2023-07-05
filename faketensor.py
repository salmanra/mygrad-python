from mygrad.nn import Layer

# the tensor extension to backprop is quite the menace

# a second order tensor of weights is the same as a "Layer" in micrograd

# b = Wx, db/dW ?
# we can reformulate this question as 
# W = Layer(nin, nout, act=False)
# x = [nin]
# b = W(x)
# p.grad for p in W.parameters()

nin = 2
nout = 2

W = Layer(nin, nout, act=False)
x = [1, 2]
c = W(x)
loss = sum(c)

loss.backward()

for p in W.parameters():
    print(p.grad)