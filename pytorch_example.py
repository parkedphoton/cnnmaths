
import torch

#random seed for three things:
#This is to reproduce result. In prod, should use timer for the seed
#See github.com/pytorch/pytorch/issues/7068
'''
import random
random.seed(42)
import numpy as np
np.random.seed(42)
import torch
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
'''



x = torch.randn(4,4)
y = x.view(16)
z = x.view(-1, 8)
print(x)
print(y)
print(z)


print(torch.cuda.is_available())

print('--------------------------')
x = torch.randn(4,4, requires_grad=True)
y = x*x
y.requires_grad_()
print(x)
print(y)
print(x.requires_grad, y.requires_grad)

z = y.mean()
z.backward()

print(x.grad)
print(y.grad)


#print(x.exp())

print('----------gradient is only retained on leaf variable: to retain gradient on intermediate variable see discuss.pytorch.org/t/why-cant-i-see-grad-of-an-intermediate-variable/94------------------')
#This is why you can keep using the same variable name x for all subsequent layers


x = torch.randn(4,4, requires_grad=True)
w = torch.empty(4,4, requires_grad=True)
y = x*x
y = y*w
z = y.mean()
z.backward()
print(x.grad)
print(y.grad)
print(w.grad)








print('---------------------------')
x = torch.randn(4,4, requires_grad=True)
x = x*x
x = x.mean()
x.backward
print(x.grad)



