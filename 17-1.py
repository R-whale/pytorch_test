import torch
import numpy as np
#tensor 向量
x = torch.empty(5, 3)
print(x)

x = torch.rand(5, 3)
print(x)

x = torch.zeros(5, 3, dtype=torch.long)
print(x)

x = torch.tensor([5.5, 3])
print(x)

x = x.new_ones(5, 3, dtype=torch.double)
print(x)

x = torch.randn_like(x, dtype=torch.float)
print(x)

print(x.size())

y = torch.rand(5,3)
result=torch.empty(5,3)
torch.add(x,y,out=result)

print(result)
y.add_(x)
print(y)
print(y[:,1])#第二列

x = torch.rand(4,4)
y = x.view(16)
z = x.view(-1,8)
print(x.size(),y.size(),z.size())

a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
a.add_(1)
print(a)
print(b)

a=np.ones(5)
b=torch.from_numpy(a)
np.add(a,1,out=a)
print(a,b)
