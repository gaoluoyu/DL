import torch

print(torch.__version__)
print(torch.cuda.is_available())

a = torch.arange(20).reshape(4, 5)
print(a)
print(a.sum())
print(a.cumsum(axis=0))
print(a.cumsum(axis=1))

b = torch.arange(5)
print(b)
c = torch.mv(a, b)
print(c)

b = torch.arange(5).reshape(5, 1)

C = torch.mm(a, b)
print(c)
