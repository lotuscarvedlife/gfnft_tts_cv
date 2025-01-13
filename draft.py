import torch

t = torch.tensor([[1,2,3],[4,5,6]], dtype=float)
a = t.cumsum(-1)
# t = t/0.9
print(a)
b = a.log_softmax(dim=0)
print(b)