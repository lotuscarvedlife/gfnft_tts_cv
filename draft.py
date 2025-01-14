import torch

# reward = torch.tensor([[1,2,3],[4,5,6]], dtype=float)
# non_term_mask = torch.tensor([[True,False,False],[True,True,False]])
# min_len = 1
# reward = torch.where(non_term_mask.cumsum(dim=-1) - 1 < min_len, -99, reward)
# print(reward)

a = {
    "c": 1,
    "b": torch.tensor([[1,2,3]])
}

device = "cuda"
a["b"] = a["b"][0]
print(a)