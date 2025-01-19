import torch

# reward = torch.tensor([[1,2,3],[4,5,6]], dtype=float)
# non_term_mask = torch.tensor([[True,False,False],[True,True,False]])
# min_len = 1
# reward = torch.where(non_term_mask.cumsum(dim=-1) - 1 < min_len, -99, reward)
# print(reward)

# a = {
#     "c": 1,
#     "b": torch.tensor([[1,2,3]])
# }

# device = "cuda"
# a["b"] = a["b"][0]
# print(a)

logPF = torch.tensor([[1,2,0,0,0],
                  [6,5,4,0,0],
                  [3,4,5,6,0]], dtype=float)
logPF = logPF - (logPF.sum(dim=0)/logPF.shape[0])

# mask = (a!=0)
# sum_non_zero = torch.where(mask, a, torch.tensor(0.0)).sum(dim=0)
# count_non_zero = mask.sum(dim=0).float()
# count_non_zero[count_non_zero == 0] = 1
# mean_non_zero = sum_non_zero / count_non_zero
# expanded_mean = mean_non_zero.expand_as(a)
# a = torch.where(mask, a - expanded_mean, a)
print(logPF)