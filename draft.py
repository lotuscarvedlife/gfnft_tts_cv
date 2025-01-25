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

logPF = logPF.gather(1, (logPF != 0).cumsum(-1).argmax(-1).unsqueeze(1))
# logPF = logPF.cumsum(1)[:, -1].unsqueeze(1)
# logPF *= torch.tensor([-1], dtype=logPF.dtype)
# logPF = torch.where(logPF<torch.tensor([-5], dtype=logPF.dtype),
#                     logPF+torch.tensor([-8000], dtype=logPF.dtype),
#                     logPF)
# # logPF = logPF - (logPF.sum(dim=0)/logPF.shape[0])

# # 创建一个形状为 [T] 的 tensor，包含从 1 到 T 的值
# # 使用广播机制将 divisors 扩展到与 tensor 相同的形状，并执行逐元素除法
# logPF[:, 1:] = logPF[:, 1:] / torch.arange(1, logPF.shape[1], dtype=logPF.dtype, device=logPF.device).unsqueeze(0)

print(logPF)