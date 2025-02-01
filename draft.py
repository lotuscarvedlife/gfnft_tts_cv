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

# logPF = torch.tensor([[1,2,0,0,0],
#                   [7,5,4,0,0],
#                   [3,4,5,6,0]], dtype=float)

# top2_values, _ = torch.topk(logPF, k=2, dim=-1)
# max_prob = top2_values[:, 0]
# second_max_prob = top2_values[:, 1]
# result_tensor = max_prob - second_max_prob
# a = [1,2]
# logPF[:, a] += 4
# logPF = logPF.cumsum(1)[:, -1].unsqueeze(1)
# logPF *= torch.tensor([-1], dtype=logPF.dtype)
# logPF = torch.where(logPF<torch.tensor([-5], dtype=logPF.dtype),
#                     logPF+torch.tensor([-8000], dtype=logPF.dtype),
#                     logPF)
# # logPF = logPF - (logPF.sum(dim=0)/logPF.shape[0])

# # 创建一个形状为 [T] 的 tensor，包含从 1 到 T 的值
# # 使用广播机制将 divisors 扩展到与 tensor 相同的形状，并执行逐元素除法
# logPF[:, 1:] = logPF[:, 1:] / torch.arange(1, logPF.shape[1], dtype=logPF.dtype, device=logPF.device).unsqueeze(0)

prob = torch.rand([1, 20])
print(prob[0])
prob_1 = prob**(torch.ones_like(prob)*3)
prob_1.log_softmax(dim=-1)
prob_2 = torch.log_softmax(prob, dim=-1)
prob_2 = prob_2*(torch.ones_like(prob)*3)

# topk_indices = torch.diag(torch.topk(prob, k=10, dim=-1)[1]).unsqueeze(-1)
print(prob_1==prob_2)