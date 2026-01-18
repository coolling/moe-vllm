import torch
tokens_per_expert = torch.tensor([5, 20, 3, 15, 8])  # 示例数据：5个专家各自的激活token数
num_experts = len(tokens_per_expert)
# 获取降序排列后的索引
sorted_indices = torch.argsort(tokens_per_expert, descending=True)
# print("排序后的专家索引:", sorted_indices[0])
for i, num_tokens in enumerate(sorted_indices):
    print(i,num_tokens[0])
# 使用 sorted_indices 来重排任何与专家相关的张量
# 例如，假设 experts_weights 形状为 [num_experts, hidden_dim, hidden_dim]
# sorted_experts_weights = experts_weights[sorted_indices]