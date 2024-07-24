import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

"""
MLP结构
"""

class EnhancedCausalModel(nn.Module):
    def __init__(self, num_agents, obs_dim, action_dim, device):
        super().__init__()
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

        self.network = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        ).to(device)

    def predict_others_actions(self, obs, action):
        return self.network(torch.cat([obs, action], dim=-1))

    def calculate_social_influence(self, obs, actions):
        if len(obs.shape) == 4:
            batch_size, episode_length, num_agents, obs_dim = obs.shape
            self.episode_length = episode_length
            self.num_agents = num_agents
        else: 
            batch_size, num_agents, obs_dim = obs.shape
        influences = []
        
        adaptive_factor = max(10, num_agents)
        
        for k in range(num_agents):
            # agent_idx = k % num_agent
            obs_k = obs[:, :, k,:]
            action_k = actions[:, :episode_length, k,:]
            p_with_k = self.predict_others_actions(obs_k, action_k).to(self.device)
            p_without_k = self.predict_others_actions(obs_k, torch.zeros_like(action_k)).to(self.device)
            # print(p_with_k.shape,p_without_k.shape )
            for _ in range(adaptive_factor):
                counterfactual_actions = torch.rand_like(action_k).to(self.device)  # Generate random actions
                p_without_k += self.predict_others_actions(obs_k, counterfactual_actions)
            p_without_k /= (adaptive_factor + 1)

            influence = F.kl_div(
                p_with_k.log_softmax(dim=-1),
                p_without_k.softmax(dim=-1),
                reduction='none'
            )
            influences.append(influence)
        
        influences = torch.stack(influences, dim=-1)
        influences = influences.mean(dim = 2, keepdim =False)
        # l2_norm = influences.norm(p=2, dim=-1, keepdim=True)
        # influences = influences / (l2_norm + 1e-8) 
        # print(influences.shape)
        influences = influences.mean(dim = -1, keepdim = False)
        influences = influences.unsqueeze(-1)
        # Calculate the sum along the second axis (dim=1)
        sums = influences.sum(dim=(1), keepdim=True)
        # print(sums.shape)

        # Normalize each value by dividing by the sum of its corresponding triplet
        influences = influences / sums

        return influences

    def calculate_social_contribution_index(self, obs, actions):
        influences = self.calculate_social_influence(obs, actions)
        return influences

    def calculate_tax_rates(self, social_contribution_index):
        # Define the condition masks
        mask_dd = social_contribution_index < 0.1
        mask_d = (social_contribution_index >= 0.1) & (social_contribution_index <= 0.3)
        mask_M = (social_contribution_index >= 0.3) & (social_contribution_index <= 0.5)
        mask_H = (social_contribution_index >= 0.5)

        # Apply the conditions to replace the values
        social_contribution_index[mask_dd] = 0.01
        social_contribution_index[mask_d] = 0.15
        social_contribution_index[mask_M] = 0.45
        social_contribution_index[mask_H] = 0.8

        # # Apply the conditions to replace the values(anti)
        # social_contribution_index[mask_dd] = 0.05
        # social_contribution_index[mask_d] = 0.1
        # social_contribution_index[mask_M] = 0.15
        # social_contribution_index[mask_H] = 0.2
        return social_contribution_index
        

    def redistribute_rewards(self, original_rewards, social_contribution_index, tax_rates, beta=0.5, alpha=1.0):
        # print("------------------------")
        # print("Tax rate shape: ", tax_rates.shape)
        # print(tax_rates)
        # print("------------------------")
        central_pool = (tax_rates * original_rewards).sum(dim=1, keepdim=True)
        # print("------------------------")
        # print("central_pools shape 1:", (tax_rates * original_rewards).shape)
        # print("central_pools shape :", central_pool.shape)
        # print(central_pool)
        # print("------------------------")
        # normalized_contributions = social_contribution_index / (social_contribution_index.sum(dim=-1, keepdim=True) + 1e-8)
        # print("N shape",normalized_contributions.shape )
        # print(normalized_contributions)
        redistributed_rewards = (1 - tax_rates) * original_rewards + social_contribution_index * central_pool
        # print("1ge",(1 - tax_rates) * original_rewards)
        # print("1ge shape",((1 - tax_rates) * original_rewards).shape)
        # print("2ge", beta * social_contribution_index * central_pool )
        # print("2ge shape", (beta * social_contribution_index * central_pool).shape )
        # return alpha * redistributed_rewards + (1 - alpha) * original_rewards
        # print("shape re:0", redistributed_rewards.shape)
        # print(redistributed_rewards)
        return redistributed_rewards


"""
使用Transformer的核心架构
加入相对位置编码，以更好地处理时序信息
使用GLU (Gated Linear Unit) 激活函数，提高模型的表达能力
加入旋转位置编码 (RoPE)，进一步增强位置感知能力

主要的改进包括：

引入了RotaryPositionalEmbedding类，实现旋转位置编码。这种编码方式可以更好地处理长序列和相对位置信息。
使用GLU（门控线性单元）替代了简单的前馈网络。GLU 可以让模型更容易学习复杂的函数，提高模型的表达能力。
在EnhancedTransformerBlock中，我们应用了旋转位置编码，并使用了GLU。
在主模型EnhancedCausalModel中，我们整合了这些改进，同时保持了原有的社会影响力计算和奖励重分配逻辑。
"""

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math

# class RotaryPositionalEmbedding(nn.Module):
#     def __init__(self, d_model, max_seq_len=1000):
#         super().__init__()
#         inv_freq = 1. / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
#         self.register_buffer('inv_freq', inv_freq)
#         self.max_seq_len = max_seq_len

#     def forward(self, x, seq_len):
#         t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
#         freqs = torch.einsum('i,j->ij', t, self.inv_freq)
#         emb = torch.cat((freqs, freqs), dim=-1)
#         return emb[None, :seq_len, :]

# def rotate_half(x):
#     x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
#     return torch.cat((-x2, x1), dim=-1)

# def apply_rotary_pos_emb(q, k, pos_emb):
#     q = q + pos_emb
#     k = k + pos_emb
#     return q, k

# class GLU(nn.Module):
#     def __init__(self, d_model):
#         super().__init__()
#         self.act = nn.GELU()
#         self.fc = nn.Linear(d_model, d_model * 2)

#     def forward(self, x):
#         x, gate = self.fc(x).chunk(2, dim=-1)
#         return x * self.act(gate)

# class EnhancedTransformerBlock(nn.Module):
#     def __init__(self, d_model, num_heads, dropout=0.1):
#         super().__init__()
#         self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.glu = GLU(d_model)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x, pos_emb):
#         q, k = apply_rotary_pos_emb(x, x, pos_emb)
#         attn_output, _ = self.attention(q, k, x)
#         x = self.norm1(x + self.dropout(attn_output))
#         glu_output = self.glu(x)
#         x = self.norm2(x + self.dropout(glu_output))
#         return x

# class EnhancedCausalModel(nn.Module):
#     def __init__(self, num_agents, obs_dim, action_dim, device, d_model=64, num_heads=4, num_layers=2, dropout=0.1):
#         super().__init__()
#         self.num_agents = num_agents
#         self.obs_dim = obs_dim
#         self.action_dim = action_dim
#         self.device = device
#         self.d_model = d_model

#         self.input_proj = nn.Linear(obs_dim + action_dim, d_model).to(device)
#         self.pos_emb = RotaryPositionalEmbedding(d_model).to(device)
#         self.transformer_layers = nn.ModuleList([
#             EnhancedTransformerBlock(d_model, num_heads, dropout).to(device) for _ in range(num_layers)
#         ])
#         self.output_proj = nn.Linear(d_model, action_dim).to(device)

#     def forward(self, obs, actions):
#         x = torch.cat([obs, actions], dim=-1).to(self.device)
#         x = self.input_proj(x)
#         x = x.permute(1, 0, 2)  # (seq_len, batch, d_model)

#         pos_emb = self.pos_emb(x, x.size(0))

#         for layer in self.transformer_layers:
#             x = layer(x, pos_emb)

#         x = x.permute(1, 0, 2)  # (batch, seq_len, d_model)
#         return self.output_proj(x)

#     def predict_others_actions(self, obs, action):
#         return self.forward(obs.unsqueeze(1), action.unsqueeze(1)).squeeze(1)

#     def calculate_social_influence(self, obs, actions):
#         obs = obs.to(self.device)
#         actions = actions.to(self.device)
#         batch_size, num_agents, obs_dim = obs.shape
#         influences = []

#         adaptive_factor = max(10, num_agents)

#         for k in range(num_agents):
#             agent_idx = k % num_agents
#             obs_k = obs[:, agent_idx]
#             action_k = actions[:, agent_idx]
#             p_with_k = self.predict_others_actions(obs_k, action_k)
#             p_without_k = self.predict_others_actions(obs_k, torch.zeros_like(action_k).to(self.device))
#             for _ in range(adaptive_factor):
#                 counterfactual_actions = torch.rand_like(action_k).to(self.device)
#                 p_without_k += self.predict_others_actions(obs_k, counterfactual_actions)
#             p_without_k /= (adaptive_factor + 1)

#             influence = F.kl_div(
#                 p_with_k.log_softmax(dim=-1),
#                 p_without_k.softmax(dim=-1),
#                 reduction='batchmean'
#             )
#             influences.append(influence.unsqueeze(-1))
#         influences = torch.stack(influences, dim=-1).to(self.device)
#         influences = F.softmax(influences, dim=-2)
#         influences = influences.unsqueeze(1)
#         return influences

#     def calculate_social_contribution_index(self, obs, actions):
#         influences = self.calculate_social_influence(obs, actions)
#         return influences

#     def calculate_tax_rates(self, social_contribution_index):
#         return torch.sigmoid(social_contribution_index)

#     def redistribute_rewards(self, original_rewards, social_contribution_index, tax_rates, beta=0.5, alpha=1.0):
#         central_pool = (tax_rates * original_rewards).sum(dim=1, keepdim=True).to(self.device)
#         normalized_contributions = social_contribution_index / (
#                     social_contribution_index.sum(dim=1, keepdim=True) + 1e-8)
#         redistributed_rewards = (1 - tax_rates) * original_rewards + beta * normalized_contributions * central_pool
#         redistributed_rewards = redistributed_rewards.sum(dim=-1, keepdim=True).to(self.device)
#         return redistributed_rewards


"""
添加了CausalGraph类来表示和管理因果关系。
在EnhancedCausalModel中集成了CausalGraph。
修改了calculate_social_influence方法,在计算影响力后更新因果图。
新增了update_causal_graph方法,根据计算的影响力更新因果图结构。
在redistribute_rewards方法中,使用因果图来调整社会贡献指数。
添加了learn_causal_structure方法,可以基于历史观察和行动来学习因果结构。
新增了get_causal_graph方法,允许外部访问当前的因果图结构。
"""


# class CausalGraph:
#     def __init__(self, num_agents):
#         self.num_agents = num_agents
#         self.edges = {i: set() for i in range(num_agents)}
#
#     def add_edge(self, from_agent, to_agent):
#         self.edges[from_agent].add(to_agent)
#
#     def get_parents(self, agent):
#         return [i for i in range(self.num_agents) if agent in self.edges[i]]
#
#
# class SocialInfluenceModel(nn.Module):
#     def __init__(self, num_agents, obs_dim, action_dim):
#         super().__init__()
#         self.network = nn.Sequential(
#             nn.Linear(obs_dim + action_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, action_dim * (num_agents - 1))
#         )
#
#     def forward(self, obs, action):
#         combined_input = torch.cat([obs, action], dim=-1)
#         # print(f"Combined input shape: {combined_input.shape}")
#         return self.network(combined_input)
#
#
# class EnhancedCausalModel(nn.Module):
#     def __init__(self, num_agents, obs_dim, action_dim, device):
#         super().__init__()
#         self.num_agents = num_agents
#         self.obs_dim = obs_dim
#         self.action_dim = action_dim
#         self.device = device
#
#         self.sim = SocialInfluenceModel(num_agents, obs_dim, action_dim).to(device)
#         self.causal_graph = CausalGraph(num_agents)
#         self.influence_threshold = 0.1  # 可调整的阈值
#
#     def calculate_social_influence(self, obs, actions):
#         # print(f"Observations shape: {obs.shape}")
#         # print(f"Actions shape: {actions.shape}")
#
#         batch_size, time_steps, obs_dim = obs.shape
#         _, _, action_dim = actions.shape
#
#         influences = torch.zeros(batch_size, time_steps, self.num_agents, self.num_agents).to(self.device)
#
#         obs_per_agent = obs_dim // self.num_agents
#         action_per_agent = action_dim // self.num_agents
#
#         adaptive_factor = max(10, self.num_agents)
#
#         for t in range(time_steps):
#             for k in range(self.num_agents):
#                 # obs_k = obs[:, t:t+1, k * obs_per_agent:(k + 1) * obs_per_agent]
#                 # action_k = actions[:, t:t+1, k * action_per_agent:(k + 1) * action_per_agent]
#                 obs_k = obs[:, k, :]
#                 action_k = actions[:, k, :]
#                 p_with_k = self.predict_others_actions(obs_k, action_k)
#                 p_without_k = self.predict_others_actions(obs_k, torch.zeros_like(action_k).to(self.device))
#                 for _ in range(adaptive_factor):  # Sample 10 counterfactual actions
#                     counterfactual_actions = torch.rand_like(action_k).to(self.device)
#                     p_without_k += self.predict_others_actions(obs_k, counterfactual_actions)
#                 p_without_k /= (adaptive_factor+1)
#
#                 for j in range(self.num_agents):
#                     if j != k:
#                         influence = F.kl_div(
#                             p_with_k.log_softmax(dim=-1),
#                             p_without_k.softmax(dim=-1),
#                             reduction='batchmean'
#                         )
#                         influences[:, t, k, j] = influence
#
#         influences = F.softmax(influences, dim=-1)
#
#         self.update_causal_graph(influences)
#
#         return influences
#
#     def update_causal_graph(self, influences):
#         mean_influence = influences.mean().item()
#         for i in range(self.num_agents):
#             for j in range(self.num_agents):
#                 if i != j and influences[:, :, i, j].mean().item() > mean_influence + self.influence_threshold:
#                     self.causal_graph.add_edge(i, j)
#
#     def predict_others_actions(self, obs, action):
#         return self.sim(obs, action)
#
#     def calculate_social_contribution_index(self, obs, actions):
#         influences = self.calculate_social_influence(obs, actions)
#         return influences
#
#     def calculate_tax_rates(self, social_contribution_index):
#         tax = torch.sigmoid(social_contribution_index)
#         # Reduce the last two dimensions by taking the mean
#         tax_new = tax.mean(dim=(2, 3), keepdim=True)
#
#         # Squeeze the dimensions to get the desired shape
#         tax_new = tax_new.squeeze(-1)
#         return tax_new
#
#     def redistribute_rewards(self, original_rewards, social_contribution_index, tax_rates, beta=0.5, alpha=1.0):
#         # print(tax_rates.shape, original_rewards.shape)
#         central_pool = (tax_rates * original_rewards).sum(dim=(1, 2), keepdim=True)
#
#         # 使用因果图调整贡献
#         adjusted_contributions = social_contribution_index.clone()
#         for i in range(self.num_agents):
#             parents = self.causal_graph.get_parents(i)
#             if parents:
#                 parent_contribution = social_contribution_index[:, :, :, parents].mean(dim=-1)
#                 adjusted_contributions[:, :, :, i] *= (1 + parent_contribution) / 2
#
#         normalized_contributions = adjusted_contributions / (adjusted_contributions.sum(dim=3, keepdim=True) + 1e-8)
#         normalized_contributions_n = normalized_contributions.mean(dim=(2, 3), keepdim=True)
#
#         # Squeeze the dimensions to get the desired shape
#         normalized_contributions_n = normalized_contributions_n.squeeze(-1)
#         redistributed_rewards = (1 - tax_rates) * original_rewards + beta * normalized_contributions_n * central_pool
#         return alpha * redistributed_rewards + (1 - alpha) * original_rewards
#
#     def learn_causal_structure(self, obs_history, action_history):
#         # 使用简化的基于相关性的因果发现
#         action_history_flat = action_history.view(self.num_agents, -1)
#         correlations = torch.corrcoef(action_history_flat)
#         threshold = correlations.mean() + correlations.std()
#
#         for i in range(self.num_agents):
#             for j in range(self.num_agents):
#                 if i != j and correlations[i, j] > threshold:
#                     self.causal_graph.add_edge(i, j)
#
#     def get_causal_graph(self):
#         return self.causal_graph.edges
