import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math

"""
MLP结构
"""


class AttentionModule(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)

    def forward(self, student_obs, teacher_obs):
        # student_obs: [batch_size, seq_len, obs_dim]
        # teacher_obs: [batch_size, seq_len, obs_dim]

        q = self.query(student_obs)  # [batch_size, seq_len, hidden_dim]
        k = self.key(teacher_obs)  # [batch_size, seq_len, hidden_dim]
        v = self.value(teacher_obs)  # [batch_size, seq_len, hidden_dim]

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)

        # Apply attention
        context = torch.matmul(attn_probs, v)

        # Compute relevance as the mean attention probability
        relevance = attn_probs.mean()  # [batch_size]

        return relevance


class EnhancedCausalModel(nn.Module):
    def __init__(self, num_agents, obs_dim, action_dim, device):
        super().__init__()
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.attention = AttentionModule(self.obs_dim,128)
        self.network = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        ).to(device)

        self.attention = AttentionModule(obs_dim, 128).to(device)

    def predict_others_actions(self, obs, action):
        return self.network(torch.cat([obs, action], dim=-1))

    def calculate_social_influence(self, obs, actions):
        if len(obs.shape) == 4:
            batch_size, episode_length, num_agents, obs_dim = obs.shape
        else:
            batch_size, num_agents, obs_dim = obs.shape
        influences = []
        # print(obs.shape, actions.shape)
        adaptive_factor = max(10, num_agents)

        for k in range(num_agents):
            obs_k = obs[:, :, k,:]
            action_k = actions[:, :episode_length, k,:]
            p_with_k = self.predict_others_actions(obs_k, action_k).to(self.device)
            p_without_k = self.predict_others_actions(obs_k, torch.zeros_like(action_k)).to(self.device)
            # print(p_with_k.shape,p_without_k.shape )
            for _ in range(adaptive_factor):
                counterfactual_actions = torch.rand_like(action_k.float()).to(self.device)  # Generate random actions
                p_without_k += self.predict_others_actions(obs_k, counterfactual_actions)
            p_without_k /= (adaptive_factor + 1)

            influence = F.kl_div(
                p_with_k.log_softmax(dim=-1),
                p_without_k.softmax(dim=-1),
                reduction='none'
            )
            influences.append(influence)

        influences = torch.stack(influences, dim=-1)
        # print(influences.shape)
        influences = influences.mean(dim = 2, keepdim= False)

        return influences

    def calculate_performance_score(self, rewards, actions):
        # rewards: [batch_size, episode_length, 1]
        # actions: [batch_size, episode_length, num_agents, action_dim]

        # 计算每个智能体的动作对全局奖励的贡献
        action_influence = torch.abs(actions.sum(dim=3))  # [batch_size, episode_length, num_agents]

        # 将奖励扩展到每个智能体
        expanded_rewards = rewards.expand(-1, -1, self.num_agents)

        # 计算每个智能体的表现得分
        performance_scores = (action_influence[:,:-1,:] * expanded_rewards).sum(dim=(0, 1))

        return performance_scores

    # def calculate_cooperation_score(self, team_reward, individual_rewards):
    #     # 计算每个智能体对团队奖励的贡献
    #     team_contribution = team_reward.unsqueeze(-1) - individual_rewards.sum(dim=-1, keepdim=True)
    #     individual_contribution = individual_rewards / (team_reward.unsqueeze(-1) + 1e-8)
    #     return torch.softmax(individual_contribution, dim=-1)

    def calculate_cooperation_score(self, rewards, actions):
        # rewards: [batch_size, episode_length, 1]
        # actions: [batch_size, episode_length, num_agents, action_dim]
        # 计算每个时间步的平均动作
        mean_actions = actions.mean(dim=2, keepdim=True)

        # 计算每个智能体的动作与平均动作的差异
        action_deviation = torch.norm(actions - mean_actions, dim=3)  # [batch_size, episode_length, num_agents]

        # 计算奖励的变化率
        reward_change = torch.diff(rewards.squeeze(-1), dim=1)  # [batch_size, episode_length-1]
        reward_change = F.pad(reward_change, (0, 1), mode='replicate')  # 填充到原始长度

        # 将奖励变化扩展到每个智能体
        expanded_reward_change = reward_change.unsqueeze(-1).expand(-1, -1, self.num_agents)

        # 计算合作得分：动作偏离度低且奖励变化正面的情况下得分高
        cooperation_scores = ((1 - action_deviation[:,:-1,:]) * F.relu(expanded_reward_change)).sum(dim=(0, 1))

        return cooperation_scores

    def calculate_innovation_score(self, actions):
        # 计算每个智能体行为的熵，衡量其创新性
        action_entropy = -torch.sum(actions * torch.log(actions + 1e-8), dim=-1)
        return action_entropy.mean(dim=(0, 1))

    def calculate_comprehensive_score(self, obs, actions, rewards):
        def min_max_normalize(x):
            return (x - x.min()) / (x.max() - x.min() + 1e-8)

        influence_scores = self.calculate_social_influence(obs, actions)
        performance_scores = self.calculate_performance_score(rewards, actions)
        cooperation_scores = self.calculate_cooperation_score(rewards, actions)
        innovation_scores = self.calculate_innovation_score(actions)

        # 对每个得分进行min-max标准化
        normalized_influence = min_max_normalize(influence_scores.mean(dim=(0, 1)))
        normalized_performance = min_max_normalize(performance_scores)
        normalized_cooperation = min_max_normalize(cooperation_scores)
        normalized_innovation = min_max_normalize(innovation_scores)

        # 综合评分，可以根据需要调整权重
        comprehensive_scores = (
                normalized_influence * 0.4 +
                normalized_performance * 0.25 +
                normalized_cooperation * 0.25 +
                normalized_innovation * 0.1
        )
        return comprehensive_scores

    # def find_most_relevant_teachers(self, bottom_agents, top_agents, batch):
    #
    #     relevance_scores = torch.zeros(len(bottom_agents), len(top_agents))
    #     for i, student in enumerate(bottom_agents):
    #         for j, teacher in enumerate(top_agents):
    #             relevance_scores[i, j] = -torch.norm(batch["obs"][:, :, student] - batch["obs"][:, :, teacher])
    #
    #     _, teacher_indices = relevance_scores.max(dim=1)
    #     return [top_agents[i] for i in teacher_indices]

    def find_most_relevant_teachers(self, bottom_agents, top_agents, batch):
        relevance_scores = torch.zeros(len(bottom_agents), len(top_agents), device=self.device)
        for i, student in enumerate(bottom_agents):
            student_obs = batch["obs"][:, :, student]
            for j, teacher in enumerate(top_agents):
                teacher_obs = batch["obs"][:, :, teacher]
                # print (student_obs.shape, teacher_obs.shape)
                relevance_scores[i, j] = self.attention(student_obs, teacher_obs)

        _, teacher_indices = relevance_scores.max(dim=1)
        return [top_agents[i] for i in teacher_indices]

    def compute_distillation_loss(self, student_actions, teacher_actions, mask):
        # 使用KL散度作为蒸馏损失
        kl_div = F.kl_div(
            student_actions.log_softmax(dim=-1),
            teacher_actions.softmax(dim=-1),
            reduction='none'
        )
        # print(kl_div.shape, mask.shape)
        return (kl_div.sum(dim=-1,keepdim = True) * mask).sum() / mask.sum()




"""
Transformer 架构
"""

# class AttentionModule(nn.Module):
#     def __init__(self, input_dim, hidden_dim):
#         super().__init__()
#         self.query = nn.Linear(input_dim, hidden_dim)
#         self.key = nn.Linear(input_dim, hidden_dim)
#         self.value = nn.Linear(input_dim, hidden_dim)
#
#     def forward(self, student_obs, teacher_obs):
#         # student_obs: [batch_size, seq_len, obs_dim]
#         # teacher_obs: [batch_size, seq_len, obs_dim]
#
#         q = self.query(student_obs)  # [batch_size, seq_len, hidden_dim]
#         k = self.key(teacher_obs)  # [batch_size, seq_len, hidden_dim]
#         v = self.value(teacher_obs)  # [batch_size, seq_len, hidden_dim]
#
#         # Compute attention scores
#         attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
#         attn_probs = F.softmax(attn_scores, dim=-1)
#
#         # Apply attention
#         context = torch.matmul(attn_probs, v)
#
#         # Compute relevance as the mean attention probability
#         relevance = attn_probs.mean()  # [batch_size]
#
#         return relevance
#
#
# class RotaryPositionalEmbedding(nn.Module):
#     def __init__(self, d_model, max_seq_len=1000):
#         super().__init__()
#         inv_freq = 1. / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
#         self.register_buffer('inv_freq', inv_freq)
#         self.max_seq_len = max_seq_len
#
#     def forward(self, x, seq_len):
#         t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
#         freqs = torch.einsum('i,j->ij', t, self.inv_freq)
#         emb = torch.cat((freqs, freqs), dim=-1)
#         return emb[None, :seq_len, :]
#
# def rotate_half(x):
#     x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
#     return torch.cat((-x2, x1), dim=-1)
#
# def apply_rotary_pos_emb(q, k, pos_emb):
#     q = q + pos_emb
#     k = k + pos_emb
#     return q, k
#
# class GLU(nn.Module):
#     def __init__(self, d_model):
#         super().__init__()
#         self.act = nn.GELU()
#         self.fc = nn.Linear(d_model, d_model * 2)
#
#     def forward(self, x):
#         x, gate = self.fc(x).chunk(2, dim=-1)
#         return x * self.act(gate)
#
#
# class EnhancedTransformerBlock(nn.Module):
#     def __init__(self, d_model, num_heads, dropout=0.1):
#         super().__init__()
#         self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.glu = GLU(d_model)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x, pos_emb):
#         q, k = apply_rotary_pos_emb(x, x, pos_emb)
#         attn_output, _ = self.attention(q, k, x)
#         x = self.norm1(x + self.dropout(attn_output))
#         glu_output = self.glu(x)
#         x = self.norm2(x + self.dropout(glu_output))
#         return x
#
#
# class EnhancedCausalModel(nn.Module):
#     def __init__(self, num_agents, obs_dim, action_dim, device, d_model=64, num_heads=4, num_layers=2, dropout=0.1):
#         super().__init__()
#         self.num_agents = num_agents
#         self.obs_dim = obs_dim
#         self.action_dim = action_dim
#         self.device = device
#         self.d_model = d_model
#
#         self.input_proj = nn.Linear(obs_dim + action_dim, d_model).to(device)
#         self.pos_emb = RotaryPositionalEmbedding(d_model).to(device)
#         self.transformer_layers = nn.ModuleList([
#             EnhancedTransformerBlock(d_model, num_heads, dropout).to(device) for _ in range(num_layers)
#         ])
#         self.output_proj = nn.Linear(d_model, action_dim).to(device)
#
#     # def forward(self, obs, actions):
#     #     x = torch.cat([obs, actions], dim=-1).to(self.device)
#     #     x = self.input_proj(x)
#     #     x = x.permute(1, 0, 2)  # (seq_len, batch, d_model)
#     #
#     #     pos_emb = self.pos_emb(x, x.size(0))
#     #
#     #     for layer in self.transformer_layers:
#     #         x = layer(x, pos_emb)
#     #
#     #     x = x.permute(1, 0, 2)  # (batch, seq_len, d_model)
#     #     return self.output_proj(x)
#
#     def predict_others_actions(self, obs, action):
#         return self.network(torch.cat([obs, action], dim=-1))
#
#     def calculate_social_influence(self, obs, actions):
#         if len(obs.shape) == 4:
#             batch_size, episode_length, num_agents, obs_dim = obs.shape
#         else:
#             batch_size, num_agents, obs_dim = obs.shape
#         influences = []
#         # print(obs.shape, actions.shape)
#         adaptive_factor = max(10, num_agents)
#
#         for k in range(num_agents):
#             obs_k = obs[:, :, k, :]
#             action_k = actions[:, :episode_length, k, :]
#             p_with_k = self.predict_others_actions(obs_k, action_k).to(self.device)
#             p_without_k = self.predict_others_actions(obs_k, torch.zeros_like(action_k)).to(self.device)
#             # print(p_with_k.shape,p_without_k.shape )
#             for _ in range(adaptive_factor):
#                 counterfactual_actions = torch.rand_like(action_k.float()).to(self.device)  # Generate random actions
#                 p_without_k += self.predict_others_actions(obs_k, counterfactual_actions)
#             p_without_k /= (adaptive_factor + 1)
#
#             influence = F.kl_div(
#                 p_with_k.log_softmax(dim=-1),
#                 p_without_k.softmax(dim=-1),
#                 reduction='none'
#             )
#             influences.append(influence)
#
#         influences = torch.stack(influences, dim=-1)
#         # print(influences.shape)
#         influences = influences.mean(dim=2, keepdim=False)
#
#         return influences
#
#     def calculate_performance_score(self, rewards, actions):
#         # rewards: [batch_size, episode_length, 1]
#         # actions: [batch_size, episode_length, num_agents, action_dim]
#
#         # 计算每个智能体的动作对全局奖励的贡献
#         action_influence = torch.abs(actions.sum(dim=3))  # [batch_size, episode_length, num_agents]
#
#         # 将奖励扩展到每个智能体
#         expanded_rewards = rewards.expand(-1, -1, self.num_agents)
#
#         # 计算每个智能体的表现得分
#         performance_scores = (action_influence[:, :-1, :] * expanded_rewards).sum(dim=(0, 1))
#
#         return performance_scores
#
#     # def calculate_cooperation_score(self, team_reward, individual_rewards):
#     #     # 计算每个智能体对团队奖励的贡献
#     #     team_contribution = team_reward.unsqueeze(-1) - individual_rewards.sum(dim=-1, keepdim=True)
#     #     individual_contribution = individual_rewards / (team_reward.unsqueeze(-1) + 1e-8)
#     #     return torch.softmax(individual_contribution, dim=-1)
#
#     def calculate_cooperation_score(self, rewards, actions):
#         # rewards: [batch_size, episode_length, 1]
#         # actions: [batch_size, episode_length, num_agents, action_dim]
#         # 计算每个时间步的平均动作
#         mean_actions = actions.mean(dim=2, keepdim=True)
#
#         # 计算每个智能体的动作与平均动作的差异
#         action_deviation = torch.norm(actions - mean_actions, dim=3)  # [batch_size, episode_length, num_agents]
#
#         # 计算奖励的变化率
#         reward_change = torch.diff(rewards.squeeze(-1), dim=1)  # [batch_size, episode_length-1]
#         reward_change = F.pad(reward_change, (0, 1), mode='replicate')  # 填充到原始长度
#
#         # 将奖励变化扩展到每个智能体
#         expanded_reward_change = reward_change.unsqueeze(-1).expand(-1, -1, self.num_agents)
#
#         # 计算合作得分：动作偏离度低且奖励变化正面的情况下得分高
#         cooperation_scores = ((1 - action_deviation[:, :-1, :]) * F.relu(expanded_reward_change)).sum(dim=(0, 1))
#
#         return cooperation_scores
#
#     def calculate_innovation_score(self, actions):
#         # 计算每个智能体行为的熵，衡量其创新性
#         action_entropy = -torch.sum(actions * torch.log(actions + 1e-8), dim=-1)
#         return action_entropy.mean(dim=(0, 1))
#
#     def calculate_comprehensive_score(self, obs, actions, rewards):
#         def min_max_normalize(x):
#             return (x - x.min()) / (x.max() - x.min() + 1e-8)
#
#         influence_scores = self.calculate_social_influence(obs, actions)
#         performance_scores = self.calculate_performance_score(rewards, actions)
#         cooperation_scores = self.calculate_cooperation_score(rewards, actions)
#         innovation_scores = self.calculate_innovation_score(actions)
#
#         # 对每个得分进行min-max标准化
#         normalized_influence = min_max_normalize(influence_scores.mean(dim=(0, 1)))
#         normalized_performance = min_max_normalize(performance_scores)
#         normalized_cooperation = min_max_normalize(cooperation_scores)
#         normalized_innovation = min_max_normalize(innovation_scores)
#
#         # 综合评分，可以根据需要调整权重
#         comprehensive_scores = (
#                 normalized_influence * 0.4 +
#                 normalized_performance * 0.25 +
#                 normalized_cooperation * 0.25 +
#                 normalized_innovation * 0.1
#         )
#         return comprehensive_scores
#
#     # def find_most_relevant_teachers(self, bottom_agents, top_agents, batch):
#     #
#     #     relevance_scores = torch.zeros(len(bottom_agents), len(top_agents))
#     #     for i, student in enumerate(bottom_agents):
#     #         for j, teacher in enumerate(top_agents):
#     #             relevance_scores[i, j] = -torch.norm(batch["obs"][:, :, student] - batch["obs"][:, :, teacher])
#     #
#     #     _, teacher_indices = relevance_scores.max(dim=1)
#     #     return [top_agents[i] for i in teacher_indices]
#
#     def find_most_relevant_teachers(self, bottom_agents, top_agents, batch):
#         relevance_scores = torch.zeros(len(bottom_agents), len(top_agents), device=self.device)
#         for i, student in enumerate(bottom_agents):
#             student_obs = batch["obs"][:, :, student]
#             for j, teacher in enumerate(top_agents):
#                 teacher_obs = batch["obs"][:, :, teacher]
#                 # print (student_obs.shape, teacher_obs.shape)
#                 relevance_scores[i, j] = self.attention(student_obs, teacher_obs)
#
#         _, teacher_indices = relevance_scores.max(dim=1)
#         return [top_agents[i] for i in teacher_indices]
#
#     def compute_distillation_loss(self, student_actions, teacher_actions, mask):
#         # 使用KL散度作为蒸馏损失
#         kl_div = F.kl_div(
#             student_actions.log_softmax(dim=-1),
#             teacher_actions.softmax(dim=-1),
#             reduction='none'
#         )
#         # print(kl_div.shape, mask.shape)
#         return (kl_div.sum(dim=-1, keepdim=True) * mask).sum() / mask.sum()
#
