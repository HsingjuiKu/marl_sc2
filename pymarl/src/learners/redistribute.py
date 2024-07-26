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
        else: 
            batch_size, num_agents, obs_dim = obs.shape
        influences = []
        print(obs.shape, actions.shape)
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
        print(influences.shape)
        influences = influences.mean(dim = 2, keepdim= False)
        
        return influences
    
    def find_most_relevant_teachers(self, bottom_agents, top_agents, batch):
        # 这个方法需要根据您的具体需求实现
        # 这里只是一个简单的示例，选择观察空间最相似的智能体作为老师
        relevance_scores = torch.zeros(len(bottom_agents), len(top_agents))
        for i, student in enumerate(bottom_agents):
            for j, teacher in enumerate(top_agents):
                relevance_scores[i, j] = -torch.norm(batch["obs"][:, :, student] - batch["obs"][:, :, teacher])

        _, teacher_indices = relevance_scores.max(dim=1)
        return [top_agents[i] for i in teacher_indices]

    def compute_distillation_loss(self, student_actions, teacher_actions, mask):
        # 使用KL散度作为蒸馏损失
        kl_div = F.kl_div(
            student_actions.log_softmax(dim=-1),
            teacher_actions.softmax(dim=-1),
            reduction='none'
        )
    
        return (kl_div.sum(dim=-1) * mask).sum() / mask.sum()
