import copy
from components.episode_buffer import EpisodeBatch
from modules.critics.maddpg import MADDPGCritic
import torch as th
from torch.optim import RMSprop, Adam
from utils.rl_utils import build_td_lambda_targets
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from redistribute import EnhancedCausalModel
import numpy as np

class MADDPGDiscreteLearner:
    def __init__(self, mac, scheme, logger, args, obs_dim, action_dim):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.mac = mac
        self.target_mac = copy.deepcopy(self.mac)
        self.agent_params = list(mac.parameters())

        self.critic = MADDPGCritic(scheme, args)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_params = list(self.critic.parameters())

        if getattr(self.args, "optimizer", "rmsprop") == "rmsprop":
            self.agent_optimiser = RMSprop(params=self.agent_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        elif getattr(self.args, "optimizer", "rmsprop") == "adam":
            self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr, eps=getattr(args, "optimizer_epsilon", 10E-8))
        else:
            raise Exception("unknown optimizer {}".format(getattr(self.args, "optimizer", "rmsprop")))

        if getattr(self.args, "optimizer", "rmsprop") == "rmsprop":
            self.critic_optimiser = RMSprop(params=self.critic_params, lr=args.critic_lr, alpha=args.optim_alpha, eps=args.optim_eps)
        elif getattr(self.args, "optimizer", "rmsprop") == "adam":
            self.critic_optimiser = Adam(params=self.critic_params, lr=args.critic_lr, eps=getattr(args, "optimizer_epsilon", 10E-8))
        else:
            raise Exception("unknown optimizer {}".format(getattr(self.args, "optimizer", "rmsprop")))

        self.log_stats_t = -self.args.learner_log_interval - 1
        self.last_target_update_episode = 0
        self.critic_training_steps = 0

        self.distillation_coef = 0.01
        self.bottom_agents = args.n_agents - int(args.n_agents / 4)
        
        # Initialize the EnhancedCausalModel for reward redistribution
        self.redistribution_model = EnhancedCausalModel(
            num_agents=self.n_agents,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            device=self.args.device
        )
        
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions_onehot"][:, :]
        terminated = batch["terminated"].float()
        mask = batch["filled"].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"][:, :-1]
        copy_mask = mask
        
        # print(mask.shape)
        obs = batch["obs"][:, :-1]
        # calculate social influence for all samples in this batch
        social_influence = self.redistribution_model.calculate_comprehensive_score(obs, actions,rewards)
        # print(social_influence.shape)
        
        # Train the critic batched
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.select_actions(batch, t_ep=t, t_env=t_env, test_mode=True)
            target_mac_out.append(target_agent_outs)
        target_mac_out = th.stack(target_mac_out, dim=1)  # Concat over time

        q_taken, _ = self.critic(batch["state"][:, :-1], actions[:, :-1])
        target_vals, _ = self.target_critic(batch["state"][:, :], target_mac_out.detach())

        q_taken = q_taken.view(batch.batch_size, -1, 1)
        target_vals = target_vals.view(batch.batch_size, -1, 1)
        targets = build_td_lambda_targets(batch["reward"], terminated, mask, target_vals, self.n_agents,
                                          self.args.gamma, self.args.td_lambda)
        mask = mask[:, :-1]
        td_error = (q_taken - targets.detach())
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask
        loss = (masked_td_error ** 2).sum() / mask.sum()

        self.critic_optimiser.zero_grad()
        loss.backward()
        critic_grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()
        self.critic_training_steps += 1


        # 根据社会影响力排序，选择表现最好和最差的智能体
        # print(social_influence.shape)
        # influence_scores = social_influence.mean(dim=(0, 1),keepdim = False)
        top_agents = social_influence.argsort(descending=True)[:self.n_agents - self.bottom_agents]
        bottom_agents = social_influence.argsort(descending=True)[-self.bottom_agents:]
        # 为每个表现较差的智能体找到最相关的榜样智能体
        # print(bottom_agents,top_agents)
        teacher_agents = self.redistribution_model.find_most_relevant_teachers(bottom_agents, top_agents, batch)
        # print(bottom_agents,teacher_agents)
        # 计算策略蒸馏损失
        distillation_loss = 0
        chosen_action_qvals = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs = self.mac.select_actions(batch, t_ep=t, t_env=t_env, test_mode=False, explore=False)
                
            for idx in range(self.n_agents):
                tem_joint_act = actions[:, t:t+1].detach().clone().view(batch.batch_size, -1, self.n_actions)
                tem_joint_act[:, idx] = agent_outs[:, idx]
                q, _ = self.critic(self._build_inputs(batch, t=t), tem_joint_act)
                chosen_action_qvals.append(q.view(batch.batch_size, -1, 1))
        chosen_action_qvals = th.stack(chosen_action_qvals, dim=1)

        # 随机采样
        sample_size = int(0.1*batch.max_seq_length) # 或其他合适的数字
        sampled_timesteps = np.random.choice(batch.max_seq_length, sample_size, replace=False)

        for student_idx, teacher_idx in zip(bottom_agents, teacher_agents):
            student_actions = []
            teacher_actions = []
        
            # 对批次中的每个时间步计算动作
            # for t in range(batch.max_seq_length):
            for t in sampled_timesteps:
                student_action = self.mac.select_actions(batch, t_ep=t, t_env=t_env, test_mode=False)[:,:,student_idx]
                teacher_action = self.mac.select_actions(batch, t_ep=t, t_env=t_env, test_mode=True)[:,:,teacher_idx]
                student_actions.append(student_action)
                teacher_actions.append(teacher_action)
        
            # 将列表转换为张量
            student_actions = th.stack(student_actions, dim=1)  # [batch_size, time_steps, action_dim]
            teacher_actions = th.stack(teacher_actions, dim=1)  # [batch_size, time_steps, action_dim]
            # print(student_actions.shape,teacher_actions.shape )
            # 计算这对学生-教师的蒸馏损失
            pair_distillation_loss = self.redistribution_model.compute_distillation_loss(student_actions, teacher_actions.detach(),
                                                                     copy_mask[:,sampled_timesteps,:])
            distillation_loss += pair_distillation_loss
            
        # Compute the actor loss
        pg_loss = -chosen_action_qvals.mean()

        # 将策略蒸馏损失添加到总损失中
        total_loss = pg_loss - self.distillation_coef * distillation_loss
        # print(pg_loss, distillation_loss, total_loss)
        # Optimise agents
        self.agent_optimiser.zero_grad()
        total_loss.backward()
        agent_grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()

        if getattr(self.args, "target_update_mode", "hard") == "hard":
            if (self.critic_training_steps - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
                self._update_targets()
                self.last_target_update_episode = self.critic_training_steps
        elif getattr(self.args, "target_update_mode", "hard") in ["soft", "exponential_moving_average"]:
            self._update_targets_soft(tau=getattr(self.args, "target_update_tau", 0.001))
        else:
            raise Exception(
                "unknown target update mode: {}!".format(getattr(self.args, "target_update_mode", "hard")))

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("critic_loss", loss.item(), t_env)
            self.logger.log_stat("critic_grad_norm", critic_grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", masked_td_error.abs().sum().item() / mask_elems, t_env)
            self.logger.log_stat("q_taken_mean", (q_taken * mask).sum().item() / mask_elems, t_env)
            self.logger.log_stat("pg_loss", pg_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", agent_grad_norm, t_env)
            self.logger.log_stat("distillation_loss", distillation_loss.item(), t_env)
            self.logger.log_stat("total_loss", total_loss.item(), t_env)
            self.log_stats_t = t_env

    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_mac.parameters(), self.mac.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        if self.args.verbose:
            self.logger.console_logger.info("Updated all target networks (soft update tau={})".format(tau))

    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        inputs = []

        # The centralized critic takes the state input, not observation
        inputs.append(batch["state"][:, t])

        if self.args.recurrent_critic:
            if self.args.obs_last_action:
                if t == 0:
                    inputs.append(th.zeros_like(batch["actions"][:, t]))
                else:
                    inputs.append(batch["actions"][:, t - 1])

        inputs = th.cat([x.reshape(bs, -1) for x in inputs], dim=1)
        return inputs

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.logger.console_logger.info("Updated all target networks")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.agent_optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        self.agent_optimiser.load_state_dict(
            th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
