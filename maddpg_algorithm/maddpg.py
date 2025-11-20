import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from utils import device_manager

class ReplayBuffer:
    """经验回放缓冲区 - 存储所有智能体的经验"""
    def __init__(self, capacity, n_agents, obs_dim, action_dim):
        self.buffer = deque(maxlen=capacity)
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
    
    def push(self, states, actions, rewards, next_states, dones):
        """
        存储经验
        states: list of arrays, 每个形状为 (obs_dim,)
        actions: list of arrays, 每个形状为 (action_dim,)
        rewards: list of floats
        next_states: list of arrays, 每个形状为 (obs_dim,)
        dones: list of bools
        """
        self.buffer.append((states, actions, rewards, next_states, dones))
    
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        
        # 随机采样
        samples = random.sample(self.buffer, batch_size)
        
        # 转换为批处理张量 - 直接在目标设备上创建
        states_batch = torch.zeros(batch_size, self.n_agents, self.obs_dim, 
                                 device=device_manager.device)
        actions_batch = torch.zeros(batch_size, self.n_agents, self.action_dim, 
                                  device=device_manager.device)
        rewards_batch = torch.zeros(batch_size, self.n_agents, 
                                  device=device_manager.device)
        next_states_batch = torch.zeros(batch_size, self.n_agents, self.obs_dim, 
                                      device=device_manager.device)
        dones_batch = torch.zeros(batch_size, self.n_agents, 
                                device=device_manager.device)
        
        for i, (states, actions, rewards, next_states, dones) in enumerate(samples):
            for j in range(self.n_agents):
                # 使用设备管理器创建张量
                states_batch[i, j] = device_manager.tensor(states[j])
                actions_batch[i, j] = device_manager.tensor(actions[j])
                rewards_batch[i, j] = device_manager.tensor(rewards[j])
                next_states_batch[i, j] = device_manager.tensor(next_states[j])
                dones_batch[i, j] = device_manager.tensor(dones[j])
        
        return states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch
    
    def __len__(self):
        return len(self.buffer)

class Actor(nn.Module):
    """Actor网络 - 输入观测，输出动作"""
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # 输出在[-1, 1]范围内
        )
    
    def forward(self, obs):
        # obs形状: (batch_size, obs_dim) 或 (obs_dim,)
        return self.net(obs)

class Critic(nn.Module):
    """Critic网络 - 输入所有智能体的观测和动作，输出Q值"""
    def __init__(self, total_obs_dim, total_action_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(total_obs_dim + total_action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, obs, actions):
        """
        obs形状: (batch_size, total_obs_dim) - 所有智能体观测的拼接
        actions形状: (batch_size, total_action_dim) - 所有智能体动作的拼接
        输出形状: (batch_size, 1)
        """
        x = torch.cat([obs, actions], dim=-1)
        return self.net(x)

class MADDPGAgent:
    """MADDPG智能体"""
    def __init__(self, obs_dim, action_dim, n_agents, agent_id, args):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.agent_id = agent_id
        self.args = args
        
        # Actor网络
        self.actor = Actor(obs_dim, action_dim, args.hidden_dim)
        self.target_actor = Actor(obs_dim, action_dim, args.hidden_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        
        # Critic网络 - 输入是所有智能体的观测和动作
        total_obs_dim = obs_dim * n_agents
        total_action_dim = action_dim * n_agents
        self.critic = Critic(total_obs_dim, total_action_dim, args.hidden_dim)
        self.target_critic = Critic(total_obs_dim, total_action_dim, args.hidden_dim)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.critic_lr)
        
        # 自动将模型移动到最佳设备
        self.actor = device_manager.move_to_device(self.actor)
        self.target_actor = device_manager.move_to_device(self.target_actor)
        self.critic = device_manager.move_to_device(self.critic)
        self.target_critic = device_manager.move_to_device(self.target_critic)
        
        # 初始化目标网络
        self.hard_update(self.target_actor, self.actor)
        self.hard_update(self.target_critic, self.critic)
        
        # 经验回放缓冲区（所有智能体共享）
        self.replay_buffer = ReplayBuffer(args.buffer_size, n_agents, obs_dim, action_dim)
    
    def hard_update(self, target, source):
        """硬更新目标网络"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    def soft_update(self, target, source, tau):
        """软更新目标网络"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
    
    def act(self, obs, noise=True):
        """
        根据观测选择动作 - 在CPU上执行（与环境交互）
        obs形状: (obs_dim,)
        返回形状: (action_dim,)
        """
        # 将观测转换为张量并移动到设备
        obs_tensor = device_manager.tensor(obs).unsqueeze(0)  # (1, obs_dim)
        
        # 使用actor网络计算动作
        with torch.no_grad():  # 不需要梯度
            action_tensor = self.actor(obs_tensor).squeeze(0)  # (action_dim,)
        
        # 将动作移回CPU并转换为numpy
        action = device_manager.numpy(action_tensor)
        
        # 添加探索噪声
        if noise:
            noise_val = np.random.normal(0, 0.1, self.action_dim)
            action = np.clip(action + noise_val, -1.0, 1.0)
        
        return action.astype(np.float32)
    
    def update(self, agents):
        """更新网络 - 在GPU上执行"""
        if len(self.replay_buffer) < self.args.batch_size:
            return 0, 0
        
        # 采样经验（已经在设备上）
        batch = self.replay_buffer.sample(self.args.batch_size)
        if batch is None:
            return 0, 0
            
        states, actions, rewards, next_states, dones = batch
        # states形状: (batch_size, n_agents, obs_dim)
        # actions形状: (batch_size, n_agents, action_dim)
        # rewards形状: (batch_size, n_agents)
        # next_states形状: (batch_size, n_agents, obs_dim)
        # dones形状: (batch_size, n_agents)
        
        batch_size = states.shape[0]
        
        # 准备Critic输入数据
        # 将所有智能体的观测和动作展平
        all_states = states.view(batch_size, -1)  # (batch_size, n_agents * obs_dim)
        all_actions = actions.view(batch_size, -1)  # (batch_size, n_agents * action_dim)
        all_next_states = next_states.view(batch_size, -1)  # (batch_size, n_agents * obs_dim)
        
        # 更新Critic
        with torch.no_grad():
            # 计算目标动作
            target_actions = []
            for i in range(self.n_agents):
                # 每个智能体的目标动作: (batch_size, action_dim)
                target_act = agents[i].target_actor(next_states[:, i, :])
                target_actions.append(target_act)
            
            # 拼接所有目标动作: (batch_size, n_agents * action_dim)
            target_actions = torch.cat(target_actions, dim=1)
            
            # 计算目标Q值
            target_q = self.target_critic(all_next_states, target_actions).squeeze()  # (batch_size,)
            target_q = rewards[:, self.agent_id] + self.args.gamma * (1 - dones[:, self.agent_id]) * target_q
        
        # 计算当前Q值
        current_q = self.critic(all_states, all_actions).squeeze()  # (batch_size,)
        critic_loss = F.mse_loss(current_q, target_q)
        
        # 更新Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # 更新Actor
        # 重新计算当前智能体的动作
        new_actions = []
        for i in range(self.n_agents):
            if i == self.agent_id:
                # 当前智能体使用actor网络
                new_act = self.actor(states[:, i, :])  # (batch_size, action_dim)
            else:
                # 其他智能体保持原动作
                new_act = actions[:, i, :].detach()  # (batch_size, action_dim)
            new_actions.append(new_act)
        
        # 拼接新动作
        new_all_actions = torch.cat(new_actions, dim=1)  # (batch_size, n_agents * action_dim)
        
        # 计算Actor损失
        actor_loss = -self.critic(all_states, new_all_actions).mean()
        
        # 更新Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # 软更新目标网络
        self.soft_update(self.target_actor, self.actor, self.args.tau)
        self.soft_update(self.target_critic, self.critic, self.args.tau)
        
        # 将损失移回CPU用于打印
        return actor_loss.item(), critic_loss.item()

class MADDPG:
    """MADDPG多智能体系统"""
    def __init__(self, env, args):
        self.env = env
        self.args = args
        
        # 获取环境信息
        env.reset()
        self.n_agents = len(env.agents)
        
        # 获取观测和动作维度（假设所有智能体相同）
        sample_agent = env.agents[0]
        self.obs_dim = env.observation_space(sample_agent).shape[0]
        self.action_dim = env.action_space(sample_agent).shape[0]
        
        print(f"智能体数量: {self.n_agents}")
        print(f"每个智能体 - 观测维度: {self.obs_dim}, 动作维度: {self.action_dim}")
        
        # 创建所有智能体
        self.agents = []
        for i in range(self.n_agents):
            agent = MADDPGAgent(self.obs_dim, self.action_dim, self.n_agents, i, args)
            self.agents.append(agent)
    
    def get_actions(self, observations, noise=True):
        """
        获取所有智能体的动作 - 在CPU上执行（与环境交互）
        observations: dict, key为智能体名，value为观测数组 (obs_dim,)
        返回: dict, key为智能体名，value为动作数组 (action_dim,)
        """
        actions = {}
        for i, agent_name in enumerate(self.env.agents):
            obs = observations[agent_name]
            action = self.agents[i].act(obs, noise)
            actions[agent_name] = action
        return actions
    
    def store_experiences(self, observations, actions, rewards, next_observations, dones):
        """
        存储经验到回放缓冲区
        所有智能体共享同一个缓冲区
        """
        # 转换为列表格式
        states_list = [observations[agent] for agent in self.env.agents]
        actions_list = [actions[agent] for agent in self.env.agents]
        rewards_list = [rewards[agent] for agent in self.env.agents]
        next_states_list = [next_observations[agent] for agent in self.env.agents]
        dones_list = [dones[agent] for agent in self.env.agents]
        
        # 存储到第一个智能体的缓冲区（所有智能体共享）
        self.agents[0].replay_buffer.push(states_list, actions_list, rewards_list, next_states_list, dones_list)
    
    def update_all_agents(self):
        """更新所有智能体 - 在GPU上执行"""
        actor_losses, critic_losses = [], []
        for agent in self.agents:
            actor_loss, critic_loss = agent.update(self.agents)
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
        
        return np.mean(actor_losses), np.mean(critic_losses)