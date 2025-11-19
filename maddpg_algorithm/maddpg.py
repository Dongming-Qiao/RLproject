import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque, namedtuple
import random

# 经验回放缓冲区
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        
        experiences = random.sample(self.buffer, batch_size)
        
        states = [torch.FloatTensor(e.state) for e in experiences]
        actions = [torch.FloatTensor(e.action) for e in experiences]
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = [torch.FloatTensor(e.next_state) for e in experiences]
        dones = torch.FloatTensor([e.done for e in experiences])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

# Actor网络
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
    
    def forward(self, obs):
        return self.net(obs)

# Critic网络
class Critic(nn.Module):
    def __init__(self, total_obs_dim, total_action_dim, hidden_dim=64):
        super(Critic, self).__init__()
        input_dim = total_obs_dim + total_action_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, obs, actions):
        x = torch.cat([obs] + actions, dim=-1)
        return self.net(x)

# Ornstein-Uhlenbeck噪声
class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()
    
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
    
    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

# MADDPG智能体
class MADDPGAgent:
    def __init__(self, obs_dim, action_dim, n_agents, agent_id, args):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.agent_id = agent_id
        self.args = args
        
        # 网络
        self.actor = Actor(obs_dim, action_dim, args.hidden_dim)
        self.target_actor = Actor(obs_dim, action_dim, args.hidden_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        
        total_obs_dim = obs_dim * n_agents
        total_action_dim = action_dim * n_agents
        self.critic = Critic(total_obs_dim, total_action_dim, args.hidden_dim)
        self.target_critic = Critic(total_obs_dim, total_action_dim, args.hidden_dim)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.critic_lr)
        
        # 初始化目标网络
        self.hard_update(self.target_actor, self.actor)
        self.hard_update(self.target_critic, self.critic)
        
        # 经验回放和噪声
        self.replay_buffer = ReplayBuffer(args.buffer_size)
        self.noise = OUNoise(action_dim)
    
    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
    
    def act(self, obs, noise=True):
        """根据观测选择动作 - 这是缺失的关键方法！"""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        action = self.actor(obs_tensor).squeeze(0).detach().numpy()
        
        if noise:
            action += self.noise.sample()
        
        # 确保动作在有效范围内并转换为numpy数组
        action = np.clip(action, -1.0, 1.0)
        return action.astype(np.float32)
    
    def target_act(self, obs):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        action = self.target_actor(obs_tensor).squeeze(0).detach().numpy()
        return np.clip(action, -1.0, 1.0).astype(np.float32)
    
    def store_experience(self, state, action, reward, next_state, done):
        """接口：存储经验"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self, agents):
        """接口：更新网络"""
        if len(self.replay_buffer) < self.args.batch_size:
            return None, None
        
        # 采样经验
        batch = self.replay_buffer.sample(self.args.batch_size)
        if batch is None:
            return None, None
            
        states, actions, rewards, next_states, dones = batch
        
        # 更新Critic
        with torch.no_grad():
            target_actions = []
            for i, agent in enumerate(agents):
                target_actions.append(torch.FloatTensor(agent.target_act(next_states[i])))
            
            target_q = rewards + self.args.gamma * (1 - dones) * self.target_critic(
                torch.cat(next_states, dim=-1), target_actions
            ).squeeze()
        
        current_q = self.critic(torch.cat(states, dim=-1), actions).squeeze()
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # 更新Actor
        new_actions = []
        for i, agent in enumerate(agents):
            if i == self.agent_id:
                new_actions.append(self.actor(states[i]))
            else:
                new_actions.append(actions[i])
        
        actor_loss = -self.critic(torch.cat(states, dim=-1), new_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # 软更新目标网络
        self.soft_update(self.target_actor, self.actor, self.args.tau)
        self.soft_update(self.target_critic, self.critic, self.args.tau)
        
        return actor_loss.item(), critic_loss.item()

# MADDPG多智能体系统
class MADDPG:
    def __init__(self, env, args):
        self.env = env
        self.args = args
        
        # 先重置环境以获取agents信息
        env.reset()
        self.n_agents = len(env.agents)
        
        print(f"智能体数量: {self.n_agents}")
        print(f"观测空间: {env.observation_space(env.agents[0]).shape}")
        print(f"动作空间: {env.action_space(env.agents[0]).shape}")
        
        # 创建所有智能体
        self.agents = []
        for i, agent_name in enumerate(env.agents):
            obs_dim = env.observation_space(agent_name).shape[0]
            action_dim = env.action_space(agent_name).shape[0]
            print(f"智能体 {agent_name}: 观测维度={obs_dim}, 动作维度={action_dim}")
            
            agent = MADDPGAgent(obs_dim, action_dim, self.n_agents, i, args)
            self.agents.append(agent)
    
    def get_actions(self, observations, noise=True):
        """接口：获取所有智能体的动作"""
        if observations is None:
            raise ValueError("observations cannot be None")

        actions = {}
        for i, agent_name in enumerate(self.env.agents):
            obs = observations[agent_name]
            if obs is None:
                raise ValueError(f"Observation for {agent_name} is None")

            action = self.agents[i].act(obs, noise)
            # 确保动作是numpy数组格式
            actions[agent_name] = np.array(action, dtype=np.float32)

        return actions
    
    def store_experiences(self, observations, actions, rewards, next_observations, dones):
        """接口：存储所有智能体的经验"""
        for i, agent_name in enumerate(self.env.agents):
            self.agents[i].store_experience(
                observations[agent_name],
                actions[agent_name],
                rewards[agent_name],
                next_observations[agent_name],
                dones[agent_name]
            )
    
    def update_all_agents(self):
        """接口：更新所有智能体"""
        actor_losses, critic_losses = [], []
        for agent in self.agents:
            losses = agent.update(self.agents)
            if losses[0] is not None:
                actor_losses.append(losses[0])
                critic_losses.append(losses[1])
        
        avg_actor_loss = np.mean(actor_losses) if actor_losses else 0
        avg_critic_loss = np.mean(critic_losses) if critic_losses else 0
        return avg_actor_loss, avg_critic_loss
    
    def save_models(self, path):
        """保存模型"""
        for i, agent in enumerate(self.agents):
            torch.save(agent.actor.state_dict(), f"{path}/actor_{i}.pth")
            torch.save(agent.critic.state_dict(), f"{path}/critic_{i}.pth")
    
    def load_models(self, path):
        """加载模型"""
        for i, agent in enumerate(self.agents):
            agent.actor.load_state_dict(torch.load(f"{path}/actor_{i}.pth"))
            agent.critic.load_state_dict(torch.load(f"{path}/critic_{i}.pth"))