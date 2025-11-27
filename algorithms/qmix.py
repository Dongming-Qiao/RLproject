import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os

from core.networks import QMixer, AgentNetwork
from core.buffer import ReplayBuffer
from core.logger import Logger
from core.utils import to_tensor, to_numpy

#额外导入环境包装器，检查动作合法性
from envs.smac_wrapper import create_smac_env

class QMIX:
    def __init__(self, config):
        """
        初始化QMIX算法

        Args:
            config: 配置字典，包含算法所有参数
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        #调试代码
        # print(config)

        #环境参数
        self.n_agents = config['env']['n_agents']
        self.obs_dim = config['env']['obs_dim']
        self.action_dim = config['env']['action_dim']
        self.state_dim = config['env']['state_dim']

        #训练参数
        self.lr = config['algorithm']['lr']
        self.gamma = config['algorithm']['gamma']
        self.batch_size = config['algorithm']['batch_size']
        self.epsilon = config['algorithm']['epsilon_start']
        self.epsilon_min = config['algorithm']['epsilon_min']
        self.epsilon_decay = config['algorithm']['epsilon_decay']
        self.target_update_interval = config['algorithm']['target_update_interval']

        #网络初始化
        self.agent_networks = [AgentNetwork(self.obs_dim, self.action_dim).to(self.device) for _ in range(self.n_agents)]
        self.mixer_network = QMixer(self.n_agents, self.state_dim).to(self.device)
        self.target_agent_networks = [AgentNetwork(self.obs_dim, self.action_dim).to(self.device) for _ in range(self.n_agents)]
        self.target_mixer_network = QMixer(self.n_agents, self.state_dim).to(self.device)

        #同步目标网络
        self.update_target_networks()

        #优化器
        self.agent_optimizers = [optim.Adam(agent.parameters(), lr=self.lr) for agent in self.agent_networks]
        self.mixer_optimizer = optim.Adam(self.mixer_network.parameters(), lr=self.lr)

        #经验回放
        self.buffer = ReplayBuffer(config['algorithm']['buffer_size'])

        #日志记录
        self.logger = Logger(config)

        self.train_step = 0

        #导入环境包装器，仅用于检查动作合法性
        # self.smac_env = create_smac_env()

    def update_target_networks(self):
        """
        同步目标网络参数
        将当前策略网络参数复制到目标网络
        """
        for i in range(self.n_agents):
            self.target_agent_networks[i].load_state_dict(self.agent_networks[i].state_dict())

        self.target_mixer_network.load_state_dict(self.mixer_network.state_dict())

    def choose_actions(self, obs_list, avail_actions, evaluate=False):
        """
        为所有智能体选择动作

        Args:
            obs_list: 所有智能体的观测列表 [n_agents, obs_dim]
            avail_actions: 所有智能体的可用动作列表 [n_agents, action_dim]
            evaluate: 是否为评估模式（评估模式下不探索）

        Returns:
            actions: 动作列表 [n_agents]
        """
        actions = []

        avail_actions_indices = [np.nonzero(avail_actions[i])[0] for i in range(self.n_agents)]

        for i in range(self.n_agents):
            obs = to_tensor(obs_list[i], self.device).unsqueeze(0)

            if not evaluate and random.random() < self.epsilon:
                #探索：随机动作，但是仅从可用动作中选择
                action = np.random.choice(avail_actions_indices[i])
            else:
                #利用：选择Q值最大的动作,但是mask掉不可用动作
                with torch.no_grad():
                    q_values = self.agent_networks[i](obs)
                    #q_values: [1, action_dim]
                    #mask不可用动作
                    q_values[0][avail_actions[i] == 0] = -float('inf')
                    action = q_values.argmax().item()

            actions.append(action)
            
        #调试代码
        #print(f"Chosen actions: {actions}")

        return actions
    
    def update_epsilon(self):
        """
        衰减探索率
        按照指数衰减方式更新epsilon值
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, batch):
        """
        执行一步训练

        Args:
            batch: 批量经验数据

        Returns:
            metrics: 训练指标字典，包含loss、epsilon等
        """
        if len(self.buffer) < self.batch_size:
            return {}
        
        #从回访缓冲区采样
        obs_batch, actions_batch, rewards_batch, next_obs_batch, done_batch, state_batch, next_state_batch = \
            self.buffer.sample(self.batch_size)
        
        #转换为张量
        obs_batch = to_tensor(obs_batch, self.device, dtype=torch.float32)  # [batch_size, n_agents, obs_dim]
        actions_batch = to_tensor(actions_batch, self.device, dtype=torch.long) # [batch_size, n_agents]
        rewards_batch = to_tensor(rewards_batch, self.device, dtype=torch.float32)   # [batch_size, 1]
        next_obs_batch = to_tensor(next_obs_batch, self.device, dtype=torch.float32) # [batch_size, n_agents, obs_dim]
        done_batch = to_tensor(done_batch, self.device, dtype=torch.float32)      # [batch_size, 1] 
        state_batch = to_tensor(state_batch, self.device, dtype=torch.float32)   # [batch_size, state_dim]
        next_state_batch = to_tensor(next_state_batch, self.device, dtype=torch.float32) # [batch_size, state_dim]

        #计算当前Q值
        current_q_values = []
        for i in range(self.n_agents):
            agent_obs = obs_batch[:, i, :]  # [batch_size, obs_dim]
            agent_qs = self.agent_networks[i](agent_obs)    # [batch_size, action_dim]
            agent_actions = actions_batch[:, i].unsqueeze(1)    # [batch_size, 1]
            agent_q = agent_qs.gather(1, agent_actions)   # [batch_size, 1]
            current_q_values.append(agent_q)

        current_q_values = torch.stack(current_q_values, dim=1) # [batch_size, n_agents, 1]
        current_q_values = current_q_values.squeeze(-1) # [batch_size, n_agents]

        #注意当前混合Q值需要梯度，但目标Q值不需要梯度，确保计算图正确
        current_total_q = self.mixer_network(current_q_values, state_batch) # [batch_size, 1]

        #计算目标Q值
        with torch.no_grad():
            next_q_values = []
            for i in range(self.n_agents):
                agent_next_obs = next_obs_batch[:, i, :]  # [batch_size, obs_dim]
                agent_next_qs = self.target_agent_networks[i](agent_next_obs)    # [batch_size, action_dim]
                next_q_values.append(agent_next_qs.max(1)[0])   # [batch_size]

            next_q_values = torch.stack(next_q_values, dim=1)   # [batch_size, n_agents]

            #混合网络计算Q值
            next_total_q = self.target_mixer_network(next_q_values, next_state_batch) # [batch_size, 1]

            target_q = rewards_batch + self.gamma * (1 - done_batch) * next_total_q  # [batch_size, 1], element-wise multiple

        #计算损失
        loss = nn.MSELoss()(current_total_q, target_q)

        #反向传播
        #清空梯度
        for optimizer in self.agent_optimizers:
            optimizer.zero_grad()
        self.mixer_optimizer.zero_grad()

        #计算梯度
        loss.backward()

        #梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.mixer_network.parameters(), self.config.get('grad_norm_clip', 10))
        for i in range(self.n_agents):
            torch.nn.utils.clip_grad_norm_(self.agent_networks[i].parameters(), self.config.get('grad_norm_clip', 10))

        #更新参数
        for optimizer in self.agent_optimizers:
            optimizer.step()
        self.mixer_optimizer.step()

        #更新目标网络
        if self.train_step % self.target_update_interval == 0:
            self.update_target_networks()

        #更新探索率
        self.update_epsilon()

        self.train_step += 1

        return {
            'loss': loss.item(),
            'epsilon': self.epsilon,
            'avg_q': current_total_q.mean().item()
        }
    
    def store_experience(self, obs, actions, rewards, next_obs, done, state, next_state):
        """
        存储经验到回放缓冲区

        Args:
            obs: 当前观测 [n_agents, obs_dim]
            actions: 执行的动作 [n_agents]
            rewards: 奖励值 [1]
            next_obs: 下一时刻观测 [n_agents, obs_dim]
            done: 是否结束episode [1]
            state: 全局状态 [state_dim]
            next_state: 下一时刻全局状态 [state_dim]
        """
        self.buffer.add(obs, actions, rewards, next_obs, done, state, next_state)

    def save_models(self, save_dir, episode):
        """
        保存模型参数

        Args:
            save_dir: 保存目录路径
            episode: 当前训练轮次，用于文件名
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for i in range(self.n_agents):
            torch.save(self.agent_networks[i].state_dict(), os.path.join(save_dir, f'agent_{i}_episode_{episode}.pth'))

        torch.save(self.mixer_network.state_dict(), os.path.join(save_dir, f'mixer_episode_{episode}.pth'))

    def load_models(self, load_dir, episode):
        """
        加载模型参数

        Args:
            load_dir: 加载目录路径
            episode: 要加载的模型轮次
        """
        for i in range(self.n_agents):
            self.agent_networks[i].load_state_dict(torch.load(os.path.join(load_dir, f'agent_{i}_episode_{episode}.pth'), map_location=self.device))
        
        self.mixer_network.load_state_dict(torch.load(os.path.join(load_dir, f'mixer_episode_{episode}.pth'), map_location=self.device))

        #同步更新目标网络
        self.update_target_networks()