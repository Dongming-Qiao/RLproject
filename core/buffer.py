import numpy as np
import torch
from collections import deque, namedtuple
import random
from typing import Dict, List, Optional, Union

# 定义经验元组
Experience = namedtuple('Experience', 
                       ['obs', 'actions', 'rewards', 'next_obs', 'dones', 'state', 'next_state'])

# MAPPO专用的经验结构
MAPPOExperience = namedtuple('MAPPOExperience',
                           ['obs', 'actions', 'rewards', 'next_obs', 'dones', 'state', 
                            'next_state', 'log_probs', 'values', 'advantages', 'returns'])

class ReplayBuffer:
    """
    通用经验回放缓冲区
    支持离散和连续动作空间
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.position = 0

    def add(self, obs, actions, rewards, next_obs, dones, state, next_state):
        """
        添加单步经验到缓冲区
        
        Args:
            obs: 当前观测 [n_agents, obs_dim]
            actions: 执行的动作 [n_agents]
            rewards: 奖励值 [1] 或 标量
            next_obs: 下一时刻观测 [n_agents, obs_dim]
            dones: 是否结束episode [1] 或 布尔值
            state: 全局状态 [state_dim]
            next_state: 下一时刻全局状态 [state_dim]
        """

        experience = Experience(obs, actions, rewards, next_obs, dones, state, next_state)
        self.buffer.append(experience)

    def sample(self, batch_size: int):
        """
        随机采样一批经验
        
        Args:
            batch_size: 批次大小
            
        Returns:
            batch: 经验元组
        """
        if len(self.buffer) < batch_size:
            return None
        
        batch = random.sample(self.buffer, batch_size)
        return self._process_batch(batch)
    
    def _process_batch(self, batch):
        """
        处理批次数据，转换为numpy数组
        
        Args:
            batch: 经验列表
            
        Returns:
            processed_batch: 处理后的批次数据
        """

        obs_batch = np.array([exp.obs for exp in batch])
        actions_batch = np.array([exp.actions for exp in batch])
        rewards_batch = np.array([exp.rewards for exp in batch])
        next_obs_batch = np.array([exp.next_obs for exp in batch])
        dones_batch = np.array([exp.dones for exp in batch])
        state_batch = np.array([exp.state for exp in batch])
        next_state_batch = np.array([exp.next_state for exp in batch])

        return (obs_batch, actions_batch, rewards_batch, next_obs_batch, 
                dones_batch, state_batch, next_state_batch)
    
    def __len__(self):
        return len(self.buffer)
    
class EpisodeReplayBuffer:
    """
    Episode级别的回放缓冲区
    专门用于需要序列数据的算法（如QMIX的DRQN）
    """
    def __init__(self, capacity: int, seq_len: int):
        self.capacity = capacity
        self.seq_len = seq_len
        self.episodes = deque(maxlen=capacity)
        self.current_episode = {
            'obs': [],
            'actions': [],
            'rewards': [],
            'next_obs': [],
            'dones': [],
            'state': [],
            'next_state': []
        }

    def add_step(self, obs, actions, rewards, next_obs, dones, state, next_state):
        """
        添加单步经验到当前episode
        
        Args:
            obs: 当前观测
            actions: 执行的动作
            rewards: 奖励值
            next_obs: 下一时刻观测
            dones: 是否结束episode
            state: 全局状态
            next_state: 下一时刻全局状态
        """
        self.current_episode['obs'].append(obs)
        self.current_episode['actions'].append(actions)
        self.current_episode['rewards'].append(rewards)
        self.current_episode['next_obs'].append(next_obs)
        self.current_episode['dones'].append(dones)
        self.current_episode['state'].append(state)
        self.current_episode['next_state'].append(next_state)

    def store_episode(self):
        """
        存储完整的episode到缓冲区
        """
        if len(self.current_episode['obs']) > 0:
            # 转换为numpy数组
            episode = {}
            for key in self.current_episode:
                episode[key] = np.array(self.current_episode[key])
            
            self.episodes.append(episode)
            self.current_episode = {key: [] for key in self.current_episode}

    def sample_sequence(self, batch_size: int):
        """
        采样序列数据
        
        Args:
            batch_size: 批次大小
            
        Returns:
            sequences: 序列数据批次
        """
        if len(self.episodes) < batch_size:
            return None
        
        # 随机选择episode
        selected_episodes = random.sample(self.episodes, batch_size)

        sequences = {
            'obs': [],
            'actions': [],
            'rewards': [],
            'next_obs': [],
            'dones': [],
            'state': [],
            'next_state': []
        }

        for episode in selected_episodes:
            ep_len = len(episode['obs'])

            # 随机选择序列起始点
            if ep_len <= self.seq_len:
                start_idx = 0
                # 如果episode长度不足，重复最后一个状态
                for key in sequences:
                    seq = episode[key]
                    if len(seq) < self.seq_len:
                        # 填充到seq_len长度
                        padding = [seq[-1]] * (self.seq_len - len(seq))
                        seq = np.concatenate([seq, padding], axis=0)
                    sequences[key].append(seq[:self.seq_len])
            else:
                start_idx = random.randint(0, ep_len - self.seq_len)
                for key in sequences:
                    sequences[key].append(episode[key][start_idx:start_idx + self.seq_len])

        # 转换为numpy数组
        for key in sequences:
            sequences[key] = np.array(sequences[key])

        return sequences
    
    def __len__(self):
        return len(self.episodes)
    
class MAPPOBuffer:
    # TODO: 实现MAPPO专用的经验回放缓冲区
    pass

class PrioritizedReplayBuffer:
    # TODO: 实现优先经验回放缓冲区
    pass

# 便捷的缓冲区工厂函数
def create_buffer(buffer_type, **kwargs):
    """
    便捷的缓冲区创建函数
    
    Args:
        buffer_type: 缓冲区类型 ('replay', 'episode', 'mappo', 'prioritized')
        **kwargs: 缓冲区参数
        
    Returns:
        buffer: 缓冲区实例
    """
    buffer_classes = {
        'replay': ReplayBuffer,
        'episode': EpisodeReplayBuffer,
        'mappo': MAPPOBuffer,
        'prioritized': PrioritizedReplayBuffer
    }

    if buffer_type not in buffer_classes:
        raise ValueError(f"Unknown buffer type: {buffer_type}")
    
    return buffer_classes[buffer_type](**kwargs)