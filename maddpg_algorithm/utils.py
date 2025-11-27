import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免图形界面问题
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
from collections import deque


class DeviceManager:
    """设备管理器 - 自动检测并使用最佳设备"""
    
    def __init__(self):
        # 自动检测可用设备
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.device_name = torch.cuda.get_device_name(0)
            print(f"检测到GPU: {self.device_name}，使用GPU训练")
        else:
            self.device = torch.device('cpu')
            self.device_name = "CPU"
            print("未检测到GPU，使用CPU训练")
    
    def tensor(self, data, dtype=torch.float32):
        """将数据转换为指定设备的张量"""
        if isinstance(data, np.ndarray):
            return torch.tensor(data, dtype=dtype, device=self.device)
        elif isinstance(data, torch.Tensor):
            return data.to(self.device, dtype=dtype)
        else:
            return torch.tensor(data, dtype=dtype, device=self.device)
    
    def numpy(self, tensor):
        """将张量转换为numpy数组（自动移到CPU）"""
        if isinstance(tensor, torch.Tensor):
            return tensor.cpu().detach().numpy()
        return tensor
    
    def move_to_device(self, model):
        """将模型移动到指定设备"""
        return model.to(self.device)
    
    def get_device(self):
        """获取当前设备"""
        return self.device


class TrainLogger:
    """训练日志记录器 - 简化版本"""
    
    def __init__(self):
        # 创建日志目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        timestamp = datetime.now().strftime("%Y%m%d_%H")
        self.log_dir = os.path.join(current_dir, f"logs_{timestamp}")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 初始化数据存储
        self.episodes = []
        self.rewards = []
        self.actor_losses = []
        self.critic_losses = []
        self.steps = []
        
        self.best_reward = -float('inf')
        print(f"训练日志目录: {self.log_dir}")
    
    def log(self, episode, reward, actor_loss, critic_loss, steps):
        """记录训练数据"""
        self.episodes.append(episode)
        self.rewards.append(float(reward))
        self.actor_losses.append(float(actor_loss))
        self.critic_losses.append(float(critic_loss))
        self.steps.append(steps)
        
        # 更新最佳奖励
        if reward > self.best_reward:
            self.best_reward = reward
    
    def save_plot(self):
        """保存训练曲线图"""
        timestamp = datetime.now().strftime("%Y%m%d_%H")
        filepath = os.path.join(self.log_dir, f"training_plot_{timestamp}.png")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # 奖励曲线
        ax1.plot(self.episodes, self.rewards, 'b-', alpha=0.7, label='Reward')
        ax1.set_ylabel('Reward')
        ax1.set_title('Training Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss曲线
        ax2.plot(self.episodes, self.actor_losses, 'r-', label='Actor Loss')
        ax2.plot(self.episodes, self.critic_losses, 'g-', label='Critic Loss')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"训练图表已保存: {filepath}")
    
    def save_data(self):
        """保存训练数据为JSON文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H")
        filepath = os.path.join(self.log_dir, f"training_data_{timestamp}.json")
        
        data = {
            'episodes': self.episodes,
            'rewards': self.rewards,
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses,
            'steps': self.steps,
            'best_reward': self.best_reward,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"训练数据已保存: {filepath}")
    
    def get_summary(self):
        """获取训练摘要"""
        if not self.episodes:
            return "暂无训练数据"
        
        return {
            'total_episodes': len(self.episodes),
            'best_reward': self.best_reward,
            'current_reward': self.rewards[-1],
            'average_reward': np.mean(self.rewards),
            'average_steps': np.mean(self.steps)
        }


class ModelSaver:
    """模型保存器"""
    
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_dir = os.path.join(current_dir, "saved_models")
        os.makedirs(self.save_dir, exist_ok=True)
    
    def save(self, maddpg, episode, reward):
        """保存模型"""
        timestamp = datetime.now().strftime("%Y%m%d_%H")
        filename = f"maddpg_ep{episode}_agent_reward{reward:.1f}_{timestamp}.pth"
        filepath = os.path.join(self.save_dir, filename)
        
        model_state = {
            'episode': episode,
            'agent_reward': reward,
            'agents': []
        }
        
        for i, agent in enumerate(maddpg.agents):
            agent_state = {
                'actor_state_dict': agent.actor.state_dict(),
                'critic_state_dict': agent.critic.state_dict(),
                'target_actor_state_dict': agent.target_actor.state_dict(),
                'target_critic_state_dict': agent.target_critic.state_dict(),
            }
            model_state['agents'].append(agent_state)
        
        torch.save(model_state, filepath)
        print(f"模型已保存: {filepath}")
        return filepath


# 全局实例
device_manager = DeviceManager()