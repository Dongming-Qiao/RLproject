import numpy as np
import matplotlib.pyplot as plt
import wandb
import os
from typing import Dict, List, Any, Optional

class Logger:
    """
    轻量级日志记录器
    支持控制台输出、Matplotlib绘图和Weights & Biases记录
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.log_config = config.get('logging', {})

        # 初始化WandB
        if self.log_config.get('use_wandb', False):
            wandb.init(
                project=self.log_config.get('project_name', 'marl-project'),
                name=self.log_config.get('run_name', 'experiment'),
                config=config
            )

        # 存储历史数据
        self.history = {
            'episode_rewards': [],
            'eval_rewards': [],
            'losses': [],
            'steps': [],
            'epsilon': []
        }

        # 创建结果目录
        self.result_dir = config['paths']['result_dir']
        os.makedirs(self.result_dir, exist_ok=True)

        print(f"Logger initialized. WandB: {self.log_config.get('use_wandb', False)}")

    def log(self, metrics: Dict[str, Any], step: int):
        """
        记录训练指标
        
        Args:
            metrics: 指标字典
            step: 当前训练步数/回合数
        """
        # 更新历史记录
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)

        # 控制台输出
        if step % self.log_config.get('log_interval', 100) == 0:
            self._console_log(metrics, step)

        # WandB记录
        if self.log_config.get('use_wandb', False):
            wandb.log(metrics, step=step)

        # 定期绘图
        if step % self.log_config.get('plot_interval', 500) == 0:
            self._plot_progress(step)

    def _console_log(self, metrics: Dict[str, Any], step: int):
        """控制台日志输出"""
        log_str = f"Step {step}: "

        for key, value in metrics.items():
            if isinstance(value, float):
                log_str += f"{key}: {value:.4f} | "
            else:
                log_str += f"{key}: {value} | "

        print(log_str[:-3])  # 移除最后的" | "

    def _plot_progress(self, step: int):
        """绘制训练进度图"""
        if len(self.history['episode_rewards']) == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'Training Progress - Step {step}')
        
        # 奖励曲线
        if self.history['episode_rewards']:
            axes[0, 0].plot(self.history['episode_rewards'])
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].grid(True)
        
        # 评估奖励曲线
        if self.history['eval_rewards']:
            axes[0, 1].plot(self.history['eval_rewards'])
            axes[0, 1].set_title('Evaluation Rewards')
            axes[0, 1].set_xlabel('Evaluation')
            axes[0, 1].set_ylabel('Avg Reward')
            axes[0, 1].grid(True)
        
        # 损失曲线
        if self.history['losses']:
            axes[1, 0].plot(self.history['losses'])
            axes[1, 0].set_title('Training Loss')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True)
        
        # 探索率曲线
        if self.history['epsilon']:
            axes[1, 1].plot(self.history['epsilon'])
            axes[1, 1].set_title('Exploration Rate')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Epsilon')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # 保存图片
        plot_path = os.path.join(self.result_dir, f'training_progress_step_{step}.png')
        plt.savefig(plot_path)
        plt.close()
        
        # 上传到WandB
        if self.log_config.get('use_wandb', False):
            wandb.log({"training_progress": wandb.Image(plot_path)}, step=step)

    def log_episode(self, episode: int, reward: float, steps: int, epsilon: float = None):
        """记录episode结果"""
        metrics = {
            'episode_reward': reward,
            'episode_steps': steps,
            'episode': episode
        }
        
        if epsilon is not None:
            metrics['epsilon'] = epsilon
        
        self.log(metrics, episode)
    
    def log_evaluation(self, eval_reward: float, step: int):
        """记录评估结果"""
        metrics = {
            'eval_reward': eval_reward
        }
        self.log(metrics, step)
    
    def log_loss(self, loss: float, step: int):
        """记录损失值"""
        metrics = {
            'loss': loss
        }
        self.log(metrics, step)
    
    def save_results(self):
        """保存结果到文件"""
        # 保存历史数据
        history_path = os.path.join(self.result_dir, 'training_history.npz')
        np.savez(history_path, **self.history)
        
        # 保存配置
        config_path = os.path.join(self.result_dir, 'config.yaml')
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f)
        
        print(f"Results saved to {self.result_dir}")
    
    def finish(self):
        """完成记录，清理资源"""
        self.save_results()
        
        if self.log_config.get('use_wandb', False):
            wandb.finish()
        
        print("Logger finished.")


class SimpleLogger:
    """
    极简日志记录器
    仅包含最基本的控制台输出功能
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.log_interval = config.get('logging', {}).get('log_interval', 100)
        self.episode_rewards = []
    
    def log(self, metrics: Dict[str, Any], step: int):
        """记录指标"""
        if step % self.log_interval == 0:
            log_str = f"Step {step}: "
            for key, value in metrics.items():
                if isinstance(value, float):
                    log_str += f"{key}: {value:.4f} | "
                else:
                    log_str += f"{key}: {value} | "
            print(log_str[:-3])
            
            # 记录奖励用于简单统计
            if 'episode_reward' in metrics:
                self.episode_rewards.append(metrics['episode_reward'])
    
    def log_episode(self, episode: int, reward: float, steps: int, epsilon: float = None):
        """记录episode结果"""
        metrics = {
            'episode_reward': reward,
            'episode_steps': steps
        }
        if epsilon is not None:
            metrics['epsilon'] = epsilon
        
        self.log(metrics, episode)
    
    def finish(self):
        """完成记录"""
        if self.episode_rewards:
            avg_reward = np.mean(self.episode_rewards)
            max_reward = np.max(self.episode_rewards)
            print(f"Training completed. Avg reward: {avg_reward:.2f}, Max reward: {max_reward:.2f}")


# 便捷的日志记录器创建函数
def create_logger(config: Dict[str, Any], logger_type: str = 'full'):
    """
    创建日志记录器
    
    Args:
        config: 配置字典
        logger_type: 记录器类型 ('full', 'simple')
    
    Returns:
        logger: 日志记录器实例
    """
    if logger_type == 'simple':
        return SimpleLogger(config)
    else:
        return Logger(config)