import os
import yaml
import torch
import numpy as np
import random
from typing import Dict, List, Any, Union, Tuple

def set_seed(seed: int):
    """
    设置随机种子以保证可重复性
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def to_tensor(x: Union[np.ndarray, List], device: torch.device = None, dtype: torch.dtype = None) -> torch.Tensor:
    """
    将numpy数组或列表转换为PyTorch张量
    
    Args:
        x: 输入数据
        device: 目标设备
        dtype: 目标数据类型
    
    Returns:
        tensor: PyTorch张量
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if isinstance(x, list):
        x = np.array(x)

    # 直接使用torch.tensor，它会自动处理数据类型转换
    tensor = torch.tensor(x, device=device, dtype=dtype)

    return tensor

def to_numpy(x: Union[torch.Tensor, List, float]) -> np.ndarray:
    """
    将PyTorch张量或其他类型转换为numpy数组
    
    Args:
        x: 输入数据
    
    Returns:
        array: numpy数组
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, list):
        return np.array(x)
    elif isinstance(x, (int, float)):
        return np.array([x])
    else:
        return x

def create_dirs(dirs: List[str]):
    """
    创建目录列表
    
    Args:
        dirs: 目录路径列表
    """
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

def save_config(config: Dict[str, Any], save_path: str):
    """
    保存配置到YAML文件
    
    Args:
        config: 配置字典
        save_path: 保存路径
    """
    with open(os.path.join(save_path, "config.yaml"), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    从YAML文件加载配置
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        config: 配置字典
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def update_config(base_config: Dict[str, Any], new_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    递归更新配置字典
    
    Args:
        base_config: 基础配置
        new_config: 新配置
    
    Returns:
        updated_config: 更新后的配置
    """
    for key, value in new_config.items():
        if isinstance(value, dict) and key in base_config and isinstance(base_config[key], dict):
            base_config[key] = update_config(base_config[key], value)
        else:
            base_config[key] = value
    return base_config

def compute_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, 
                gamma: float = 0.99, gae_lambda: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算广义优势估计(GAE)
    
    Args:
        rewards: 奖励序列 [T, n_agents]
        values: 值函数序列 [T, n_agents]
        dones: 终止标志序列 [T, 1]
        gamma: 折扣因子
        gae_lambda: GAE参数
    
    Returns:
        advantages: 优势函数 [T, n_agents]
        returns: 回报 [T, n_agents]
    """
    T = len(rewards)
    advantages = np.zeros_like(rewards)
    last_advantage = 0
    
    for t in reversed(range(T)):
        if t == T - 1:
            next_non_terminal = 1.0 - dones[t]
            next_value = 0.0  # 假设最后状态价值为0
        else:
            next_non_terminal = 1.0 - dones[t]
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        advantages[t] = last_advantage = delta + gamma * gae_lambda * next_non_terminal * last_advantage
    
    returns = advantages + values
    return advantages, returns

def normalize(x: np.ndarray) -> np.ndarray:
    """
    标准化数据（零均值，单位方差）
    
    Args:
        x: 输入数据
    
    Returns:
        normalized_x: 标准化后的数据
    """
    return (x - np.mean(x)) / (np.std(x) + 1e-8)


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float):
    """
    软更新目标网络参数
    
    Args:
        target: 目标网络
        source: 源网络
        tau: 更新系数
    """
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

def hard_update(target: torch.nn.Module, source: torch.nn.Module):
    """
    硬更新目标网络参数（完全复制）
    
    Args:
        target: 目标网络
        source: 源网络
    """
    target.load_state_dict(source.state_dict())

def get_optimizer(model: torch.nn.Module, optimizer_type: str = "adam", lr: float = 1e-3, **kwargs):
    """
    获取优化器
    
    Args:
        model: 模型
        optimizer_type: 优化器类型
        lr: 学习率
        **kwargs: 其他优化器参数
    
    Returns:
        optimizer: 优化器实例
    """
    if optimizer_type.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, **kwargs)
    elif optimizer_type.lower() == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr, **kwargs)
    elif optimizer_type.lower() == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, **kwargs)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
def get_activation_fn(activation: str):
    """
    获取激活函数
    
    Args:
        activation: 激活函数名称
    
    Returns:
        activation_fn: 激活函数
    """
    if activation == "relu":
        return torch.nn.ReLU()
    elif activation == "tanh":
        return torch.nn.Tanh()
    elif activation == "sigmoid":
        return torch.nn.Sigmoid()
    elif activation == "leaky_relu":
        return torch.nn.LeakyReLU()
    elif activation == "elu":
        return torch.nn.ELU()
    else:
        raise ValueError(f"Unsupported activation function: {activation}")
    
def calculate_grad_norm(model: torch.nn.Module) -> float:
    """
    计算模型梯度范数
    
    Args:
        model: 模型
    
    Returns:
        grad_norm: 梯度范数
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def one_hot_encode(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    一热编码
    
    Args:
        labels: 标签 [batch_size] 或 [batch_size, seq_len]
        num_classes: 类别数量
    
    Returns:
        one_hot: 一热编码 [batch_size, num_classes] 或 [batch_size, seq_len, num_classes]
    """
    if len(labels.shape) == 1:
        return np.eye(num_classes)[labels]
    else:
        batch_size, seq_len = labels.shape
        one_hot = np.zeros((batch_size, seq_len, num_classes))
        for i in range(batch_size):
            one_hot[i] = np.eye(num_classes)[labels[i]]
        return one_hot
    
def flatten_dict(nested_dict: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    展平嵌套字典
    
    Args:
        nested_dict: 嵌套字典
        parent_key: 父键
        sep: 分隔符
    
    Returns:
        flattened_dict: 展平后的字典
    """
    items = []
    for k, v in nested_dict.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def print_model_info(model: torch.nn.Module, name: str = "Model"):
    """
    打印模型信息
    
    Args:
        model: 模型
        name: 模型名称
    """
    print(f"=== {name} ===")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print("=" * 50)

def get_device() -> torch.device:
    """
    获取可用设备
    
    Returns:
        device: PyTorch设备
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def batch_observations(observations: List[np.ndarray]) -> np.ndarray:
    """
    批量处理观测数据
    
    Args:
        observations: 观测数据列表
    
    Returns:
        batched_obs: 批量观测数据
    """
    return np.stack(observations)

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                   episode: int, filepath: str):
    """
    保存训练检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        episode: 当前训练轮次
        filepath: 保存路径
    """
    torch.save({
        'episode': episode,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filepath)

def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                   filepath: str) -> int:
    """
    加载训练检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        filepath: 检查点路径
    
    Returns:
        episode: 训练轮次
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['episode']