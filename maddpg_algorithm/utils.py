import torch
import numpy as np

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

# 全局设备管理器实例 - 自动初始化
device_manager = DeviceManager()