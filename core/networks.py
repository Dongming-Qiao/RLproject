import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Union

class BaseNetwork(nn.Module):
    """基础网络类，提供通用功能"""

    def __init__(self):
        super().__init__()

    def init_weights(self, init_type='orthogonal', gain=1.0):
        """权重初始化"""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if init_type == 'orthogonal':
                    nn.init.orthogonal_(module.weight, gain=gain)
                elif init_type == 'xavier':
                    nn.init.xavier_uniform_(module.weight, gain=gain)
                elif init_type == 'normal':
                    nn.init.normal_(module.weight, std=gain)

                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

class MLP(BaseNetwork):
    """多层感知机"""

    def __init__(self, input_dim, hidden_dims, output_dim, activation='relu', output_activation=None, dropout=0.0):
        super().__init__()

        layers = []
        prev_dim = input_dim

        #隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU())

            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        #输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        if output_activation == 'softmax':
            layers.append(nn.Softmax(dim=-1))
        elif output_activation == 'tanh':
            layers.append(nn.Tanh())
        elif output_activation == 'sigmoid':
            layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)
        self.init_weights()

    def forward(self, x):
        return self.network(x)
    
class RNN(BaseNetwork):
    """RNN网络（支持LSTM/GRU）"""

    def __init__(self, input_dim, hidden_dim, output_dim, rnn_type='lstm', num_layers=1, bidirectional=False, dropout=0.0):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional

        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_fiest=True, bidirectional=bidirectional, dropout=dropout)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout)
        else:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}")
        
        self.init_weights()

    def forward(self, x, hidden_state=None):
        # x: [batch_size, seq_len, input_dim]
        output, hidden = self.rnn(x, hidden_state)
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        """初始化隐藏状态"""

        num_directions = 2 if self.bidirectional else 1
        if self.rnn_type == 'lstm':
            h_0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_dim).to(device)
            c_0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_dim).to(device)
            return (h_0, c_0)
        else: # GRU
            h_0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_dim).to(device)
            return h_0
        
class AgentNetwork(BaseNetwork):
    """QMIX智能体网络（DRQN）"""

    def __init__(self, obs_dim, action_dim, rnn_hidden_dim=64, fc_hidden_dim=64):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.rnn_hidden_dim = rnn_hidden_dim

        #特征提取层
        self.feature_net = nn.Sequential(
            nn.Linear(obs_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.Linear(fc_hidden_dim, rnn_hidden_dim),
            nn.ReLU()
        )

        #RNN层
        self.rnn = nn.GRU(rnn_hidden_dim, rnn_hidden_dim, batch_first=True)

        #输出层
        self.q_net = nn.Linear(rnn_hidden_dim, action_dim)

        self.init_weights()

    def forward(self, obs, hidden_state=None):
        # obs: [batch_size, seq_len, obs_dim] or [batch_size, obs_dim]

        #调试代码，打印obs的形状
        #print(obs.shape)

        if len(obs.shape) == 2:
            #单步输入，增加序列维度
            obs = obs.unsqueeze(1)
            single_step = True
        else:
            single_step = False

        batch_size, seq_len= obs.shape[0], obs.shape[1]

        #特征提取
        features = self.feature_net(obs.reshape(-1, self.obs_dim))
        features = features.reshape(batch_size, seq_len, -1)

        #RNN前向传播
        if hidden_state is None:
            hidden_state = self.init_hidden(batch_size, obs.device)

        rnn_out, hidden_state = self.rnn(features, hidden_state)

        #Q值计算
        q_values = self.q_net(rnn_out.reshape(-1, self.rnn_hidden_dim))
        q_values = q_values.reshape(batch_size, seq_len, self.action_dim)

        if single_step:
            q_values = q_values.squeeze(1)

        #q_values的维度： [batch_size, seq_len, action_dim] or [batch_size, action_dim]
        return q_values
    
    def init_hidden(self, batch_size, device):
        """初始化隐藏状态"""
        return torch.zeros(1, batch_size, self.rnn_hidden_dim).to(device)
    
class QMixer(BaseNetwork):
    """QMIX混合网络"""

    def __init__(self, n_agents, state_dim, embedding_dim=32, hypernet_hidden_dim=64):
        super().__init__()
        
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.embedding_dim = embedding_dim

        #超网络1：生成混合权重
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_hidden_dim),
            nn.ReLU(),
            nn.Linear(hypernet_hidden_dim, n_agents * embedding_dim)
        )

        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_hidden_dim),
            nn.ReLU(),
            nn.Linear(hypernet_hidden_dim, embedding_dim)
        )

        #超网络2：生成混合偏置
        self.hyper_b1 = nn.Linear(state_dim, embedding_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_hidden_dim),
            nn.ReLU(),
            nn.Linear(hypernet_hidden_dim, 1)
        )

        self.init_weights()

    def forward(self, agent_qs, states):
        # agent_qs: [batch_size, n_agents]
        # states: [batch_size, state_dim]

        batch_size = agent_qs.size(0)

        #第一层
        w1 = torch.abs(self.hyper_w1(states))  # [batch_size, n_agents * embedding_dim]
        w1 = w1.view(batch_size, self.n_agents, self.embedding_dim)

        b1 = self.hyper_b1(states)  # [batch_size, embedding_dim]
        b1 = b1.view(batch_size, 1, self.embedding_dim)

        #智能体Q值与第一层权重相乘
        hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)

        #第二层
        w2 = torch.abs(self.hyper_w2(states))  # [batch_size, embedding
        w2 = w2.view(batch_size, self.embedding_dim, 1)

        b2 = self.hyper_b2(states)  # [batch_size, 1]
        b2 = b2.view(batch_size, 1, 1)

        #输出总Q值
        q_total = torch.bmm(hidden, w2) + b2  # [batch_size, 1, 1]
        q_total = q_total.squeeze(-1) # [batch_size, 1]

        return q_total

#TODO:后续可以添加其他算法的网络，如MAPPO,基于Transformer的网络等