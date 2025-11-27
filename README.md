## RLproject

强化学习课程大作业。大作业主要研究多智能体强化学习算法，使用平台为pettingzoo，选择场景为waterworld。

- 文献调研，选择可能运行的多智能体算法。
    - 可能的算法:MADDPG,MAPPO,MADQN
- 复现这些算法，并将其指标作为基线算法的benchmark。
- 在此基础上思考算法改进，并与基线算法进行对比。

计划文件架构
```text
MARL-Project/
├── algorithms/           # 每个算法一个文件
│   ├── qmix.py          # QMIX完整实现
│   ├── mappo.py         # MAPPO完整实现  
│   └── iql.py           # IQL完整实现
├── envs/
│   └── smac_wrapper.py  # 环境封装
├── core/                # 核心组件（合并common和utils）
│   ├── networks.py      # 网络结构
│   ├── buffer.py        # 经验回放
│   ├── logger.py        # 日志和可视化
│   └── utils.py         # 工具函数
├── run.py               # 单一运行脚本（训练+评估）
├── configs/             # 配置文件
│   ├── qmix.yaml
│   ├── mappo.yaml
│   └── iql.yaml
├── models/              # 模型存储
├── results/             # 实验结果
└── requirements.txt
```

调用算法的指令：
```bash
# 训练
python run.py --algo qmix --config configs/qmix.yaml --train

# 评估  
python run.py --algo qmix --config configs/qmix.yaml --eval --model_path models/qmix.pth
```