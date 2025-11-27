import numpy as np
from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper

class SMACv2Wrapper:
    """
    SMACv2环境封装类
    提供统一的接口供MARL算法使用
    """
    def __init__(self, config):
        """
        初始化SMACv2环境
        
        Args:
            config: 环境配置字典
        """
        self.config = config
        self.env_config = config.get('env', {})
        
        #创建环境
        distribution_config = {
            "n_units": 5,
            "n_enemies": 5,
            "team_gen": {
                "dist_type": "weighted_teams",
                "unit_types": ["marine", "marauder", "medivac"],
                "exception_unit_types": ["medivac"],
                "weights": [0.45, 0.45, 0.1],
                "observe": True,
            },
            "start_positions": {
                "dist_type": "surrounded_and_reflect",
                "p": 0.5,
                "n_enemies": 5,
                "map_x": 32,
                "map_y": 32,
            },
        }

        #调试代码
        # print(self.env_config.get('scenario', '10gen_terran'))

        self.env = StarCraftCapabilityEnvWrapper(
            capability_config=distribution_config,
            map_name=self.env_config.get('scenario', '10gen_terran'),
            debug=self.env_config.get('debug', False),
            conic_fov=self.env_config.get('conic_fov', False),
            obs_own_pos=self.env_config.get('obs_own_pos', True),
            use_unit_ranges=self.env_config.get('use_unit_ranges', True),
            min_attack_range=self.env_config.get('min_attack_range', 2),
        )

        #获取环境信息
        env_info  =self.env.get_env_info()
        self.n_agents = env_info['n_agents']
        self.n_actions = env_info['n_actions']
        self.episode_limit = env_info['episode_limit']

        #状态和观测维度，可根据实际环境调整
        self.state_dim = env_info['state_shape']
        self.obs_dim = env_info['obs_shape']
        self.action_dim = self.n_actions

        self.steps = 0
        self.episode_reward = 0

    def reset(self):
        """
        重置环境
        
        Returns:
            obs: 所有智能体的观测 [n_agents, obs_dim]
            state: 全局状态 [state_dim]
        """
        self.env.reset()
        self.steps = 0
        self.episode_reward = 0

        obs = self.get_obs()
        state = self.get_state()

        return obs, state
    
    def step(self, actions):
        """
        执行一步动作
        
        Args:
            actions: 所有智能体的动作列表 [n_agents]
            
        Returns:
            next_obs: 下一时刻观测 [n_agents, obs_dim]
            next_state: 下一时刻全局状态 [state_dim]
            rewards: 奖励值 [1]
            done: 是否结束episode [1]
            info: 额外信息字典
        """

        # print(f"HAHAHAHA the actions are :{actions}")

        reward, terminated, info = self.env.step(actions)
        self.steps += 1
        self.episode_reward += reward

        next_obs = self.get_obs()
        next_state = self.get_state()

        #检查是否达到最大步数
        truncated = self.steps >= self.episode_limit
        done = terminated or truncated

        #返回格式统一的奖励与完成标志
        return next_obs, next_state, np.array([reward]), np.array([done]), info
    
    def get_obs(self):
        """
        获取所有智能体的观测
        
        Returns:
            obs: 观测数组 [n_agents, obs_dim]
        """
        obs = []
        for agent_id in range(self.n_agents):
            agent_obs = self.env.get_obs()[agent_id]
            obs.append(agent_obs)

        return np.array(obs)
    
    def get_state(self):
        """
        获取全局状态
        
        Returns:
            state: 全局状态数组 [state_dim]
        """
        return self.env.get_state()
    
    def get_avail_actions(self):
        """
        获取所有智能体的可用动作
        
        Returns:
            avail_actions: 可用动作矩阵 [n_agents, n_actions]
        """
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.env.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return np.array(avail_actions)
    
    def get_avail_agent_actions(self, agent_id):
        """
        获取指定智能体的可用动作
        
        Args:
            agent_id: 智能体ID
            
        Returns:
            avail_actions: 可用动作向量 [n_actions]
        """
        return self.env.get_avail_agent_actions(agent_id)
    
    def render(self):
        """渲染环境"""
        self.env.render()

    def close(self):
        """关闭环境"""
        self.env.close()

    def get_env_info(self):
        """
        获取环境信息
        
        Returns:
            env_info: 环境信息字典
        """
        return {
            'n_agents': self.n_agents,
            'n_actions': self.n_actions,
            'state_dim': self.state_dim,
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'episode_limit': self.episode_limit
        }
    
    def get_stats(self):
        """
        获取环境统计信息
        
        Returns:
            stats: 统计信息字典
        """
        return {
            'episode_reward': self.episode_reward,
            'steps': self.steps,
            'battle_won': getattr(self.env, 'battle_won', False)
        }
    
def create_smac_env(config):
    """
    创建SMACv2环境的工厂函数
    
    Args:
        config: 环境配置
        
    Returns:
        env: SMACv2环境实例
    """
    return SMACv2Wrapper(config)