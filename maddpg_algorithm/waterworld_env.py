from pettingzoo.sisl import waterworld_v4
import numpy as np

class WaterworldEnv:
    def __init__(self, render_mode=None, **env_args):
        self.env = waterworld_v4.env(render_mode=render_mode, **env_args)
        self._agents = None
        self._is_reset = False  # 确保这里初始化了
    
    @property
    def agents(self):
        if self._agents is None:
            raise AttributeError("Environment must be reset before accessing agents")
        return self._agents
    
    def reset(self):
        """重置环境并返回所有智能体的观测"""
        observations = self.env.reset()
        self._agents = self.env.agents
        self._is_reset = True  # 确保这个被设置
        
        # 获取所有智能体的观测
        observations_dict = {}
        for agent in self.agents:
            observations_dict[agent] = self.env.observe(agent)
        return observations_dict
    
    def step(self, actions):
        if not self._is_reset:
            raise RuntimeError("Environment must be reset before stepping")
        
        # 按照 agent_iter() 的顺序逐个执行动作
        for agent in self.env.agent_iter():
            # 检查智能体是否已经死亡
            if self.env.terminations[agent] or self.env.truncations[agent]:
                # 死亡智能体必须传递 None
                action = None
            elif agent in actions:
                action = actions[agent]
                # 确保动作是numpy数组格式
                if isinstance(action, (list, np.ndarray)):
                    action = np.array(action, dtype=np.float32)
            else:
                # 如果没有该agent的动作，传递None
                action = None
            
            self.env.step(action)
        
        # 获取新的观测、奖励、终止状态
        next_observations = {}
        for agent in self.agents:
            next_observations[agent] = self.env.observe(agent)
        
        rewards = {agent: self.env.rewards[agent] for agent in self.agents}
        dones = {agent: self.env.terminations[agent] for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        
        return next_observations, rewards, dones, infos
    
    def observation_space(self, agent):
        return self.env.observation_space(agent)
    
    def action_space(self, agent):
        return self.env.action_space(agent)
    
    def render(self):
        self.env.render()
    
    def close(self):
        self.env.close()
        self._agents = None