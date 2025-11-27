import time
from pettingzoo.mpe import simple_world_comm_v3

def random_policy_demo():
    # 创建环境
    env = simple_world_comm_v3.env(render_mode="human", max_cycles=100, continuous_actions=True)
    
    # 重置环境
    env.reset()
    
    print("智能体列表:", env.agents)
    print("环境开始运行...")
    
    # 运行环境
    for agent in env.agent_iter():
        # 获取当前智能体的观察、奖励、终止和截断信息
        observation, reward, termination, truncation, info = env.last()

        print(info)
        
        print(f"当前智能体: {agent}, 奖励: {reward:.3f}, 终止: {termination}, 截断: {truncation}")
        
        # 如果智能体已经终止或截断，动作为None
        if termination or truncation:
            action = None
        else:
            # 随机策略：从动作空间中随机采样一个动作
            if env.continuous_actions:
                # 连续动作空间：采样连续值
                action = env.action_space(agent).sample()
            else:
                # 离散动作空间：采样离散动作
                action = env.action_space(agent).sample()
            
            print(f"  选择的动作: {action}")
        
        # 执行动作
        env.step(action)
        
        # 稍微延迟以便观察
        time.sleep(0.025)
    
    print("环境运行结束")
    env.close()

def parallel_random_demo():
    """使用并行API的版本"""
    print("\n=== 使用并行API ===")
    
    env = simple_world_comm_v3.parallel_env(render_mode="human", max_cycles=50)
    observations, infos = env.reset()
    
    for i in range(50):
        # 为所有智能体生成随机动作
        actions = {}
        for agent in env.agents:
            if env.continuous_actions:
                actions[agent] = env.action_space(agent).sample()
            else:
                actions[agent] = env.action_space(agent).sample()
        
        # 并行执行所有动作
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        print(f"步骤 {i}:")
        for agent in env.agents:
            print(f"  {agent}: 奖励={rewards[agent]:.3f}")
        
        time.sleep(0.05)
        
        # 检查是否所有智能体都结束了
        if all(terminations.values()) or all(truncations.values()):
            break
    
    env.close()

if __name__ == "__main__":
    # 运行基础版本
    random_policy_demo()
    
    # 运行并行版本
    # parallel_random_demo()