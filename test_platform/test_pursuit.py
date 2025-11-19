from pettingzoo.sisl import pursuit_v4
import time

def pursuit_demo():
    # 创建环境 - 启用部分可观测（迷雾效果）
    env = pursuit_v4.env(
        render_mode="human",
        x_size=16,          # 地图大小
        y_size=16,
        max_cycles=1000,    # 最大步数
        shared_reward=True, # 共享奖励
        n_evaders=2,        # 逃跑者数量
        n_pursuers=4,       # 追捕者数量
        obs_range=5         # 观测范围 - 这就是"迷雾"！
        # catch_range 参数已在新版本中移除
    )
    
    env.reset()
    
    print("=== Pursuit 环境信息 ===")
    print(f"智能体数量: {len(env.possible_agents)}")
    print(f"观测范围: 5x5 (部分可观测)")
    print(f"地图大小: 16x16")
    print(f"动作空间: 5个方向 (上、下、左、右、停留)")
    print("=====================")
    
    for i in range(500):  # 运行500步
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            
            if termination or truncation:
                action = None
            else:
                # 随机策略 - 在实际研究中会被RL算法替代
                action = env.action_space(agent).sample()
            
            env.step(action)
            
            # 显示每个智能体的局部观测信息
            if i % 50 == 0 and agent == env.agents[0]:
                print(f"步骤 {i}, 智能体 {agent}: 奖励={reward}, 观测形状={observation.shape}")
        
        time.sleep(0.05)
        
        if all(env.terminations.values()) or all(env.truncations.values()):
            print("游戏结束!")
            break
    
    env.close()

if __name__ == "__main__":
    pursuit_demo()