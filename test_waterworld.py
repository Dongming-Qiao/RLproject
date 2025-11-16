from pettingzoo.sisl import waterworld_v4
import time

def waterworld_demo():
    # 使用更简单的参数配置
    env = waterworld_v4.env(
        render_mode="human",
        n_pursuers=2,           # 减少追捕者数量
        n_evaders=4,            # 减少食物数量
        n_poisons=3,            # 减少毒物数量
        n_coop=1,               # 简化合作要求
        max_cycles=500          # 减少最大步数
    )
    
    env.reset()
    
    print("=== Waterworld 环境信息 ===")
    print(f"智能体: {len(env.possible_agents)} 个追捕者")
    print(f"动作空间: 连续控制")
    print("=========================")
    
    for i in range(300):  # 减少总步数
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            
            if termination or truncation:
                action = None
            else:
                action = env.action_space(agent).sample()  # 随机动作
            
            env.step(action)
            
            if i % 50 == 0 and agent == env.agents[0]:
                print(f"步骤 {i}, {agent}: 奖励={reward:.3f}")
        
        time.sleep(0.01)
        
        if all(env.terminations.values()) or all(env.truncations.values()):
            print("游戏结束!")
            break
    
    env.close()

if __name__ == "__main__":
    waterworld_demo()