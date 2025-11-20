from pettingzoo.sisl import waterworld_v4
import time

def waterworld_demo():
    env = waterworld_v4.env(
        render_mode="human",
        n_pursuers=2,
        n_evaders=4,
        n_poisons=3,
        n_coop=1,
        max_cycles=500
    )
    
    env.reset()
    
    print("=== Waterworld 环境信息 ===")
    print(f"智能体: {len(env.possible_agents)} 个追捕者")
    print(f"动作空间: 连续控制")
    print("=========================")
    
    step_count = 0
    max_steps = 300
    
    # 使用 env.agent_iter() 作为主循环
    for agent in env.agent_iter():
        if step_count >= max_steps:
            break
            
        obs, reward, termination, truncation, info = env.last()

        # #打印观测信息的形状
        # print(f"{agent} 观测形状: {obs.shape}")
        # #数据结构类型
        # print(f"{agent} 观测数据类型: {type(obs)}")

        # #打印奖励形状和数据结构类型
        # print(f"{agent} 奖励: {reward}, 数据类型: {type(reward)}")

        # #打印info形状和数据结构类型
        # print(f"{agent} info: {info}, 数据类型: {type(info)}")
        
        if termination or truncation:
            action = None
            print(f"{agent} 已终止或截断，传递 None 动作")
        else:
            # 随机
            action = env.action_space(agent).sample()

        # #打印动作形状和数据结构类型
        # print(f"{agent} 动作: {action}, 数据类型: {type(action)}")
        
        env.step(action)
        
        # 打印信息（可选，避免输出太多）
        if step_count % 50 == 0:
            print(f"步骤 {step_count}, {agent}: 奖励={reward:.3f}")
        
        step_count += 1
        
        # 短暂延迟以便观察
        if step_count % 10 == 0:  # 每10步延迟一次，避免太频繁
            time.sleep(0.01)
    
    print(f"模拟结束，共执行 {step_count} 步")
    env.close()

if __name__ == "__main__":
    waterworld_demo()