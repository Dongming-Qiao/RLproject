from pettingzoo.sisl import multiwalker_v9
import time
import numpy as np

def multiwalker_demo():
    """
    Multiwalker 最简单演示
    3个双足机器人合作运输物体的物理模拟环境
    """
    
    # 创建环境 - 移除了不存在的参数
    env = multiwalker_v9.env(
        render_mode="human",
        n_walkers=3,           # 机器人数量
        position_noise=1e-3,   # 位置噪声
        angle_noise=1e-3,      # 角度噪声
        forward_reward=1.0,    # 前进奖励
        fall_reward=-100.0,    # 摔倒惩罚
        terminate_on_fall=True,# 摔倒终止
        max_cycles=500         # 最大步数
        # 移除了 terminate_on_goal 参数
    )
    
    # 初始化环境
    env.reset()
    
    print("=== Multiwalker 环境信息 ===")
    print(f"智能体数量: {len(env.possible_agents)}")
    print(f"动作空间: 连续控制 (每个智能体 4 个关节)")
    print(f"观测空间: 位置、速度、角度等状态信息")
    print(f"目标: 协调行走，共同前进")
    print("============================")
    
    # 运行环境
    total_steps = 0
    for i in range(300):  # 运行300步
        for agent in env.agent_iter():
            # 获取当前状态
            observation, reward, termination, truncation, info = env.last()
            
            # 选择动作 (随机策略)
            if termination or truncation:
                action = None
            else:
                # 连续动作空间：为每个关节生成随机力矩
                action = env.action_space(agent).sample()
            
            # 执行动作
            env.step(action)
            
            # 显示信息
            if total_steps % 50 == 0:
                print(f"步骤 {total_steps}, 智能体 {agent}:")
                print(f"  奖励: {reward:.3f}")
                print(f"  动作范围: [{action.min():.3f}, {action.max():.3f}]")
            
            total_steps += 1
        
        # 延迟以便观察
        time.sleep(0.02)
        
        # 检查是否所有智能体都结束
        if all(env.terminations.values()) or all(env.truncations.values()):
            print("所有智能体都结束了!")
            break
    
    # 关闭环境
    env.close()
    print("Demo 运行完成!")

def multiwalker_quick_test():
    """最简化的测试版本"""
    env = multiwalker_v9.env(render_mode="human", n_walkers=2, max_cycles=200)
    env.reset()
    
    print("快速测试 Multiwalker...")
    for i in range(100):
        for agent in env.agent_iter():
            obs, reward, done, trunc, info = env.last()
            action = env.action_space(agent).sample() if not (done or trunc) else None
            env.step(action)
        time.sleep(0.03)
    
    env.close()
    print("测试完成!")

if __name__ == "__main__":
    # 运行完整demo
    multiwalker_demo()
    
    # 或者运行快速测试
    # multiwalker_quick_test()