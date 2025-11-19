from pettingzoo.butterfly import knights_archers_zombies_v10
import time
import random

def knights_archers_zombies_demo():
    """
    骑士弓箭手打僵尸最简演示
    合作防御僵尸进攻的战略游戏
    """
    
    # 创建环境
    env = knights_archers_zombies_v10.env(
        render_mode="human",
        spawn_rate=20,          # 僵尸生成速度
        max_cycles=1000,        # 最大步数
        num_archers=2,          # 弓箭手数量
        num_knights=2           # 骑士数量
    )
    
    # 初始化环境
    env.reset()
    
    print("=== 骑士弓箭手打僵尸环境信息 ===")
    print(f"智能体数量: {len(env.possible_agents)}")
    print(f"角色: 2个弓箭手 + 2个骑士")
    print(f"目标: 合作防御僵尸进攻")
    print(f"动作空间: 移动 + 攻击")
    print("==============================")
    
    # 运行环境
    for i in range(500):  # 运行500步
        for agent in env.agent_iter():
            # 获取当前状态
            observation, reward, termination, truncation, info = env.last()
            
            # 选择动作
            if termination or truncation:
                action = None
            else:
                # 随机动作：0-7是移动，8是攻击
                action = env.action_space(agent).sample()
            
            # 执行动作
            env.step(action)
            
            # 显示奖励信息
            if reward != 0:
                print(f"步骤 {i}, {agent}: 获得奖励 {reward}")
        
        # 延迟以便观察
        time.sleep(0.03)
        
        # 检查是否所有智能体都结束
        if all(env.terminations.values()) or all(env.truncations.values()):
            print("游戏结束!")
            break
    
    # 关闭环境
    env.close()
    print("Demo 运行完成!")

def kaz_simple_test():
    """更简化的版本"""
    env = knights_archers_zombies_v10.env(
        render_mode="human",
        max_cycles=300
    )
    
    env.reset()
    print("快速测试骑士弓箭手打僵尸...")
    
    for i in range(200):
        for agent in env.agent_iter():
            obs, reward, done, trunc, info = env.last()
            action = env.action_space(agent).sample() if not (done or trunc) else None
            env.step(action)
        
        time.sleep(0.02)
    
    env.close()
    print("测试完成!")

if __name__ == "__main__":
    # 运行完整demo
    knights_archers_zombies_demo()
    
    # 或者运行快速测试
    # kaz_simple_test()