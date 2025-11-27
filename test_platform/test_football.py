from pettingzoo.butterfly import google_research_football_v2
import time
import numpy as np

# 创建环境
env = google_research_football_v2.env(render_mode="human")

# 初始化环境
env.reset()

print("球队中的智能体:", env.agents)
print("智能体数量:", len(env.agents))

# 循环运行环境
max_steps = 1000
step_count = 0

for agent in env.agent_iter():
    if step_count > max_steps:
        break
        
    # 获取当前agent的观察、奖励等信息
    observation, reward, termination, truncation, info = env.last()
    
    print(f"\n步骤 {step_count}, 当前球员: {agent}")
    print(f"奖励: {reward}")

    # 如果游戏结束，动作设为None，否则随机采样一个动作
    if termination or truncation:
        action = None
    else:
        # 足球环境的动作空间较大，随机选择
        action = env.action_space(agent).sample()
    
    # 执行动作
    env.step(action)
    
    # 增加步数计数
    step_count += 1
    
    # 如果所有agent都结束退出循环
    if all(env.terminations.values()) or all(env.truncations.values()):
        print("比赛结束!")
        break

# 关闭环境
env.close()