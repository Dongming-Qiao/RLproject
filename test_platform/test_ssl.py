from pettingzoo.mpe import simple_speaker_listener_v4
import time

# 创建环境
env = simple_speaker_listener_v4.env(render_mode="human")

# 初始化环境
env.reset()

print("环境中的智能体:", env.agents)
print("说话者动作空间:", env.action_space('speaker_0'))
print("倾听者动作空间:", env.action_space('listener_0'))

# 循环运行环境
for agent in env.agent_iter():
    # 获取当前agent的观察、奖励等信息
    observation, reward, termination, truncation, info = env.last()
    
    print(f"\n当前智能体: {agent}")
    print(f"观察空间维度: {observation.shape}")
    print(f"奖励: {reward}")

    # 如果游戏结束，动作设为None，否则随机采样一个动作
    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample()
    
    print(f"执行动作: {action}")
    
    # 执行动作
    env.step(action)
    
    # 渲染
    env.render()
    
    # 暂停以便观察
    time.sleep(0.25)

    # 如果所有agent都结束退出循环
    if all(env.terminations.values()) or all(env.truncations.values()):
        print("Episode结束!")
        break

# 关闭环境
env.close()