from pettingzoo.butterfly import pistonball_v6
import time

#创建环境
env = pistonball_v6.env(render_mode="human")

#初始化环境
env.reset()

#循环运行环境
for agent in env.agent_iter():
    #获取当前agent的观察、奖励、终止标志等信息。
    observation, reward, termination, truncation, info = env.last()

    #如果游戏结束，动作设为None，否则随机采样一个动作
    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample()
    
    #执行动作
    env.step(action)

    #暂停0.1秒以便观察
    time.sleep(0.005)

    #如果所有agent都结束退出循环
    if all(env.terminations.values()) or all(env.truncations.values()):
        break

#关闭环境
env.close()