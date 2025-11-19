from pettingzoo.classic import tictactoe_v3

#初始化tictacoe环境
env = tictactoe_v3.env(render_mode="human")

#重置环境
env.reset()

print("可用智能体:", env.possible_agents)
print("动作空间", env.action_space(env.agents[0]))

#执行一个完整的游戏
for agent in env.agent_iter():
    #获取当前状态信息
    observation, reward, termination, truncation, info = env.last()

    #打印信息
    print(f"\n当前玩家:{agent}")
    print(f"观察空间形状:{observation['observation'].shape}")
    print(f"奖励:{reward},是否结束{termination}")

    #如果游戏结束，动作为None
    if termination or truncation:
        action = None
    else:
        #否则，随机选择一个合法动作
        action = env.action_space(agent).sample()
        print(f"选择动作:{action}")

    #执行动作
    env.step(action)

    #渲染游戏画面
    env.render()

    #如果所有智能体都结束，退出循环
    if all(env.terminations.values()) or all(env.truncations.values()):
        print("游戏结束")
        break

#关闭环境
env.close()