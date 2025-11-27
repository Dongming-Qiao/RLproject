from pettingzoo.classic import hanabi_v5
import time
import numpy as np

def test_hanabi():
    # 创建环境
    env = hanabi_v5.env(render_mode="human")
    
    # 初始化环境
    env.reset()

    print("=" * 50)
    print("Hanabi 卡牌游戏测试")
    print("=" * 50)
    print("游戏玩家:", env.agents)
    print("玩家数量:", len(env.agents))

    # 先执行一步来获取有效的游戏状态信息
    first_agent = env.agents[0]
    observation, reward, termination, truncation, info = env.last()
    
    print("\n初始状态:")
    print("观察类型:", type(observation))
    
    # 安全地获取游戏信息
    if info:
        print("可用信息键:", list(info.keys()))
        
        # 使用更安全的方式获取游戏状态
        life_tokens = info.get('life_tokens', '未知')
        information_tokens = info.get('information_tokens', '未知')
        fireworks = info.get('fireworks', '未知')
        deck_size = info.get('deck_size', '未知')
        
        print(f"剩余生命: {life_tokens}")
        print(f"剩余提示令牌: {information_tokens}")
        print(f"烟花堆状态: {fireworks}")
        print(f"牌堆剩余: {deck_size}")
    else:
        print("暂无游戏状态信息")

    # 循环运行环境
    max_steps = 30  # 限制最大步数
    step_count = 0

    for agent in env.agent_iter():
        if step_count >= max_steps:
            print(f"\n达到最大步数 {max_steps}，结束测试")
            break
            
        # 获取当前agent的观察、奖励等信息
        observation, reward, termination, truncation, info = env.last()
        
        print(f"\n{'='*30}")
        print(f"步骤 {step_count} - 当前玩家: {agent}")
        print(f"奖励: {reward}")
        print(f"终止: {termination}, 截断: {truncation}")
        
        # 显示观察信息
        if isinstance(observation, dict):
            print("观察字典键:", list(observation.keys()))
            if 'vectorized' in observation:
                print(f"向量化观察形状: {observation['vectorized'].shape}")
        
        # 显示游戏状态信息
        if info:
            print("游戏信息:")
            for key, value in info.items():
                print(f"  - {key}: {value}")
        
        # 如果游戏结束，动作设为None，否则随机采样一个动作
        if termination or truncation:
            action = None
            print("游戏结束!")
        else:
            # 处理动作选择
            if isinstance(observation, dict) and 'action_mask' in observation:
                # 使用动作掩码选择有效动作
                action_mask = observation['action_mask']
                valid_actions = np.where(action_mask == 1)[0]
                if len(valid_actions) > 0:
                    action = np.random.choice(valid_actions)
                    print(f"从 {len(valid_actions)} 个有效动作中选择: {action}")
                else:
                    action = 0
                    print("警告: 没有有效动作，使用默认动作0")
            else:
                action = env.action_space(agent).sample()
                print(f"随机选择动作: {action}")
        
        # 执行动作
        env.step(action)
        
        # 渲染
        env.render()
        
        # 暂停以便观察
        time.sleep(2.0)
        
        step_count += 1

        # 如果所有agent都结束退出循环
        if all(env.terminations.values()) or all(env.truncations.values()):
            print("\n游戏正常结束!")
            break

    print("\n" + "="*50)
    print("测试完成")
    print("="*50)

    # 关闭环境
    env.close()

if __name__ == "__main__":
    test_hanabi()