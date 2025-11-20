from maddpg import MADDPG
from pettingzoo.sisl import waterworld_v4
import argparse
import numpy as np
# 不需要显式导入device_manager，因为maddpg.py已经导入了

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--actor_lr', type=float, default=1e-3)
    parser.add_argument('--critic_lr', type=float, default=1e-3)
    parser.add_argument('--buffer_size', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--tau', type=float, default=0.01)
    parser.add_argument('--max_episode_len', type=int, default=500)
    parser.add_argument('--episodes', type=int, default=10000)
    # 移除了 --use_gpu 参数
    
    args = parser.parse_args()
    
    # 创建环境
    env = waterworld_v4.env(
        render_mode=None,
        n_pursuers=2,      # 2个智能体
        n_evaders=4,       # 4个目标
        n_poisons=3,       # 3个毒物
        n_coop=1,          # 需要合作的目标数
        max_cycles=args.max_episode_len
    )
    
    # 创建MADDPG - 设备会自动检测和设置
    maddpg = MADDPG(env, args)
    
    print("开始训练...")

    # 训练循环保持不变...
    for episode in range(args.episodes):
        # 重置环境
        env.reset()
        observations = {agent: env.observe(agent) for agent in env.agents}
        episode_reward = 0
        episode_steps = 0

        for agent_name in env.agent_iter():
            # 获取所有智能体的动作（只在第一个智能体时计算）
            if agent_name == env.agents[0]:
                actions = maddpg.get_actions(observations, noise=True)
                next_observations = {}
                rewards = {}
                dones = {}
            
            # 执行动作
            if agent_name in actions:
                action = actions[agent_name]
            else:
                action = None
            
            # 检查智能体是否终止
            if env.terminations[agent_name] or env.truncations[agent_name]:
                action = None

            env.step(action)

            # 获取环境反馈
            obs, reward, termination, truncation, info = env.last()

            # 更新信息
            next_observations[agent_name] = obs
            rewards[agent_name] = reward
            dones[agent_name] = termination or truncation

            # 当处理到最后一个智能体时，进行学习
            if agent_name == env.agents[-1]:
                # 存储经验
                maddpg.store_experiences(observations, actions, rewards, next_observations, dones)

                # 更新网络
                actor_loss, critic_loss = maddpg.update_all_agents()

                # 更新状态
                observations = next_observations
                episode_reward += sum(rewards.values())
                episode_steps += 1

                # 检查是否结束
                if all(dones.values()) or episode_steps >= args.max_episode_len:
                    break

        # 输出训练信息
        if episode % 10 == 0:
            print(f"Episode {episode:4d} | Steps: {episode_steps:3d} | Reward: {episode_reward:7.2f}")
        
        if episode % 100 == 0 and episode > 0:
            print(f"=== Episode {episode} 详细统计 ===")
            print(f"总步数: {episode_steps}")
            print(f"总奖励: {episode_reward:.2f}")
            print("=" * 30)
    
    env.close()
    print("训练完成!")

if __name__ == "__main__":
    train()