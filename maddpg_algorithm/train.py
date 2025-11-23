from maddpg import MADDPG
from pettingzoo.mpe import simple_world_comm_v3
import argparse
import numpy as np
from collections import deque
from utils import device_manager, TrainLogger, ModelSaver

from tqdm import tqdm


def create_environment(args):
    """创建训练环境"""
    return simple_world_comm_v3.env(
        num_good=2,
        num_adversaries=4,
        num_obstacles=1,
        num_food=2,
        max_cycles=args.max_episode_len,
        num_forests=2,
        continuous_actions=True,
        render_mode=None
    )


def run_episode(env, maddpg, max_episode_len):
    """运行单个episode并返回统计数据"""
    env.reset()
    observations = {agent: env.observe(agent) for agent in env.agents}
    
    agent_episode_reward = 0
    adversary_episode_reward = 0
    episode_steps = 0
    episode_actor_loss = 0
    episode_critic_loss = 0
    update_count = 0

    buffer_size = len(maddpg.agents[0].replay_buffer)
    should_train = buffer_size >= maddpg.args.batch_size

    for _ in range(max_episode_len):
        # 建立rewards字典
        next_observations = {}
        rewards = {}
        dones = {}
        terminations = {}
        truncations = {}

        # 获取所有智能体的动作
        actions = maddpg.get_actions(observations, noise=True)
        # 执行动作
        for agent in env.agents:
            action = actions.get(agent)
            if env.terminations[agent] or env.truncations[agent]:
                action = None
            env.step(action)

        # 获取环境反馈
        for agent in env.agents:
            obs, reward, termination, truncation, info = env.last(agent)

            # 根据阵营计算奖励
            # 好的智能体名字为agen_0 ...,对抗着里面有adversary单词，其中领导对抗者为leadadversary_0
            if 'adversary' in agent:
                adversary_episode_reward += reward
            else:
                agent_episode_reward += reward

            next_observations[agent] = obs
            rewards[agent] = reward
            terminations[agent] = termination
            truncations[agent] = truncation
            dones[agent] = termination or truncation

        # 存储经验
        maddpg.store_experiences(observations, actions, rewards, next_observations, dones)

        # 更新网络（只在有足够经验时）
        if should_train:
            actor_loss, critic_loss = maddpg.update_all_agents()
            if actor_loss is not None and critic_loss is not None:
                episode_actor_loss += actor_loss
                episode_critic_loss += critic_loss
                update_count += 1
        
        # 更新状态
        observations = next_observations
        spisode_steps += 1

        # 检查是否结束
        if all(dones.values()):
            break

    avg_actor_loss = episode_actor_loss / update_count if update_count > 0 else 0
    avg_critic_loss = episode_critic_loss / update_count if update_count > 0 else 0
    
    return agent_episode_reward, adversary_episode_reward, episode_steps, avg_actor_loss, avg_critic_loss


def train():
    """主训练函数"""
    parser = argparse.ArgumentParser()
    
    # 训练参数 - 调整为更合理的值进行测试
    parser.add_argument('--hidden_dim', type=int, default=128)  # 测试时先用小网络
    parser.add_argument('--actor_lr', type=float, default=5e-4)
    parser.add_argument('--critic_lr', type=float, default=1e-3)
    parser.add_argument('--buffer_size', type=int, default=10000)  # 测试时先用小缓冲区
    parser.add_argument('--batch_size', type=int, default=256)  # 增加批大小避免形状问题
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--tau', type=float, default=0.01)
    parser.add_argument('--exploration_noise', type=float, default=0.05)  # 探索噪声
    parser.add_argument('--max_episode_len', type=int, default=100)  # 增加到合理的长度
    parser.add_argument('--episodes', type=int, default=2000)  # 测试时先用少量episodes
    
    # 日志和保存参数
    parser.add_argument('--save_interval', type=int, default=100)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--plot_interval', type=int, default=100)
    
    args = parser.parse_args()
    
    # 初始化组件
    env = create_environment(args)
    maddpg = MADDPG(env, args)
    logger = TrainLogger()
    model_saver = ModelSaver()
    
    # 训练统计
    best_reward = -float('inf')
    agent_reward_window = deque(maxlen=100)
    adversary_reward_window = deque(maxlen=100)

    # 设置种子，先设置为42以保证结果可复现
    np.random.seed(42)
    
    print("开始训练..., 环境为:simple world_comm_v3")
    print(f"超参数: {vars(args)}")

    # 创建进度条
    pbar = tqdm(range(args.episodes), desc="训练进度")

    # 训练循环
    for episode in range(args.episodes):
        try:
            # 运行episode - 传入max_episode_len参数
            agent_reward, adversary_reward, steps, actor_loss, critic_loss = run_episode(env, maddpg, args.max_episode_len)
            
            # 更新统计
            agent_reward_window.append(agent_reward)
            adversary_reward_window.append(adversary_reward)
            recent_avg_agent_reward = np.mean(agent_reward_window) if agent_reward_window else agent_reward
            recent_avg_adversary_reward = np.mean(adversary_reward_window) if adversary_reward_window else adversary_reward
            
            # 记录日志 - 先仅记录智能体奖励
            logger.log(episode, agent_reward, actor_loss, critic_loss, steps)

            # 更新进度条
            pbar.set_postfix({
                'Agent Reward': f"{agent_reward:.2f}",
                'Adv Reward': f"{adversary_reward:.2f}",
                'Avg Agent': f"{recent_avg_agent_reward:.2f}",
                'Avg Adv': f"{recent_avg_adversary_reward:.2f}",
                'step': f"{steps}",
                'Actor Loss': f"{actor_loss:.4f}",
                'Critic Loss': f"{critic_loss:.4f}"
            })
            pbar.update(1)
            
            # 输出进度
            if episode % args.log_interval == 0:
                print(f"\nEpisode {episode:5d} | Steps: {steps:3d} | "
                      f"Agent Reward: {agent_reward:7.2f} | Adversary Reward: {adversary_reward:7.2f} | "
                      f"Avg Agent reward: {recent_avg_agent_reward:7.2f} | Avg Adversary reward: {recent_avg_adversary_reward:7.2f} | "
                      f"Actor Loss: {actor_loss:.4f} | Critic Loss: {critic_loss:.4f}")
            
            # 保存图表和数据
            if episode % args.plot_interval == 0 and episode > 0:
                logger.save_plot()
                logger.save_data()
                print("save the plot and data.")
            
            # 保存模型
            if episode % args.save_interval == 0:
                # 先直接保存模型
                model_saver.save(maddpg, episode, recent_avg_agent_reward)
        
        except Exception as e:
            print(f"Episode {episode} 发生错误: {e}")
            continue
    
    # 训练结束
    env.close()
    logger.save_plot()
    logger.save_data()
    model_saver.save(maddpg, args.episodes, recent_avg_agent_reward)
    
    print("训练完成!")
    summary = logger.get_summary()
    print("\n=== 最终统计 ===")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    train()