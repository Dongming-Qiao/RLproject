from maddpg import MADDPG
from pettingzoo.mpe import simple_world_comm_v3  # 改为MPE环境
import argparse
import numpy as np
from collections import deque
from utils import device_manager, TrainLogger, ModelSaver
from tqdm import tqdm

def create_environment(args):
    """创建MPE训练环境"""
    return simple_world_comm_v3.env(
        num_good=2,           # 好的智能体数量
        num_adversaries=4,    # 对手智能体数量
        num_obstacles=1,      # 障碍物数量
        num_food=2,           # 食物数量
        max_cycles=args.max_episode_len,
        num_forests=2,        # 森林数量
        continuous_actions=True,  # 使用连续动作空间
        render_mode=None
    )

def run_episode(env, maddpg, max_episode_len, episode):
    """运行单个episode并返回统计数据"""
    env.reset()
    observations = {agent: env.observe(agent) for agent in env.agents}
    
    episode_reward = 0
    episode_steps = 0
    episode_actor_loss = 0
    episode_critic_loss = 0
    update_count = 0

    # 检查缓冲区大小，决定是否进行训练
    buffer_size = len(maddpg.agents[0].replay_buffer)
    should_train = buffer_size >= maddpg.args.batch_size
    
    for _ in range(max_episode_len):
        # 获取所有智能体的动作
        actions = maddpg.get_actions(observations, noise=True)
        
        # 执行动作
        env.step(actions)
        
        # 获取环境反馈
        next_observations = {}
        rewards = {}
        dones = {}
        terminations = {}
        truncations = {}
        
        for agent in env.agents:
            obs, reward, termination, truncation, info = env.last(agent)
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
        episode_reward += sum(rewards.values())
        episode_steps += 1
        
        # 检查是否结束
        if all(dones.values()):
            break

    # 计算平均损失
    avg_actor_loss = episode_actor_loss / update_count if update_count > 0 else 0
    avg_critic_loss = episode_critic_loss / update_count if update_count > 0 else 0
    
    return episode_reward, episode_steps, avg_actor_loss, avg_critic_loss, buffer_size, should_train

def train():
    """主训练函数"""
    parser = argparse.ArgumentParser()
    
    # 训练参数 - 优化为快速验证
    parser.add_argument('--hidden_dim', type=int, default=128)      # 减少网络大小
    parser.add_argument('--actor_lr', type=float, default=1e-3)     # 提高学习率
    parser.add_argument('--critic_lr', type=float, default=1e-3)    # 提高学习率
    parser.add_argument('--buffer_size', type=int, default=10000)   # 减小缓冲区
    parser.add_argument('--batch_size', type=int, default=256)      # 适当批大小
    parser.add_argument('--gamma', type=float, default=0.95)        # 折扣因子
    parser.add_argument('--tau', type=float, default=0.01)          # 目标网络更新率
    parser.add_argument('--exploration_noise', type=float, default=0.1)  # 探索噪声
    parser.add_argument('--max_episode_len', type=int, default=100) # 减少episode长度
    parser.add_argument('--episodes', type=int, default=2000)       # 减少总episodes
    
    # 日志和保存参数
    parser.add_argument('--save_interval', type=int, default=200)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--plot_interval', type=int, default=200)
    
    args = parser.parse_args()
    
    # 初始化组件
    env = create_environment(args)
    maddpg = MADDPG(env, args)
    logger = TrainLogger()
    model_saver = ModelSaver()
    
    # 训练统计
    best_reward = -float('inf')
    reward_window = deque(maxlen=50)  # 减小窗口大小
    
    print("开始训练MPE Simple World Comm环境...")
    print(f"超参数: {vars(args)}")

    # 创建进度条
    pbar = tqdm(total=args.episodes, desc="训练进度", unit="episode")

    # 训练循环
    for episode in range(args.episodes):
        try:
            # 运行episode
            reward, steps, actor_loss, critic_loss, buffer_size, should_train = run_episode(
                env, maddpg, args.max_episode_len, episode)
            
            # 更新统计
            reward_window.append(reward)
            recent_avg_reward = np.mean(reward_window) if reward_window else reward
            
            # 记录日志
            logger.log(episode, reward, actor_loss, critic_loss, steps)

            # 更新进度条
            status = "训练" if should_train else f"收集经验({buffer_size}/{args.batch_size})"
            pbar.set_postfix({
                '状态': status,
                '奖励': f'{reward:.1f}',
                '平均奖励': f'{recent_avg_reward:.1f}',
                '步数': steps,
                'Actor损失': f'{actor_loss:.3f}' if actor_loss else 'N/A',
                'Critic损失': f'{critic_loss:.3f}' if critic_loss else 'N/A'
            })
            pbar.update(1)
            
            # 输出进度
            if episode % args.log_interval == 0:
                print(f"\nEpisode {episode:5d} | Steps: {steps:3d} | "
                      f"Reward: {reward:7.2f} | Avg: {recent_avg_reward:7.2f} | "
                      f"Actor: {actor_loss:.4f} | Critic: {critic_loss:.4f}")
            
            # 保存图表和数据
            if (episode % args.plot_interval == 0) and (episode > 0):
                logger.save_plot()
                logger.save_data()
                print("保存训练图表和数据...")
            
            # 保存模型
            if episode % args.save_interval == 0 and episode > 0:
                if recent_avg_reward > best_reward:
                    best_reward = recent_avg_reward
                    model_saver.save(maddpg, episode, recent_avg_reward)
                    print(f"保存最佳模型，Episode {episode}, 平均奖励: {recent_avg_reward:.2f}")
        
        except Exception as e:
            print(f"Episode {episode} 发生错误: {e}")
            continue
    
    # 训练结束
    env.close()
    logger.save_plot()
    logger.save_data()
    model_saver.save(maddpg, args.episodes, recent_avg_reward)
    
    print("\n训练完成!")
    summary = logger.get_summary()
    print("=== 最终统计 ===")
    for key, value in summary.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    train()