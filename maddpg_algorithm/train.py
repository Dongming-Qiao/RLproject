from maddpg import MADDPG
from waterworld_env import WaterworldEnv
import argparse

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
    
    args = parser.parse_args()
    
    # 创建环境
    env = WaterworldEnv(
        render_mode=None,
        n_pursuers=2,
        n_evaders=4,
        n_poisons=3, 
        n_coop=1,
        max_cycles=args.max_episode_len
    )
    
    # 创建MADDPG
    maddpg = MADDPG(env, args)
    
    print("开始训练...")
    
    for episode in range(args.episodes):
        # 重置环境并获取观测
        observations = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        for step in range(args.max_episode_len):
            # 获取动作
            actions = maddpg.get_actions(observations, noise=True)
            
            # 环境交互 - 现在env.step会按正确顺序处理所有agent
            next_observations, rewards, dones, infos = env.step(actions)
            
            # 存储经验
            maddpg.store_experiences(observations, actions, rewards, next_observations, dones)
            
            # 更新网络
            actor_loss, critic_loss = maddpg.update_all_agents()
            
            observations = next_observations
            episode_reward += sum(rewards.values())
            episode_steps += 1
            
            # 检查是否结束
            if all(dones.values()):
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