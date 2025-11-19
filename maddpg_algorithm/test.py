from waterworld_env import WaterworldEnv
from maddpg import MADDPG
import argparse

def test():
    # 加载训练时的参数
    args = argparse.Namespace(
        hidden_dim=64,
        actor_lr=1e-3,
        critic_lr=1e-3,
        buffer_size=100000,
        batch_size=1024,
        gamma=0.95,
        tau=0.01,
        max_episode_len=500
    )
    
    # 创建带渲染的环境
    env = WaterworldEnv(
        render_mode="human",
        n_pursuers=2,
        n_evaders=4,
        n_poisons=3,
        n_coop=1,
        max_cycles=args.max_episode_len
    )
    
    # 创建MADDPG智能体系统
    maddpg = MADDPG(env, args)
    
    # 加载训练好的模型（如果有）
    # maddpg.load_models("./models/episode_1000")
    
    print("开始测试...")
    
    for episode in range(5):  # 测试5个回合
        observations = env.reset()
        episode_reward = 0
        
        for step in range(args.max_episode_len):
            # 获取动作（测试时不加噪声）
            actions = maddpg.get_actions(observations, noise=False)
            
            # 环境交互
            next_observations, rewards, dones, infos = env.step(actions)
            
            observations = next_observations
            episode_reward += sum(rewards.values())
            
            if all(dones.values()):
                break
        
        print(f"测试回合 {episode + 1}: 总奖励 = {episode_reward:.2f}")
    
    env.close()

if __name__ == "__main__":
    test()