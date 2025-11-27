import argparse
import yaml
import torch
import numpy as np
from envs.smac_wrapper import SMACv2Wrapper
from algorithms.qmix import QMIX
from core.logger import Logger
from core.utils import save_config, create_dirs

def train(config):
    """训练QMIX算法"""
    #创建环境和算法
    env = SMACv2Wrapper(config)
    algo = QMIX(config)
    logger = Logger(config)

    print(f"start to train qmix, env is {config['env']['scenario']}")
    print(f"num of agents: {config['env']['n_agents']}, action space: {env.n_actions}")

    for episode in range(config['training']['total_episodes']):
        #重置环境
        obs, state = env.reset()
        episode_reward = 0
        steps = 0

        while True:
            # 获取可用动作
            avail_actions = env.get_avail_actions()

            # 选择动作
            actions = algo.choose_actions(obs, avail_actions, evaluate=False)

            # 执行动作
            next_obs, next_state, reward, done, info = env.step(actions)

            # 存储经验
            algo.store_experience(obs, actions, reward, next_obs, done, state, next_state)

            # 训练
            metrics = algo.train(None)  # 内部会从buffer采样

            # 更新状态
            obs, state = next_obs, next_state
            episode_reward += reward[0]
            steps += 1

            if done[0]:
                break

        # 记录日志
        if episode % config['training']['log_interval'] == 0:
            log_data = {
                'episode': episode,
                'reward': episode_reward,
                'steps': steps,
                'epsilon': algo.epsilon
            }
            if metrics:
                log_data.update(metrics)
            logger.log(log_data, episode)

            print(f"Episode {episode}, Reward: {episode_reward:.2f}, Steps: {steps}, Epsilon: {algo.epsilon:.3f}")

        # 评估
        if episode % config['training']['evaluate_interval'] == 0:
            eval_reward = evaluate(algo, env, config)
            logger.log({'eval_reward': eval_reward}, episode)
            print(f"评估结果 - Episode {episode}, 平均奖励: {eval_reward:.2f}")

        # 保存模型
        if episode % config['training']['save_interval'] == 0:
            algo.save_models(config['paths']['model_dir'], episode)

    env.close()

def evaluate(algo, env, config):
    """评估算法性能"""
    total_reward = 0
    n_episodes = config['evaluation']['eval_episodes']

    for _ in range(n_episodes):
        obs, state = env.reset()
        episode_reward = 0

        while True:
            avail_actions = env.get_avail_actions()
            actions = algo.choose_actions(obs, avail_actions, evaluate=True)
            next_obs, next_state, reward, done, info = env.step(actions)

            obs, state = next_obs, next_state
            episode_reward += reward[0]
            
            if done[0]:
                break

        total_reward += episode_reward

    return total_reward / n_episodes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='qmix', help='name of the algorithm to use')
    parser.add_argument('--config', type=str, required=True, help='the path of the config file')
    parser.add_argument('--train', action='store_true', help='train mode')
    parser.add_argument('--eval', action='store_true', help='evaluation mode')
    parser.add_argument('--model_path', type=str, help='the path of the trained model for evaluation')

    args = parser.parse_args()

    #加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    #创建必要的目录
    create_dirs([config['paths']['model_dir'], config['paths']['result_dir']])

    if args.train:
        # 训练模式
        save_config(config, config['paths']['result_dir'])
        train(config)
    elif args.eval:
        # 评估模式
        if not args.model_path:
            raise ValueError("need to point out --model_path in evaluation mode")
        
        env = SMACv2Wrapper(config)
        algo = QMIX(config)

        # 加载模型
        algo.load_models(args.model_path, 'latest')

        # 运行评估
        eval_reward = evaluate(algo, env, config)

        print(f"final evaluation over, average reward: {eval_reward:.2f}")

        env.close()

if __name__ == '__main__':
    main()