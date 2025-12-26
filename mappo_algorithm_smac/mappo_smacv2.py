#!/usr/bin/env python3
"""
Production-quality MAPPO baseline for SMACv2.

Single-file baseline that:
 - Uses centralized critic (global state + agent id) to produce per-agent values
 - Shared actor for all agents (parameter sharing)
 - Action masking (SMACv2 avail actions)
 - GAE, PPO clipping, multiple epochs, minibatches
 - TensorBoard logging (rewards, wins, losses, entropy, value loss)
 - Model save/load
"""

import os
import time
import argparse
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# --- SMACv2 import (assumes installed in venv) ---
from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper

# --------------------------
# Hyperparameters / Config
# --------------------------
DEFAULT_CONFIG = {
    "map_name": "10gen_terran",
    "capability_config": {
        "n_units": 5,
        "n_enemies": 5,
        "team_gen": {
            "dist_type": "weighted_teams",
            "unit_types": ["marine", "marauder", "medivac"],
            "exception_unit_types": ["medivac"],
            "weights": [0.45, 0.45, 0.1],
            "observe": True,
        },
        "start_positions": {
            "dist_type": "surrounded_and_reflect",
            "p": 0.5,
            "n_enemies": 5,
            "map_x": 32,
            "map_y": 32,
        },
    },
    "conic_fov": False,
    "obs_own_pos": True,
    "use_unit_ranges": True,
    "min_attack_range": 2,

    # RL hyperparams
    "rollout_steps": 512,          # T
    "ppo_epochs": 5,               # K
    "minibatch_size": 8000,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "lr": 5e-4,
    "value_coef": 0.5,
    "ent_coef": 0.01,
    "max_grad_norm": 10.0,
    "total_episodes": 5000,
    "save_interval": 100,         # save every N episodes
    "log_interval": 1,
    "eval_interval": 200,
    "seed": 1,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_dir": "models",
    "tb_dir": "runs",
}

# --------------------------
# Utilities
# --------------------------
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# --------------------------
# Networks
# --------------------------
class Actor(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, obs, action_mask=None):
        # obs: [B, obs_dim]
        logits = self.net(obs)
        if action_mask is not None:
            # mask: 1=available, 0=not
            logits = logits.masked_fill(action_mask == 0, -1e10)
        return logits  # [B, n_actions]

class CentralizedCritic(nn.Module):
    def __init__(self, state_dim, n_agents, hidden=64):
        # critic input will be state concatenated with agent one-hot (state_dim + n_agents)
        super().__init__()
        self.input_dim = state_dim + n_agents
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, state_agent):
        # state_agent: [B, state_dim + n_agents]
        return self.net(state_agent).squeeze(-1)  # [B]

# --------------------------
# Rollout Buffer
# --------------------------
class RolloutBuffer:
    def __init__(self, rollout_steps, n_agents, obs_dim, state_dim, n_actions, device):
        self.T = rollout_steps
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.device = device

        self.clear()

    def clear(self):
        # Lists of length <= T
        self.obs = []            # each element: [n_agents, obs_dim]
        self.states = []         # each element: [state_dim]
        self.actions = []        # each element: [n_agents]
        self.action_masks = []   # each: [n_agents, n_actions]
        self.logprobs = []       # each: [n_agents]
        self.rewards = []        # each: [n_agents] (team reward repeated)
        self.dones = []          # each: scalar done flag
        self.values = []         # each: [n_agents] values computed at step

        self.advantages = None
        self.returns = None

    def add(self, obs, state, actions, action_masks, logprobs, rewards, done, values):
        # obs: array [n_agents, obs_dim]
        # state: array [state_dim]
        # actions: array [n_agents]
        # action_masks: array [n_agents, n_actions]
        # logprobs: array [n_agents]
        # rewards: array [n_agents] or scalar repeated
        # done: bool
        # values: array [n_agents] (critic per agent)
        self.obs.append(np.array(obs, dtype=np.float32))
        self.states.append(np.array(state, dtype=np.float32))
        self.actions.append(np.array(actions, dtype=np.int64))
        self.action_masks.append(np.array(action_masks, dtype=np.float32))
        self.logprobs.append(np.array(logprobs, dtype=np.float32))
        self.rewards.append(np.array(rewards, dtype=np.float32))
        self.dones.append(float(done))
        self.values.append(np.array(values, dtype=np.float32))

    def compute_gae(self, last_values, gamma=0.99, lam=0.95):
        # last_values: [n_agents] values for final bootstrap
        T = len(self.rewards)
        n = self.n_agents

        advantages = np.zeros((T, n), dtype=np.float32)
        last_gae = np.zeros(n, dtype=np.float32)

        for t in reversed(range(T)):
            if t == T - 1:
                next_values = last_values
                next_nonterminal = 1.0 - self.dones[t]
            else:
                next_values = self.values[t + 1]
                next_nonterminal = 1.0 - self.dones[t + 1]

            delta = self.rewards[t] + gamma * next_values * next_nonterminal - self.values[t]
            last_gae = delta + gamma * lam * next_nonterminal * last_gae
            advantages[t] = last_gae

        reward_arr = np.array(self.rewards, dtype=np.float32)  # shape [T, n_agents]
        reward_mean = reward_arr.mean()
        reward_std = reward_arr.std() + 1e-8
        reward_arr = (reward_arr - reward_mean) / reward_std
        self.rewards = reward_arr.tolist()

        # compute normalized returns
        returns = advantages + np.array(self.values)
        returns_mean = returns.mean()
        returns_std = returns.std() + 1e-8
        self.returns = ((returns - returns_mean) / returns_std).tolist()
        self.advantages = advantages  # can optionally normalize advantage

    def get_tensors(self):
        """
        Returns flattened tensors shaped for training:
         - obs_flat: [T * n_agents, obs_dim]
         - state_agent_flat: [T * n_agents, state_dim + n_agents]
         - actions_flat: [T * n_agents]
         - masks_flat: [T * n_agents, n_actions]
         - old_logprobs_flat: [T * n_agents]
         - returns_flat: [T * n_agents]
         - advantages_flat: [T * n_agents]
        """
        T = len(self.obs)
        if T == 0:
            raise RuntimeError("RolloutBuffer.get_tensors() called but buffer is empty. Did you collect data?")

        # obs: shape [T, n_agents, obs_dim] -> flatten to [T*n_agents, obs_dim]
        obs_arr = np.stack(self.obs, axis=0)
        obs_flat = obs_arr.reshape(-1, self.obs_dim)

        # states: [T, state_dim] -> repeat per agent -> [T*n_agents, state_dim]
        states_arr = np.stack(self.states, axis=0)
        states_rep = np.repeat(states_arr, repeats=self.n_agents, axis=0)  # [T*n, state_dim]

        # create agent one-hot, shape [n_agents, n_agents]
        agent_ids = np.eye(self.n_agents, dtype=np.float32)  # [n_agents, n_agents]
        agent_ids_rep = np.tile(agent_ids, (T, 1))            # [T*n, n_agents]

        # state_agent concatenation
        state_agent_flat = np.concatenate([states_rep, agent_ids_rep], axis=1)

        # actions, masks, logprobs, returns, advantages: stack and flatten per agent
        actions_arr = np.stack(self.actions, axis=0).reshape(-1)        # [T*n]
        masks_arr = np.stack(self.action_masks, axis=0).reshape(-1, self.n_actions)  # [T*n, n_actions]
        logp_arr = np.stack(self.logprobs, axis=0).reshape(-1)          # [T*n]
        returns_arr = np.stack(self.returns, axis=0).reshape(-1)        # [T*n]
        adv_arr = np.stack(self.advantages, axis=0).reshape(-1)         # [T*n]

        # values stored originally as [T, n]
        values_arr = np.stack(self.values, axis=0).reshape(-1)

        # convert to torch tensors on device
        device = self.device
        obs_flat_t = torch.tensor(obs_flat, dtype=torch.float32, device=device)
        state_agent_t = torch.tensor(state_agent_flat, dtype=torch.float32, device=device)
        actions_t = torch.tensor(actions_arr, dtype=torch.long, device=device)
        masks_t = torch.tensor(masks_arr, dtype=torch.float32, device=device)
        old_logp_t = torch.tensor(logp_arr, dtype=torch.float32, device=device)
        returns_t = torch.tensor(returns_arr, dtype=torch.float32, device=device)
        adv_t = torch.tensor(adv_arr, dtype=torch.float32, device=device)
        values_t = torch.tensor(values_arr, dtype=torch.float32, device=device)

        return obs_flat_t, state_agent_t, actions_t, masks_t, old_logp_t, returns_t, adv_t, values_t

# --------------------------
# MAPPO Agent (actor + critic + update)
# --------------------------
class MAPPO:
    def __init__(self, obs_dim, state_dim, n_actions, n_agents, cfg):
        self.device = torch.device(cfg["device"])
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.n_agents = n_agents

        # networks
        self.actor = Actor(obs_dim, n_actions).to(self.device)
        self.critic = CentralizedCritic(state_dim, n_agents).to(self.device)

        # optimizer
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=cfg["lr"]
        )

        # hyperparams
        self.clip_eps = cfg["clip_eps"]
        self.value_coef = cfg["value_coef"]
        self.ent_coef = cfg["ent_coef"]
        self.max_grad_norm = cfg["max_grad_norm"]
        self.ppo_epochs = cfg["ppo_epochs"]
        self.minibatch_size = cfg["minibatch_size"]

    def select_action(self, obs_np, masks_np):
        # obs_np: [n_agents, obs_dim]
        # masks_np: [n_agents, n_actions]
        obs = torch.tensor(obs_np, dtype=torch.float32, device=self.device)
        masks = torch.tensor(masks_np, dtype=torch.float32, device=self.device)

        logits = self.actor(obs, action_mask=masks)  # [n_agents, n_actions]
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()                     # [n_agents]
        logp = dist.log_prob(actions)               # [n_agents]

        # critic values: we need per-agent values using global state injected later in train loop
        return actions.detach().cpu().numpy(), logp.detach().cpu().numpy()

    def compute_values_for_step(self, state_np):
        # returns per-agent values for a given state by repeating state and feeding agent one-hots
        state = torch.tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, state_dim]
        states_rep = state.repeat(self.n_agents, 1)  # [n_agents, state_dim]
        agent_ids = torch.eye(self.n_agents, device=self.device)
        state_agent = torch.cat([states_rep, agent_ids], dim=1)  # [n_agents, state_dim + n_agents]
        values = self.critic(state_agent)  # [n_agents]
        return values.detach().cpu().numpy()

    def update(self, buffer: RolloutBuffer, cfg):
        tensors = buffer.get_tensors()
        if tensors is None:
            return

        obs_flat, state_agent_flat, actions_flat, masks_flat, old_logp_flat, returns_flat, adv_flat, values_flat = tensors

        # normalize advantages
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        N = obs_flat.size(0)  # total samples T*n_agents

        # PPO update: multiple epochs, minibatch
        inds = np.arange(N)
        clip_eps = self.clip_eps

        # store old log probs passed externally (we have old_logp_flat)
        for epoch in range(self.ppo_epochs):
            np.random.shuffle(inds)
            for start in range(0, N, self.minibatch_size):
                mb_inds = inds[start:start + self.minibatch_size]
                mb_inds = torch.tensor(mb_inds, dtype=torch.long, device=self.device)

                mb_obs = obs_flat[mb_inds]           # [M, obs_dim]
                mb_state_agent = state_agent_flat[mb_inds]  # [M, state_dim + n_agents]
                mb_actions = actions_flat[mb_inds]
                mb_masks = masks_flat[mb_inds]
                mb_old_logp = old_logp_flat[mb_inds]
                mb_returns = returns_flat[mb_inds]
                mb_adv = adv_flat[mb_inds]

                # new log probs from actor
                logits = self.actor(mb_obs, action_mask=mb_masks)  # [M, n_actions]
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(mb_actions)

                ratio = torch.exp(new_logp - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # value loss
                values_pred = self.critic(mb_state_agent)  # [M]
                value_loss = ((mb_returns - values_pred) ** 2).mean()

                entropy = dist.entropy().mean()

                loss = policy_loss + self.value_coef * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()),
                                         self.max_grad_norm)
                self.optimizer.step()

        # Return scalar metrics for logging
        with torch.no_grad():
            # compute full-batch losses for monitoring (cheap-ish)
            logits_all = self.actor(obs_flat, action_mask=masks_flat)
            dist_all = torch.distributions.Categorical(logits=logits_all)
            logp_all = dist_all.log_prob(actions_flat)
            ratio_all = torch.exp(logp_all - old_logp_flat)
            surr_all = torch.minimum(ratio_all * adv_flat, torch.clamp(ratio_all, 1 - clip_eps, 1 + clip_eps) * adv_flat)
            policy_loss_all = -surr_all.mean().item()
            values_pred_all = self.critic(state_agent_flat)
            value_loss_all = ((returns_flat - values_pred_all) ** 2).mean().item()
            entropy_all = dist_all.entropy().mean().item()

        return policy_loss_all, value_loss_all, entropy_all

    def save(self, path):
        ensure_dir(os.path.dirname(path))
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict()
        }, path)

    def load(self, path, map_location=None):
        checkpoint = torch.load(path, map_location=self.device if map_location is None else map_location)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])

# --------------------------
# Training Loop
# --------------------------
def train(cfg):
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"])
    ensure_dir(cfg["model_dir"])
    writer = SummaryWriter(cfg["tb_dir"])

    # Build env
    env = StarCraftCapabilityEnvWrapper(
        capability_config=cfg["capability_config"],
        map_name=cfg["map_name"],
        debug=False,
        conic_fov=cfg["conic_fov"],
        obs_own_pos=cfg["obs_own_pos"],
        use_unit_ranges=cfg["use_unit_ranges"],
        min_attack_range=cfg["min_attack_range"],
    )

    env_info = env.get_env_info()
    n_agents = env_info["n_agents"]
    n_actions = env_info["n_actions"]
    obs_dim = env_info["obs_shape"]
    state_dim = env_info["state_shape"]

    print(f"Env: agents={n_agents}, actions={n_actions}, obs_dim={obs_dim}, state_dim={state_dim}")
    agent = MAPPO(obs_dim, state_dim, n_actions, n_agents, cfg)

    buffer = RolloutBuffer(cfg["rollout_steps"], n_agents, obs_dim, state_dim, n_actions, device)
    buffer.device = device

    reward_history = []
    win_history = []
    ep_rewards_window = deque(maxlen=100)
    ep_wins_window = deque(maxlen=100)

    total_steps = 0
    start_time = time.time()

    for episode in range(1, cfg["total_episodes"] + 1):
        # reset env
        obs_list = None
        state = None
        env.reset()
        terminated = False
        episode_reward = 0.0
        episode_won = False

        step = 0
        # collect rollout up to rollout_steps or until done
        while (step < cfg["rollout_steps"]) and (not terminated):
            obs = env.get_obs()         # list of n_agents obs arrays
            state = env.get_state()     # global state
            masks = [env.get_avail_agent_actions(i) for i in range(n_agents)]

            obs_arr = np.stack(obs)     # [n_agents, obs_dim]
            masks_arr = np.stack(masks)

            # select action & logp
            actions, logps = agent.select_action(obs_arr, masks_arr)  # actions: [n_agents], logps: [n_agents]

            # step environment with actions (list or np array)
            reward, terminated, info = env.step(actions)
            # In SMACv2 wrapper, reward returned is team reward (scalar). Convert to per-agent
            per_agent_reward = np.array([reward] * n_agents, dtype=np.float32)

            # compute critic values for this state (per-agent)
            values = agent.compute_values_for_step(state)  # [n_agents]

            # store in buffer
            buffer.add(obs_arr, state, actions, masks_arr, logps, per_agent_reward, terminated, values)

            episode_reward += reward
            total_cumulative_reward = sum(reward_history)
            # battle_won info may be in info
            episode_won = bool(info.get("battle_won", episode_won))
            step += 1
            total_steps += 1

        # bootstrap value for last state (for GAE)
        if not terminated:
            last_values = agent.compute_values_for_step(env.get_state())  # [n_agents]
        else:
            last_values = np.zeros(n_agents, dtype=np.float32)

        # compute advantages & returns
        buffer.compute_gae(last_values, gamma=cfg["gamma"], lam=cfg["gae_lambda"])
        
        adv_mean = buffer.advantages.mean()
        adv_std = buffer.advantages.std() + 1e-8
        buffer.advantages = (buffer.advantages - adv_mean) / adv_std

        # perform PPO update
        metrics = agent.update(buffer, cfg)
        if metrics is None:
            # nothing to update
            buffer.clear()
            continue

        policy_loss, value_loss, entropy = metrics

        # logging and housekeeping
        reward_history.append(episode_reward)
        win_history.append(int(episode_won))
        ep_rewards_window.append(episode_reward)
        ep_wins_window.append(int(episode_won))

        writer.add_scalar("train/episode_reward", episode_reward, episode)
        writer.add_scalar("train/episode_win", int(episode_won), episode)
        writer.add_scalar("train/avg_100_reward", np.mean(ep_rewards_window), episode)
        writer.add_scalar("train/avg_100_winrate", np.mean(ep_wins_window), episode)
        writer.add_scalar("train/policy_loss", policy_loss, episode)
        writer.add_scalar("train/value_loss", value_loss, episode)
        writer.add_scalar("train/entropy", entropy, episode)
        writer.add_scalar("train/cumulative_reward", total_cumulative_reward, episode)

        if episode % cfg["log_interval"] == 0:
            elapsed = time.time() - start_time
            print(f"Ep {episode:04d} | reward {episode_reward:.3f} | win {int(episode_won)} | "
                  f"avg100 {np.mean(ep_rewards_window):.3f} | steps {total_steps} | time {elapsed:.1f}s")

        # save models periodically
        if episode % cfg["save_interval"] == 0:
            ckpt_path = os.path.join(cfg["model_dir"], f"mappo_ep{episode}.pt")
            agent.save(ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

        # reset buffer for next episode
        buffer.clear()

        # optional evaluation loop (not implemented here) could be added at cfg["eval_interval"]

    writer.close()
    # final save
    ckpt_path = os.path.join(cfg["model_dir"], f"mappo_final.pt")
    agent.save(ckpt_path)
    print("Training finished. Model saved to", ckpt_path)

# --------------------------
# CLI
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=DEFAULT_CONFIG["total_episodes"])
    parser.add_argument("--map", type=str, default=DEFAULT_CONFIG["map_name"])
    parser.add_argument("--device", type=str, default=DEFAULT_CONFIG["device"])
    parser.add_argument("--save-dir", type=str, default=DEFAULT_CONFIG["model_dir"])
    parser.add_argument("--tb-dir", type=str, default=DEFAULT_CONFIG["tb_dir"])
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG.copy()
    cfg["total_episodes"] = args.episodes
    cfg["map_name"] = args.map
    cfg["device"] = args.device
    cfg["model_dir"] = args.save_dir
    cfg["tb_dir"] = args.tb_dir

    ensure_dir(cfg["model_dir"])
    ensure_dir(cfg["tb_dir"])

    train(cfg)

if __name__ == "__main__":
    main()
