import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


device = torch.device("cuda")

class RolloutBuffer:
    def __init__(self, max_steps, n_agents):
        self.max_steps = max_steps
        self.n_agents = n_agents

        self.obs = []
        self.states = []
        self.actions = []
        self.action_masks = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def add(self, obs, state, action, action_mask, logp, reward, done, value):
        self.obs.append(obs)
        self.states.append(state)
        self.actions.append(action)
        self.action_masks.append(action_mask)
        self.logprobs.append(logp)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def compute_gae(self, last_value, gamma=0.99, lam=0.95):
        T = len(self.rewards)

        advantages = np.zeros((T, self.n_agents), dtype=np.float32)
        last_gae = np.zeros(self.n_agents, dtype=np.float32)

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = last_value
                next_nonterminal = 1 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_nonterminal = 1 - self.dones[t]

            delta = (
                self.rewards[t]
                + gamma * next_value * next_nonterminal
                - self.values[t]
            )

            last_gae = delta + gamma * lam * next_nonterminal * last_gae
            advantages[t] = last_gae

        returns = advantages + np.array(self.values)

        self.advantages = advantages
        self.returns = returns

    def to_torch(self):
        return (
            torch.tensor(self.obs, dtype=torch.float32).reshape(-1, self.n_agents, self.obs[0].shape[-1]).to(device),
            torch.tensor(self.states, dtype=torch.float32).reshape(-1, self.states[0].shape[-1]).to(device),
            torch.tensor(self.actions, dtype=torch.long).reshape(-1, self.n_agents).to(device),
            torch.tensor(self.action_masks, dtype=torch.float32).reshape(-1, self.n_agents, -1).to(device),
            torch.tensor(self.logprobs, dtype=torch.float32).reshape(-1, self.n_agents).to(device),
            torch.tensor(self.returns, dtype=torch.float32).reshape(-1, self.n_agents).to(device),
            torch.tensor(self.advantages, dtype=torch.float32).reshape(-1, self.n_agents).to(device),
        )
