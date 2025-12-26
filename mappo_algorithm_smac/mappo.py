import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

device = torch.device("cuda")

class Actor(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, obs, action_mask=None):
        logits = self.net(obs)

        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, -1e10)

        return logits


class Critic(nn.Module):
    def __init__(self, state_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)

class MAPPOAgent:
    def __init__(self, obs_dim, state_dim, n_actions, n_agents, lr=5e-4, clip_eps=0.2):
        self.n_agents = n_agents
        self.n_actions = n_actions

        self.actor = Actor(obs_dim, n_actions).to(device)
        self.critic = Critic(state_dim).to(device)
        

        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr
        )

        self.clip_eps = clip_eps

    def select_action(self, obs, action_mask):
        obs_t = torch.tensor(obs, dtype=torch.float32).to(device)          # [n_agents, obs_dim]
        mask_t = torch.tensor(action_mask, dtype=torch.float32).to(device) # [n_agents, n_actions]

        logits = self.actor(obs_t, action_mask=mask_t)
        dist = torch.distributions.Categorical(logits=logits)

        action = dist.sample()
        logp = dist.log_prob(action)

        return action.cpu().numpy(), logp.cpu().numpy()

    def evaluate_actions(self, obs, action_masks, actions):
        logits = self.actor(obs, action_mask=action_masks)
        dist = torch.distributions.Categorical(logits=logits)

        logp = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        values = self.critic(obs.reshape(obs.size(0) * obs.size(1), -1))

        return logp, entropy, values

    def update(self, batch):
        obs, states, actions, masks, returns, advantages = batch

        B, N, _ = obs.shape

        obs = obs.to(device)
        states = states.to(device)
        actions = actions.to(device)
        masks = masks.to(device)
        returns = returns.to(device)
        advantages = ((advantages - advantages.mean()) / (advantages.std() + 1e-8)).to(device)

        logits = self.actor(obs, action_mask=masks)
        dist = torch.distributions.Categorical(logits=logits)

        new_logp = dist.log_prob(actions)
        old_logp = new_logp.detach()  # use stored logp if you prefer

        ratio = torch.exp(new_logp - old_logp)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages

        policy_loss = -torch.min(surr1, surr2).mean()

        values = self.critic(states)
        value_loss = ((returns.reshape(-1, N) - values.reshape(B, N)) ** 2).mean()

        loss = policy_loss + value_loss * 0.5 - dist.entropy().mean() * 0.01

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.optimizer.step()
