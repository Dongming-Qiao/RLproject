from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper
import numpy as np
import time
from buffer import RolloutBuffer
from mappo import MAPPOAgent
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

device = torch.device("cuda")

def train_smacv2():
    distribution_config = {
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
    }

    env = StarCraftCapabilityEnvWrapper(
        capability_config=distribution_config,
        map_name="10gen_terran",
        debug=False,
        conic_fov=False,
        obs_own_pos=True,
        use_unit_ranges=True,
        min_attack_range=2,
    )

    info = env.get_env_info()
    n_agents = info["n_agents"]
    n_actions = info["n_actions"]
    obs_dim = info["obs_shape"]
    state_dim = info["state_shape"]

    agent = MAPPOAgent(obs_dim, state_dim, n_actions, n_agents)

    MAX_STEPS = 120  # rollout length
    buffer = RolloutBuffer(MAX_STEPS, n_agents)

    for episode in range(5000):

        env.reset()
        terminated = False
        episode_reward = 0

        step = 0

        while not terminated and step < MAX_STEPS:
            obs = env.get_obs()                 # list of n_agents arrays
            state = env.get_state()             # global state
            masks = [env.get_avail_agent_actions(i) for i in range(n_agents)]

            obs_arr = np.stack(obs)
            mask_arr = np.stack(masks)

            actions, logps = agent.select_action(obs_arr, mask_arr)

            reward, terminated, _ = env.step(actions)

            values = agent.critic(torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0)).cpu().item()

            buffer.add(
                obs_arr,
                state,
                actions,
                mask_arr,
                logps,
                np.array([reward] * n_agents),
                float(terminated),
                np.array([values] * n_agents)
            )

            episode_reward += reward
            step += 1

        # bootstrap
        last_state = env.get_state()
        last_value = agent.critic(torch.tensor(last_state, dtype=torch.float32).to(device).unsqueeze(0)).cpu().item()

        buffer.compute_gae(last_value)
        batch = buffer.to_torch()

        agent.update(batch)

        buffer = RolloutBuffer(MAX_STEPS, n_agents)

        print(f"Episode {episode} reward = {episode_reward}")


if __name__ == "__main__":
    train_smacv2()
