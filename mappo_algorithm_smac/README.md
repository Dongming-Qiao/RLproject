Run code with `python mappo_smacv2.py`

See visualized results with `tensorboard --logdir runs`

venv activation `source /home/fireray/smac_env/bin/activate`

pip install git+https://github.com/devindeng94/smac-hard.git

selfplay customization options:
first import smacv2_selfplay_inject
then when initializing env
    env = StarCraftCapabilityEnvWrapper(
        map_name="10gen_terran",
        capability_config=cfg["capability_config"],
        self_play=False,
    )
    difficulty="7" 
    self play must choose opponent
    opponent_type=opponent_fn
    opponent_fn = create_policy_opponent(
        actor_class=Actor,
        checkpoint_path="models/mappo_ep100.pt",
        device="cuda"
    )

    when saving model, can also restart environment with new saved checkpoint every 100 (or set save frequency) to do continuous self play

For customizeable maps, choose StarCraftCapabilityEnvWrapper, otherwise for maps like '8m' choose StarCraftEnv 
    # env = StarCraft2Env(
    #     map_name="8m",
    #     self_play=False,      # Use SC2 AI
    #     difficulty="7"        # 1=VeryEasy, 7=VeryHard, A=CheatInsane
    # )

tested that with or without imports, self play, different wrappers, results are the same. 

Default Config
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
    "rollout_steps": 128,          # T
    "ppo_epochs": 5,               # K
    "minibatch_size": 256,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.1,
    "lr": 5e-4,
    "value_coef": 0.5,
    "ent_coef": 0.05,
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


DEFAULT_CONFIG = {
        "map_name": "3m",
    "capability_config": {
        "n_units": 3,
        "n_enemies": 3
    },
    "conic_fov": False,
    "obs_own_pos": True,
    "use_unit_ranges": True,
    "min_attack_range": 2,

    # RL hyperparams
    "rollout_steps": 128,          # T
    "ppo_epochs": 5,               # K
    "minibatch_size": 256,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.1,
    "lr": 5e-4,
    "value_coef": 0.5,
    "ent_coef": 0.01,
    "max_grad_norm": 10.0,
    "total_episodes": 10000,
    "save_interval": 100,         # save every N episodes
    "log_interval": 1,
    "eval_interval": 200,
    "seed": 1,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_dir": "models",
    "tb_dir": "runs",
}
