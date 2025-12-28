"""
SMAC v2 Self-Play Inject (Minimal)
==================================

This inject is MINIMAL - it preserves ALL original SMAC v2 features:
- Conic FOV
- Stochastic health
- All observation/reward logic
- Everything else

It ONLY:
1. Fixes the _kill_all_units hang bug
2. Adds opponent actions before the original step() runs

USAGE:
    import smacv2_selfplay_inject  # FIRST
    
    from smacv2.env.starcraft2.starcraft2 import StarCraft2Env
    # or with wrapper
    from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper

OPTIONS:

    # SC2 built-in AI (default - no changes to behavior)
    env = StarCraft2Env(map_name="8m", self_play=False, difficulty="7")
    
    # Heuristic opponent
    env = StarCraft2Env(map_name="8m", self_play=True, opponent_type="heuristic")
    
    # Trained policy opponent
    from smacv2_selfplay_inject import create_policy_opponent
    opponent = create_policy_opponent(Actor, "models/checkpoint.pt", "cuda")
    env = StarCraft2Env(map_name="8m", self_play=True, opponent_type=opponent)

DIFFICULTY (for self_play=False):
    "1"=VeryEasy, "2"=Easy, "3"=Medium, "4"=MediumHard, "5"=Hard,
    "6"=Harder, "7"=VeryHard, "8"=CheatVision, "9"=CheatMoney, "A"=CheatInsane
"""

import math
import numpy as np
from s2clientprotocol import common_pb2 as sc_common
from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import raw_pb2 as r_pb
from s2clientprotocol import debug_pb2 as d_pb

ACTIONS = {
    "move": 16,
    "attack": 23,
    "stop": 4,
    "heal": 386,
}


# =============================================================================
# Opponent Policies
# =============================================================================

class HeuristicOpponent:
    """Attack nearest enemy within range, else move toward nearest."""
    
    def __init__(self, env):
        self.env = env
    
    def get_sc2_actions(self):
        sc_actions = []
        
        for e_id, e_unit in self.env.enemies.items():
            if e_unit.health <= 0:
                continue
            
            # Find nearest alive ally
            min_dist = float('inf')
            target = None
            
            for a_id, a_unit in self.env.agents.items():
                if a_unit.health <= 0:
                    continue
                dist = math.hypot(
                    e_unit.pos.x - a_unit.pos.x,
                    e_unit.pos.y - a_unit.pos.y
                )
                if dist < min_dist:
                    min_dist = dist
                    target = a_unit
            
            if target is None:
                continue
            
            # Check if in attack range
            try:
                shoot_range = self.env.unit_shoot_range(e_id + self.env.n_agents)
            except:
                shoot_range = 6  # default
            
            if min_dist <= shoot_range:
                cmd = r_pb.ActionRawUnitCommand(
                    ability_id=ACTIONS["attack"],
                    target_unit_tag=target.tag,
                    unit_tags=[e_unit.tag],
                    queue_command=False,
                )
            else:
                cmd = r_pb.ActionRawUnitCommand(
                    ability_id=ACTIONS["move"],
                    target_world_space_pos=sc_common.Point2D(
                        x=target.pos.x, y=target.pos.y
                    ),
                    unit_tags=[e_unit.tag],
                    queue_command=False,
                )
            
            sc_actions.append(sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=cmd)))
        
        return sc_actions


class RandomOpponent:
    """Random actions - 70% attack, 30% move."""
    
    def __init__(self, env):
        self.env = env
        import random
        self.random = random
    
    def get_sc2_actions(self):
        sc_actions = []
        
        for e_id, e_unit in self.env.enemies.items():
            if e_unit.health <= 0:
                continue
            
            alive_allies = [a for a in self.env.agents.values() if a.health > 0]
            
            if alive_allies and self.random.random() < 0.7:
                target = self.random.choice(alive_allies)
                cmd = r_pb.ActionRawUnitCommand(
                    ability_id=ACTIONS["attack"],
                    target_unit_tag=target.tag,
                    unit_tags=[e_unit.tag],
                    queue_command=False,
                )
            else:
                dx = self.random.uniform(-2, 2)
                dy = self.random.uniform(-2, 2)
                cmd = r_pb.ActionRawUnitCommand(
                    ability_id=ACTIONS["move"],
                    target_world_space_pos=sc_common.Point2D(
                        x=e_unit.pos.x + dx, y=e_unit.pos.y + dy
                    ),
                    unit_tags=[e_unit.tag],
                    queue_command=False,
                )
            
            sc_actions.append(sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=cmd)))
        
        return sc_actions


class DoNothingOpponent:
    """Enemies stand still - for debugging/baseline."""
    
    def __init__(self, env):
        self.env = env
    
    def get_sc2_actions(self):
        return []


class PolicyOpponent:
    """Opponent controlled by trained RL policy."""
    
    def __init__(self, env, policy_fn):
        self.env = env
        self.policy_fn = policy_fn
    
    def get_sc2_actions(self):
        obs_list = []
        avail_list = []
        alive_enemy_ids = []
        
        for e_id, e_unit in self.env.enemies.items():
            if e_unit.health <= 0:
                continue
            
            alive_enemy_ids.append(e_id)
            obs = self._get_enemy_obs(e_id)
            avail = self._get_enemy_avail_actions(e_id)
            obs_list.append(obs)
            avail_list.append(avail)
        
        if not obs_list:
            return []
        
        obs_arr = np.stack(obs_list)
        avail_arr = np.stack(avail_list)
        actions = self.policy_fn(obs_arr, avail_arr)
        
        sc_actions = []
        for i, e_id in enumerate(alive_enemy_ids):
            sc_action = self._action_to_sc2(e_id, actions[i])
            if sc_action:
                sc_actions.append(sc_action)
        
        return sc_actions
    
    def _get_enemy_obs(self, e_id):
        """Build observation for enemy from its perspective."""
        e_unit = self.env.enemies[e_id]
        
        try:
            sight_range = self.env.unit_sight_range(e_id + self.env.n_agents)
        except:
            sight_range = 9
        
        move_feats = np.ones(4, dtype=np.float32)
        
        n_targets = self.env.n_agents
        target_feats = np.zeros((n_targets, 4), dtype=np.float32)
        
        for a_id, a_unit in self.env.agents.items():
            if a_unit.health <= 0:
                continue
            dist = math.hypot(e_unit.pos.x - a_unit.pos.x, e_unit.pos.y - a_unit.pos.y)
            if dist < sight_range:
                target_feats[a_id, 0] = a_unit.health / a_unit.health_max
                target_feats[a_id, 1] = dist / sight_range
                target_feats[a_id, 2] = (a_unit.pos.x - e_unit.pos.x) / sight_range
                target_feats[a_id, 3] = (a_unit.pos.y - e_unit.pos.y) / sight_range
        
        n_allies = self.env.n_enemies - 1
        ally_feats = np.zeros((n_allies, 4), dtype=np.float32)
        
        ally_idx = 0
        for other_id, other_unit in self.env.enemies.items():
            if other_id == e_id:
                continue
            if other_unit.health > 0:
                dist = math.hypot(e_unit.pos.x - other_unit.pos.x, e_unit.pos.y - other_unit.pos.y)
                if dist < sight_range:
                    ally_feats[ally_idx, 0] = other_unit.health / other_unit.health_max
                    ally_feats[ally_idx, 1] = dist / sight_range
                    ally_feats[ally_idx, 2] = (other_unit.pos.x - e_unit.pos.x) / sight_range
                    ally_feats[ally_idx, 3] = (other_unit.pos.y - e_unit.pos.y) / sight_range
            ally_idx += 1
        
        own_feats = np.array([e_unit.health / e_unit.health_max], dtype=np.float32)
        
        return np.concatenate([move_feats, target_feats.flatten(), ally_feats.flatten(), own_feats])
    
    def _get_enemy_avail_actions(self, e_id):
        """Get available actions for enemy unit."""
        e_unit = self.env.enemies[e_id]
        n_actions = 6 + self.env.n_agents
        avail = np.zeros(n_actions, dtype=np.float32)
        
        if e_unit.health <= 0:
            avail[0] = 1
            return avail
        
        avail[1] = 1  # stop
        avail[2:6] = 1  # movement
        
        try:
            shoot_range = self.env.unit_shoot_range(e_id + self.env.n_agents)
        except:
            shoot_range = 6
        
        for a_id, a_unit in self.env.agents.items():
            if a_unit.health <= 0:
                continue
            dist = math.hypot(e_unit.pos.x - a_unit.pos.x, e_unit.pos.y - a_unit.pos.y)
            if dist <= shoot_range:
                avail[6 + a_id] = 1
        
        return avail
    
    def _action_to_sc2(self, e_id, action):
        """Convert action index to SC2 command."""
        e_unit = self.env.enemies[e_id]
        
        if action == 0:
            return None
        elif action == 1:
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=ACTIONS["stop"],
                unit_tags=[e_unit.tag],
                queue_command=False,
            )
        elif action in [2, 3, 4, 5]:
            directions = {2: (0, 2), 3: (0, -2), 4: (2, 0), 5: (-2, 0)}
            dx, dy = directions[action]
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=ACTIONS["move"],
                target_world_space_pos=sc_common.Point2D(x=e_unit.pos.x + dx, y=e_unit.pos.y + dy),
                unit_tags=[e_unit.tag],
                queue_command=False,
            )
        else:
            target_id = action - 6
            target_unit = self.env.agents.get(target_id)
            if target_unit and target_unit.health > 0:
                cmd = r_pb.ActionRawUnitCommand(
                    ability_id=ACTIONS["attack"],
                    target_unit_tag=target_unit.tag,
                    unit_tags=[e_unit.tag],
                    queue_command=False,
                )
            else:
                return None
        
        return sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=cmd))


# =============================================================================
# Helper to create policy opponent
# =============================================================================

def create_policy_opponent(actor_class, checkpoint_path, device="cpu"):
    """
    Create policy function from saved checkpoint.
    
    Args:
        actor_class: Your Actor network class
        checkpoint_path: Path to .pt file
        device: "cpu" or "cuda"
    
    Returns:
        policy_fn for use as opponent_type
    """
    import torch
    
    device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    actor_state = checkpoint.get("actor", checkpoint)
    
    # Infer dimensions
    for key in actor_state.keys():
        if "net.0.weight" in key or "0.weight" in key:
            obs_dim = actor_state[key].shape[1]
            break
    for key in reversed(list(actor_state.keys())):
        if "weight" in key:
            n_actions = actor_state[key].shape[0]
            break
    
    print(f"[create_policy_opponent] obs_dim={obs_dim}, n_actions={n_actions}")
    
    actor = actor_class(obs_dim, n_actions).to(device)
    actor.load_state_dict(actor_state)
    actor.eval()
    
    def policy_fn(obs_arr, avail_arr):
        with torch.no_grad():
            # Handle dimension mismatch
            if obs_arr.shape[1] < obs_dim:
                pad = np.zeros((obs_arr.shape[0], obs_dim - obs_arr.shape[1]), dtype=np.float32)
                obs_arr = np.concatenate([obs_arr, pad], axis=1)
            elif obs_arr.shape[1] > obs_dim:
                obs_arr = obs_arr[:, :obs_dim]
            
            if avail_arr.shape[1] < n_actions:
                pad = np.zeros((avail_arr.shape[0], n_actions - avail_arr.shape[1]), dtype=np.float32)
                avail_arr = np.concatenate([avail_arr, pad], axis=1)
            elif avail_arr.shape[1] > n_actions:
                avail_arr = avail_arr[:, :n_actions]
            
            # Normalize
            obs_mean = obs_arr.mean(axis=0, keepdims=True)
            obs_std = obs_arr.std(axis=0, keepdims=True) + 1e-8
            obs_arr = (obs_arr - obs_mean) / obs_std
            
            obs_t = torch.tensor(obs_arr, dtype=torch.float32, device=device)
            avail_t = torch.tensor(avail_arr, dtype=torch.float32, device=device)
            
            logits = actor(obs_t, action_mask=avail_t)
            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()
            
            return actions.cpu().numpy().tolist()
    
    return policy_fn


# =============================================================================
# Monkey-patch (MINIMAL)
# =============================================================================

_PATCHED = False

def patch_starcraft2env():
    """
    Minimal patch:
    1. Fix _kill_all_units hang
    2. Add opponent actions BEFORE original step() - preserves ALL original logic
    """
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True
    
    from smacv2.env.starcraft2.starcraft2 import StarCraft2Env
    
    # Store originals
    _original_init = StarCraft2Env.__init__
    _original_step = StarCraft2Env.step
    _original_reset = StarCraft2Env.reset
    
    # === FIX 1: _kill_all_units hang ===
    def _kill_all_units_fixed(self):
        """Fixed: uses tracked units, no infinite loop."""
        units_alive = [
            unit.tag for unit in self.agents.values() if unit.health > 0
        ] + [
            unit.tag for unit in self.enemies.values() if unit.health > 0
        ]
        
        if units_alive:
            debug_command = [
                d_pb.DebugCommand(kill_unit=d_pb.DebugKillUnit(tag=units_alive))
            ]
            self._controller.debug(debug_command)
        
        self._controller.step(2)
        self._obs = self._controller.observe()
    
    StarCraft2Env._kill_all_units = _kill_all_units_fixed
    
    # === Extended __init__ ===
    def _new_init(self, self_play=False, opponent_type="heuristic", **kwargs):
        self._self_play = self_play
        self._opponent_type = opponent_type
        self._opponent = None
        
        # Call original init - ALL original parameters preserved
        _original_init(self, **kwargs)
        
        if self._self_play:
            self._init_opponent()
    
    def _init_opponent(self):
        if self._opponent_type == "heuristic":
            self._opponent = HeuristicOpponent(self)
        elif self._opponent_type == "random":
            self._opponent = RandomOpponent(self)
        elif self._opponent_type == "nothing":
            self._opponent = DoNothingOpponent(self)
        elif callable(self._opponent_type):
            self._opponent = PolicyOpponent(self, self._opponent_type)
        else:
            raise ValueError(f"Unknown opponent_type: {self._opponent_type}")
    
    # === MINIMAL step override ===
    def _new_step(self, actions):
        """
        MINIMAL override:
        - If self_play=True: send opponent actions FIRST, then call original step()
        - If self_play=False: call original step() directly (SC2 AI handles enemies)
        
        This preserves ALL original SMAC v2 logic: conic FOV, stochastic health, etc.
        """
        # Send opponent actions BEFORE original step processes agent actions
        if getattr(self, '_self_play', False) and self._opponent is not None:
            opponent_sc_actions = self._opponent.get_sc2_actions()
            if opponent_sc_actions:
                req = sc_pb.RequestAction(actions=opponent_sc_actions)
                self._controller.actions(req)
        
        # Call ORIGINAL step() - preserves ALL SMAC v2 features
        return _original_step(self, actions)
    
    # === Extended reset ===
    def _new_reset(self, episode_config={}):
        result = _original_reset(self, episode_config)
        
        if getattr(self, '_self_play', False) and self._opponent is None:
            self._init_opponent()
        
        return result
    
    # Apply patches
    StarCraft2Env.__init__ = _new_init
    StarCraft2Env.step = _new_step
    StarCraft2Env.reset = _new_reset
    StarCraft2Env._init_opponent = _init_opponent
    
    print("[smacv2_selfplay_inject] Patched: _kill_all_units fix + minimal self-play")


# Auto-patch on import
patch_starcraft2env()