from __future__ import annotations

import gymnasium as gym

from . import agents
from .opendoor_ffwsg2_rl_env_cfg import OpenDoorFFWSG2EnvCfg, OpenDoorFFWSG2EnvCfg_PLAY


gym.register(
    id="Isaac-OpenDoor-FFWSG2-RL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": OpenDoorFFWSG2EnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OpenDoorFFWSG2PPORunnerCfg",
    },
)


gym.register(
    id="Isaac-OpenDoor-FFWSG2-RL-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": OpenDoorFFWSG2EnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OpenDoorFFWSG2PPORunnerCfg",
    },
)
