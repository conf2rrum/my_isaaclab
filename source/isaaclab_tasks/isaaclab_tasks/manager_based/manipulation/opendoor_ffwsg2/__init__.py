from __future__ import annotations

import gymnasium as gym

from . import agents
from .opendoor_ffwsg2_env_cfg import OpenDoorFFWSG2EnvCfg, OpenDoorFFWSG2EnvCfg_PLAY


gym.register(
    id="Isaac-OpenDoor-FFWSG2-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": OpenDoorFFWSG2EnvCfg,
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_low_dim.json",
    },
)


gym.register(
    id="Isaac-OpenDoor-FFWSG2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": OpenDoorFFWSG2EnvCfg_PLAY,
    },
)
