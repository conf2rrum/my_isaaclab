from __future__ import annotations

import gymnasium as gym

from .g1_microwave_env_cfg import G1OpenMicrowaveEnvCfg, G1OpenMicrowaveEnvCfg_PLAY


gym.register(
    id="Isaac-G1-Open-Microwave-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1OpenMicrowaveEnvCfg,
    },
)


gym.register(
    id="Isaac-G1-Open-Microwave-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1OpenMicrowaveEnvCfg_PLAY,
    },
)
