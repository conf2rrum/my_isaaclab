from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg


def joint_opened(
    env,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("microwave"),
) -> torch.Tensor:
    """Return True when the selected microwave door joint is opened beyond `threshold`.

    This is intentionally simple: it checks the raw joint position of the articulated
    microwave asset. Tune `threshold` after you inspect the actual hinge joint range
    of your microwave asset in Isaac Sim.
    """
    asset = env.scene[asset_cfg.name]

    joint_ids = asset_cfg.joint_ids
    if joint_ids is None or len(joint_ids) == 0:
        joint_pos = asset.data.joint_pos
    else:
        joint_pos = asset.data.joint_pos[:, joint_ids]

    if joint_pos.ndim == 1:
        joint_pos = joint_pos.unsqueeze(-1)

    return torch.any(joint_pos >= threshold, dim=1)


