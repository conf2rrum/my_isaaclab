from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices.device_base import DeviceBase, DevicesCfg
from isaaclab.devices.openxr import OpenXRDeviceCfg, XrCfg
from isaaclab.devices.openxr.retargeters.manipulator.gripper_retargeter import GripperRetargeterCfg
from isaaclab.devices.openxr.xr_cfg import XrAnchorRotationMode
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg, DifferentialInverseKinematicsActionCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_ISAACLAB_ROOT = Path(__file__).resolve().parents[6]

# Asset locations.
SERVER_RACK_USD_PATH = str(_ISAACLAB_ROOT / "custom_assets/network_rack_v4.usd")
FFW_SG2_USD_PATH = str(_ISAACLAB_ROOT / "custom_assets/robots/FFW_SG2/FFW_SG2.usd")

# Server rack articulation.
RACK_DOOR_JOINT = "door_hinge"
RACK_DOOR_BODY_NAME = "glass_door_link"
RACK_DOOR_HANDLE_OFFSET = (0.054, -0.55, 0.0)
RACK_CLOSED_JOINT_POS = 0.0
RACK_MAX_OPEN_JOINT_POS = 2.61799
RACK_TRAIN_OPEN_TARGET = 1.20
RACK_SUCCESS_OPEN_THRESHOLD = RACK_TRAIN_OPEN_TARGET
RACK_POS = (0.65, 0.0, 0.0)
# RACK_ROT_WXYZ = (1.0, 0.0, 0.0, 0.0)
RACK_ROT_WXYZ = (0.70710678, 0.0, 0.0, -0.70710678)

# Demo visualization boundaries. These are visual-only so they do not change the RL physics.
DEMO_WALL_X_MIN = -0.55
DEMO_WALL_X_MAX = 1.45
DEMO_WALL_Y_MIN = -1.45
DEMO_WALL_Y_MAX = 0.75
DEMO_WALL_THICKNESS = 0.035
DEMO_WALL_HEIGHT = 0.8
DEMO_FLOOR_THICKNESS = 0.01
DEMO_GROUND_SIZE = 80.0
DEMO_GROUND_THICKNESS = 0.012
DEMO_GROUND_COLOR = (0.24, 0.27, 0.30)
DEMO_WALL_COLOR_PALETTE = (
    (0.38, 0.67, 0.84),
    (0.86, 0.48, 0.42),
    (0.48, 0.74, 0.52),
    (0.88, 0.72, 0.36),
)
DEMO_WALL_PRIM_NAMES = ("DemoWallNorth", "DemoWallSouth", "DemoWallEast", "DemoWallWest")

# Robot defaults. Adjust these names once the FFW SG2 USD is in place.
FIX_ROBOT_BASE = True
ROBOT_POS = (0.3, -1.0, 0.0)
ROBOT_ROT_WXYZ = (0.70710678, 0.0, 0.0, 0.70710678)
ROBOT_LIFT_LOWEST_JOINT_POS = -0.4
ROBOT_ARM_JOINT_NAMES = ["arm_r_joint[1-7]"]
ROBOT_EE_BODY_NAME = "arm_r_link7"
ROBOT_GRIPPER_JOINT_NAMES = ["gripper_r_joint[1-4]"]
ROBOT_GRIPPER_OPEN_COMMAND = {"gripper_r_joint.*": 0.0}
ROBOT_GRIPPER_CLOSE_COMMAND = {"gripper_r_joint.*": 1.0}
ROBOT_DEFAULT_JOINT_POS = {
    "arm_r_joint1": 0.638451,
    "arm_r_joint2": -0.924706,
    "arm_r_joint3": -0.737492,
    "arm_r_joint4": -2.572742,
    "arm_r_joint5": -1.302020,
    "arm_r_joint6": 0.350000,
    "arm_r_joint7": -0.527065,
    "gripper_r_joint.*": 0.0,
    "head_joint1": 0.694821,
    "head_joint2": -0.350000,
    "lift_joint": ROBOT_LIFT_LOWEST_JOINT_POS,
}


def _resolve_joint_ids(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    asset = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    unresolved = (
        joint_ids is None
        or joint_ids == slice(None)
        or (not isinstance(joint_ids, slice | int) and len(joint_ids) == 0)
    )
    if asset_cfg.joint_names is not None and unresolved:
        joint_ids, _ = asset.find_joints(asset_cfg.joint_names)
    if joint_ids is None or (not isinstance(joint_ids, slice | int) and len(joint_ids) == 0):
        raise ValueError(f"No joints matched {asset_cfg.joint_names} on scene asset '{asset_cfg.name}'.")
    return joint_ids


def _joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, _resolve_joint_ids(env, asset_cfg)]
    if joint_pos.ndim == 1:
        return joint_pos
    return joint_pos[:, 0]


def _joint_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, _resolve_joint_ids(env, asset_cfg)]
    if joint_vel.ndim == 1:
        return joint_vel
    return joint_vel[:, 0]


def _constant_vec(env: ManagerBasedRLEnv, values: tuple[float, ...]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32, device=env.device).unsqueeze(0).expand(env.num_envs, -1)


def _body_id(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, body_name: str) -> int:
    asset = env.scene[asset_cfg.name]
    body_ids, _ = asset.find_bodies(body_name)
    if len(body_ids) == 0:
        raise ValueError(f"Body '{body_name}' was not found in scene asset '{asset_cfg.name}'.")
    return body_ids[0]


def body_pos(env: ManagerBasedRLEnv, body_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    body_id = _body_id(env, asset_cfg, body_name)
    return asset.data.body_pos_w[:, body_id] - env.scene.env_origins


def body_quat(env: ManagerBasedRLEnv, body_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    body_id = _body_id(env, asset_cfg, body_name)
    return asset.data.body_quat_w[:, body_id]


def body_pose(env: ManagerBasedRLEnv, body_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return torch.cat((body_pos(env, body_name, asset_cfg), body_quat(env, body_name, asset_cfg)), dim=-1)


def body_lin_vel(env: ManagerBasedRLEnv, body_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    body_id = _body_id(env, asset_cfg, body_name)
    return asset.data.body_lin_vel_w[:, body_id]


def asset_root_pose(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return torch.cat((asset.data.root_pos_w - env.scene.env_origins, asset.data.root_quat_w), dim=-1)


def door_hinge_pos(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("server_rack"),
) -> torch.Tensor:
    return body_pos(env, RACK_DOOR_BODY_NAME, asset_cfg)


def door_handle_pos(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("server_rack"),
    handle_offset: tuple[float, float, float] = RACK_DOOR_HANDLE_OFFSET,
) -> torch.Tensor:
    hinge_pos = door_hinge_pos(env, asset_cfg)
    door_quat = body_quat(env, RACK_DOOR_BODY_NAME, asset_cfg)
    handle_offset_t = _constant_vec(env, handle_offset)
    return hinge_pos + math_utils.quat_apply(door_quat, handle_offset_t)


def door_handle_quat(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("server_rack"),
) -> torch.Tensor:
    return body_quat(env, RACK_DOOR_BODY_NAME, asset_cfg)


def door_handle_pose(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("server_rack"),
) -> torch.Tensor:
    return torch.cat((door_handle_pos(env, asset_cfg), door_handle_quat(env, asset_cfg)), dim=-1)


def rel_eef_to_handle(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    rack_cfg: SceneEntityCfg = SceneEntityCfg("server_rack"),
) -> torch.Tensor:
    return door_handle_pos(env, rack_cfg) - body_pos(env, ROBOT_EE_BODY_NAME, robot_cfg)


def door_open_fraction(
    env: ManagerBasedRLEnv,
    open_target: float = RACK_TRAIN_OPEN_TARGET,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("server_rack", joint_names=[RACK_DOOR_JOINT]),
) -> torch.Tensor:
    door_pos = torch.clamp(_joint_pos(env, asset_cfg), min=0.0, max=RACK_MAX_OPEN_JOINT_POS)
    return torch.clamp(door_pos / open_target, min=0.0, max=1.0).unsqueeze(-1)


def _handle_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.norm(rel_eef_to_handle(env), dim=-1)


def _door_local_eef_error(env: ManagerBasedRLEnv) -> torch.Tensor:
    door_quat = door_handle_quat(env)
    eef_pos = body_pos(env, ROBOT_EE_BODY_NAME)
    handle_pos = door_handle_pos(env)
    return math_utils.quat_apply_inverse(door_quat, eef_pos - handle_pos)


def _gripper_closed_amount(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=ROBOT_GRIPPER_JOINT_NAMES),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, _resolve_joint_ids(env, asset_cfg)]
    if joint_pos.ndim == 1:
        joint_pos = joint_pos.unsqueeze(-1)
    return torch.mean(torch.clamp(joint_pos, min=0.0, max=1.0), dim=-1)


def gripper_closed_obs(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=ROBOT_GRIPPER_JOINT_NAMES),
) -> torch.Tensor:
    return _gripper_closed_amount(env, asset_cfg).unsqueeze(-1)


def handle_distance_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.20,
    near_bonus_threshold: float = 0.06,
) -> torch.Tensor:
    distance = _handle_distance(env)
    reward = 1.0 / (1.0 + torch.square(distance / std))
    return reward + 0.5 * (distance < near_bonus_threshold).float()


def handle_precision_reward(
    env: ManagerBasedRLEnv,
    x_std: float = 0.08,
    y_std: float = 0.045,
    z_std: float = 0.12,
) -> torch.Tensor:
    local_error = _door_local_eef_error(env)
    scaled_error = torch.stack(
        (local_error[:, 0] / x_std, local_error[:, 1] / y_std, local_error[:, 2] / z_std), dim=-1
    )
    return torch.exp(-torch.sum(torch.square(scaled_error), dim=-1))


def approach_from_handle_front_reward(
    env: ManagerBasedRLEnv,
    min_front_offset: float = -0.02,
    y_std: float = 0.08,
    z_std: float = 0.16,
) -> torch.Tensor:
    local_error = _door_local_eef_error(env)
    front_gate = torch.sigmoid((local_error[:, 0] - min_front_offset) / 0.025)
    yz_error = torch.square(local_error[:, 1] / y_std) + torch.square(local_error[:, 2] / z_std)
    return front_gate * torch.exp(-yz_error)


def grasp_handle_reward(env: ManagerBasedRLEnv, close_distance: float = 0.075) -> torch.Tensor:
    near_handle = (_handle_distance(env) < close_distance).float()
    return near_handle * _gripper_closed_amount(env)


def far_gripper_close_penalty(env: ManagerBasedRLEnv, far_distance: float = 0.16) -> torch.Tensor:
    far_from_handle = (_handle_distance(env) > far_distance).float()
    return far_from_handle * _gripper_closed_amount(env)


def _handle_tangent_dir(env: ManagerBasedRLEnv) -> torch.Tensor:
    door_quat = door_handle_quat(env)
    hinge_pos = door_hinge_pos(env)
    handle_pos = door_handle_pos(env)
    hinge_axis = math_utils.quat_apply(door_quat, _constant_vec(env, (0.0, 0.0, 1.0)))
    radius = handle_pos - hinge_pos
    tangent = torch.cross(hinge_axis, radius, dim=-1)
    return tangent / torch.norm(tangent, dim=-1, keepdim=True).clamp(min=1.0e-6)


def pull_handle_velocity_reward(
    env: ManagerBasedRLEnv,
    close_distance: float = 0.10,
    max_speed: float = 0.35,
) -> torch.Tensor:
    near_and_closed = (_handle_distance(env) < close_distance).float() * _gripper_closed_amount(env)
    eef_vel = body_lin_vel(env, ROBOT_EE_BODY_NAME)
    tangent = _handle_tangent_dir(env)
    pull_speed = torch.sum(eef_vel * tangent, dim=-1)
    return near_and_closed * torch.clamp(pull_speed / max_speed, min=0.0, max=1.0)


def door_open_velocity_reward(
    env: ManagerBasedRLEnv,
    max_velocity: float = 0.8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("server_rack", joint_names=[RACK_DOOR_JOINT]),
) -> torch.Tensor:
    door_vel = _joint_vel(env, asset_cfg)
    near_and_closed = (_handle_distance(env) < 0.12).float() * _gripper_closed_amount(env)
    return near_and_closed * torch.clamp(door_vel / max_velocity, min=0.0, max=1.0)


def door_open_progress_reward(
    env: ManagerBasedRLEnv,
    open_target: float = RACK_TRAIN_OPEN_TARGET,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("server_rack", joint_names=[RACK_DOOR_JOINT]),
) -> torch.Tensor:
    progress = torch.clamp(_joint_pos(env, asset_cfg) / open_target, min=0.0, max=1.0)
    return 0.5 * progress + 0.5 * torch.square(progress)


def door_stage_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("server_rack", joint_names=[RACK_DOOR_JOINT]),
) -> torch.Tensor:
    door_pos = _joint_pos(env, asset_cfg)
    return (
        0.25 * (door_pos > 0.15).float()
        + 0.50 * (door_pos > 0.35).float()
        + 0.75 * (door_pos > 0.65).float()
        + 1.00 * (door_pos > RACK_SUCCESS_OPEN_THRESHOLD).float()
        + 1.50 * (door_pos > RACK_TRAIN_OPEN_TARGET).float()
    )


def door_success_bonus(
    env: ManagerBasedRLEnv,
    threshold: float = RACK_SUCCESS_OPEN_THRESHOLD,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("server_rack", joint_names=[RACK_DOOR_JOINT]),
) -> torch.Tensor:
    return (_joint_pos(env, asset_cfg) >= threshold).float()


def door_slam_penalty(
    env: ManagerBasedRLEnv,
    velocity_limit: float = 1.4,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("server_rack", joint_names=[RACK_DOOR_JOINT]),
) -> torch.Tensor:
    excess_speed = torch.clamp(torch.abs(_joint_vel(env, asset_cfg)) - velocity_limit, min=0.0)
    return torch.square(excess_speed)


def rack_slide_motion_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("server_rack", joint_names=[".*slide"]),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    joint_ids = _resolve_joint_ids(env, asset_cfg)
    joint_pos = asset.data.joint_pos[:, joint_ids]
    joint_vel = asset.data.joint_vel[:, joint_ids]
    if joint_pos.ndim == 1:
        joint_pos = joint_pos.unsqueeze(-1)
        joint_vel = joint_vel.unsqueeze(-1)
    return torch.sum(torch.square(joint_pos), dim=-1) + 0.05 * torch.sum(torch.square(joint_vel), dim=-1)


def eef_too_far_from_handle(
    env: ManagerBasedRLEnv,
    threshold: float = 0.4,
) -> torch.Tensor:
    return _handle_distance(env) > threshold


def joint_opened(
    env: ManagerBasedRLEnv,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("server_rack", joint_names=[RACK_DOOR_JOINT]),
    positive_direction: bool = True,
    max_abs_velocity: float | None = None,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    joint_ids = _resolve_joint_ids(env, asset_cfg)

    joint_pos = asset.data.joint_pos[:, joint_ids]
    if joint_pos.ndim == 1:
        joint_pos = joint_pos.unsqueeze(-1)

    if positive_direction:
        opened = torch.any(joint_pos >= threshold, dim=1)
    else:
        opened = torch.any(joint_pos <= threshold, dim=1)

    if max_abs_velocity is not None:
        joint_vel = torch.abs(asset.data.joint_vel[:, joint_ids])
        if joint_vel.ndim == 1:
            joint_vel = joint_vel.unsqueeze(-1)
        opened = torch.logical_and(opened, torch.all(joint_vel <= max_abs_velocity, dim=1))

    return opened


def apply_grasp_friction_materials(
    env: ManagerBasedRLEnv,
    env_ids,
    static_friction: float = 1.5,
    dynamic_friction: float = 1.2,
) -> None:
    stage = sim_utils.get_current_stage()
    material_path = "/World/PhysicsMaterials/HighGrip"
    material_cfg = sim_utils.RigidBodyMaterialCfg(
        static_friction=static_friction,
        dynamic_friction=dynamic_friction,
        restitution=0.0,
        friction_combine_mode="max",
        restitution_combine_mode="min",
    )
    material_cfg.func(material_path, material_cfg)

    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        prim_path_lower = prim_path.lower()
        is_handle_or_door = "/serverrack" in prim_path_lower and "glass_door_link" in prim_path_lower
        is_right_gripper = "/robot" in prim_path_lower and "gripper_r" in prim_path_lower
        if is_handle_or_door or is_right_gripper:
            sim_utils.bind_physics_material(prim_path, material_path, stage=stage, stronger_than_descendants=True)


def apply_demo_wall_materials(
    env: ManagerBasedRLEnv,
    env_ids,
    palette: tuple[tuple[float, float, float], ...] = DEMO_WALL_COLOR_PALETTE,
) -> None:
    """Apply checkerboard colors to per-environment demo boundaries."""

    from pxr import Gf, UsdGeom, UsdShade, Vt

    stage = sim_utils.get_current_stage()
    spacing = getattr(env.scene.cfg, "env_spacing", 1.0)
    origins = env.scene.env_origins.detach().cpu()
    min_x = torch.min(origins[:, 0]).item()
    min_y = torch.min(origins[:, 1]).item()

    def set_display_color(mesh_path: str, color: tuple[float, float, float], opacity: float = 1.0) -> None:
        prim = stage.GetPrimAtPath(mesh_path)
        if not prim.IsValid():
            return

        binding_api = UsdShade.MaterialBindingAPI(prim)
        if hasattr(binding_api, "UnbindAllBindings"):
            binding_api.UnbindAllBindings()
        gprim = UsdGeom.Gprim(prim)
        gprim.CreateDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*color)]))
        gprim.CreateDisplayOpacityAttr().Set(Vt.FloatArray([opacity]))

    for env_id, origin in enumerate(origins):
        col = int(round((origin[0].item() - min_x) / spacing))
        row = int(round((origin[1].item() - min_y) / spacing))
        color_id = (row + col) % len(palette)
        wall_color = palette[color_id]
        floor_color = tuple(0.35 + 0.45 * channel for channel in wall_color)

        for prim_name in DEMO_WALL_PRIM_NAMES:
            mesh_path = f"/World/envs/env_{env_id}/{prim_name}/geometry/mesh"
            set_display_color(mesh_path, wall_color, opacity=1.0)

        floor_mesh_path = f"/World/envs/env_{env_id}/DemoFloor/geometry/mesh"
        set_display_color(floor_mesh_path, floor_color, opacity=0.35)

    set_display_color("/World/DemoGround/geometry/mesh", DEMO_GROUND_COLOR, opacity=1.0)


class OpenXRHandTrackingVisualizer(RetargeterBase):
    """Visualizes raw OpenXR hand joints without changing the action vector."""

    def __init__(self, cfg: OpenXRHandTrackingVisualizerCfg):
        super().__init__(cfg)
        self._enable_visualization = cfg.enable_visualization
        if self._enable_visualization:
            marker_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/ffw_sg2_hand_markers",
                markers={
                    "joint": sim_utils.SphereCfg(
                        radius=0.008,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 1.0)),
                    ),
                },
            )
            self._markers = VisualizationMarkers(marker_cfg)

    def retarget(self, data: dict) -> torch.Tensor:
        if self._enable_visualization:
            joint_positions = []
            for hand in (DeviceBase.TrackingTarget.HAND_LEFT, DeviceBase.TrackingTarget.HAND_RIGHT):
                hand_poses = data.get(hand, {})
                joint_positions.extend([pose[:3] for pose in hand_poses.values()])
            if joint_positions:
                self._markers.visualize(
                    translations=torch.tensor(joint_positions, dtype=torch.float32, device=self._sim_device)
                )

        return torch.empty(0, dtype=torch.float32, device=self._sim_device)

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        return [RetargeterBase.Requirement.HAND_TRACKING]


@dataclass
class OpenXRHandTrackingVisualizerCfg(RetargeterCfg):
    enable_visualization: bool = True
    retargeter_type: type[RetargeterBase] = OpenXRHandTrackingVisualizer


class CalibratedWristPoseDeltaRetargeter(RetargeterBase):
    """Converts OpenXR wrist motion to stable relative pose commands."""

    def __init__(self, cfg: CalibratedWristPoseDeltaRetargeterCfg):
        super().__init__(cfg)
        self.bound_hand = cfg.bound_hand
        self._position_scale = torch.tensor(cfg.position_scale, dtype=torch.float32, device=self._sim_device)
        self._rotation_scale = cfg.rotation_scale
        self._position_deadband = cfg.position_deadband
        self._rotation_deadband = cfg.rotation_deadband
        self._max_frame_delta = cfg.max_frame_delta
        self._max_frame_rotation = cfg.max_frame_rotation
        self._convert_openxr_to_robot_frame = cfg.convert_openxr_to_robot_frame
        self._previous_wrist_pos = None
        self._previous_wrist_quat = None

    def retarget(self, data: dict) -> torch.Tensor:
        hand_data = data.get(self.bound_hand, {})
        wrist = hand_data.get("wrist")
        if wrist is None:
            return torch.zeros(6, dtype=torch.float32, device=self._sim_device)

        wrist_pos = torch.tensor(wrist[:3], dtype=torch.float32, device=self._sim_device)
        wrist_quat = torch.tensor(wrist[3:7], dtype=torch.float32, device=self._sim_device)
        wrist_quat = wrist_quat / torch.linalg.norm(wrist_quat).clamp(min=1e-6)

        if self._previous_wrist_pos is None or self._previous_wrist_quat is None:
            self._previous_wrist_pos = wrist_pos.clone()
            self._previous_wrist_quat = wrist_quat.clone()
            return torch.zeros(6, dtype=torch.float32, device=self._sim_device)

        delta_pos = wrist_pos - self._previous_wrist_pos
        delta_quat = math_utils.quat_mul(
            wrist_quat.unsqueeze(0), math_utils.quat_inv(self._previous_wrist_quat).unsqueeze(0)
        )
        delta_rot = math_utils.axis_angle_from_quat(delta_quat).squeeze(0)
        self._previous_wrist_pos = wrist_pos.clone()
        self._previous_wrist_quat = wrist_quat.clone()

        delta_pos_norm = torch.linalg.norm(delta_pos)
        if delta_pos_norm < self._position_deadband or delta_pos_norm > self._max_frame_delta:
            delta_pos = torch.zeros(3, dtype=torch.float32, device=self._sim_device)

        if self._convert_openxr_to_robot_frame:
            # Hand poses arrive in the Isaac world frame, while relative IK commands are in the robot base frame.
            # The FFW SG2 starts at +90 deg yaw, so world vectors -> robot vectors are Rz(-90) * vector.
            delta_pos = torch.stack((delta_pos[1], -delta_pos[0], delta_pos[2]))
            delta_rot = torch.stack((delta_rot[1], -delta_rot[0], delta_rot[2]))

        delta_rot_norm = torch.linalg.norm(delta_rot)
        if delta_rot_norm < self._rotation_deadband or delta_rot_norm > self._max_frame_rotation:
            delta_rot = torch.zeros(3, dtype=torch.float32, device=self._sim_device)

        return torch.cat((delta_pos * self._position_scale, delta_rot * self._rotation_scale))

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        return [RetargeterBase.Requirement.HAND_TRACKING]


@dataclass
class CalibratedWristPoseDeltaRetargeterCfg(RetargeterCfg):
    bound_hand: DeviceBase.TrackingTarget = DeviceBase.TrackingTarget.HAND_RIGHT
    position_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    rotation_scale: float = 1.0
    position_deadband: float = 0.003
    rotation_deadband: float = 0.01
    max_frame_delta: float = 0.08
    max_frame_rotation: float = 0.5
    convert_openxr_to_robot_frame: bool = True
    retargeter_type: type[RetargeterBase] = CalibratedWristPoseDeltaRetargeter


@configclass
class OpenDoorFFWSG2SceneCfg(InteractiveSceneCfg):
    """Scene for opening the server rack door with an FFW SG2 robot."""

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=FFW_SG2_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=0.5,
                max_contact_impulse=150.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1,
                fix_root_link=FIX_ROBOT_BASE,
            ),
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=ROBOT_POS,
            rot=ROBOT_ROT_WXYZ,
            joint_pos=ROBOT_DEFAULT_JOINT_POS,
            joint_vel={".*": 0.0},
        ),
        actuators={
            "lift": ImplicitActuatorCfg(
                joint_names_expr=["lift_joint"],
                velocity_limit_sim=0.2,
                effort_limit_sim=2000.0,
                stiffness=2000.0,
                damping=200.0,
            ),
            "DY_80": ImplicitActuatorCfg(
                joint_names_expr=[
                    "arm_l_joint[1-2]",
                    "arm_r_joint[1-2]",
                ],
                velocity_limit_sim=15.0,
                effort_limit_sim=61.4,
                stiffness=300.0,
                damping=25.0,
            ),
            "DY_70": ImplicitActuatorCfg(
                joint_names_expr=[
                    "arm_l_joint[3-6]",
                    "arm_r_joint[3-6]",
                ],
                velocity_limit_sim=15.0,
                effort_limit_sim=31.7,
                stiffness=300.0,
                damping=20.0,
            ),
            "DP-42": ImplicitActuatorCfg(
                joint_names_expr=[
                    "arm_l_joint7",
                    "arm_r_joint7",
                ],
                velocity_limit_sim=6.0,
                effort_limit_sim=5.1,
                stiffness=100.0,
                damping=5.0,
            ),
            "gripper_master": ImplicitActuatorCfg(
                joint_names_expr=["gripper_l_joint1", "gripper_r_joint1"],
                velocity_limit_sim=6.0,
                effort_limit_sim=100.0,
                stiffness=400.0,
                damping=8.0,
            ),
            "gripper_slave": ImplicitActuatorCfg(
                joint_names_expr=["gripper_l_joint[2-4]", "gripper_r_joint[2-4]"],
                effort_limit_sim=70.0,
                stiffness=8.0,
                damping=1.0,
            ),
            "head": ImplicitActuatorCfg(
                joint_names_expr=["head_joint1", "head_joint2"],
                velocity_limit_sim=2.0,
                effort_limit_sim=30.0,
                stiffness=150.0,
                damping=3.0,
            ),
        },
    )

    server_rack = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/ServerRack",
        spawn=sim_utils.UsdFileCfg(
            usd_path=SERVER_RACK_USD_PATH,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=5.0,
                max_angular_velocity=4.0,
                max_depenetration_velocity=0.5,
                max_contact_impulse=150.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=RACK_POS,
            rot=RACK_ROT_WXYZ,
            joint_pos={
                RACK_DOOR_JOINT: RACK_CLOSED_JOINT_POS,
                ".*slide": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "door": ImplicitActuatorCfg(
                joint_names_expr=[RACK_DOOR_JOINT],
                effort_limit_sim=80.0,
                velocity_limit_sim=1.2,
                stiffness=0.5,
                damping=1.0,
                armature=0.05,
                friction=0.4,
                dynamic_friction=0.3,
                viscous_friction=0.2,
            ),
            "slides": ImplicitActuatorCfg(
                joint_names_expr=[".*slide"],
                effort_limit_sim=40.0,
                velocity_limit_sim=5.0,
                stiffness=200.0,
                damping=1.0,
            ),
        },
    )

    demo_floor = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/DemoFloor",
        spawn=sim_utils.CuboidCfg(
            size=(
                DEMO_WALL_X_MAX - DEMO_WALL_X_MIN,
                DEMO_WALL_Y_MAX - DEMO_WALL_Y_MIN,
                DEMO_FLOOR_THICKNESS,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(
                0.5 * (DEMO_WALL_X_MIN + DEMO_WALL_X_MAX),
                0.5 * (DEMO_WALL_Y_MIN + DEMO_WALL_Y_MAX),
                0.5 * DEMO_FLOOR_THICKNESS,
            ),
        ),
    )

    demo_wall_north = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/DemoWallNorth",
        spawn=sim_utils.CuboidCfg(
            size=(DEMO_WALL_X_MAX - DEMO_WALL_X_MIN, DEMO_WALL_THICKNESS, DEMO_WALL_HEIGHT),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(
                0.5 * (DEMO_WALL_X_MIN + DEMO_WALL_X_MAX),
                DEMO_WALL_Y_MAX,
                0.5 * DEMO_WALL_HEIGHT,
            ),
        ),
    )

    demo_wall_south = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/DemoWallSouth",
        spawn=sim_utils.CuboidCfg(
            size=(DEMO_WALL_X_MAX - DEMO_WALL_X_MIN, DEMO_WALL_THICKNESS, DEMO_WALL_HEIGHT),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(
                0.5 * (DEMO_WALL_X_MIN + DEMO_WALL_X_MAX),
                DEMO_WALL_Y_MIN,
                0.5 * DEMO_WALL_HEIGHT,
            ),
        ),
    )

    demo_wall_east = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/DemoWallEast",
        spawn=sim_utils.CuboidCfg(
            size=(DEMO_WALL_THICKNESS, DEMO_WALL_Y_MAX - DEMO_WALL_Y_MIN, DEMO_WALL_HEIGHT),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(
                DEMO_WALL_X_MAX,
                0.5 * (DEMO_WALL_Y_MIN + DEMO_WALL_Y_MAX),
                0.5 * DEMO_WALL_HEIGHT,
            ),
        ),
    )

    demo_wall_west = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/DemoWallWest",
        spawn=sim_utils.CuboidCfg(
            size=(DEMO_WALL_THICKNESS, DEMO_WALL_Y_MAX - DEMO_WALL_Y_MIN, DEMO_WALL_HEIGHT),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(
                DEMO_WALL_X_MIN,
                0.5 * (DEMO_WALL_Y_MIN + DEMO_WALL_Y_MAX),
                0.5 * DEMO_WALL_HEIGHT,
            ),
        ),
    )

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(visible=False),
    )

    demo_ground = AssetBaseCfg(
        prim_path="/World/DemoGround",
        spawn=sim_utils.CuboidCfg(
            size=(DEMO_GROUND_SIZE, DEMO_GROUND_SIZE, DEMO_GROUND_THICKNESS),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.5 * DEMO_GROUND_THICKNESS)),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class ActionsCfg:
    """Right-arm relative pose IK plus binary SG2 gripper command."""

    arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=ROBOT_ARM_JOINT_NAMES,
        body_name=ROBOT_EE_BODY_NAME,
        scale=(0.04, 0.04, 0.04, 0.25, 0.25, 0.25),
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
    )

    gripper_action = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=ROBOT_GRIPPER_JOINT_NAMES,
        open_command_expr=ROBOT_GRIPPER_OPEN_COMMAND,
        close_command_expr=ROBOT_GRIPPER_CLOSE_COMMAND,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=base_mdp.last_action)
        robot_joint_pos = ObsTerm(func=base_mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_joint_vel = ObsTerm(func=base_mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        right_eef_pos = ObsTerm(func=body_pos, params={"body_name": ROBOT_EE_BODY_NAME})
        right_eef_quat = ObsTerm(func=body_quat, params={"body_name": ROBOT_EE_BODY_NAME})
        right_eef_lin_vel = ObsTerm(func=body_lin_vel, params={"body_name": ROBOT_EE_BODY_NAME})
        door_handle_pos = ObsTerm(func=door_handle_pos)
        rel_eef_handle = ObsTerm(func=rel_eef_to_handle)
        door_open_fraction = ObsTerm(func=door_open_fraction)
        gripper_closed = ObsTerm(func=gripper_closed_obs)
        gripper_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ROBOT_GRIPPER_JOINT_NAMES)},
        )
        door_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("server_rack", joint_names=[RACK_DOOR_JOINT])},
        )
        door_joint_vel = ObsTerm(
            func=base_mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("server_rack", joint_names=[RACK_DOOR_JOINT])},
        )
        rack_pose = ObsTerm(func=asset_root_pose, params={"asset_cfg": SceneEntityCfg("server_rack")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class DatagenInfoCfg(ObsGroup):
        right_eef_pose = ObsTerm(func=body_pose, params={"body_name": ROBOT_EE_BODY_NAME})
        rack_pose = ObsTerm(func=asset_root_pose, params={"asset_cfg": SceneEntityCfg("server_rack")})
        door_handle_pose = ObsTerm(func=door_handle_pose)
        door_handle_pos = ObsTerm(func=door_handle_pos)
        door_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("server_rack", joint_names=[RACK_DOOR_JOINT])},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    datagen_info: DatagenInfoCfg | None = None


@configclass
class RewardsCfg:
    """Dense rewards for reaching, gripping, pulling, and opening the server-rack door."""

    approach_handle = RewTerm(func=handle_distance_reward, weight=4.0)
    precise_handle_pose = RewTerm(func=handle_precision_reward, weight=2.0)
    approach_from_front = RewTerm(func=approach_from_handle_front_reward, weight=1.0)
    grasp_handle = RewTerm(func=grasp_handle_reward, weight=3.0)
    close_far_from_handle = RewTerm(func=far_gripper_close_penalty, weight=-0.4)
    pull_handle_velocity = RewTerm(func=pull_handle_velocity_reward, weight=2.0)
    door_open_velocity = RewTerm(func=door_open_velocity_reward, weight=2.0)
    door_open_progress = RewTerm(func=door_open_progress_reward, weight=4.0)
    success = RewTerm(func=door_success_bonus, weight=200.0)
    alive_penalty = RewTerm(func=base_mdp.is_alive, weight=-0.05)
    door_slam = RewTerm(func=door_slam_penalty, weight=-1.0)
    rack_slide_motion = RewTerm(func=rack_slide_motion_penalty, weight=-0.25)
    action_rate_l2 = RewTerm(func=base_mdp.action_rate_l2, weight=-0.01)
    action_l2 = RewTerm(func=base_mdp.action_l2, weight=-0.005)
    arm_joint_vel = RewTerm(
        func=base_mdp.joint_vel_l2,
        weight=-2.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ROBOT_ARM_JOINT_NAMES)},
    )
    gripper_joint_vel = RewTerm(
        func=base_mdp.joint_vel_l2,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ROBOT_GRIPPER_JOINT_NAMES)},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    success = DoneTerm(
        func=joint_opened,
        params={
            "asset_cfg": SceneEntityCfg("server_rack", joint_names=[RACK_DOOR_JOINT]),
            "threshold": RACK_SUCCESS_OPEN_THRESHOLD,
            "positive_direction": True,
        },
    )
    abandoned = DoneTerm(
        func=eef_too_far_from_handle,
        params={"threshold": 1.2},
    )


@configclass
class EventCfg:
    apply_grasp_friction = EventTerm(
        func=apply_grasp_friction_materials,
        mode="startup",
    )

    apply_demo_wall_materials = EventTerm(
        func=apply_demo_wall_materials,
        mode="startup",
    )

    set_door_mass = EventTerm(
        func=base_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("server_rack", body_names=[RACK_DOOR_BODY_NAME]),
            "mass_distribution_params": (4.0, 6.0),
            "operation": "abs",
            "recompute_inertia": True,
        },
    )

    reset_all = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})
    reset_robot_arm = EventTerm(
        func=base_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=ROBOT_ARM_JOINT_NAMES),
            "position_range": (-0.03, 0.03),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class OpenDoorFFWSG2EnvCfg(ManagerBasedRLEnvCfg):
    scene: OpenDoorFFWSG2SceneCfg = OpenDoorFFWSG2SceneCfg(num_envs=100, env_spacing=2.5, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    commands = None
    curriculum = None

    xr: XrCfg | None = None

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 12.0
        self.viewer.eye = (1.8, -1.8, 1.4)
        self.viewer.lookat = (0.65, 0.0, 0.9)

        self.sim.dt = 1 / 200
        self.sim.render_interval = 2
        self.sim.physx.solve_articulation_contact_last = True
        self.sim.physx.enable_stabilization = True
        self.sim.physx.min_velocity_iteration_count = 1

    def _configure_xr_teleop(self):
        self.xr = XrCfg(
            anchor_pos=(0.0, 0.0, 0.0),
            anchor_rot=(1.0, 0.0, 0.0, 0.0),
        )
        self.xr.anchor_prim_path = "/World/envs/env_0/Robot"
        self.xr.fixed_anchor_height = True
        self.xr.anchor_rotation_mode = XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED

        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        CalibratedWristPoseDeltaRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT,
                            position_scale=(0.8, 0.8, 0.8),
                            rotation_scale=0.7,
                            position_deadband=0.002,
                            rotation_deadband=0.015,
                            max_frame_delta=0.08,
                            max_frame_rotation=0.45,
                            convert_openxr_to_robot_frame=True,
                            sim_device=self.sim.device,
                        ),
                        GripperRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT,
                            sim_device=self.sim.device,
                        ),
                        OpenXRHandTrackingVisualizerCfg(
                            enable_visualization=True,
                            sim_device=self.sim.device,
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
            }
        )


@configclass
class OpenDoorFFWSG2EnvCfg_PLAY(OpenDoorFFWSG2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.observations.datagen_info = ObservationsCfg.DatagenInfoCfg()
        self._configure_xr_teleop()
