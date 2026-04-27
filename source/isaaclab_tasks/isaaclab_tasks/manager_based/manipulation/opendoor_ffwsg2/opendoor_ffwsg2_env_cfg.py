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
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_ISAACLAB_ROOT = Path(__file__).resolve().parents[6]

# Asset locations.
SERVER_RACK_USD_PATH = str(_ISAACLAB_ROOT / "custom_assets/server_lack_primitives_v2/server_rack_v2_plane.usd")
FFW_SG2_USD_PATH = str(_ISAACLAB_ROOT / "custom_assets/robots/ffw_sg2/FFW_SG2.usd")

# Server rack articulation.
RACK_DOOR_JOINT = "door_hinge"
RACK_FULLY_OPEN_JOINT_POS = 0
RACK_SUCCESS_OPEN_THRESHOLD = 1.20
RACK_POS = (0.65, 0.0, 0.0)
RACK_ROT_WXYZ = (1.0, 0.0, 0.0, 0.0)

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


def asset_root_pose(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return torch.cat((asset.data.root_pos_w - env.scene.env_origins, asset.data.root_quat_w), dim=-1)


def joint_opened(
    env: ManagerBasedRLEnv,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("server_rack", joint_names=[RACK_DOOR_JOINT]),
    positive_direction: bool = True,
    max_abs_velocity: float | None = None,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    if asset_cfg.joint_names is not None and joint_ids == slice(None):
        joint_ids, _ = asset.find_joints(asset_cfg.joint_names)
    elif joint_ids is None:
        joint_ids, _ = asset.find_joints(asset_cfg.joint_names)
    elif not isinstance(joint_ids, slice | int) and len(joint_ids) == 0:
        joint_ids, _ = asset.find_joints(asset_cfg.joint_names)
    if not isinstance(joint_ids, slice | int) and len(joint_ids) == 0:
        raise ValueError(f"No joints matched {asset_cfg.joint_names} on scene asset '{asset_cfg.name}'.")

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


class CalibratedWristDeltaRetargeter(RetargeterBase):
    """Converts OpenXR wrist motion to stable relative position commands."""

    def __init__(self, cfg: CalibratedWristDeltaRetargeterCfg):
        super().__init__(cfg)
        self.bound_hand = cfg.bound_hand
        self._position_scale = torch.tensor(cfg.position_scale, dtype=torch.float32, device=self._sim_device)
        self._position_deadband = cfg.position_deadband
        self._max_frame_delta = cfg.max_frame_delta
        self._convert_openxr_to_robot_frame = cfg.convert_openxr_to_robot_frame
        self._previous_wrist_pos = None

    def retarget(self, data: dict) -> torch.Tensor:
        hand_data = data.get(self.bound_hand, {})
        wrist = hand_data.get("wrist")
        if wrist is None:
            return torch.zeros(3, dtype=torch.float32, device=self._sim_device)

        wrist_pos = torch.tensor(wrist[:3], dtype=torch.float32, device=self._sim_device)
        if self._previous_wrist_pos is None:
            self._previous_wrist_pos = wrist_pos.clone()
            return torch.zeros(3, dtype=torch.float32, device=self._sim_device)

        delta = wrist_pos - self._previous_wrist_pos
        self._previous_wrist_pos = wrist_pos.clone()

        delta_norm = torch.linalg.norm(delta)
        if delta_norm < self._position_deadband or delta_norm > self._max_frame_delta:
            return torch.zeros(3, dtype=torch.float32, device=self._sim_device)

        if self._convert_openxr_to_robot_frame:
            # Hand poses arrive in the Isaac world frame (Z-up), while the IK command is in the robot base frame.
            # The FFW SG2 starts at +90 deg yaw, so world delta -> robot delta is Rz(-90) * delta.
            delta = torch.stack((delta[1], -delta[0], delta[2]))

        return delta * self._position_scale

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        return [RetargeterBase.Requirement.HAND_TRACKING]


@dataclass
class CalibratedWristDeltaRetargeterCfg(RetargeterCfg):
    bound_hand: DeviceBase.TrackingTarget = DeviceBase.TrackingTarget.HAND_RIGHT
    position_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    position_deadband: float = 0.003
    max_frame_delta: float = 0.08
    convert_openxr_to_robot_frame: bool = True
    retargeter_type: type[RetargeterBase] = CalibratedWristDeltaRetargeter


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
                max_contact_impulse=25.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
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
                effort_limit_sim=1000000.0,
                stiffness=10000.0,
                damping=100.0,
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
                velocity_limit_sim=2.2,
                effort_limit_sim=30.0,
                stiffness=100.0,
                damping=4.0,
            ),
            "gripper_slave": ImplicitActuatorCfg(
                joint_names_expr=["gripper_l_joint[2-4]", "gripper_r_joint[2-4]"],
                effort_limit_sim=20.0,
                stiffness=2.0,
                damping=0.5,
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
                max_contact_impulse=25.0,
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
                RACK_DOOR_JOINT: RACK_FULLY_OPEN_JOINT_POS,
                ".*slide": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "door": ImplicitActuatorCfg(
                joint_names_expr=[RACK_DOOR_JOINT],
                effort_limit_sim=250.0,
                velocity_limit_sim=0.6,
                stiffness=2.0,
                damping=250.0,
                armature=1.0,
                friction=3.0,
                dynamic_friction=2.0,
                viscous_friction=15.0,
            ),
            "slides": ImplicitActuatorCfg(
                joint_names_expr=[".*slide"],
                effort_limit_sim=40.0,
                velocity_limit_sim=5.0,
                stiffness=200.0,
                damping=20.0,
            ),
        },
    )

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class ActionsCfg:
    """Right-arm absolute pose IK plus binary SG2 gripper command."""

    arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=ROBOT_ARM_JOINT_NAMES,
        body_name=ROBOT_EE_BODY_NAME,
        controller=DifferentialIKControllerCfg(command_type="position", use_relative_mode=True, ik_method="dls"),
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
        right_eef_pos = ObsTerm(func=body_pos, params={"body_name": ROBOT_EE_BODY_NAME})
        right_eef_quat = ObsTerm(func=body_quat, params={"body_name": ROBOT_EE_BODY_NAME})
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
            self.concatenate_terms = False

    @configclass
    class DatagenInfoCfg(ObsGroup):
        right_eef_pose = ObsTerm(func=body_pose, params={"body_name": ROBOT_EE_BODY_NAME})
        rack_pose = ObsTerm(func=asset_root_pose, params={"asset_cfg": SceneEntityCfg("server_rack")})
        door_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("server_rack", joint_names=[RACK_DOOR_JOINT])},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    datagen_info: DatagenInfoCfg = DatagenInfoCfg()


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


@configclass
class EventCfg:
    reset_all = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})


@configclass
class OpenDoorFFWSG2EnvCfg(ManagerBasedRLEnvCfg):
    scene: OpenDoorFFWSG2SceneCfg = OpenDoorFFWSG2SceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    commands = None
    rewards = None
    curriculum = None

    xr: XrCfg = XrCfg(
        anchor_pos=(0.0, 0.0, 0.0),
        anchor_rot=(1.0, 0.0, 0.0, 0.0),
    )

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.viewer.eye = (1.8, -1.8, 1.4)
        self.viewer.lookat = (0.65, 0.0, 0.9)

        self.sim.dt = 1 / 200
        self.sim.render_interval = 2
        self.sim.physx.solve_articulation_contact_last = True
        self.sim.physx.enable_stabilization = True
        self.sim.physx.min_velocity_iteration_count = 1

        self.xr.anchor_prim_path = "/World/envs/env_0/Robot"
        self.xr.fixed_anchor_height = True
        self.xr.anchor_rotation_mode = XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED

        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        CalibratedWristDeltaRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT,
                            position_scale=(0.35, 0.35, 0.35),
                            position_deadband=0.003,
                            max_frame_delta=0.03,
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
