# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import tempfile

import torch
from pink.tasks import DampingTask, FrameTask

import carb

import isaaclab.controllers.utils as ControllerUtils
import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.pink_ik import NullSpacePostureTask, PinkIKControllerCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import ManusViveCfg, OpenXRDeviceCfg, XrCfg
from isaaclab.devices.openxr.retargeters.humanoid.fourier.gr1t2_retargeter import GR1T2RetargeterCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnv

from . import mdp

from isaaclab_assets.robots.fourier import GR1T2_HIGH_PD_CFG  # isort: skip


##
# Scene definition
##
@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the GR1T2 Pick Place Base Scene."""

    # Table
    packing_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.55, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    object = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Microwave",
        spawn=sim_utils.UsdFileCfg(
            # 1. URDF를 임포트해서 만든 '전자레인지 USD 파일'의 실제 경로로 변경해야 합니다.
            usd_path="/home/lgc/projects/IsaacLab/custom_assets/microwave_222.usd", 
            activate_contact_sensors=False,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
            ),
            scale=(0.8, 0.9, 0.8),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.19524, 0.89, 1.23555),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                # 2. 캐비닛 조인트 대신, 전자레인지 URDF에 있는 회전 조인트 이름을 사용합니다.
                ".*door_hinge_joint.*": 0.0, 
            },
        ),
        actuators={
            "door": ImplicitActuatorCfg(
                # 3. 구동(Actuation)할 조인트 이름도 동일하게 맞춰줍니다.
                joint_names_expr=[".*door_hinge_joint.*"], 
                # 4. URDF에 정의된 effort 한계값을 반영합니다.
                effort_limit_sim=2000.0, 
                stiffness=10.0, 
                damping=2.5,
            ),
        },
    )

    # Humanoid robot configured for pick-place manipulation tasks
    robot: ArticulationCfg = GR1T2_HIGH_PD_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0, 0.93),
            rot=(0.7071, 0, 0, 0.7071),
            joint_pos={
                # right-arm
                "right_shoulder_pitch_joint": 0.0,
                "right_shoulder_roll_joint": 0.0,
                "right_shoulder_yaw_joint": 0.0,
                "right_elbow_pitch_joint": -1.5708,
                "right_wrist_yaw_joint": 0.0,
                "right_wrist_roll_joint": 0.0,
                "right_wrist_pitch_joint": 0.0,
                # left-arm
                "left_shoulder_pitch_joint": 0.0,
                "left_shoulder_roll_joint": 0.0,
                "left_shoulder_yaw_joint": 0.0,
                "left_elbow_pitch_joint": -1.5708,
                "left_wrist_yaw_joint": 0.0,
                "left_wrist_roll_joint": 0.0,
                "left_wrist_pitch_joint": 0.0,
                # --
                "head_.*": 0.0,
                "waist_.*": 0.0,
                ".*_hip_.*": 0.0,
                ".*_knee_.*": 0.0,
                ".*_ankle_.*": 0.0,
                "R_.*": 0.0,
                "L_.*": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
    )

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    upper_body_ik = PinkInverseKinematicsActionCfg(
        pink_controlled_joint_names=[
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_pitch_joint",
            "left_wrist_yaw_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_pitch_joint",
            "right_wrist_yaw_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
        ],
        hand_joint_names=[
            "L_index_proximal_joint",
            "L_middle_proximal_joint",
            "L_pinky_proximal_joint",
            "L_ring_proximal_joint",
            "L_thumb_proximal_yaw_joint",
            "R_index_proximal_joint",
            "R_middle_proximal_joint",
            "R_pinky_proximal_joint",
            "R_ring_proximal_joint",
            "R_thumb_proximal_yaw_joint",
            "L_index_intermediate_joint",
            "L_middle_intermediate_joint",
            "L_pinky_intermediate_joint",
            "L_ring_intermediate_joint",
            "L_thumb_proximal_pitch_joint",
            "R_index_intermediate_joint",
            "R_middle_intermediate_joint",
            "R_pinky_intermediate_joint",
            "R_ring_intermediate_joint",
            "R_thumb_proximal_pitch_joint",
            "L_thumb_distal_joint",
            "R_thumb_distal_joint",
        ],
        target_eef_link_names={
            "left_wrist": "left_hand_pitch_link",
            "right_wrist": "right_hand_pitch_link",
        },
        # the robot in the sim scene we are controlling
        asset_name="robot",
        # Configuration for the IK controller
        # The frames names are the ones present in the URDF file
        # The urdf has to be generated from the USD that is being used in the scene
        controller=PinkIKControllerCfg(
            articulation_name="robot",
            base_link_name="base_link",
            num_hand_joints=22,
            show_ik_warnings=False,
            # Determines whether Pink IK solver will fail due to a joint limit violation
            fail_on_joint_limit_violation=False,
            variable_input_tasks=[
                FrameTask(
                    "GR1T2_fourier_hand_6dof_left_hand_pitch_link",
                    position_cost=8.0,  # [cost] / [m]
                    orientation_cost=1.0,  # [cost] / [rad]
                    lm_damping=12,  # dampening for solver for step jumps
                    gain=0.5,
                ),
                FrameTask(
                    "GR1T2_fourier_hand_6dof_right_hand_pitch_link",
                    position_cost=8.0,  # [cost] / [m]
                    orientation_cost=1.0,  # [cost] / [rad]
                    lm_damping=12,  # dampening for solver for step jumps
                    gain=0.5,
                ),
                DampingTask(
                    cost=0.5,  # [cost] * [s] / [rad]
                ),
                NullSpacePostureTask(
                    cost=0.5,
                    lm_damping=1,
                    controlled_frames=[
                        "GR1T2_fourier_hand_6dof_left_hand_pitch_link",
                        "GR1T2_fourier_hand_6dof_right_hand_pitch_link",
                    ],
                    controlled_joints=[
                        "left_shoulder_pitch_joint",
                        "left_shoulder_roll_joint",
                        "left_shoulder_yaw_joint",
                        "left_elbow_pitch_joint",
                        "right_shoulder_pitch_joint",
                        "right_shoulder_roll_joint",
                        "right_shoulder_yaw_joint",
                        "right_elbow_pitch_joint",
                        "waist_yaw_joint",
                        "waist_pitch_joint",
                        "waist_roll_joint",
                    ],
                ),
            ],
            fixed_input_tasks=[],
            xr_enabled=bool(carb.settings.get_settings().get("/app/xr/enabled")),
        ),
    )


# 1. 상단에 함수 추가
def get_object_pose(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return torch.cat([asset.data.root_pos_w, asset.data.root_quat_w], dim=-1)

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        robot_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        robot_root_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_root_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot")})
        object_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("object")})
        object_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("object")})
        robot_links_state = ObsTerm(func=mdp.get_all_robot_link_state)

        left_eef_pos = ObsTerm(func=mdp.get_eef_pos, params={"link_name": "left_hand_roll_link"})
        left_eef_quat = ObsTerm(func=mdp.get_eef_quat, params={"link_name": "left_hand_roll_link"})
        right_eef_pos = ObsTerm(func=mdp.get_eef_pos, params={"link_name": "right_hand_roll_link"})
        right_eef_quat = ObsTerm(func=mdp.get_eef_quat, params={"link_name": "right_hand_roll_link"})

        hand_joint_state = ObsTerm(func=mdp.get_robot_joint_state, params={"joint_names": ["R_.*", "L_.*"]})
        head_joint_state = ObsTerm(
            func=mdp.get_robot_joint_state,
            params={"joint_names": ["head_pitch_joint", "head_roll_joint", "head_yaw_joint"]},
        )

        # ObservationsCfg 내부
        object = ObsTerm(
            func=mdp.object_obs,
            params={"left_eef_link_name": "left_hand_roll_link", "right_eef_link_name": "right_hand_roll_link"}, # asset_cfg 삭제!
        )
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class DatagenInfoCfg(ObsGroup):
        object_pose = ObsTerm(
            func=get_object_pose,
            params={"asset_cfg": SceneEntityCfg("object")}
        )
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    datagen_info: DatagenInfoCfg = DatagenInfoCfg()

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("object")}
    )

    success = DoneTerm(
        func=mdp.task_done_open_microwave,
        params={
            # task_link_name과 기존 pick_place 변수들은 모두 지워줍니다!
            
            # 대신 새 함수에서 조절하고 싶은 값을 여기에 적어줍니다.
            "target_angle": -0.7,  # 문이 열리는 목표 각도 (-1.5 라디안)
            "min_vel": 0.1,        # 멈춤 판정 속도
            
            # (참고: microwave_cfg나 door_joint_name은 함수에 기본값이 있으니 
            # 여기서 굳이 안 적어줘도 알아서 작동합니다.)
        },
    )


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": [-0.01, 0.01],
                "y": [-0.01, 0.01],
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class PickPlaceGR1T2EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the GR1T2 environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()
    events = EventCfg()

    # Unused managers
    commands = None
    rewards = None
    curriculum = None

    # Position of the XR anchor in the world frame
    xr: XrCfg = XrCfg(
        anchor_pos=(0.0, 0.0, 0.0),
        anchor_rot=(1.0, 0.0, 0.0, 0.0),
    )

    # OpenXR hand tracking has 26 joints per hand
    NUM_OPENXR_HAND_JOINTS = 26

    # Temporary directory for URDF files
    temp_urdf_dir = tempfile.gettempdir()

    # Idle action to hold robot in default pose
    # Action format: [left arm pos (3), left arm quat (4), right arm pos (3), right arm quat (4),
    #                 left hand joint pos (11), right hand joint pos (11)]
    idle_action = torch.tensor(
        [
            -0.22878,
            0.2536,
            1.0953,
            0.5,
            0.5,
            -0.5,
            0.5,
            0.22878,
            0.2536,
            1.0953,
            0.5,
            0.5,
            -0.5,
            0.5,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 6
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1 / 120  # 120Hz
        self.sim.render_interval = 2

        # Convert USD to URDF and change revolute joints to fixed
        temp_urdf_output_path, temp_urdf_meshes_output_path = ControllerUtils.convert_usd_to_urdf(
            self.scene.robot.spawn.usd_path, self.temp_urdf_dir, force_conversion=True
        )

        # Set the URDF and mesh paths for the IK controller
        self.actions.upper_body_ik.controller.urdf_path = temp_urdf_output_path
        self.actions.upper_body_ik.controller.mesh_path = temp_urdf_meshes_output_path

        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        GR1T2RetargeterCfg(
                            enable_visualization=True,
                            # number of joints in both hands
                            num_open_xr_hand_joints=2 * self.NUM_OPENXR_HAND_JOINTS,
                            sim_device=self.sim.device,
                            hand_joint_names=self.actions.upper_body_ik.hand_joint_names,
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
                "manusvive": ManusViveCfg(
                    retargeters=[
                        GR1T2RetargeterCfg(
                            enable_visualization=True,
                            num_open_xr_hand_joints=2 * 26,
                            sim_device=self.sim.device,
                            hand_joint_names=self.actions.upper_body_ik.hand_joint_names,
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
            }
        )
