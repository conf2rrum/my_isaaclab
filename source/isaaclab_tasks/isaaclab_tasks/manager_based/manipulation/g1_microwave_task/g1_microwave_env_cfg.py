from __future__ import annotations

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import OpenXRDeviceCfg, XrCfg
from isaaclab.devices.openxr.retargeters.humanoid.unitree.g1_lower_body_standing import G1LowerBodyStandingRetargeterCfg
from isaaclab.devices.openxr.retargeters.humanoid.unitree.g1_motion_controller_locomotion import (
    G1LowerBodyStandingMotionControllerRetargeterCfg,
)
from isaaclab.devices.openxr.retargeters.humanoid.unitree.trihand.g1_upper_body_motion_ctrl_retargeter import (
    G1TriHandUpperBodyMotionControllerRetargeterCfg,
)
from isaaclab.devices.openxr.retargeters.humanoid.unitree.trihand.g1_upper_body_retargeter import (
    G1TriHandUpperBodyRetargeterCfg,
)
from isaaclab.devices.openxr.xr_cfg import XrAnchorRotationMode
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, retrieve_file_path

from isaaclab_assets.robots.unitree import G1_29DOF_CFG
from isaaclab_tasks.manager_based.locomanipulation.pick_place import mdp as locomanip_mdp
from isaaclab_tasks.manager_based.locomanipulation.pick_place.configs.action_cfg import AgileBasedLowerBodyActionCfg
from isaaclab_tasks.manager_based.locomanipulation.pick_place.configs.agile_locomotion_observation_cfg import (
    AgileTeacherPolicyObservationsCfg,
)
from isaaclab_tasks.manager_based.locomanipulation.pick_place.configs.pink_controller_cfg import (
    G1_UPPER_BODY_IK_ACTION_CFG,
)
from isaaclab_tasks.manager_based.manipulation.pick_place import mdp as manip_mdp

from . import microwave_mdp

# -----------------------------------------------------------------------------
# User-tunable constants
# -----------------------------------------------------------------------------
# Replace this with your own articulated microwave USD.
MICROWAVE_USD_PATH = "/workspace/assets/microwave.usd"

# Update these two names to match your microwave asset.
MICROWAVE_DOOR_JOINT = "joint_revolute_l_0_abstract_0_1"
SUCCESS_OPEN_THRESHOLD = -1.0

# Rough starter layout. Adjust after the first visual test.
TABLE_POS = (0.72, 0.0, 0.0)
MICROWAVE_POS = (0.80, 0.0, 0.82)
MICROWAVE_ROT_WXYZ = (1.0, 0.0, 0.0, 0.0)

# Keep the robot in place for stable teleop data collection.
FIX_ROBOT_BASE = True


@configclass
class G1MicrowaveSceneCfg(InteractiveSceneCfg):
    """Simple static scene for opening a microwave door with Unitree G1."""

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=TABLE_POS, rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    microwave = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Microwave",
        spawn=sim_utils.UsdFileCfg(
            usd_path=MICROWAVE_USD_PATH,
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=MICROWAVE_POS,
            rot=MICROWAVE_ROT_WXYZ,
            joint_pos={MICROWAVE_DOOR_JOINT: 0.0},
        ),
    )

    robot: ArticulationCfg = G1_29DOF_CFG.copy()

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
    """Action specs.

    We reuse the official G1 upper-body Pink IK action plus the standing lower-body
    action used in the locomanipulation demo, because that already matches Isaac Lab's
    OpenXR retargeters for Meta Quest hand tracking / controllers.
    """

    upper_body_ik = G1_UPPER_BODY_IK_ACTION_CFG

    lower_body_joint_pos = AgileBasedLowerBodyActionCfg(
        asset_name="robot",
        joint_names=[
            ".*_hip_.*_joint",
            ".*_knee_joint",
            ".*_ankle_.*_joint",
        ],
        policy_output_scale=0.25,
        obs_group_name="lower_body_policy",
        policy_path=f"{ISAACLAB_NUCLEUS_DIR}/Policies/Agile/agile_locomotion.pt",
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=manip_mdp.last_action)
        robot_joint_pos = ObsTerm(func=base_mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_root_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_root_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot")})

        right_eef_pos = ObsTerm(func=manip_mdp.get_eef_pos, params={"link_name": "right_wrist_yaw_link"})
        right_eef_quat = ObsTerm(func=manip_mdp.get_eef_quat, params={"link_name": "right_wrist_yaw_link"})
        left_eef_pos = ObsTerm(func=manip_mdp.get_eef_pos, params={"link_name": "left_wrist_yaw_link"})
        left_eef_quat = ObsTerm(func=manip_mdp.get_eef_quat, params={"link_name": "left_wrist_yaw_link"})

        hand_joint_state = ObsTerm(func=manip_mdp.get_robot_joint_state, params={"joint_names": [".*_hand.*"]})

        microwave_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("microwave", joint_names=[MICROWAVE_DOOR_JOINT])},
        )
        microwave_joint_vel = ObsTerm(
            func=base_mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("microwave", joint_names=[MICROWAVE_DOOR_JOINT])},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    lower_body_policy: AgileTeacherPolicyObservationsCfg = AgileTeacherPolicyObservationsCfg()


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=locomanip_mdp.time_out, time_out=True)
    success = DoneTerm(
        func=microwave_mdp.joint_opened,
        params={
            "asset_cfg": SceneEntityCfg("microwave", joint_names=[MICROWAVE_DOOR_JOINT]),
            "threshold": SUCCESS_OPEN_THRESHOLD,
        },
    )


@configclass
class G1OpenMicrowaveEnvCfg(ManagerBasedRLEnvCfg):
    scene: G1MicrowaveSceneCfg = G1MicrowaveSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands = None
    rewards = None
    curriculum = None
    terminations: TerminationsCfg = TerminationsCfg()

    xr: XrCfg = XrCfg(
        anchor_pos=(0.0, 0.0, -0.35),
        anchor_rot=(1.0, 0.0, 0.0, 0.0),
    )

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.viewer.eye = (1.8, 1.4, 1.4)
        self.viewer.lookat = (0.65, 0.0, 0.85)

        self.sim.dt = 1 / 200
        self.sim.render_interval = 2

        # Fix the robot base to make teleop collection much more stable for a first version.
        self.scene.robot = self.scene.robot.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.articulation_props.fix_root_link = FIX_ROBOT_BASE
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.75)
        self.scene.robot.init_state.rot = (0.7071, 0.0, 0.0, 0.7071)
        self.scene.robot.init_state.joint_pos.update(
            {
                "right_shoulder_pitch_joint": 0.35,
                "right_elbow_joint": 0.75,
                "left_shoulder_pitch_joint": 0.20,
                "left_elbow_joint": 0.45,
                ".*_hip_pitch_joint": -0.10,
                ".*_knee_joint": 0.30,
                ".*_ankle_pitch_joint": -0.20,
            }
        )

        urdf_omniverse_path = (
            f"{ISAACLAB_NUCLEUS_DIR}/Controllers/LocomanipulationAssets/"
            "unitree_g1_kinematics_asset/g1_29dof_with_hand_only_kinematics.urdf"
        )
        self.actions.upper_body_ik.controller.urdf_path = retrieve_file_path(urdf_omniverse_path)

        self.xr.anchor_prim_path = "/World/envs/env_0/Robot/pelvis"
        self.xr.fixed_anchor_height = True
        self.xr.anchor_rotation_mode = XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED

        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        G1TriHandUpperBodyRetargeterCfg(
                            enable_visualization=True,
                            num_open_xr_hand_joints=2 * 26,
                            sim_device=self.sim.device,
                            hand_joint_names=self.actions.upper_body_ik.hand_joint_names,
                        ),
                        G1LowerBodyStandingRetargeterCfg(sim_device=self.sim.device),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
                "motion_controllers": OpenXRDeviceCfg(
                    retargeters=[
                        G1TriHandUpperBodyMotionControllerRetargeterCfg(
                            enable_visualization=True,
                            sim_device=self.sim.device,
                            hand_joint_names=self.actions.upper_body_ik.hand_joint_names,
                        ),
                        G1LowerBodyStandingMotionControllerRetargeterCfg(sim_device=self.sim.device),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
            }
        )


@configclass
class G1OpenMicrowaveEnvCfg_PLAY(G1OpenMicrowaveEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


