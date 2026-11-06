import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

# Joint PD gains (stiffness/damping): single source of truth in x2_config/joint_gains.yaml
import sys as _sys
from pathlib import Path as _Path
_d = _Path(__file__).resolve().parent
_cands = [_d] + [p / "x2_config" for p in [_d, *_d.parents]]
_gdir = next((c for c in _cands if (c / "joint_gains.py").exists()), None)
if _gdir is None:
    raise FileNotFoundError("joint_gains.py not found (expected in x2_config/)")
_sys.path.insert(0, str(_gdir))
from joint_gains import load_gains as _load_gains
_GAINS = _load_gains()
_KP, _KD = _GAINS["rl_kp"], _GAINS["rl_kd"]

X2_URDF_DIR = os.path.join(os.path.dirname(__file__), "x2_description")


@configclass
class AgibotArticulationCfg(ArticulationCfg):
    joint_sdk_names: list[str] = None
    soft_joint_pos_limit_factor = 0.9


@configclass
class AgibotUrdfFileCfg(sim_utils.UrdfFileCfg):
    fix_base: bool = False
    activate_contact_sensors: bool = True
    replace_cylinders_with_capsules = True
    joint_drive = sim_utils.UrdfConverterCfg.JointDriveCfg(
        gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
    )
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=4,
    )
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    )

    def replace_asset(self, meshes_dir, urdf_path):
        tmp_meshes_dir = "/tmp/IsaacLab/agibot_rl_lab/meshes"
        if os.path.exists(tmp_meshes_dir):
            os.remove(tmp_meshes_dir)
        os.makedirs("/tmp/IsaacLab/agibot_rl_lab", exist_ok=True)
        os.symlink(meshes_dir, tmp_meshes_dir)

        self.asset_path = "/tmp/IsaacLab/agibot_rl_lab/robot.urdf"
        if os.path.exists(self.asset_path):
            os.remove(self.asset_path)
        os.symlink(urdf_path, self.asset_path)


AGIBOT_X2_29DOF_CFG = AgibotArticulationCfg(
    spawn=AgibotUrdfFileCfg(
        asset_path=f"{X2_URDF_DIR}/x2_ultra_simple_collision.urdf",
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.95),
        joint_pos={
            ".*_hip_pitch_joint": -0.312,
            ".*_knee_joint": 0.669,
            ".*_ankle_pitch_joint": -0.363,
            ".*_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.2,
            "right_shoulder_roll_joint": -0.2,
            ".*_elbow_joint": -0.3,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    # WARNING: these are SIM gains; if real-robot calibrated gains are obtained,
    #   update here + deploy_x2.py and retrain.
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit={
                ".*_hip_yaw_joint": 120.0,
                ".*_hip_roll_joint": 120.0,
                ".*_hip_pitch_joint": 120.0,
                ".*_knee_joint": 120.0,
            },
            velocity_limit_sim=12.0,
            stiffness={j: _KP[j] for j in (
                "left_hip_pitch_joint", "right_hip_pitch_joint",
                "left_hip_roll_joint", "right_hip_roll_joint",
                "left_hip_yaw_joint", "right_hip_yaw_joint",
                "left_knee_joint", "right_knee_joint",
            )},
            damping={j: _KD[j] for j in (
                "left_hip_pitch_joint", "right_hip_pitch_joint",
                "left_hip_roll_joint", "right_hip_roll_joint",
                "left_hip_yaw_joint", "right_hip_yaw_joint",
                "left_knee_joint", "right_knee_joint",
            )},
            armature=0.01,
            # sim2real: explicit PD + command delay models real command->torque latency.
            # Unit = physics steps (sim.dt=0.005 -> 1 step=5ms). [1,6]=5-30ms, sampled per env reset.
            # min_delay=1 (not 0): zero delay is the sim2real gap itself, so every env has latency.
            min_delay=1,
            max_delay=6,
        ),
        "feet": DelayedPDActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit={
                ".*_ankle_pitch_joint": 36.0,
                ".*_ankle_roll_joint": 24.0,
            },
            velocity_limit_sim=15.0,
            stiffness={j: _KP[j] for j in (
                "left_ankle_pitch_joint", "right_ankle_pitch_joint",
                "left_ankle_roll_joint", "right_ankle_roll_joint",
            )},
            damping={j: _KD[j] for j in (
                "left_ankle_pitch_joint", "right_ankle_pitch_joint",
                "left_ankle_roll_joint", "right_ankle_roll_joint",
            )},
            armature=0.01,
            min_delay=1,
            max_delay=6,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
            effort_limit_sim=48.0,
            velocity_limit_sim=13.0,
            stiffness={j: _KP[j] for j in (
                "waist_roll_joint", "waist_pitch_joint",
            )},
            damping={j: _KD[j] for j in (
                "waist_roll_joint", "waist_pitch_joint",
            )},
            armature=0.01,
        ),
        "waist_yaw": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint"],
            effort_limit_sim=120.0,
            velocity_limit_sim=12.0,
            stiffness={j: _KP[j] for j in ("waist_yaw_joint",)},
            damping={j: _KD[j] for j in ("waist_yaw_joint",)},
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 36.0,
                ".*_shoulder_roll_joint": 36.0,
                ".*_shoulder_yaw_joint": 24.0,
                ".*_elbow_joint": 24.0,
                ".*_wrist_roll_joint": 4.8,
                ".*_wrist_pitch_joint": 4.8,
                ".*_wrist_yaw_joint": 24.0,
            },
            velocity_limit_sim=15.0,
            stiffness={j: _KP[j] for j in (
                "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
                "left_shoulder_roll_joint", "right_shoulder_roll_joint",
                "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
                "left_elbow_joint", "right_elbow_joint",
                "left_wrist_roll_joint", "right_wrist_roll_joint",
                "left_wrist_pitch_joint", "right_wrist_pitch_joint",
                "left_wrist_yaw_joint", "right_wrist_yaw_joint",
            )},
            damping={j: _KD[j] for j in (
                "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
                "left_shoulder_roll_joint", "right_shoulder_roll_joint",
                "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
                "left_elbow_joint", "right_elbow_joint",
                "left_wrist_roll_joint", "right_wrist_roll_joint",
                "left_wrist_pitch_joint", "right_wrist_pitch_joint",
                "left_wrist_yaw_joint", "right_wrist_yaw_joint",
            )},
            armature=0.01,
        ),
    },
    # fmt: off
    joint_sdk_names=[
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_pitch_joint",
        "waist_roll_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_yaw_joint",
        "left_wrist_pitch_joint",
        "left_wrist_roll_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_yaw_joint",
        "right_wrist_pitch_joint",
        "right_wrist_roll_joint",
        "head_yaw_joint",
        "head_pitch_joint",
    ],
    # fmt: on
)
