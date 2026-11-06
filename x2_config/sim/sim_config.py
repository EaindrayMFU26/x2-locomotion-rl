"""Sim2sim-specific config for `rl/sim2sim/sim2sim_x2.py`.

MuJoCo model paths + joint order + sim-only scalars. Kept out of sim2sim_x2.py so
the script body stays logic-only. Dev-only (ships with x2_config/ but unused on the
robot). Divergent-from-deploy scalars (GRAVITY_PITCH_LIMIT, SIM_DT) intentionally live here.
"""

import os

from policy_defs import POLICY_DT

# repo root = x2_config/../ (this file lives in x2_config/sim/)
_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

ONNX_PATH = os.path.join(_REPO_ROOT, "x2_deploy_pkg/models/policy.onnx")
SCENE_XML = os.path.join(_REPO_ROOT, "x2_urdf/X2_URDF-v1.3.0/scene.xml")

MUJOCO_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
    "head_yaw_joint", "head_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint",
]

GRAVITY_PITCH_LIMIT = 0.72  # sim-specific: sin(0.8rad~46deg), aligns training bad_orientation limit_angle=0.8 (deploy uses 1.0)
SIM_DT = 0.001
DECIMATION = int(POLICY_DT / SIM_DT)
