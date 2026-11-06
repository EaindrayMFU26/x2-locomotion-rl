"""Shared X2 policy/robot constants — single source of truth for sim2sim + deploy.

Consumed by `rl/sim2sim/sim2sim_x2.py` and `x2_deploy_pkg/deploy_x2.py` (both used to
copy these verbatim). Values mirror the training asset config
(`agibot_rl_lab/.../assets/robots/agibot.py`) and the exported `deploy.yaml`.

Only values that are IDENTICAL between sim2sim and deploy live here. Intentionally
divergent knobs stay local to each script:
  - GRAVITY_PITCH_LIMIT   (sim 0.72 / deploy 1.0)
  - simulation timestep    (sim SIM_DT=0.001+DECIMATION=20 / deploy CONTROL_DT=0.005)
  - ACTION_DELTA_LIMIT     (deploy-only rate limit)
  - MUJOCO_JOINT_NAMES     (sim-only) / JOINT_AREAS + topics (deploy-only)
"""

# --- Joint orderings -------------------------------------------------------
# Observation joint order (31): matches training obs joint_pos_rel/joint_vel_rel.
ISAACLAB_JOINT_ORDER = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_pitch_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_roll_joint",
    "left_knee_joint",
    "right_knee_joint",
    "head_yaw_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "head_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
]

# Action joint order (29): policy output order (excludes head_yaw/head_pitch).
ACTION_JOINT_ORDER = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_pitch_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_roll_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
]

# --- Default joint positions ----------------------------------------------
# RL baseline: obs joint_pos_rel subtracts this, action offset adds it.
DEFAULT_JOINT_POS = {
    "left_hip_pitch_joint": -0.312, "right_hip_pitch_joint": -0.312,
    "left_knee_joint": 0.669, "right_knee_joint": 0.669,
    "left_ankle_pitch_joint": -0.363, "right_ankle_pitch_joint": -0.363,
    "left_shoulder_pitch_joint": 0.0, "right_shoulder_pitch_joint": 0.0,
    "left_shoulder_roll_joint": 0.2, "right_shoulder_roll_joint": -0.2,
    "left_elbow_joint": -0.3, "right_elbow_joint": -0.3,
}

# JOINT_DEFAULT (half-squat prep) pose: full-body target for the JOINT_DEFAULT FSM state.
JOINT_DEFAULT_POS = {
    "left_hip_pitch_joint": -0.05, "right_hip_pitch_joint": -0.05,
    "left_hip_roll_joint": 0.0, "right_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0, "right_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.1, "right_knee_joint": 0.1,
    "left_ankle_pitch_joint": -0.05, "right_ankle_pitch_joint": -0.05,
    "left_ankle_roll_joint": 0.0, "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0, "waist_pitch_joint": 0.0, "waist_roll_joint": 0.0,
    "left_shoulder_pitch_joint": 0.4, "right_shoulder_pitch_joint": 0.4,
    "left_shoulder_roll_joint": 0.0, "right_shoulder_roll_joint": 0.0,
    "left_shoulder_yaw_joint": 0.0, "right_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": -1.2, "right_elbow_joint": -1.2,
    "left_wrist_yaw_joint": 0.0, "right_wrist_yaw_joint": 0.0,
    "left_wrist_pitch_joint": 0.0, "right_wrist_pitch_joint": 0.0,
    "left_wrist_roll_joint": 0.0, "right_wrist_roll_joint": 0.0,
    "head_yaw_joint": 0.0, "head_pitch_joint": 0.0,
}

# --- Observation spec ------------------------------------------------------
OBS_TERM_DIMS = {
    "base_ang_vel": 3,
    "projected_gravity": 3,
    "velocity_commands": 3,
    "joint_pos_rel": 31,
    "joint_vel_rel": 31,
    "last_action": 29,
}

# --- Control / policy scalars ---------------------------------------------
ACTION_SCALE = 0.25            # policy action -> joint target delta scale
POLICY_DT = 0.02               # policy inference period (50 Hz)
HISTORY_LENGTH = 5             # obs history frames stacked
JOINT_DEFAULT_DURATION = 1.5   # JOINT_DEFAULT interpolation duration (s)

# --- Observation scaling / clipping (must match training obs terms) --------
ANG_VEL_SCALE = 0.25           # base_ang_vel obs scale
JOINT_VEL_SCALE = 0.05         # joint_vel_rel obs scale
OBS_CLIP = 100.0               # obs clamp to [-OBS_CLIP, OBS_CLIP] before policy
ACTION_CLIP = 5.0              # raw policy output clamp to [-ACTION_CLIP, ACTION_CLIP]
