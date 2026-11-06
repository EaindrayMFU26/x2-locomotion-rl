"""Deploy-specific config for `x2_deploy_pkg/deploy_x2.py`.

ROS/HAL transport map + deploy-only scalars. Kept out of deploy_x2.py so the
script body stays logic-only. Ships to the robot inside x2_config/ (deploy_x2.py
adds x2_config/shared + x2_config/deploy to sys.path). Divergent-from-sim scalars
(e.g. GRAVITY_PITCH_LIMIT) intentionally live here, not in shared policy_defs.
"""

ONNX_PATH = "models/policy.onnx"

JOINT_AREAS = {
    "leg": {
        "state_topic": "/aima/hal/joint/leg/state",
        "cmd_topic": "/aima/hal/joint/leg/command",
        "joints": [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        ],
    },
    "waist": {
        "state_topic": "/aima/hal/joint/waist/state",
        "cmd_topic": "/aima/hal/joint/waist/command",
        "joints": [
            "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
        ],
    },
    "arm": {
        "state_topic": "/aima/hal/joint/arm/state",
        "cmd_topic": "/aima/hal/joint/arm/command",
        "joints": [
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint", "left_elbow_joint",
            "left_wrist_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint", "right_elbow_joint",
            "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint",
        ],
    },
    "head": {
        "state_topic": "/aima/hal/joint/head/state",
        "cmd_topic": "/aima/hal/joint/head/command",
        "joints": [
            "head_yaw_joint", "head_pitch_joint",
        ],
    },
}

ALL_JOINT_NAMES = [j for area in JOINT_AREAS.values() for j in area["joints"]]

IMU_TOPIC = "/aima/hal/imu/torso/state"

GRAVITY_PITCH_LIMIT = 1.0   # deploy-specific tilt limit (sim uses 0.72)
ACTION_DELTA_LIMIT = 0.5    # per-tick action rate limit
CONTROL_DT = 0.005          # motor command loop period (200 Hz)
