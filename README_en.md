# AgiBot X2 Locomotion RL Training & Validation

**English** | [简体中文](README.md)

> Full-stack open-source code for AgiBot X2 humanoid algorithm modules — Locomotion module

The shipped model `x2_deploy_pkg/models/policy.onnx` (the default in-repo model) has passed on-robot validation across the full velocity range: \|a\| matches sim2sim expectations, no ESTOP triggered, no jitter.

| Dimension | Validated range | Result |
|---|---|---|
| cmd_vx (forward / backward) | −0.3 ~ 0.5 m/s full range | ✅ Stable, incl. standing still & high speed |
| cmd_vy (A/D lateral) | ±0.3 m/s | ✅ Stable |
| cmd_vyaw (Q/E turn) | ±0.2 rad/s | ✅ Stable |
| Composite (vx + vyaw) | combinations within the above | ✅ Stable |
| Speed switching (continuous key adjust) | within the above | ✅ Stable |
| Long-duration run | continuous | ✅ Stable |

---

## Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Robot Configuration](#robot-configuration)
- [Technical Architecture](#technical-architecture)
- [Directory Structure](#directory-structure)
- [Environment Dependencies](#environment-dependencies)
- [License and Third-Party Components](#license-and-third-party-components)

---

## Overview

This project delivers the complete RL training and simulation-validation pipeline for the AgiBot X2 (lx2501 / T2.5, 29-DoF humanoid) locomotion module, and has been deployed on the real robot:

```
X2 URDF → Isaac Lab config → PPO training → ONNX export → MuJoCo Sim2Sim validation → on-robot deployment
```

---

## Quick Start

> The shortest path to get going.

### First-time install

Unitree's `unitree_rl_lab` is now a **submodule** under `third_party/` (read-only, pinned at 4960b84, reusing its mdp); AgiBot X2 code is a standalone `rl/agibot_rl_lab` package.

```bash
git clone --recursive <repo>            # if already cloned: git submodule update --init
conda activate isaaclab
pip install -e third_party/unitree_rl_lab/source/unitree_rl_lab   # Unitree (read-only, reuse its mdp)
pip install -e rl/agibot_rl_lab/source/agibot_rl_lab              # AgiBot X2 (robot/tasks/PPO/scripts)
```

### Training

```bash
conda activate isaaclab
export ISAACLAB_PATH=~/IsaacLab
cd ~/mc-rl/rl/agibot_rl_lab
python scripts/rsl_rl/train.py --task Agibot-X2-29dof-Velocity --num_envs 4096 --headless
```

Training takes ~15-19 h (RTX 5090, iterating to 40k-50k; `max_iterations` defaults to 50000). Checkpoint selection & ONNX/JIT export: for final/latest, use `rl/agibot_rl_lab/scripts/rsl_rl/play.py --export-only` (auto-export); to pick the best (by `Train/mean_reward` moving average), use `rl/train/export_best.py`. Policy network: 500 → 512 → 256 → 128 → 29 (ELU hidden layers).

### MuJoCo Sim2Sim validation

```bash
conda activate isaaclab
cd ~/mc-rl

python rl/sim2sim/sim2sim_x2.py --no-fsm --cmd_vx 0.3            # skip FSM, go straight to RL walking
python rl/sim2sim/sim2sim_x2.py                                   # full FSM flow (WASD+QE real-time keyboard control)
MUJOCO_GL=egl python rl/sim2sim/sim2sim_x2.py --no-fsm --cmd_vx 0.3 --record sim_walk.mp4 --duration 10  # headless recording
```

The default model is the shipped deployment model `x2_deploy_pkg/models/policy.onnx` (set by `ONNX_PATH` in `x2_config/sim/sim_config.py`); to evaluate a model exported from a specific training run, pass `--onnx <run>/exported/policy.onnx` explicitly. Acceptance criteria: at cmd_vx=0.2, ~meter-scale displacement over 10 s (actual ~0.16 m/s), steady-state \|a\| ≈ 3.2–3.7, pelvis height stable around 0.68. `sim2sim_x2.py` also supports two fidelity tests: external-force disturbance (`I/J/K/L/U` + `1`–`5`) and IMU pitch bias (`--imu_pitch_bias`).

### On-robot Sim2Real deployment

Requires a ROS2 + aimdk_msgs environment (the industrial-PC path). First use `deploy_to_board.sh` to scp the deployment files to the board, then run through the FSM flow on the board:

| Step | Command | Notes |
|------|------|------|
| Step 0 | `bash x2_deploy_pkg/deploy_to_board.sh <robot-user> <robot-ip>` | One-command sync of deploy files to the board + model md5 double-check + remote env_check |
| Step 0.5 | `ssh <robot-user>@<robot-ip> "aima em get-app mc"` | Check mc state: Running → next `stop-app mc`; Idle → `aima em start-app mc && sleep 2 && aima em stop-app mc` |
| Step 1 | `python3 deploy_x2.py` | Enter FSM: damping → half-squat → stand → RL stand-in-place (DAMPING/STANDUP folded into the state machine; no separate test scripts) |
| Step 2 | `python3 deploy_x2.py` (press W to walk) | RL walking (first W jumps to initial_cmd_vx=0.2, then W/S to accel/decel) |
| Teardown | `ssh <robot-user>@<robot-ip> "aima em start-app mc"` | Start the robot's built-in MC, restore normal operation |

FSM (deploy_x2.py): `WAITING → DAMPING → (Enter/Space) → JOINT_DEFAULT → STANDUP → (1.5s) → RL_ACTIVE`. Damping softening and the two-stage stand-up are handled inside the state machine (using damping_kd / standup gains from `x2_config/joint_gains.yaml`).
Keyboard control (RL_ACTIVE): WASD = move (first W jumps from 0 to `initial_cmd_vx=0.2`, vx clamped to [-0.3, 0.5]), QE = turn (±0.2), A/D = lateral (±0.3), Space = stand in place, Backspace = emergency return to DAMPING, Esc = exit.
RC gamepad control (optional): add `--enable_rc_proxy` at launch to drive cmd_vx/vy/vyaw with the stock gamepad sticks instead of keyboard WASD+QE; gamepad combos also advance the FSM (`L2+X`→JOINT_DEFAULT, `L2+↑`→STANDUP, `L1+R1+Create button`→back to DAMPING). Params: `--rc_velocity_topic` (default `/aima/mc/locomotion/velocity`), `--rc_timeout` (default 0.5 s; zeroes velocity on timeout). Requires the built-in MC already stopped and the stock RC still running.
Safety: damping start, cosine-interpolated stand-up, action clip ±5.0, action delta clip ±0.5, `DAMPING_KD=4.0`, tilt fallback auto-ESTOP (deploy `GRAVITY_PITCH_LIMIT=1.0`, triggers only when the gravity-projection component nears its ~90° limit — an extreme fallback only; sim2sim uses 0.72 ≈ 46°), Backspace e-stop, Ctrl+C safe exit.

---

## Robot Configuration

> All reference tables below are cross-checked against source / URDF / MJCF; the values are core references — do not change arbitrarily.

### Joint definitions (31 joints, 29 in RL)

| Group | Joint | Count | Kp | Kd | effort |
|----|------|------|----|----|--------|
| Leg | hip_pitch | 2 | 120 | 5 | 120 |
| Leg | hip_roll | 2 | 100 | 4 | 120 |
| Leg | hip_yaw | 2 | 100 | 4 | 120 |
| Leg | knee | 2 | 150 | 5 | 120 |
| Ankle | ankle_pitch | 2 | 40 | 2 | 36 |
| Ankle | ankle_roll | 2 | 40 | 2 | 24 |
| Waist | waist_yaw | 1 | 40.18 | 2.56 | 120 |
| Waist | waist_pitch/roll | 2 | 200 | 2 | 48 |
| Shoulder/Elbow | shoulder_*/elbow | 8 | 50 | 3 | 24-36 |
| Wrist | wrist_* | 6 | 20 | 2 | 4.8-24 |
| Head | head_yaw/pitch | 2 | 0 | 2 | —(not in RL; KD=2 is only a joint_gains.yaml placeholder) |

> KP/KD single source of truth: `x2_config/joint_gains.yaml` (loaded by `joint_gains.py`), shared by training `agibot.py` / sim2sim / deploy `deploy_x2.py` — change once, all three stay in sync. `rl_gains` (used by RL_ACTIVE + STANDUP) are **simulation values, not hardware-measured**; after obtaining hardware system-identification values, edit that YAML and retrain. `standup_gains` (JOINT_DEFAULT) and `damping_kd` are measured values. effort comes from the X2 URDF / MJCF.

### Observations (500-dim)

Single-frame obs = 100 dims × 5-frame history = 500 dims:

| Obs term | Dim | scale | Source |
|--------|------|-------|------|
| base_ang_vel | 3 | 0.25 | IMU gyroscope |
| projected_gravity | 3 | 1.0 | IMU accelerometer |
| velocity_commands | 3 | 1.0 | gamepad / host |
| joint_pos_rel | 31 | 1.0 | encoder (relative to default position) |
| joint_vel_rel | 31 | 0.05 | encoder |
| last_action | 29 | 1.0 | self-memory |

History concatenation: per-term (each obs term keeps its own CircularBuffer, oldest first).

### Action space (29-dim)

`target_pos = action × 0.25 + default_pos`

| Environment | sim_dt | PD freq | decimation | Policy freq |
|------|--------|---------|------------|---------|
| Isaac Lab training | 0.005s | 200 Hz | 4 | 50 Hz |
| MuJoCo Sim2Sim | 0.001s | 1000 Hz | 20 | 50 Hz |
| Real robot | ~0.001s | ~1000 Hz (driver inner loop) | 200 Hz command (`CONTROL_DT=0.005`) | 50 Hz |

### Isaac Lab joint ordering

URDF pattern matching produces an interleaved order (left/right alternating), different from the URDF definition order:

| idx | Joint | idx | Joint | idx | Joint |
|-----|------|-----|------|-----|------|
| 0 | L_hip_pitch | 11 | head_yaw | 21 | L_shoulder_yaw |
| 1 | R_hip_pitch | 12 | L_shoulder_pitch | 22 | R_shoulder_yaw |
| 2 | waist_yaw | 13 | R_shoulder_pitch | 23 | L_elbow |
| 3 | L_hip_roll | 14 | L_ankle_pitch | 24 | R_elbow |
| 4 | R_hip_roll | 15 | R_ankle_pitch | 25 | L_wrist_yaw |
| 5 | waist_pitch | 16 | head_pitch | 26 | R_wrist_yaw |
| 6 | L_hip_yaw | 17 | L_shoulder_roll | 27 | L_wrist_pitch |
| 7 | R_hip_yaw | 18 | R_shoulder_roll | 28 | R_wrist_pitch |
| 8 | waist_roll | 19 | L_ankle_roll | 29 | L_wrist_roll |
| 9 | L_knee | 20 | R_ankle_roll | 30 | R_wrist_roll |
| 10 | R_knee | | | | |

Action = drop idx 11 (head_yaw) and idx 16 (head_pitch), leaving 29.

### Training recipe (key reward weights)

Configured in `velocity_env_cfg.py`, reusing unitree_rl_lab's mdp:

| Item | Value | Purpose |
|---|------|------|
| `track_lin_vel_xy` (std) | √0.1 ≈ 0.316 | Linear-velocity tracking (smaller std = stricter, helps low-speed) |
| `track_ang_vel_z` | 1.0 | Yaw-rate tracking |
| `base_angular_velocity` | −0.15 | Suppress torso sway |
| `action_rate` | −0.1 | Smooth actions, suppress jitter |
| `joint_deviation_arms` | −0.3 | Constrain arms, reduce irrelevant swing |
| `base_height` (target) | 0.68 | Target pelvis standing height |
| Termination `base_height` (min) | 0.3 | Below this height = fall, early termination |

Other reward terms keep the mdp defaults. The X2 recipe adds training-time IMU pitch-bias domain randomization on the projected-gravity observation (local mdp extension `projected_gravity_with_bias`). Sweep weights in batch with `rl/train/batch_train.py` (`EXPERIMENTS` list at the top, `MAX_ITER=40000` per group).

---

## Technical Architecture

- **rsl_rl 5.0 adaptation**: Actor/Critic are separate `MLPModel` (`RslRlMLPModelCfg`); ONNX export uses the built-in `MLPModel.as_onnx()`; requires `handle_deprecated_rsl_rl_cfg()` for legacy compatibility.
- **Obs history concatenation**: Isaac Lab uses **per-term history** — each obs term keeps its own `CircularBuffer` (oldest first); with `flatten_history_dim=True` it is reshaped to `(batch, history_len * dim)` and finally concatenated.
- **Isaac Lab joint ordering**: URDF regex pattern-matching expansion produces the interleaved order, hard-coded in `deploy_x2.py`'s `ISAACLAB_JOINT_ORDER/ACTION_JOINT_ORDER`.
- **ROS2 topic interface**:

| Topic | Type | Direction | Notes |
|-------|------|------|------|
| `/aima/hal/joint/{leg,waist,arm,head}/state` | `JointStateArray` | subscribe | joint states |
| `/aima/hal/joint/{leg,waist,arm,head}/command` | `JointCommandArray` | publish | joint commands |
| `/aima/hal/imu/torso/state` | `sensor_msgs/Imu` | subscribe | torso IMU (the "torso" topic is actually at the hip/pelvis, used in training) |

QoS: state subscriptions use `BEST_EFFORT`, command publishers use `RELIABLE`; `_sensors_ready()` only requires the leg/waist/arm joint states + IMU to be ready (head not required). `deploy_x2.py` sends/receives topics directly via rclpy, not `ros2 service call` (which is unresponsive on X2).

---

## Directory Structure

```
mc-rl/
├── README.md                    ← Chinese README (overview / reference / index)
├── README_en.md                 ← this file (English)
│
├── rl/                          ← training / simulation validation
│   ├── agibot_rl_lab/           ← AgiBot X2 training extension package (IsaacLab extension, pip install -e)
│   │   ├── scripts/rsl_rl/
│   │   │   ├── train.py         ← training entry point
│   │   │   ├── play.py          ← inference + ONNX export
│   │   │   └── cli_args.py      ← training / inference CLI args
│   │   └── source/agibot_rl_lab/agibot_rl_lab/
│   │       ├── assets/robots/agibot.py                              ← X2 ArticulationCfg (KP/KD/effort)
│   │       ├── tasks/locomotion/robots/x2/29dof/velocity_env_cfg.py ← training env config (reward/obs/termination)
│   │       └── tasks/locomotion/agents/rsl_rl_ppo_cfg.py            ← X2 PPO (rsl_rl 5.0 MLPModel)
│   │       (logs output to rl/agibot_rl_lab/logs/..., gitignored locally)
│   ├── sim2sim/
│   │   └── sim2sim_x2.py        ← MuJoCo Sim2Sim validation (viewer/record/fidelity test bench)
│   └── train/
│       ├── batch_train.py       ← unattended batch training (runs multiple reward-weight sets in sequence)
│       └── export_best.py       ← pick highest mean_reward checkpoint from tensorboard and export ONNX+JIT
│
├── x2_deploy_pkg/               ← on-robot deployment package (runtime files scp'd to the board; also ships x2_config/)
│   ├── deploy_x2.py             ← on-robot deploy (ROS2 FSM, 500 Hz; DAMPING/STANDUP stages + optional RC gamepad)
│   ├── env_check.sh             ← pre-deploy environment self-check (incl. optional RC checks)
│   ├── deploy_to_board.sh       ← one-command deploy: scp + model md5 double-check + remote env_check
│   └── models/policy.onnx       ← deployment model
│
├── x2_config/                   ← shared config layer (gains / policy constants / keyboard / sim & deploy config; shipped with the deploy package)
│   ├── joint_gains.py           ← gain loader (shared by agibot.py / deploy_x2.py / sim2sim)
│   ├── joint_gains.yaml         ← rl/standup gains + damping_kd (single source of truth)
│   ├── qos_override.yaml        ← rosbag QoS override
│   ├── shared/                  ← shared by sim2sim and deploy
│   │   ├── policy_defs.py       ← joint order / default pose / obs spec / scale & clip constants
│   │   └── keyboard.py          ← keyboard teleop base (KeyListener + velocity stepping)
│   ├── sim/
│   │   └── sim_config.py        ← MuJoCo paths / joint order / sim-only scalars (used by sim2sim)
│   └── deploy/
│       ├── deploy_config.py     ← ROS/HAL topic mapping + deploy-only scalars
│       └── rc_control_proxy.py  ← RC gamepad bridge (subscribes velocity + impersonates SetMcAction/GetMcAction service shims)
│
├── third_party/
│   └── unitree_rl_lab/          ← Unitree training framework (submodule, read-only, pinned 4960b84; X2 reuses its mdp)
│
└── x2_urdf/                     ← X2 URDF + MuJoCo XML
```

---

## Environment Dependencies

Two **mutually exclusive** environments (root cause: ROS2 Humble binds system Python 3.10 while IsaacLab requires Python 3.11 — they cannot coexist in one interpreter):

- **Training machine** (conda `isaaclab`, Python 3.11): IsaacSim 5.1.0 + IsaacLab 2.3.2 + rsl_rl 5.0.1 + MuJoCo ≥3.0 + PyTorch(CUDA) + onnxruntime, NVIDIA driver ≥580, RTX 5090. Runs training / export / sim2sim.
- **Deployment machine** (system Python 3.10, no conda/IsaacLab): ROS2 Humble + aimdk_msgs + onnxruntime ≥1.16 + numpy. Runs `deploy_x2.py`.

The only artifact passed between the two environments is the exported `x2_deploy_pkg/models/policy.onnx` (produced on the training machine, loaded on the deployment machine). Pre-deploy self-check: `x2_deploy_pkg/env_check.sh`.

---

## License and Third-Party Components

This project is licensed under the **Apache License 2.0**; full text in [LICENSE](LICENSE) at the repo root.

Copyright 2026 AgiBot. *(Publishing entity / copyright attribution subject to the company's final confirmation.)*

This repository contains or derives from the following third-party components; each is governed by the license declaration within its own directory:

| Component | License | Notes |
|------|--------|------|
| `third_party/unitree_rl_lab/` (submodule) | Apache-2.0 | Unitree [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab) (pinned 4960b84, read-only); see `third_party/unitree_rl_lab/LICENCE` |
| `rl/agibot_rl_lab/` | Apache-2.0 | AgiBot X2 extension package; its scripts/PPO derive from Unitree's unitree_rl_lab above |
| IsaacLab (training dependency) | BSD-3-Clause | see `third_party/unitree_rl_lab/doc/licenses/isaaclab-license.txt` |
| ONNX Runtime | MIT | see `third_party/unitree_rl_lab/doc/licenses/onnxruntime-license.txt`; runtime binaries are not committed — fetch via `third_party/unitree_rl_lab/deploy/thirdparty/fetch_onnxruntime.sh` |
| `x2_urdf/` | — | AgiBot X2 robot URDF assets |
