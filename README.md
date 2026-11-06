# 灵犀 X2 Locomotion RL 训练与验证

[English](README_en.md) | **简体中文**

> 灵犀 X2 人形机器人各算法模块全链路代码开源 —— Locomotion 模块

部署模型 `x2_deploy_pkg/models/policy.onnx`（随仓默认模型）已在全速度维度实机验证通过：\|a\| 与 sim2sim 预期吻合，未触发 ESTOP，无震颤。

| 维度 | 验证范围 | 结果 |
|---|---|---|
| cmd_vx（前进 / 后退） | −0.3 ~ 0.5 m/s 全程 | ✅ 稳定，含静止站立与高速 |
| cmd_vy（A/D 横移） | ±0.3 m/s | ✅ 稳定 |
| cmd_vyaw（Q/E 转向） | ±0.2 rad/s | ✅ 稳定 |
| 复合命令（vx + vyaw） | 上述范围内组合 | ✅ 稳定 |
| 速度切换（连续按键调速） | 上述范围内 | ✅ 稳定 |
| 长时间运行 | 连续运行 | ✅ 稳定 |

---

## 目录

- [项目概述](#项目概述)
- [快速开始](#快速开始)
- [机器人配置](#机器人配置)
- [技术架构](#技术架构)
- [目录结构](#目录结构)
- [环境依赖](#环境依赖)
- [许可证与第三方组件](#许可证与第三方组件)

---

## 项目概述

本项目完成了灵犀 X2（lx2501 / T2.5，29-DoF 人形）locomotion 模块的完整 RL 训练与仿真验证链路，并已上真机部署：

```
X2 URDF → Isaac Lab 配置 → PPO 训练 → ONNX 导出 → MuJoCo Sim2Sim 验证 → 实机部署
```

---

## 快速开始

> 以下为最短上手路径。

### 首次安装

宇树 `unitree_rl_lab` 现为 `third_party/` 下的 **submodule**（只读，钉 4960b84，复用其 mdp）；灵犀 X2 代码独立成 `rl/agibot_rl_lab` 包。

```bash
git clone --recursive <repo>            # 已克隆则: git submodule update --init
conda activate isaaclab
pip install -e third_party/unitree_rl_lab/source/unitree_rl_lab   # 宇树(只读, 复用其 mdp)
pip install -e rl/agibot_rl_lab/source/agibot_rl_lab              # 灵犀 X2(机器人/任务/PPO/脚本)
```

### 训练

```bash
conda activate isaaclab
export ISAACLAB_PATH=~/IsaacLab
cd ~/mc-rl/rl/agibot_rl_lab
python scripts/rsl_rl/train.py --task Agibot-X2-29dof-Velocity --num_envs 4096 --headless
```

训练约 15-19 小时（RTX 5090，迭代至 40k-50k，`max_iterations` 默认 50000）。checkpoint 选择与 ONNX/JIT 导出：final/latest 用 `rl/agibot_rl_lab/scripts/rsl_rl/play.py --export-only` 自动导出；选 best（按 `Train/mean_reward` 滑动平均）用 `rl/train/export_best.py`。策略网络结构 500 → 512 → 256 → 128 → 29（隐藏层 ELU）。

### MuJoCo Sim2Sim 验证

```bash
conda activate isaaclab
cd ~/mc-rl

python rl/sim2sim/sim2sim_x2.py --no-fsm --cmd_vx 0.3            # 跳过 FSM，直接 RL 行走
python rl/sim2sim/sim2sim_x2.py                                   # 完整 FSM 流程 (WASD+QE 实时键盘控制)
MUJOCO_GL=egl python rl/sim2sim/sim2sim_x2.py --no-fsm --cmd_vx 0.3 --record sim_walk.mp4 --duration 10  # 无 GUI 录制
```

默认模型为随仓部署模型 `x2_deploy_pkg/models/policy.onnx`（由 `x2_config/sim/sim_config.py` 的 `ONNX_PATH` 指定）；如需评估某次训练导出的模型，显式传 `--onnx <run>/exported/policy.onnx`。验收核心判据：cmd_vx=0.2 时 10s 产生米级位移（实速约 0.16 m/s）、稳态 \|a\| 约 3.2–3.7、pelvis 高度稳定在 0.68 附近。`sim2sim_x2.py` 另支持外力抗扰（`I/J/K/L/U` + `1`–`5`）与 IMU pitch 偏置（`--imu_pitch_bias`）两类保真度测试。

### 实机 Sim2Real 部署

需要 ROS2 + aimdk_msgs 环境（工控机路径）。先用 `deploy_to_board.sh` 把部署件 scp 到板子，再在板上按 FSM 流程跑：

| 步骤 | 命令 | 说明 |
|------|------|------|
| Step 0 | `bash x2_deploy_pkg/deploy_to_board.sh <robot-user> <robot-ip>` | 一键同步部署件到板子 + 模型 md5 双校验 + 远端 env_check |
| Step 0.5 | `ssh <robot-user>@<robot-ip> "aima em get-app mc"` | 确认 mc state：Running → 下一步 `stop-app mc`；Idle → `aima em start-app mc && sleep 2 && aima em stop-app mc` |
| Step 1 | `python3 deploy_x2.py` | 进 FSM：阻尼 → 半蹲 → 站立 → RL 原地站稳（DAMPING/STANDUP 已并入状态机，无需独立测试脚本）|
| Step 2 | `python3 deploy_x2.py` (按 W 开走) | RL 行走（首次 W 跳到 initial_cmd_vx=0.2，再按 W/S 加减速）|
| 收场 | `ssh <robot-user>@<robot-ip> "aima em start-app mc"` | 启动机器人自带 MC，恢复正常业务态 |

FSM 状态机（deploy_x2.py）：`WAITING → DAMPING → (Enter/Space) → JOINT_DEFAULT → STANDUP → (1.5s) → RL_ACTIVE`。阻尼软化、两段式站立均由状态机内部完成（用 `x2_config/joint_gains.yaml` 的 damping_kd / standup 增益）。
键盘控制（RL_ACTIVE）：WASD=前后左右（首次 W 由 0 跳到 `initial_cmd_vx=0.2`，vx 限幅 [-0.3, 0.5]），QE=转向（±0.2），A/D 横移（±0.3），Space=原地站稳，Backspace=紧急回 DAMPING，Esc=退出。
RC 手柄控制（可选）：启动加 `--enable_rc_proxy`，由原厂手柄摇杆替代键盘 WASD+QE 驱动 cmd_vx/vy/vyaw；手柄组合键也可推进 FSM（`L2+X`→JOINT_DEFAULT，`L2+↑`→STANDUP，`L1+R1+创造键`→回 DAMPING）。相关参数 `--rc_velocity_topic`（默认 `/aima/mc/locomotion/velocity`）、`--rc_timeout`（默认 0.5s，超时清零速度）。使用前需已停自带 MC 且原厂 RC 仍在运行。
安全机制：阻尼启动、Cosine 插值站立、Action clip ±5.0、Action delta clip ±0.5、`DAMPING_KD=4.0`、倾倒兜底自动 ESTOP（部署 `GRAVITY_PITCH_LIMIT=1.0`，重力投影分量达上限约 90° 才触发，仅作极端兜底；sim2sim 用 0.72 约 46°）、Backspace 急停、Ctrl+C 安全退出。

---

## 机器人配置

> 以下参考表均经源码 / URDF / MJCF 实测核对，数值为核心参考，请勿擅改。

### 关节定义 (31 关节，29 个参与 RL)

| 组 | 关节 | 数量 | Kp | Kd | effort |
|----|------|------|----|----|--------|
| 腿部 | hip_pitch | 2 | 120 | 5 | 120 |
| 腿部 | hip_roll | 2 | 100 | 4 | 120 |
| 腿部 | hip_yaw | 2 | 100 | 4 | 120 |
| 腿部 | knee | 2 | 150 | 5 | 120 |
| 脚踝 | ankle_pitch | 2 | 40 | 2 | 36 |
| 脚踝 | ankle_roll | 2 | 40 | 2 | 24 |
| 腰部 | waist_yaw | 1 | 40.18 | 2.56 | 120 |
| 腰部 | waist_pitch/roll | 2 | 200 | 2 | 48 |
| 肩/肘 | shoulder_*/elbow | 8 | 50 | 3 | 24-36 |
| 手腕 | wrist_* | 6 | 20 | 2 | 4.8-24 |
| 头部 | head_yaw/pitch | 2 | 0 | 2 | —（不参与 RL；KD=2 仅为 joint_gains.yaml 占位） |

> KP/KD 单一真源 `x2_config/joint_gains.yaml`（`joint_gains.py` 加载），训练 `agibot.py` / sim2sim / 部署 `deploy_x2.py` 三方共用，改一处三方同步生效。其中 `rl_gains`（RL_ACTIVE + STANDUP 用）为**仿真值、非真机实测**，取得真机系统辨识值后改该 YAML 重训即可；`standup_gains`（JOINT_DEFAULT 用）与 `damping_kd` 为实测值。effort 来源于 X2 URDF / MJCF。

### 观测量 (500 维)

单帧 obs = 100 维 × 5 帧历史 = 500 维：

| 观测项 | 维度 | scale | 来源 |
|--------|------|-------|------|
| base_ang_vel | 3 | 0.25 | IMU 陀螺仪 |
| projected_gravity | 3 | 1.0 | IMU 加速度计 |
| velocity_commands | 3 | 1.0 | 遥控器/上位机 |
| joint_pos_rel | 31 | 1.0 | 编码器（相对默认位置） |
| joint_vel_rel | 31 | 0.05 | 编码器 |
| last_action | 29 | 1.0 | 自身记忆 |

History 拼接方式：per-term（每个 obs term 独立维护 CircularBuffer，oldest first）。

### 动作空间 (29 维)

`target_pos = action × 0.25 + default_pos`

| 环境 | sim_dt | PD 频率 | decimation | 策略频率 |
|------|--------|---------|------------|---------|
| Isaac Lab 训练 | 0.005s | 200 Hz | 4 | 50 Hz |
| MuJoCo Sim2Sim | 0.001s | 1000 Hz | 20 | 50 Hz |
| 真机 | ~0.001s | ~1000 Hz（驱动器内环） | 下发 200 Hz（`CONTROL_DT=0.005`） | 50 Hz |

### Isaac Lab 关节排序

URDF pattern matching 产生交错排列（left/right 交替），与 URDF 定义顺序不同：

| idx | 关节 | idx | 关节 | idx | 关节 |
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

Action = 去除 idx 11 (head_yaw) 和 idx 16 (head_pitch)，剩余 29 个。

### 训练配方（关键 reward 权重）

在 `velocity_env_cfg.py` 配置，复用 unitree_rl_lab 的 mdp：

| 项 | 取值 | 作用 |
|---|------|------|
| `track_lin_vel_xy`（std） | √0.1 ≈ 0.316 | 线速度跟踪（std 越小越严，利于低速） |
| `track_ang_vel_z` | 1.0 | 转向角速度跟踪 |
| `base_angular_velocity` | −0.15 | 抑制躯干晃动 |
| `action_rate` | −0.1 | 平滑动作、抑制抖动 |
| `joint_deviation_arms` | −0.3 | 约束手臂、减少无关摆动 |
| `base_height`（target） | 0.68 | 目标骨盆站立高度 |
| 终止 `base_height`（min） | 0.3 | 骨盆低于此高度判摔倒、提前终止 |

其余 reward 项沿用 mdp 默认。X2 配方在重力投影观测上加入训练期 IMU pitch 偏置域随机化（本地 mdp 扩展 `projected_gravity_with_bias`）。批量扫权重用 `rl/train/batch_train.py`（顶部 `EXPERIMENTS` 列表，每组 `MAX_ITER=40000`）。

---

## 技术架构

- **rsl_rl 5.0 适配**：Actor/Critic 是独立的 `MLPModel`（`RslRlMLPModelCfg`）；ONNX 导出用 `MLPModel.as_onnx()` 内置方法；需 `handle_deprecated_rsl_rl_cfg()` 处理旧版兼容。
- **Obs History 拼接**：Isaac Lab 用 **per-term history**——每个 obs term 独立维护 `CircularBuffer`（oldest first），`flatten_history_dim=True` 时 reshape 为 `(batch, history_len * dim)`，最终 cat 拼接。
- **Isaac Lab 关节排序**：URDF regex pattern matching 展开产生交错排列，固化在 `deploy_x2.py` 的 `ISAACLAB_JOINT_ORDER/ACTION_JOINT_ORDER`。
- **ROS2 Topic 接口**：

| Topic | 类型 | 方向 | 说明 |
|-------|------|------|------|
| `/aima/hal/joint/{leg,waist,arm,head}/state` | `JointStateArray` | 订阅 | 关节状态 |
| `/aima/hal/joint/{leg,waist,arm,head}/command` | `JointCommandArray` | 发布 | 关节指令 |
| `/aima/hal/imu/torso/state` | `sensor_msgs/Imu` | 订阅 | 躯干 IMU（torso topic = 髋部/骨盆，训练用） |

QoS：state 订阅 `BEST_EFFORT`，command 发布 `RELIABLE`；`_sensors_ready()` 仅要求 leg/waist/arm 三组 joint state + IMU 就绪（head 非必需）。`deploy_x2.py` 用 rclpy 直接收发 topic，不用 `ros2 service call`（在 X2 上会无响应）。

---

## 目录结构

```
mc-rl/
├── README.md                    ← 本文件（概述 / 参考 / 索引）
│
├── rl/                          ← 训练 / 仿真验证
│   ├── agibot_rl_lab/           ← 灵犀 X2 训练扩展包（IsaacLab extension，pip install -e）
│   │   ├── scripts/rsl_rl/
│   │   │   ├── train.py         ← 训练入口
│   │   │   ├── play.py          ← 推理 + ONNX 导出
│   │   │   └── cli_args.py      ← 训练 / 推理命令行参数
│   │   └── source/agibot_rl_lab/agibot_rl_lab/
│   │       ├── assets/robots/agibot.py                              ← X2 ArticulationCfg（KP/KD/effort）
│   │       ├── tasks/locomotion/robots/x2/29dof/velocity_env_cfg.py ← 训练环境配置（reward/obs/termination）
│   │       └── tasks/locomotion/agents/rsl_rl_ppo_cfg.py            ← X2 PPO（rsl_rl 5.0 MLPModel）
│   │       （日志输出 rl/agibot_rl_lab/logs/...，本地 gitignore）
│   ├── sim2sim/
│   │   └── sim2sim_x2.py        ← MuJoCo Sim2Sim 验证（viewer/record/保真度测试台）
│   └── train/
│       ├── batch_train.py       ← 无人值守批量训练（多组 reward 权重顺序跑）
│       └── export_best.py       ← 从 tensorboard 选 mean_reward 最高的 checkpoint 并导出 ONNX+JIT
│
├── x2_deploy_pkg/               ← 实机部署包（scp 到板子的运行件；同时下发 x2_config/）
│   ├── deploy_x2.py             ← 实机部署（ROS2 FSM，500 Hz；DAMPING/STANDUP 阶段 + 可选 RC 手柄接入）
│   ├── env_check.sh             ← 部署前环境自检（含可选 RC 自检）
│   ├── deploy_to_board.sh       ← 一键部署：scp + 模型 md5 双校验 + 远端 env_check
│   └── models/policy.onnx       ← 部署模型
│
├── x2_config/                   ← 共享配置层（增益 / 策略常量 / 键盘 / sim & deploy 配置；deploy 时随包下发）
│   ├── joint_gains.py           ← 增益加载器（agibot.py / deploy_x2.py / sim2sim 共用）
│   ├── joint_gains.yaml         ← rl/standup 增益 + damping_kd（单一真源）
│   ├── qos_override.yaml        ← rosbag QoS override
│   ├── shared/                  ← sim2sim 与 deploy 共用
│   │   ├── policy_defs.py       ← 关节序 / 默认位姿 / obs 规格 / scale & clip 常量
│   │   └── keyboard.py          ← 键盘 teleop 底座（KeyListener + 速度步进）
│   ├── sim/
│   │   └── sim_config.py        ← MuJoCo 路径 / 关节序 / sim-only 标量（sim2sim 用）
│   └── deploy/
│       ├── deploy_config.py     ← ROS/HAL 话题映射 + deploy-only 标量
│       └── rc_control_proxy.py  ← RC 手柄接入（订阅速度 + 伪装 SetMcAction/GetMcAction 服务替身）
│
├── third_party/
│   └── unitree_rl_lab/          ← 宇树训练框架（submodule，只读，钉 4960b84；X2 复用其 mdp）
│
└── x2_urdf/                     ← X2 URDF + MuJoCo XML
```

---

## 环境依赖

两套**互斥**环境（根因：ROS2 Humble 绑定系统 Python 3.10，IsaacLab 要求 Python 3.11，无法共存于同一解释器）：

- **训练机**（conda `isaaclab`，Python 3.11）：IsaacSim 5.1.0 + IsaacLab 2.3.2 + rsl_rl 5.0.1 + MuJoCo ≥3.0 + PyTorch(CUDA) + onnxruntime，NVIDIA 驱动 ≥580、RTX 5090。跑训练 / 导出 / sim2sim。
- **部署机**（系统 Python 3.10，不装 conda/IsaacLab）：ROS2 Humble + aimdk_msgs + onnxruntime ≥1.16 + numpy。跑 `deploy_x2.py`。

两套环境唯一的传递产物是导出的 `x2_deploy_pkg/models/policy.onnx`（训练机产出、部署机加载）。部署前环境自检见 `x2_deploy_pkg/env_check.sh`。

---

## 许可证与第三方组件

本项目以 **Apache License 2.0** 授权，全文见根目录 [LICENSE](LICENSE)。

Copyright 2026 AgiBot（灵犀）。*（发布主体/版权归属以公司最终确认为准。）*

本仓库包含或派生自以下第三方组件，各组件以其目录内的许可证声明为准：

| 组件 | 许可证 | 说明 |
|------|--------|------|
| `third_party/unitree_rl_lab/`（submodule） | Apache-2.0 | 宇树 [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab)（钉 4960b84，只读），见 `third_party/unitree_rl_lab/LICENCE` |
| `rl/agibot_rl_lab/` | Apache-2.0 | 灵犀 X2 扩展包；其 scripts/PPO 派生自上述宇树 unitree_rl_lab |
| IsaacLab（训练依赖） | BSD-3-Clause | 见 `third_party/unitree_rl_lab/doc/licenses/isaaclab-license.txt` |
| ONNX Runtime | MIT | 见 `third_party/unitree_rl_lab/doc/licenses/onnxruntime-license.txt`；运行时二进制不入库，用 `third_party/unitree_rl_lab/deploy/thirdparty/fetch_onnxruntime.sh` 下载 |
| `x2_urdf/` | — | 灵犀 X2 机器人 URDF 资产 |