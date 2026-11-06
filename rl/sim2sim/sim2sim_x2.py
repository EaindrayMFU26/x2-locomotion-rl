import argparse
import collections
import enum
import os
import select
import sys
import termios
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort

# Config in x2_config/ (joint_gains at top; shared/ + sim/ subdirs). Non-package:
# put each needed subdir on sys.path, then flat-import.
import sys as _sys
from pathlib import Path as _Path
_d = _Path(__file__).resolve().parent
_cfg = next((p / "x2_config" for p in [_d, *_d.parents]
             if (p / "x2_config" / "joint_gains.py").exists()), None)
if _cfg is None:
    raise FileNotFoundError("x2_config/joint_gains.py not found in any parent dir")
for _sub in (_cfg, _cfg / "shared", _cfg / "sim"):
    _sys.path.insert(0, str(_sub))
from joint_gains import load_gains as _load_gains
_G = _load_gains()
KP, KD = _G["rl_kp"], _G["rl_kd"]
JOINT_DEFAULT_KP, JOINT_DEFAULT_KD = _G["standup_kp"], _G["standup_kd"]
DAMPING_KD = _G["damping_kd"]              # scalar default (fallback for joints not in the map)
DAMPING_KD_MAP = _G["damping_kd_map"]      # resolved per-joint damping (mirror mc special_joints)

from policy_defs import (
    ISAACLAB_JOINT_ORDER, ACTION_JOINT_ORDER, DEFAULT_JOINT_POS, JOINT_DEFAULT_POS,
    OBS_TERM_DIMS, ACTION_SCALE, POLICY_DT, HISTORY_LENGTH, JOINT_DEFAULT_DURATION,
    ANG_VEL_SCALE, JOINT_VEL_SCALE, OBS_CLIP, ACTION_CLIP,
)
from keyboard import KeyListener, step_velocity, SIM_HELP_TEXT
from sim_config import (
    ONNX_PATH, SCENE_XML, MUJOCO_JOINT_NAMES,
    GRAVITY_PITCH_LIMIT, SIM_DT, DECIMATION,
)


class State(enum.Enum):
    DAMPING = 0
    JOINT_DEFAULT = 1
    STANDUP = 2
    RL_ACTIVE = 3    # AimDK STAND_DEFAULT + LOCOMOTION_DEFAULT unified (v0.8.0+)
    ESTOP = 4


def quat_rotate_inverse(q, v):
    w, x, y, z = q[0], q[1], q[2], q[3]
    q_vec = np.array([x, y, z])
    a = v * (2.0 * w * w - 1.0)
    b = np.cross(q_vec, v) * w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


def get_gravity_orientation(quat):
    gravity_world = np.array([0.0, 0.0, -1.0])
    return quat_rotate_inverse(quat, gravity_world)


class TermHistory:
    def __init__(self, dim, max_len):
        self.dim = dim
        self.max_len = max_len
        self.buf = collections.deque(maxlen=max_len)
        for _ in range(max_len):
            self.buf.append(np.zeros(dim, dtype=np.float32))

    def append(self, data):
        self.buf.append(data.astype(np.float32))

    def flatten(self):
        return np.concatenate(list(self.buf), axis=0)

    def reset(self):
        self.buf.clear()
        for _ in range(self.max_len):
            self.buf.append(np.zeros(self.dim, dtype=np.float32))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, default=ONNX_PATH)
    parser.add_argument("--xml", type=str, default=SCENE_XML)
    parser.add_argument("--cmd_vx", type=float, default=0.3,
                        help="fixed forward velocity (--no-fsm mode; ignored in FSM mode, use +/- keys)")
    parser.add_argument("--cmd_vy", type=float, default=0.0)
    parser.add_argument("--cmd_vyaw", type=float, default=0.0)
    parser.add_argument("--initial_cmd_vx", type=float, default=0.2,
                        help="first '+' in RL_ACTIVE jumps from 0 to this (policy stable only at >=0.2; <0.2 stands still)")
    parser.add_argument("--cmd_vx_step", type=float, default=0.1,
                        help="'+/-' speed step")
    parser.add_argument("--cmd_vx_max", type=float, default=1.0)
    parser.add_argument("--cmd_vx_min", type=float, default=-1.0)
    parser.add_argument("--cmd_vy_step", type=float, default=0.1,
                        help="A/D lateral_velocity step")
    parser.add_argument("--cmd_vyaw_step", type=float, default=0.1,
                        help="Q/E angular_velocity step")
    parser.add_argument("--no-fsm", action="store_true", help="Skip FSM, start RL immediately")
    parser.add_argument("--record", type=str, default=None, help="Record video to file (e.g. sim.mp4). Uses EGL offscreen rendering.")
    parser.add_argument("--duration", type=float, default=10.0, help="Sim duration in seconds (only for --record mode)")
    parser.add_argument("--imu_pitch_bias", type=float, default=0.0,
                        help="Simulate IMU pitch zero-point bias: add to projected_gravity[x]. "
                             "Positive = robot tilts forward; negative = backward. "
                             "e.g. --imu_pitch_bias 0.08 simulates real-robot +0.08 bias")
    args = parser.parse_args()

    mjmodel = mujoco.MjModel.from_xml_path(args.xml)
    mjmodel.opt.timestep = SIM_DT
    mjdata = mujoco.MjData(mjmodel)

    mj_joint_ids = {}
    for name in MUJOCO_JOINT_NAMES:
        jid = mujoco.mj_name2id(mjmodel, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"Joint {name} not found in MuJoCo model")
        mj_joint_ids[name] = jid

    mj_actuator_ids = {}
    for name in MUJOCO_JOINT_NAMES:
        motor_name = f"motor_{name}"
        aid = mujoco.mj_name2id(mjmodel, mujoco.mjtObj.mjOBJ_ACTUATOR, motor_name)
        if aid >= 0:
            mj_actuator_ids[name] = aid

    mujoco.mj_resetData(mjmodel, mjdata)
    pelvis_jnt_id = mujoco.mj_name2id(mjmodel, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
    pelvis_qpos_adr = mjmodel.jnt_qposadr[pelvis_jnt_id]
    mjdata.qpos[pelvis_qpos_adr + 2] = 0.57

    for name in MUJOCO_JOINT_NAMES:
        jid = mj_joint_ids[name]
        qpos_adr = mjmodel.jnt_qposadr[jid]
        mjdata.qpos[qpos_adr] = DEFAULT_JOINT_POS.get(name, 0.0)

    mujoco.mj_forward(mjmodel, mjdata)

    session = ort.InferenceSession(args.onnx)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    term_histories = collections.OrderedDict()
    for term_name, dim in OBS_TERM_DIMS.items():
        term_histories[term_name] = TermHistory(dim, HISTORY_LENGTH)

    last_action = np.zeros(29, dtype=np.float32)
    prev_action = np.zeros(29, dtype=np.float32)
    target_joint_pos = {name: DEFAULT_JOINT_POS.get(name, 0.0) for name in MUJOCO_JOINT_NAMES}
    velocity_cmd = np.array([args.cmd_vx, args.cmd_vy, args.cmd_vyaw], dtype=np.float32)

    state = State.DAMPING
    if args.no_fsm:
        state = State.RL_ACTIVE
    step_count = 0
    interp_start_pos = {}
    interp_elapsed = 0.0
    policy_count = 0
    warmup_done = False
    # user_cmd flags set by input_listener
    user_cmd = {"advance": False, "damping": False, "exit": False, "push": None}
    lock = threading.Lock()

    PUSH_DURATION = int(0.1 / SIM_DT)  # 100ms impulse
    push_remaining = 0
    push_force = np.zeros(3)
    pelvis_body_id_ref = [None]

    def get_quat():
        pelvis_body_id = mujoco.mj_name2id(mjmodel, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        return mjdata.xquat[pelvis_body_id].copy()

    def get_ang_vel():
        sensor_gyro_id = mujoco.mj_name2id(mjmodel, mujoco.mjtObj.mjOBJ_SENSOR, "body-angular-velocity")
        if sensor_gyro_id >= 0:
            gyro_adr = mjmodel.sensor_adr[sensor_gyro_id]
            return mjdata.sensordata[gyro_adr:gyro_adr + 3].copy()
        pelvis_body_id = mujoco.mj_name2id(mjmodel, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        ang_vel_world = mjdata.cvel[pelvis_body_id][3:6].copy() if mjdata.cvel.shape[1] == 6 else np.zeros(3)
        return quat_rotate_inverse(get_quat(), ang_vel_world)

    def get_joint_pos(name):
        jid = mj_joint_ids[name]
        return mjdata.qpos[mjmodel.jnt_qposadr[jid]]

    def get_joint_vel(name):
        jid = mj_joint_ids[name]
        return mjdata.qvel[mjmodel.jnt_dofadr[jid]]

    def apply_damping():
        for name in MUJOCO_JOINT_NAMES:
            if name not in mj_actuator_ids:
                continue
            aid = mj_actuator_ids[name]
            jid = mj_joint_ids[name]
            vel = mjdata.qvel[mjmodel.jnt_dofadr[jid]]
            kd = DAMPING_KD_MAP.get(name, DAMPING_KD)
            mjdata.ctrl[aid] = -kd * vel

    def apply_pd(pos_targets, kp_dict, kd_dict):
        for name in MUJOCO_JOINT_NAMES:
            if name not in mj_actuator_ids:
                continue
            aid = mj_actuator_ids[name]
            jid = mj_joint_ids[name]
            pos = mjdata.qpos[mjmodel.jnt_qposadr[jid]]
            vel = mjdata.qvel[mjmodel.jnt_dofadr[jid]]
            tgt = pos_targets.get(name, 0.0)
            kp = kp_dict.get(name, 0.0)
            kd = kd_dict.get(name, 0.0)
            mjdata.ctrl[aid] = kp * (tgt - pos) - kd * vel

    def compute_obs_terms():
        ang_vel_local = get_ang_vel()
        base_ang_vel = (ang_vel_local * ANG_VEL_SCALE).astype(np.float32)
        quat = get_quat()
        projected_gravity = get_gravity_orientation(quat).astype(np.float32)
        projected_gravity[0] += args.imu_pitch_bias  # simulate IMU pitch zero-point offset
        joint_pos_all = np.zeros(31, dtype=np.float32)
        joint_vel_all = np.zeros(31, dtype=np.float32)
        for i, name in enumerate(ISAACLAB_JOINT_ORDER):
            default_pos = DEFAULT_JOINT_POS.get(name, 0.0)
            joint_pos_all[i] = get_joint_pos(name) - default_pos
            joint_vel_all[i] = get_joint_vel(name) * JOINT_VEL_SCALE
        return {
            "base_ang_vel": base_ang_vel,
            "projected_gravity": projected_gravity,
            "velocity_commands": velocity_cmd.copy(),
            "joint_pos_rel": joint_pos_all,
            "joint_vel_rel": joint_vel_all,
            "last_action": last_action.copy(),
        }

    def build_obs():
        obs_parts = []
        for term_name in OBS_TERM_DIMS:
            obs_parts.append(term_histories[term_name].flatten())
        return np.concatenate(obs_parts).reshape(1, -1).astype(np.float32)

    def check_safety():
        grav = get_gravity_orientation(get_quat())
        return abs(grav[0]) < GRAVITY_PITCH_LIMIT and abs(grav[1]) < GRAVITY_PITCH_LIMIT

    def input_listener():
        """Non-blocking raw keyboard input via termios. Each key takes effect immediately, no Enter needed."""

        def _handle_key(key: str):
            with lock:
                if key == "?":
                    print(SIM_HELP_TEXT)
                elif key == "\x03":  # Ctrl+C: raw mode disables ISIG so no SIGINT is sent; treat as clean exit
                    user_cmd["exit"] = True
                    print("[USER] Ctrl+C -> exit")
                elif key == "\x1b":  # Esc (distinguish bare Esc from arrow-key escape sequence)
                    ready, _, _ = select.select([sys.stdin], [], [], 0.0)
                    if ready:
                        while True:
                            r, _, _ = select.select([sys.stdin], [], [], 0.0)
                            if r:
                                sys.stdin.read(1)
                            else:
                                break
                    else:
                        user_cmd["exit"] = True
                        print("[USER] Esc -> exit")
                elif key in ("\x7f", "\x08"):  # Backspace
                    user_cmd["damping"] = True
                    print("[USER] Backspace -> DAMPING")
                elif key in ("\r", " "):  # Enter or Space
                    if key == " ":
                        # Space: RL_ACTIVE zeros velocity; DAMPING advances FSM
                        old = [float(velocity_cmd[0]), float(velocity_cmd[1]), float(velocity_cmd[2])]
                        if any(abs(v) > 1e-6 for v in old):
                            velocity_cmd[0] = 0.0
                            velocity_cmd[1] = 0.0
                            velocity_cmd[2] = 0.0
                            print("[USER] Space: all velocity -> 0 (stand)")
                        else:
                            user_cmd["advance"] = True
                    else:
                        user_cmd["advance"] = True
                elif key.lower() == "w":
                    old = float(velocity_cmd[0])
                    if abs(old) < 1e-6:
                        new = args.initial_cmd_vx
                    else:
                        new = old + args.cmd_vx_step
                    new = max(args.cmd_vx_min, min(new, args.cmd_vx_max))
                    velocity_cmd[0] = new
                    print(f"[USER] W forward {old:+.2f} -> {new:+.2f}")
                elif key.lower() == "s":
                    old = float(velocity_cmd[0])
                    new = step_velocity(old, -args.cmd_vx_step, args.cmd_vx_min, args.cmd_vx_max)
                    velocity_cmd[0] = new
                    print(f"[USER] S forward {old:+.2f} -> {new:+.2f}")
                elif key.lower() == "a":
                    old = float(velocity_cmd[1])
                    new = step_velocity(old, -args.cmd_vy_step, -0.3, 0.3)
                    velocity_cmd[1] = new
                    print(f"[USER] A lateral {old:+.2f} -> {new:+.2f}")
                elif key.lower() == "d":
                    old = float(velocity_cmd[1])
                    new = step_velocity(old, args.cmd_vy_step, -0.3, 0.3)
                    velocity_cmd[1] = new
                    print(f"[USER] D lateral {old:+.2f} -> {new:+.2f}")
                elif key.lower() == "q":
                    old = float(velocity_cmd[2])
                    new = step_velocity(old, -args.cmd_vyaw_step, -0.2, 0.2)
                    velocity_cmd[2] = new
                    print(f"[USER] Q yaw {old:+.2f} -> {new:+.2f}")
                elif key.lower() == "e":
                    old = float(velocity_cmd[2])
                    new = step_velocity(old, args.cmd_vyaw_step, -0.2, 0.2)
                    velocity_cmd[2] = new
                    print(f"[USER] E yaw {old:+.2f} -> {new:+.2f}")
                elif key in "12345":
                    user_cmd["push_scale"] = int(key)
                    print(f"[USER] Push scale -> {key}x (={int(key)*100}N)")
                elif key.lower() == "i":
                    user_cmd["push"] = np.array([1.0, 0.0, 0.0]) * user_cmd.get("push_scale", 1) * 50.0
                    print(f"[USER] Push FORWARD {np.linalg.norm(user_cmd['push']):.0f}N")
                elif key.lower() == "k":
                    user_cmd["push"] = np.array([-1.0, 0.0, 0.0]) * user_cmd.get("push_scale", 1) * 50.0
                    print(f"[USER] Push BACKWARD {np.linalg.norm(user_cmd['push']):.0f}N")
                elif key.lower() == "j":
                    user_cmd["push"] = np.array([0.0, 1.0, 0.0]) * user_cmd.get("push_scale", 1) * 50.0
                    print(f"[USER] Push LEFT {np.linalg.norm(user_cmd['push']):.0f}N")
                elif key.lower() == "l":
                    user_cmd["push"] = np.array([0.0, -1.0, 0.0]) * user_cmd.get("push_scale", 1) * 50.0
                    print(f"[USER] Push RIGHT {np.linalg.norm(user_cmd['push']):.0f}N")
                elif key.lower() == "u":
                    angle = np.random.uniform(0, 2 * np.pi)
                    user_cmd["push"] = np.array([np.cos(angle), np.sin(angle), 0.0]) * user_cmd.get("push_scale", 1) * 150.0
                    print(f"[USER] Push RANDOM {np.linalg.norm(user_cmd['push']):.0f}N angle={np.degrees(angle):.0f}deg")

        # raw mode (cbreak=False): sim overrides stdout \n->\r\n; loop runs until process exit
        KeyListener(_handle_key, cbreak=False).run()

    # Capture original (cooked) terminal settings so we can guarantee a restore on any
    # exit path: input_listener runs as a daemon thread and the os._exit() at the end of
    # main() skips its finally/atexit cleanup. Guard for non-tty (piped / record CI) use.
    try:
        _stdin_fd = sys.stdin.fileno()
        _orig_term = termios.tcgetattr(_stdin_fd)
    except Exception:
        _stdin_fd, _orig_term = None, None

    input_thread = threading.Thread(target=input_listener, daemon=True)
    input_thread.start()

    def kv(label, value):
        print(f"  {label:<15}: {value}")

    print("=" * 60)
    print("  X2 Sim2Sim with FSM (deploy logic test)")
    print("-" * 60)
    kv("ONNX", args.onnx)
    kv("initial cmd_vx", f"{args.initial_cmd_vx:g} m/s")
    kv("cmd_vx step", f"{args.cmd_vx_step:g} m/s   range [{args.cmd_vx_min:g}, {args.cmd_vx_max:g}]")
    kv("cmd_vy step", f"{args.cmd_vy_step:g} m/s")
    kv("cmd_vyaw step", f"{args.cmd_vyaw_step:g} rad/s")
    if args.imu_pitch_bias != 0.0:
        kv("imu_pitch_bias", f"{args.imu_pitch_bias:+.4f} rad  (proj_gravity[x] += {args.imu_pitch_bias:+.4f})")
    print("=" * 60)
    if args.record:
        print(f"  Mode: RECORD -> {args.record} ({args.duration}s)")
    else:
        print("  FSM: DAMPING -> [Enter/Space] -> JOINT_DEFAULT -> STANDUP -> RL_ACTIVE")
        print("       RL_ACTIVE: WASD=move, QE=turn, Space=stand in place, Backspace=back to DAMPING")
        print("       Robot tilt -> ESTOP (auto damping)")
        print("       press '?' for full key list")
    print("=" * 60)
    if args.no_fsm:
        print("[FSM] Skipped. RL_ACTIVE immediately.")
    elif not args.record:
        print("[FSM] Starting in DAMPING mode (kp=0, kd=5). Press Enter...")

    print("[Init] Settling for 0.5s with PD hold...")
    settle_target = {n: DEFAULT_JOINT_POS.get(n, 0.0) for n in MUJOCO_JOINT_NAMES}
    for _ in range(500):
        apply_pd(settle_target, KP, KD)
        mujoco.mj_step(mjmodel, mjdata)
    pelvis_body_id = mujoco.mj_name2id(mjmodel, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    print(f"[Init] Settled. pelvis_z = {mjdata.xpos[pelvis_body_id][2]:.4f}m")

    def sim_step_logic(advance, damping_req):
        nonlocal state, interp_elapsed, interp_start_pos, policy_count, step_count, warmup_done
        nonlocal push_remaining, push_force

        # Handle external push
        with lock:
            if user_cmd["push"] is not None:
                push_force = user_cmd["push"].copy()
                push_remaining = PUSH_DURATION
                user_cmd["push"] = None
        torso_bid = mujoco.mj_name2id(mjmodel, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        if push_remaining > 0:
            mjdata.xfrc_applied[torso_bid, :3] = push_force
            push_remaining -= 1
        else:
            mjdata.xfrc_applied[torso_bid, :3] = 0.0

        if state == State.DAMPING:
            apply_damping()
            if advance:
                grav = get_gravity_orientation(get_quat())
                fell = abs(grav[2] + 1.0) > 0.5
                if fell:
                    print("[FSM] Robot is on the ground. Resetting pose (sim only)...")
                    mujoco.mj_resetData(mjmodel, mjdata)
                    mjdata.qpos[pelvis_qpos_adr + 2] = 0.57
                    for name in MUJOCO_JOINT_NAMES:
                        jid = mj_joint_ids[name]
                        mjdata.qpos[mjmodel.jnt_qposadr[jid]] = DEFAULT_JOINT_POS.get(name, 0.0)
                    mujoco.mj_forward(mjmodel, mjdata)
                    for _ in range(500):
                        apply_pd(settle_target, KP, KD)
                        mujoco.mj_step(mjmodel, mjdata)
                    interp_start_pos = {name: get_joint_pos(name) for name in MUJOCO_JOINT_NAMES}
                    interp_elapsed = 0.0
                    state = State.STANDUP
                    print("[FSM] Reset done -> STANDUP (to RL default pos)")
                else:
                    interp_start_pos = {name: get_joint_pos(name) for name in MUJOCO_JOINT_NAMES}
                    interp_elapsed = 0.0
                    state = State.JOINT_DEFAULT
                    print("[FSM] DAMPING -> JOINT_DEFAULT")

        elif state == State.JOINT_DEFAULT:
            interp_elapsed += SIM_DT
            progress = min(interp_elapsed / JOINT_DEFAULT_DURATION, 1.0)
            progress = progress * progress * (3.0 - 2.0 * progress)
            interp_pos = {}
            for name in MUJOCO_JOINT_NAMES:
                s = interp_start_pos.get(name, 0.0)
                t = JOINT_DEFAULT_POS.get(name, 0.0)
                interp_pos[name] = s + (t - s) * progress
            apply_pd(interp_pos, JOINT_DEFAULT_KP, JOINT_DEFAULT_KD)
            if interp_elapsed >= JOINT_DEFAULT_DURATION:
                interp_start_pos = {name: get_joint_pos(name) for name in MUJOCO_JOINT_NAMES}
                interp_elapsed = 0.0
                state = State.STANDUP
                print("[FSM] JOINT_DEFAULT -> STANDUP (to RL default pos)")

        elif state == State.STANDUP:
            interp_elapsed += SIM_DT
            standup_dur = 1.5
            progress = min(interp_elapsed / standup_dur, 1.0)
            progress = 0.5 * (1 - np.cos(np.pi * progress))
            interp_pos = {}
            for name in MUJOCO_JOINT_NAMES:
                s = interp_start_pos.get(name, 0.0)
                t = DEFAULT_JOINT_POS.get(name, 0.0)
                interp_pos[name] = s + (t - s) * progress
            apply_pd(interp_pos, KP, KD)
            if interp_elapsed >= standup_dur:
                for th in term_histories.values():
                    th.reset()
                last_action[:] = 0.0
                prev_action[:] = 0.0
                policy_count = 0
                warmup_done = False
                velocity_cmd[0] = 0.0   # stand first: forward_velocity=0 on entering RL_ACTIVE
                state = State.RL_ACTIVE
                print("[FSM] STANDUP -> RL_ACTIVE (fwd=0, standing; '+' to walk)")

        elif state == State.RL_ACTIVE:
            if not check_safety():
                for name in MUJOCO_JOINT_NAMES:
                    target_joint_pos[name] = DEFAULT_JOINT_POS.get(name, 0.0)
                state = State.ESTOP
                print("[FSM] -> ESTOP! Robot tilted too far.")
            elif damping_req:
                for name in MUJOCO_JOINT_NAMES:
                    target_joint_pos[name] = DEFAULT_JOINT_POS.get(name, 0.0)
                state = State.DAMPING
                print("[FSM] RL_ACTIVE -> DAMPING (user 'd')")
            else:
                if step_count % DECIMATION == 0:
                    terms = compute_obs_terms()
                    for term_name, val in terms.items():
                        term_histories[term_name].append(val)

                    obs = np.clip(build_obs(), -OBS_CLIP, OBS_CLIP)
                    output = session.run([output_name], {input_name: obs})[0]
                    action = np.clip(output[0], -ACTION_CLIP, ACTION_CLIP)

                    if not warmup_done:
                        if policy_count < HISTORY_LENGTH:
                            action[:] = 0.0
                        else:
                            warmup_done = True
                            print(f"[RL] Warmup done ({HISTORY_LENGTH} frames)")

                    last_action[:] = action

                    for i, name in enumerate(ACTION_JOINT_ORDER):
                        default_pos = DEFAULT_JOINT_POS.get(name, 0.0)
                        target_joint_pos[name] = float(action[i]) * ACTION_SCALE + default_pos

                    policy_count += 1
                    if policy_count % 50 == 0:
                        print(f"[RL_ACTIVE] step={policy_count}, "
                              f"cmd=(vx={velocity_cmd[0]:+.2f}, vy={velocity_cmd[1]:+.2f}, "
                              f"vyaw={velocity_cmd[2]:+.2f}), |a|={np.linalg.norm(action):.3f}")

                apply_pd(target_joint_pos, KP, KD)

        elif state == State.ESTOP:
            apply_damping()
            if advance or damping_req:
                state = State.DAMPING
                print("[FSM] ESTOP -> DAMPING (user reset)")

        mujoco.mj_step(mjmodel, mjdata)
        step_count += 1

    try:
        if args.record:
            _run_record(mjmodel, mjdata, args, sim_step_logic)
        else:
            _run_viewer(mjmodel, mjdata, args, sim_step_logic, lock, user_cmd, lambda: step_count)
    finally:
        # Guarantee terminal restore (input_listener is a daemon; os._exit skips its finally).
        if _orig_term is not None:
            termios.tcsetattr(_stdin_fd, termios.TCSADRAIN, _orig_term)
        sys.stdout.flush()

    # mujoco.viewer (GLFW/OpenGL) segfaults during interpreter shutdown when the render
    # thread races GL-context teardown. Hard-exit to skip that cleanup path entirely.
    os._exit(0)


def _run_viewer(mjmodel, mjdata, args, sim_step_logic, lock, user_cmd, get_step_count):
    with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:
        start_time = time.time()
        while viewer.is_running():
            with lock:
                advance = user_cmd["advance"]
                damping_req = user_cmd["damping"]
                exit_req = user_cmd["exit"]
                user_cmd["advance"] = False
                user_cmd["damping"] = False
                user_cmd["exit"] = False

            if exit_req:
                print("[USER] Exiting viewer...")
                break

            sim_step_logic(advance, damping_req)

            sc = get_step_count()
            if sc % DECIMATION == 0:
                viewer.sync()

            sim_time = sc * SIM_DT
            real_time = time.time() - start_time
            if sim_time > real_time:
                time.sleep(sim_time - real_time)


def _run_record(mjmodel, mjdata, args, sim_step_logic):
    import imageio

    os.environ.setdefault("MUJOCO_GL", "egl")

    vid_w, vid_h = 1280, 720
    mjmodel.vis.global_.offwidth = vid_w
    mjmodel.vis.global_.offheight = vid_h
    renderer = mujoco.Renderer(mjmodel, height=vid_h, width=vid_w)

    pelvis_body_id = mujoco.mj_name2id(mjmodel, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    settle_steps = int(1.0 / SIM_DT)
    print(f"[Record] Settling for 1s ({settle_steps} steps)...")
    for _ in range(settle_steps):
        sim_step_logic(False, False)

    total_steps = int(args.duration / SIM_DT)
    render_every = DECIMATION
    fps = int(1.0 / POLICY_DT)
    frames = []

    heights = []
    start_pos = mjdata.xpos[pelvis_body_id].copy()
    start_quat = mjdata.xquat[pelvis_body_id].copy()

    print(f"[Record] Running {args.duration}s sim, rendering every {render_every} steps @ {fps} fps...")
    for i in range(total_steps):
        sim_step_logic(False, False)

        if i % render_every == 0:
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = pelvis_body_id
            cam.distance = 3.0
            cam.azimuth = 135
            cam.elevation = -20
            cam.lookat[:] = mjdata.xpos[pelvis_body_id]
            renderer.update_scene(mjdata, camera=cam)
            frames.append(renderer.render().copy())
            heights.append(mjdata.xpos[pelvis_body_id][2])

        if (i + 1) % (total_steps // 10) == 0:
            pct = int((i + 1) / total_steps * 100)
            print(f"[Record] {pct}%...")

    end_pos = mjdata.xpos[pelvis_body_id].copy()
    end_quat = mjdata.xquat[pelvis_body_id].copy()
    displacement = end_pos - start_pos
    dist = np.sqrt(displacement[0]**2 + displacement[1]**2)
    end_yaw = np.arctan2(2*(end_quat[0]*end_quat[3]+end_quat[1]*end_quat[2]),
                          1-2*(end_quat[2]**2+end_quat[3]**2))
    start_yaw = np.arctan2(2*(start_quat[0]*start_quat[3]+start_quat[1]*start_quat[2]),
                            1-2*(start_quat[2]**2+start_quat[3]**2))
    yaw_change = np.degrees(end_yaw - start_yaw)
    h_arr = np.array(heights)

    print(f"\n[Metrics] Distance: {dist:.3f}m  (dx={displacement[0]:.3f}, dy={displacement[1]:.3f})")
    print(f"[Metrics] Yaw change: {yaw_change:.1f}°")
    print(f"[Metrics] Avg speed: {dist/args.duration:.3f} m/s")
    print(f"[Metrics] Pelvis height: {h_arr.mean():.4f} ± {h_arr.std():.4f}m (min={h_arr.min():.4f})")
    print(f"[Metrics] Lateral drift: {abs(displacement[1]):.3f}m")

    print(f"\n[Record] Writing {len(frames)} frames to {args.record}...")
    writer = imageio.get_writer(args.record, fps=fps, quality=8)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    print(f"[Record] Done! -> {args.record}")


if __name__ == "__main__":
    main()
