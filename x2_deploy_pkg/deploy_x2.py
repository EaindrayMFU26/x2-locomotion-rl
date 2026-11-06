import argparse
import collections
import enum
import select
import sys
import threading
import time

import numpy as np
import onnxruntime as ort
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Imu

try:
    from aimdk_msgs.msg import JointCommandArray, JointStateArray, JointCommand
except ImportError:
    print("ERROR: aimdk_msgs not found.")
    print("On the robot: source the AimDK workspace, then retry.")
    print("On dev machine: export PYTHONPATH=/path/to/aimdk_msgs/parent:$PYTHONPATH")
    sys.exit(1)

# PD gains: single source of truth in x2_config/joint_gains.py
import sys as _sys
from pathlib import Path as _Path
_d = _Path(__file__).resolve().parent
# in-repo: repo_root/x2_config/...; on-robot: ~/x2_deploy/x2_config/... (deploy_to_board ships the dir)
# Non-package: put joint_gains dir + shared/ + deploy/ on sys.path, then flat-import.
_cfg = next((p / "x2_config" for p in [_d, *_d.parents]
             if (p / "x2_config" / "joint_gains.py").exists()), None)
if _cfg is None:
    raise FileNotFoundError("x2_config/joint_gains.py not found in any parent dir")
for _sub in (_cfg, _cfg / "shared", _cfg / "deploy"):
    _sys.path.insert(0, str(_sub))
from joint_gains import load_gains as _load_gains
_G = _load_gains()

DEPLOY_KP, DEPLOY_KD = _G["rl_kp"], _G["rl_kd"]
JOINT_DEFAULT_KP, JOINT_DEFAULT_KD = _G["standup_kp"], _G["standup_kd"]
DAMPING_KD = _G["damping_kd"]   # scalar default (backward compat with old scalar yaml)
DAMPING_KD_MAP = _G.get("damping_kd_map")  # per-joint resolved map; None if old-style yaml

from policy_defs import (
    ISAACLAB_JOINT_ORDER, ACTION_JOINT_ORDER, DEFAULT_JOINT_POS, JOINT_DEFAULT_POS,
    OBS_TERM_DIMS, ACTION_SCALE, POLICY_DT, HISTORY_LENGTH, JOINT_DEFAULT_DURATION,
    ANG_VEL_SCALE, JOINT_VEL_SCALE, OBS_CLIP, ACTION_CLIP,
)
from keyboard import KeyListener, step_velocity, DEPLOY_HELP_TEXT
from deploy_config import (
    ONNX_PATH, JOINT_AREAS, ALL_JOINT_NAMES, IMU_TOPIC,
    GRAVITY_PITCH_LIMIT, ACTION_DELTA_LIMIT, CONTROL_DT,
)

# RC gamepad integration (optional, enabled with --enable_rc_proxy). rc_control_proxy
# and deploy_config both live in x2_config/deploy/ (already on sys.path), so import
# them flat here.
from rc_control_proxy import (
    RcState, RcControlProxy,
    MC_DAMPING_DEFAULT, MC_SOFT_EMERGENCY_STOP,
    MC_JOINT_DEFAULT, MC_STAND_DEFAULT, MC_LOCOMOTION_DEFAULT,
)
try:
    from aimdk_msgs.srv import SetMcAction, GetMcAction
    from aimdk_msgs.msg import (
        McLocomotionVelocity, McAction, McActionInfo, McActionStatus,
    )
    _RC_MSGS_OK = True
except Exception:
    _RC_MSGS_OK = False

# Speed-range query service (the RC calls it every cycle; without it the RC sends no
# velocity). May be absent from the slimmed-down aimdk_msgs, so try it separately.
try:
    from aimdk_msgs.srv import GetMcLocomotionSpeedStatus
    from aimdk_msgs.msg import McLocomotionSpeedMode
    _RC_SPEED_OK = True
except Exception:
    _RC_SPEED_OK = False


class State(enum.Enum):
    WAITING = 0
    DAMPING = 1
    JOINT_DEFAULT = 2
    STANDUP = 3
    RL_ACTIVE = 4     # cmd_vx=0 stands in place, cmd_vx>0.09 walks; auto-switches by velocity command
    ESTOP = 5


def fsm_state_to_mc_action(state: "State") -> int:
    """Map the deploy FSM state to the MC action value reported to the RC (used by GetMcAction)."""
    if state in (State.WAITING, State.DAMPING):
        return MC_DAMPING_DEFAULT      # 3
    if state == State.JOINT_DEFAULT:
        return MC_JOINT_DEFAULT        # 100
    if state == State.STANDUP:
        return MC_STAND_DEFAULT        # 200
    if state == State.RL_ACTIVE:
        return MC_LOCOMOTION_DEFAULT   # 300
    if state == State.ESTOP:
        return MC_SOFT_EMERGENCY_STOP  # 2
    return MC_DAMPING_DEFAULT


def quat_rotate_inverse(q_wxyz, v):
    w, x, y, z = q_wxyz[0], q_wxyz[1], q_wxyz[2], q_wxyz[3]
    q_vec = np.array([x, y, z])
    a = v * (2.0 * w * w - 1.0)
    b = np.cross(q_vec, v) * w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


def get_gravity_orientation(quat_wxyz):
    gravity_world = np.array([0.0, 0.0, -1.0])
    return quat_rotate_inverse(quat_wxyz, gravity_world)


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


class X2RLDeployNode(Node):
    def __init__(self, args):
        super().__init__("x2_rl_deploy")
        self.args = args
        self.lock = threading.Lock()

        self.session = ort.InferenceSession(args.onnx)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.get_logger().info(f"Loaded ONNX: {args.onnx}")

        self.joint_pos = {}
        self.joint_vel = {}
        self.imu_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
        self.imu_ang_vel = np.zeros(3)
        self.imu_ready = False
        self.joints_ready = set()

        self.term_histories = collections.OrderedDict()
        for term_name, dim in OBS_TERM_DIMS.items():
            self.term_histories[term_name] = TermHistory(dim, HISTORY_LENGTH)

        self.last_action = np.zeros(29, dtype=np.float32)
        self.prev_action = np.zeros(29, dtype=np.float32)
        self.target_joint_pos = {name: DEFAULT_JOINT_POS.get(name, 0.0) for name in ALL_JOINT_NAMES}
        # velocity_cmd: [forward, lateral, angular]. Always starts at forward=0; press '+' to walk
        self.velocity_cmd = np.array([0.0, args.cmd_vy, args.cmd_vyaw], dtype=np.float32)
        self.initial_cmd_vx = float(args.initial_cmd_vx)
        self.cmd_vx_step = float(args.cmd_vx_step)
        self.cmd_vx_max = float(args.cmd_vx_max)
        self.cmd_vx_min = float(args.cmd_vx_min)
        self.cmd_vy_step = float(args.cmd_vy_step)
        self.cmd_vyaw_step = float(args.cmd_vyaw_step)
        self.damping_kd = float(args.damping_kd)
        # resolved per-joint damping: yaml overrides win, CLI value as default baseline
        if DAMPING_KD_MAP is not None:
            overrides = _G.get("damping_kd_overrides", {})
            self.damping_kd_map = {
                j: overrides[j] if j in overrides else self.damping_kd
                for j in DAMPING_KD_MAP
            }
        else:
            self.damping_kd_map = None

        self.state = State.WAITING
        self.tick_count = 0
        self.interp_start_pos = {}
        self.interp_elapsed = 0.0
        self.joint_default_done = False
        self.policy_count = 0
        self.warmup_done = False

        sub_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )
        pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.joint_pubs = {}
        for area_name, area_cfg in JOINT_AREAS.items():
            self.create_subscription(
                JointStateArray, area_cfg["state_topic"],
                lambda msg, an=area_name: self._joint_state_cb(msg, an),
                sub_qos,
            )
            self.joint_pubs[area_name] = self.create_publisher(
                JointCommandArray, area_cfg["cmd_topic"], pub_qos,
            )

        self.create_subscription(Imu, IMU_TOPIC, self._imu_cb, sub_qos)

        # ---- RC gamepad integration ----
        self.rc_enabled = bool(getattr(args, "enable_rc_proxy", False))
        self.rc_state = None
        self.rc_proxy = None
        if self.rc_enabled:
            if not _RC_MSGS_OK:
                self.get_logger().error(
                    "[RC] enable_rc_proxy set but aimdk_msgs srv/msg import failed; RC disabled")
                self.rc_enabled = False
            else:
                self.rc_state = RcState(timeout=float(args.rc_timeout))
                _deps = {
                    "McLocomotionVelocity": McLocomotionVelocity,
                    "SetMcAction": SetMcAction,
                    "GetMcAction": GetMcAction,
                    "McActionInfo": McActionInfo,
                    "McAction": McAction,
                    "McActionStatus": McActionStatus,
                }
                _speed_status_service = None
                _get_speed_bounds = None
                if _RC_SPEED_OK:
                    _deps["GetMcLocomotionSpeedStatus"] = GetMcLocomotionSpeedStatus
                    _deps["McLocomotionSpeedMode"] = McLocomotionSpeedMode
                    _speed_status_service = "aimdk_5Fmsgs/srv/GetMcLocomotionSpeedStatus"
                    _get_speed_bounds = self._rc_get_speed_bounds
                else:
                    self.get_logger().warn(
                        "[RC] This machine's aimdk_msgs has no GetMcLocomotionSpeedStatus; "
                        "the RC may fail to query the speed range and send no velocity "
                        "(gamepad walking will be unavailable)")
                self.rc_proxy = RcControlProxy(
                    node=self,
                    state=self.rc_state,
                    on_action=self._rc_on_action,
                    get_fsm_action_value=self._rc_get_fsm_action_value,
                    deps=_deps,
                    velocity_topic=args.rc_velocity_topic,
                    set_action_service="aimdk_5Fmsgs/srv/SetMcAction",
                    get_action_service="aimdk_5Fmsgs/srv/GetMcAction",
                    logger=self.get_logger(),
                    speed_status_service=_speed_status_service,
                    get_speed_bounds=_get_speed_bounds,
                )
                self.rc_proxy.setup()
                self.get_logger().warn(
                    "[RC] proxy enabled. Confirm the stock MC is stopped and the stock RC is still running.")

        self.control_timer = self.create_timer(CONTROL_DT, self._control_loop)
        self.get_logger().info("Waiting for joint states and IMU...")

    def _joint_state_cb(self, msg: JointStateArray, area_name: str):
        with self.lock:
            for js in msg.joints:
                self.joint_pos[js.name] = js.position
                self.joint_vel[js.name] = js.velocity
            self.joints_ready.add(area_name)

    def _imu_cb(self, msg: Imu):
        with self.lock:
            q = msg.orientation
            self.imu_quat_wxyz = np.array([q.w, q.x, q.y, q.z])
            w = msg.angular_velocity
            self.imu_ang_vel = np.array([w.x, w.y, w.z])
            self.imu_ready = True

    def _sensors_ready(self):
        if not self.imu_ready:
            return False
        required = {"leg", "waist", "arm"}
        return required.issubset(self.joints_ready)

    def _check_safety(self):
        grav = get_gravity_orientation(self.imu_quat_wxyz)
        pitch_proxy = abs(grav[0])
        roll_proxy = abs(grav[1])
        if pitch_proxy > GRAVITY_PITCH_LIMIT or roll_proxy > GRAVITY_PITCH_LIMIT:
            return False
        return True

    def _publish_damping(self):
        for area_name, area_cfg in JOINT_AREAS.items():
            cmd_msg = JointCommandArray()
            for jname in area_cfg["joints"]:
                j = JointCommand()
                j.name = jname
                j.position = 0.0
                j.velocity = 0.0
                j.effort = 0.0
                j.stiffness = 0.0
                if self.damping_kd_map is not None:
                    j.damping = self.damping_kd_map.get(jname, self.damping_kd)
                else:
                    j.damping = self.damping_kd
                cmd_msg.joints.append(j)
            self.joint_pubs[area_name].publish(cmd_msg)

    def _publish_position(self, positions, kp_dict, kd_dict):
        for area_name, area_cfg in JOINT_AREAS.items():
            cmd_msg = JointCommandArray()
            for jname in area_cfg["joints"]:
                j = JointCommand()
                j.name = jname
                j.position = float(positions.get(jname, 0.0))
                j.velocity = 0.0
                j.effort = 0.0
                j.stiffness = float(kp_dict.get(jname, 20.0))
                j.damping = float(kd_dict.get(jname, 1.0))
                cmd_msg.joints.append(j)
            self.joint_pubs[area_name].publish(cmd_msg)

    def _enter_damping(self):
        prev = self.state
        self.state = State.DAMPING
        self.get_logger().info(f"[FSM] {prev.name} -> DAMPING")

    def _enter_joint_default(self):
        self.interp_start_pos = dict(self.joint_pos)
        self.interp_elapsed = 0.0
        self.joint_default_done = False
        self.state = State.JOINT_DEFAULT
        self.get_logger().info("[FSM] DAMPING -> JOINT_DEFAULT (half-squat prep)")

    def _enter_standup(self):
        self.interp_start_pos = dict(self.joint_pos)
        self.interp_elapsed = 0.0
        self.state = State.STANDUP
        self.get_logger().info("[FSM] JOINT_DEFAULT -> STANDUP (to RL default pos)")

    def _enter_rl_active(self):
        for th in self.term_histories.values():
            th.reset()
        self.last_action[:] = 0.0
        self.prev_action[:] = 0.0
        self.policy_count = 0
        self.warmup_done = False
        self.velocity_cmd[0] = 0.0   # stand-first: enter RL_ACTIVE with forward_velocity=0
        self.state = State.RL_ACTIVE
        self.get_logger().info("[FSM] STANDUP -> RL_ACTIVE (forward=0, standing; '+' to walk)")

    def _enter_estop(self):
        self.state = State.ESTOP
        self.get_logger().warn("[FSM] -> ESTOP! Robot tilted too far. Sending damping.")

    def _rc_get_fsm_action_value(self) -> int:
        with self.lock:
            return fsm_state_to_mc_action(self.state)

    def _rc_get_speed_bounds(self):
        """Speed range for the RC (fwd_min,fwd_max,lat_min,lat_max,ang_min,ang_max).

        Kept consistent with the clamping in _apply_rc_velocity: forward uses
        cmd_vx_min/max, lateral [-0.3,0.3], yaw [-0.2,0.2].
        """
        return (self.cmd_vx_min, self.cmd_vx_max, -0.3, 0.3, -0.2, 0.2)

    def _rc_on_action(self, event: str) -> None:
        """Intent expressed by the RC via SetMcAction; must honor the existing FSM guards and safety."""
        with self.lock:
            state = self.state
            if event == "damping":
                if state in (State.RL_ACTIVE, State.JOINT_DEFAULT,
                             State.STANDUP, State.ESTOP):
                    self._enter_damping()
                return
            if event == "prepare":
                if state == State.DAMPING:
                    self._enter_joint_default()
                return
            if event == "locomotion":
                if state == State.JOINT_DEFAULT and self.joint_default_done:
                    self._enter_standup()
                return

    def _apply_rc_velocity(self) -> None:
        """Called inside the RL_ACTIVE control loop: fresh RC velocity overrides velocity_cmd.

        When the RC has never connected (no velocity ever received) it does not take
        over at all -- the keyboard fallback is kept. Only after the RC has connected
        and then timed out is velocity zeroed, to stop the robot from walking on after
        the gamepad drops out.
        """
        if not self.rc_enabled or self.rc_state is None:
            return
        # RC has never received velocity: don't take over, leave it to keyboard control
        if not self.rc_state.has_ever_received():
            return
        v = self.rc_state.get_velocity(now=time.monotonic())
        if v is None:
            # RC connected and then timed out: zero out to prevent walking on
            self.velocity_cmd[0] = 0.0
            self.velocity_cmd[1] = 0.0
            self.velocity_cmd[2] = 0.0
            return
        fwd, lat, ang = v
        self.velocity_cmd[0] = float(np.clip(fwd, self.cmd_vx_min, self.cmd_vx_max))
        self.velocity_cmd[1] = float(np.clip(lat, -0.3, 0.3))
        self.velocity_cmd[2] = float(np.clip(ang, -0.2, 0.2))

    def _compute_obs_terms(self):
        ang_vel_local = self.imu_ang_vel.copy()
        base_ang_vel = (ang_vel_local * ANG_VEL_SCALE).astype(np.float32)

        quat = self.imu_quat_wxyz.copy()
        projected_gravity = get_gravity_orientation(quat).astype(np.float32)
        # IMU pitch bias correction: set via --imu_pitch_bias (rad equivalent in gravity projection).
        # Measure: hold robot visually upright, record grav.x; that value is the bias.
        projected_gravity[0] -= self.args.imu_pitch_bias

        joint_pos_all = np.zeros(31, dtype=np.float32)
        joint_vel_all = np.zeros(31, dtype=np.float32)
        for i, name in enumerate(ISAACLAB_JOINT_ORDER):
            default_pos = DEFAULT_JOINT_POS.get(name, 0.0)
            joint_pos_all[i] = self.joint_pos.get(name, default_pos) - default_pos
            joint_vel_all[i] = self.joint_vel.get(name, 0.0) * JOINT_VEL_SCALE

        return {
            "base_ang_vel": base_ang_vel,
            "projected_gravity": projected_gravity,
            "velocity_commands": self.velocity_cmd.copy(),
            "joint_pos_rel": joint_pos_all,
            "joint_vel_rel": joint_vel_all,
            "last_action": self.last_action.copy(),
        }

    def _build_obs(self):
        obs_parts = []
        for term_name in OBS_TERM_DIMS:
            obs_parts.append(self.term_histories[term_name].flatten())
        return np.concatenate(obs_parts).reshape(1, -1).astype(np.float32)

    def _control_loop(self):
        with self.lock:
            self.tick_count += 1

            if self.state == State.WAITING:
                if self._sensors_ready():
                    self.get_logger().info("[FSM] Sensors ready -> entering DAMPING mode")
                    self._enter_damping()
                elif self.tick_count % 250 == 0:
                    self.get_logger().info(
                        f"Waiting... IMU={self.imu_ready}, joints={self.joints_ready}"
                    )
                return

            if self.state == State.DAMPING:
                self._publish_damping()
                if self.tick_count % 250 == 0:
                    self.get_logger().info(
                        "[DAMPING] Press Enter to start..."
                    )
                return

            if self.state == State.JOINT_DEFAULT:
                if not self.joint_default_done:
                    self.interp_elapsed += CONTROL_DT
                    progress = min(self.interp_elapsed / JOINT_DEFAULT_DURATION, 1.0)
                    progress = progress * progress * (3.0 - 2.0 * progress)

                    interp_pos = {}
                    for name in ALL_JOINT_NAMES:
                        start = self.interp_start_pos.get(name, 0.0)
                        target = JOINT_DEFAULT_POS.get(name, 0.0)
                        interp_pos[name] = start + (target - start) * progress

                    self._publish_position(interp_pos, JOINT_DEFAULT_KP, JOINT_DEFAULT_KD)

                    if self.interp_elapsed >= JOINT_DEFAULT_DURATION:
                        self.joint_default_done = True
                        self.get_logger().info(
                            "[JOINT_DEFAULT] done — press Enter to enter STANDUP once stable"
                        )
                    elif self.tick_count % 250 == 0:
                        self.get_logger().info(
                            f"[JOINT_DEFAULT] progress={progress*100:.0f}%"
                        )
                else:
                    self._publish_position(JOINT_DEFAULT_POS, JOINT_DEFAULT_KP, JOINT_DEFAULT_KD)
                    if self.tick_count % 500 == 0:
                        self.get_logger().info("[JOINT_DEFAULT hold] press Enter to enter STANDUP")
                return

            if self.state == State.STANDUP:
                self.interp_elapsed += CONTROL_DT
                standup_dur = 1.5
                progress = min(self.interp_elapsed / standup_dur, 1.0)
                progress = 0.5 * (1 - np.cos(np.pi * progress))

                interp_pos = {}
                for name in ALL_JOINT_NAMES:
                    start = self.interp_start_pos.get(name, 0.0)
                    target = DEFAULT_JOINT_POS.get(name, 0.0)
                    interp_pos[name] = start + (target - start) * progress

                self._publish_position(interp_pos, DEPLOY_KP, DEPLOY_KD)

                if self.interp_elapsed >= standup_dur:
                    self._enter_rl_active()
                elif self.tick_count % 250 == 0:
                    self.get_logger().info(
                        f"[STANDUP] progress={progress*100:.0f}%"
                    )
                return

            if self.state == State.RL_ACTIVE:
                if not self._check_safety():
                    self._enter_estop()
                    self._publish_damping()
                    return

                self._apply_rc_velocity()

                policy_tick = int(POLICY_DT / CONTROL_DT)
                if self.tick_count % policy_tick == 0:
                    terms = self._compute_obs_terms()
                    for term_name, val in terms.items():
                        self.term_histories[term_name].append(val)

                    obs = self._build_obs()
                    obs = np.clip(obs, -OBS_CLIP, OBS_CLIP)

                    output = self.session.run([self.output_name], {self.input_name: obs})[0]
                    action = np.clip(output[0], -ACTION_CLIP, ACTION_CLIP)

                    if not self.warmup_done:
                        if self.policy_count < HISTORY_LENGTH:
                            action[:] = 0.0
                        else:
                            self.warmup_done = True
                            self.get_logger().info(f"[RL] Warmup done ({HISTORY_LENGTH} frames)")

                    self.last_action[:] = action

                    for i, name in enumerate(ACTION_JOINT_ORDER):
                        default_pos = DEFAULT_JOINT_POS.get(name, 0.0)
                        desired = float(action[i]) * ACTION_SCALE + default_pos
                        prev = self.target_joint_pos.get(name, default_pos)
                        delta = np.clip(desired - prev, -ACTION_DELTA_LIMIT, ACTION_DELTA_LIMIT)
                        self.target_joint_pos[name] = prev + delta

                    self.policy_count += 1
                    if self.policy_count % 50 == 0:
                        grav = terms["projected_gravity"]
                        ang = terms["base_ang_vel"] / ANG_VEL_SCALE  # undo scale for display
                        self.get_logger().info(
                            f"[RL_ACTIVE] step={self.policy_count}, "
                            f"cmd=(vx={self.velocity_cmd[0]:+.2f}, vy={self.velocity_cmd[1]:+.2f}, "
                            f"vyaw={self.velocity_cmd[2]:+.2f}), |a|={np.linalg.norm(action):.3f}\n"
                            f"  projected_gravity=[{grav[0]:+.4f}, {grav[1]:+.4f}, {grav[2]:+.4f}]"
                            f"  (upright: [~0, ~0, ~-1]; x<0 leaning back, x>0 leaning forward)\n"
                            f"  imu_ang_vel     =[{ang[0]:+.4f}, {ang[1]:+.4f}, {ang[2]:+.4f}] rad/s"
                        )

                self._publish_position(self.target_joint_pos, DEPLOY_KP, DEPLOY_KD)
                return

            if self.state == State.ESTOP:
                self._publish_damping()
                if self.tick_count % 500 == 0:
                    self.get_logger().warn("[ESTOP] Damping active. Ctrl+C to exit.")
                return


def input_listener(node: "X2RLDeployNode"):
    """Non-blocking single-key input (delegates terminal handling to KeyListener).
    Each keypress takes effect immediately, no Enter needed."""

    def _handle_key(key: str):
        with node.lock:
            state = node.state

            if key == "?":
                print(DEPLOY_HELP_TEXT)
                return

            # Esc -> exit (DAMPING/ESTOP/WAITING)
            # Arrow keys etc. send escape sequences (\x1b[...); distinguish bare Esc from a sequence
            if key == "\x1b":
                # Check for trailing bytes (escape sequence like \x1b[A up arrow)
                ready, _, _ = select.select([sys.stdin], [], [], 0.0)
                if ready:
                    # It's an escape sequence; consume remaining bytes and ignore
                    while True:
                        r, _, _ = select.select([sys.stdin], [], [], 0.0)
                        if r:
                            sys.stdin.read(1)
                        else:
                            break
                    return
                if state in (State.DAMPING, State.WAITING, State.ESTOP):
                    node.get_logger().info("[USER] Esc -> exit")
                    rclpy.shutdown()
                else:
                    node.get_logger().warn(
                        f"[USER] Esc invalid in {state.name}; Backspace to DAMPING first, then Esc"
                    )
                return

            # Backspace -> DAMPING (emergency brake)
            if key in ("\x7f", "\x08"):
                if state in (State.RL_ACTIVE, State.JOINT_DEFAULT, State.STANDUP, State.ESTOP):
                    node.get_logger().info(f"[USER] Backspace {state.name} -> DAMPING")
                    node._enter_damping()
                else:
                    node.get_logger().warn(f"[USER] Backspace invalid in {state.name}")
                return

            # Enter/Space -> FSM advance (DAMPING) or zero velocity (RL_ACTIVE)
            # cbreak keeps ICRNL so Enter arrives as \n; raw gives \r. Accept both to be safe.
            if key in ("\r", "\n", " "):
                if state == State.DAMPING:
                    node.get_logger().info("[USER] Enter/Space: DAMPING -> JOINT_DEFAULT")
                    node._enter_joint_default()
                elif state == State.JOINT_DEFAULT:
                    if node.joint_default_done:
                        node.get_logger().info("[USER] Enter/Space: JOINT_DEFAULT -> STANDUP")
                        node._enter_standup()
                    else:
                        node.get_logger().info("[USER] JOINT_DEFAULT interp not done, please wait")
                elif state == State.RL_ACTIVE:
                    if key == " ":
                        # Space in RL_ACTIVE: zero all velocities
                        node.velocity_cmd[0] = 0.0
                        node.velocity_cmd[1] = 0.0
                        node.velocity_cmd[2] = 0.0
                        node.get_logger().info("[USER] Space: all velocity -> 0 (stand)")
                    else:
                        node.get_logger().info(
                            "[USER] Enter invalid in RL_ACTIVE (safety). Backspace to DAMPING, WASD/QE to set speed"
                        )
                else:
                    node.get_logger().info(f"[USER] Enter/Space invalid in {state.name}")
                return

            # Velocity control - RL_ACTIVE only
            if state != State.RL_ACTIVE:
                if key.lower() in ("w", "a", "s", "d", "q", "e"):
                    node.get_logger().warn(
                        f"[USER] velocity key '{key}' only works in RL_ACTIVE, current {state.name}"
                    )
                return

            old_vx = float(node.velocity_cmd[0])
            old_vy = float(node.velocity_cmd[1])
            old_vyaw = float(node.velocity_cmd[2])

            if key.lower() == "w":
                if abs(old_vx) < 1e-6:
                    new_vx = node.initial_cmd_vx
                else:
                    new_vx = old_vx + node.cmd_vx_step
                new_vx = max(node.cmd_vx_min, min(new_vx, node.cmd_vx_max))
                node.velocity_cmd[0] = new_vx
                node.get_logger().info(f"[USER] W forward {old_vx:+.2f} -> {new_vx:+.2f}")

            elif key.lower() == "s":
                new_vx = step_velocity(old_vx, -node.cmd_vx_step, node.cmd_vx_min, node.cmd_vx_max)
                node.velocity_cmd[0] = new_vx
                node.get_logger().info(f"[USER] S forward {old_vx:+.2f} -> {new_vx:+.2f}")

            elif key.lower() == "a":
                new_vy = step_velocity(old_vy, -node.cmd_vy_step, -0.3, 0.3)
                node.velocity_cmd[1] = new_vy
                node.get_logger().info(f"[USER] A lateral {old_vy:+.2f} -> {new_vy:+.2f}")

            elif key.lower() == "d":
                new_vy = step_velocity(old_vy, node.cmd_vy_step, -0.3, 0.3)
                node.velocity_cmd[1] = new_vy
                node.get_logger().info(f"[USER] D lateral {old_vy:+.2f} -> {new_vy:+.2f}")

            elif key.lower() == "q":
                new_vyaw = step_velocity(old_vyaw, -node.cmd_vyaw_step, -0.2, 0.2)
                node.velocity_cmd[2] = new_vyaw
                node.get_logger().info(f"[USER] Q yaw {old_vyaw:+.2f} -> {new_vyaw:+.2f}")

            elif key.lower() == "e":
                new_vyaw = step_velocity(old_vyaw, node.cmd_vyaw_step, -0.2, 0.2)
                node.velocity_cmd[2] = new_vyaw
                node.get_logger().info(f"[USER] E yaw {old_vyaw:+.2f} -> {new_vyaw:+.2f}")

    # cbreak mode (keeps ISIG so Ctrl+C exits, and OPOST/ONLCR so log \n aligns);
    # loop until rclpy shuts down. Per-key dispatch stays local in _handle_key.
    KeyListener(_handle_key, cbreak=True, should_continue=rclpy.ok).run()


def main():
    parser = argparse.ArgumentParser(description="X2 Sim2Real RL Deploy (Python)")
    parser.add_argument("--onnx", type=str, default=ONNX_PATH)
    parser.add_argument("--cmd_vx", type=float, default=0.0,
                        help="DEPRECATED: RL_ACTIVE always starts at forward=0; use '+' to set speed. Kept for back-compat, no effect")
    parser.add_argument("--cmd_vy", type=float, default=0.0,
                        help="initial lateral_velocity (tunable at runtime with A/D)")
    parser.add_argument("--cmd_vyaw", type=float, default=0.0,
                        help="initial angular_velocity (tunable at runtime with Q/E)")
    parser.add_argument("--initial_cmd_vx", type=float, default=0.2,
                        help="forward_velocity that first '+' jumps to from 0 in RL_ACTIVE "
                             "(default 0.2; policy walks stably only at >=0.2, stays still below)")
    parser.add_argument("--cmd_vx_step", type=float, default=0.1,
                        help="'+/-' step size for forward_velocity (default 0.1)")
    parser.add_argument("--cmd_vx_max", type=float, default=0.5,
                        help="forward_velocity upper limit (default 0.5, conservative vs training range [-0.5, 1.0])")
    parser.add_argument("--cmd_vx_min", type=float, default=-0.3,
                        help="forward_velocity lower limit (default -0.3)")
    parser.add_argument("--cmd_vy_step", type=float, default=0.1,
                        help="A/D step size for lateral_velocity (default 0.1)")
    parser.add_argument("--cmd_vyaw_step", type=float, default=0.1,
                        help="Q/E step size for angular_velocity (default 0.1)")
    parser.add_argument("--imu_pitch_bias", type=float, default=0.0,
                        help="IMU pitch bias correction: subtract this from projected_gravity[x]. "
                             "Measure: hold robot visually upright, record grav.x; that value is the bias. "
                             "Example: --imu_pitch_bias 0.08")
    parser.add_argument("--damping_kd", type=float, default=DAMPING_KD,
                        help=f"Damping coefficient in DAMPING/ESTOP state (default: {DAMPING_KD}, validated on hardware)")
    parser.add_argument("--enable_rc_proxy", action="store_true",
                        help="Enable RC gamepad integration: subscribe to /aima/mc/locomotion/velocity and impersonate SetMcAction/GetMcAction. Stop the stock MC first.")
    parser.add_argument("--rc_velocity_topic", type=str,
                        default="/aima/mc/locomotion/velocity",
                        help="RC velocity-command topic")
    parser.add_argument("--rc_timeout", type=float, default=0.5,
                        help="RC velocity timeout in seconds; after timeout the RL velocity is zeroed")
    args = parser.parse_args()

    if abs(args.cmd_vx) > 1e-6:
        print(f"\033[33m[WARN] --cmd_vx={args.cmd_vx} deprecated, RL_ACTIVE starts at 0. Use '+' to ramp to {args.initial_cmd_vx} to walk.\033[0m")

    rclpy.init()
    node = X2RLDeployNode(args)

    input_thread = threading.Thread(target=input_listener, args=(node,), daemon=False)
    input_thread.start()

    print("=" * 70)
    print("  X2 Sim2Real RL Deploy (AimDK-aligned FSM v2)")
    print(f"  ONNX:    {args.onnx}")
    print(f"  initial_cmd_vx / step / [min, max]:")
    print(f"           {args.initial_cmd_vx} / {args.cmd_vx_step} / "
          f"[{args.cmd_vx_min}, {args.cmd_vx_max}] m/s")
    print(f"  vy_step / vyaw_step: {args.cmd_vy_step} / {args.cmd_vyaw_step}")
    print(f"  damping_kd: {args.damping_kd} (DAMPING/ESTOP)")
    print(f"  freq:    {1.0/POLICY_DT:.0f} Hz policy, {1.0/CONTROL_DT:.0f} Hz control")
    print("=" * 70)
    print("  FSM: WAITING -> DAMPING -> [Enter #1] -> JOINT_DEFAULT -> [Enter #2] -> STANDUP -> RL_ACTIVE")
    print("       RL_ACTIVE: WASD=move, QE=turn, Space=stand in place, Backspace=back to DAMPING")
    print("       ESTOP (tilt>limit): auto -> DAMPING")
    print("       Esc: DAMPING/ESTOP/WAITING -> exit")
    print(f"  quick help: type '?' for full key table")
    print("=" * 70)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.get_logger().info("Shutting down, sending final damping commands...")
            with node.lock:
                for _ in range(5):
                    node._publish_damping()
                    time.sleep(0.01)
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        input_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
