"""RC gamepad bridge for `x2_deploy_pkg/deploy_x2.py` — opt-in via --enable_rc_proxy.

Translates the SOC0 stock RC's output into deploy_x2's internal control without
touching the policy-inference path. Provides RcState (pure-logic velocity/action
container, unit-testable, no ROS dependency) and RcControlProxy (binds an rclpy node:
velocity subscription + SetMcAction/GetMcAction/GetMcLocomotionSpeedStatus service
shims that impersonate the stock MC). Ships to the robot inside x2_config/ alongside
deploy_config.py; deploy_x2.py adds x2_config/deploy to sys.path and flat-imports it.
"""

from dataclasses import dataclass


@dataclass
class RcCommand:
    forward: float
    lateral: float
    angular: float
    stamp: float


class RcState:
    """Velocity/action state container.

    The current deployment uses rclpy's single-threaded executor, so the velocity
    callback and the control loop run serially on the same thread; no internal
    locking is needed. RcCommand is replaced wholesale each time (never mutated in
    place), so reads and writes are inherently safe. If this is ever switched to a
    MultiThreadedExecutor with callbacks in concurrent groups, add locking inside
    this class.
    """

    def __init__(self, timeout: float):
        self._timeout = float(timeout)
        self._cmd: RcCommand | None = None
        self.last_action_value: int = 0

    def update_velocity(self, forward: float, lateral: float, angular: float, now: float) -> None:
        self._cmd = RcCommand(float(forward), float(lateral), float(angular), float(now))

    def has_ever_received(self) -> bool:
        """Whether at least one RC velocity has been received. Used to distinguish
        'RC never connected' from 'connected then timed out'."""
        return self._cmd is not None

    def get_velocity(self, now: float):
        if self._cmd is None:
            return None
        if now - self._cmd.stamp > self._timeout:
            return None
        return (self._cmd.forward, self._cmd.lateral, self._cmd.angular)

    def set_action(self, value: int) -> None:
        self.last_action_value = int(value)


# --- MC action constants (int32, from aimdk_msgs/McAction) ---
MC_PASSIVE_DEFAULT = 1
MC_SOFT_EMERGENCY_STOP = 2
MC_DAMPING_DEFAULT = 3
MC_ZERO_TORQUE_DEFAULT = 4
MC_JOINT_DEFAULT = 100
MC_STAND_DEFAULT = 200
MC_LOCOMOTION_DEFAULT = 300
MC_RUN_DEFAULT = 301

# value -> action name string. The RC's action_ruler matches against the string
# name (action_desc) in publish_velocity's action_list rather than the int value,
# so the GetMcAction response must populate action_desc.
MC_ACTION_NAME = {
    MC_PASSIVE_DEFAULT: "PASSIVE_DEFAULT",
    MC_SOFT_EMERGENCY_STOP: "PASSIVE_DEFAULT",
    MC_DAMPING_DEFAULT: "DAMPING_DEFAULT",
    MC_ZERO_TORQUE_DEFAULT: "DAMPING_DEFAULT",
    MC_JOINT_DEFAULT: "JOINT_DEFAULT",
    MC_STAND_DEFAULT: "STAND_DEFAULT",
    MC_LOCOMOTION_DEFAULT: "LOCOMOTION_DEFAULT",
    MC_RUN_DEFAULT: "RUN_DEFAULT",
}

_DAMPING_SET = {MC_PASSIVE_DEFAULT, MC_SOFT_EMERGENCY_STOP,
                MC_DAMPING_DEFAULT, MC_ZERO_TORQUE_DEFAULT}
_PREPARE_SET = {MC_JOINT_DEFAULT, MC_STAND_DEFAULT}
_LOCOMOTION_SET = {MC_LOCOMOTION_DEFAULT, MC_RUN_DEFAULT}


def action_to_event(value: int) -> str:
    """Map the MC action value requested by the RC to a deploy FSM event category."""
    v = int(value)
    if v in _DAMPING_SET:
        return "damping"
    if v in _PREPARE_SET:
        return "prepare"
    if v in _LOCOMOTION_SET:
        return "locomotion"
    return "ignore"


import time as _time

try:
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
    _HAVE_RCLPY = True
except Exception:  # allow importing this module in a unit-test env without rclpy
    _HAVE_RCLPY = False


def find_conflicting_mc_providers(node, service_name, self_node_name=None):
    """Enumerate all nodes and find the [stock MC] (node name contains 'mc') that
    provides service_name.

    The standard name SetMcAction is registered by several nodes --
    MC/RC(soc0_rc)/interaction_slave/task_manager/ourselves, etc. -- but only the
    stock MC contending with this shim makes routing ambiguous; other same-named
    providers are normal. self_node_name excludes ourselves. Returns
    (mc_providers, other_providers), whose elements are fully-qualified node names.
    """
    target = "/" + service_name.lstrip("/")
    mc_providers, other_providers = [], []
    for name, ns in node.get_node_names_and_namespaces():
        full = (ns.rstrip("/") + "/" + name) if ns not in ("", "/") else "/" + name
        if self_node_name is not None and name == self_node_name:
            continue
        try:
            srvs = node.get_service_names_and_types_by_node(name, ns)
        except Exception:
            continue  # the node may have disappeared between enumeration steps
        if not any(sn == target for sn, _ in srvs):
            continue
        (mc_providers if "mc" in name.lower() else other_providers).append(full)
    return mc_providers, other_providers


class RcControlProxy:
    def __init__(self, node, state, on_action, get_fsm_action_value, deps,
                 velocity_topic, set_action_service, get_action_service, logger=None,
                 speed_status_service=None, get_speed_bounds=None):
        self._node = node
        self._state = state
        self._on_action = on_action
        self._get_fsm_action_value = get_fsm_action_value
        self._deps = deps
        self._velocity_topic = velocity_topic
        self._set_action_service = set_action_service
        self._get_action_service = get_action_service
        self._log = logger
        self._first_action_logged = False
        # Before sending each velocity the RC calls GetMcLocomotionSpeedStatus to
        # fetch the speed range; once the MC is stopped this service is missing,
        # which makes the RC stop sending velocity. Providing a shim that returns a
        # valid range restores it. Optional; skipped automatically when deps are missing.
        self._speed_status_service = speed_status_service
        self._get_speed_bounds = get_speed_bounds
        self._first_speed_query_logged = False

    def _now(self) -> float:
        return _time.monotonic()

    def _make_sub_qos(self):
        if not _HAVE_RCLPY:
            return 10
        return QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )

    def setup(self) -> None:
        d = self._deps
        # Pre-startup check: enumerate all nodes and find only the [stock MC] that
        # actually provides the standard SetMcAction name. The RC itself /
        # interaction_slave / task_manager / ourselves also register the same name,
        # which is normal and not warned about; only the stock MC (node name contains
        # 'mc') contending for the same service name makes RC request routing ambiguous.
        try:
            target = "/" + self._set_action_service.lstrip("/")
            mc, other = find_conflicting_mc_providers(
                self._node, self._set_action_service,
                self_node_name=self._node.get_name())
            if mc:
                if self._log:
                    self._log.error(
                        f"[RC] Stock MC is still providing {target} (nodes: {', '.join(mc)}) -- "
                        f"two services with the same name make control routing ambiguous; "
                        f"stop the stock MC before enabling the RC proxy.")
            elif self._log:
                extra = f"; other same-named providers are harmless: {', '.join(other)}" if other else ""
                self._log.info(f"[RC] No stock MC contending for {target}{extra}")
        except Exception as e:
            if self._log:
                self._log.warn(f"[RC] Service-conflict check failed (ignored): {e}")
        self._node.create_subscription(
            d["McLocomotionVelocity"], self._velocity_topic,
            self._velocity_cb, self._make_sub_qos())
        self._node.create_service(
            d["SetMcAction"], self._set_action_service, self._set_action_cb)
        self._node.create_service(
            d["GetMcAction"], self._get_action_service, self._get_action_cb)
        # Optional: speed-range query shim. The RC queries it every cycle; if it is
        # missing the RC stops sending velocity.
        if (self._speed_status_service and self._get_speed_bounds
                and "GetMcLocomotionSpeedStatus" in d):
            self._node.create_service(
                d["GetMcLocomotionSpeedStatus"], self._speed_status_service,
                self._speed_status_cb)
        if self._log:
            self._log.info(
                f"[RC] proxy ready: sub={self._velocity_topic}, "
                f"srv={self._set_action_service}, {self._get_action_service}, "
                f"{self._speed_status_service}")

    def _velocity_cb(self, msg) -> None:
        self._state.update_velocity(
            msg.forward_velocity, msg.lateral_velocity, msg.angular_velocity,
            now=self._now())

    def _set_action_cb(self, request, response):
        value = int(request.command.action.value)
        self._state.set_action(value)
        event = action_to_event(value)
        if not self._first_action_logged:
            self._first_action_logged = True
            if self._log:
                self._log.info("[RC] First SetMcAction received; RC->SOC1 FSM link is up")
        if self._log:
            self._log.info(f"[RC] SetMcAction value={value} -> event={event}")
        if event != "ignore":
            try:
                self._on_action(event)
            except Exception as e:  # a service callback must not raise
                if self._log:
                    self._log.error(f"[RC] on_action failed: {e}")
        response.response.header.code = 0
        response.response.message = "ok"
        try:
            from aimdk_msgs.msg import CommonState as _CS
            response.response.status.value = _CS.SUCCESS
        except Exception:
            pass
        return response

    def _get_action_cb(self, request, response):
        try:
            value = int(self._get_fsm_action_value())
            response.info.current_action.value = value
            # The RC matches action_desc (the string name) against publish_velocity's
            # action_list, so the name must be filled in correctly
            response.info.action_desc = MC_ACTION_NAME.get(value, "")
            response.info.status.value = self._deps["McActionStatus"].RUNNING
            response.header.code = 0
        except Exception as e:
            if self._log:
                self._log.error(f"[RC] GetMcAction failed: {e}")
            response.header.code = 1
        return response

    def _fill_speed_status(self, status, bounds):
        """Fill (fwd_min,fwd_max,lat_min,lat_max,ang_min,ang_max) into a McLocomotionSpeedStatus."""
        fmin, fmax, lmin, lmax, amin, amax = bounds
        status.forward_bounds.min_value = float(fmin)
        status.forward_bounds.max_value = float(fmax)
        status.lateral_bounds.min_value = float(lmin)
        status.lateral_bounds.max_value = float(lmax)
        status.angular_bounds.min_value = float(amin)
        status.angular_bounds.max_value = float(amax)
        # low speed mode is fine for speed_mode; the RC just needs a valid non-NONE value
        try:
            status.speed_mode.value = self._deps["McLocomotionSpeedMode"].MC_LOCOMOTION_LOW_SPEED
        except Exception:
            pass
        return status

    def _speed_status_cb(self, request, response):
        """RC queries the current speed range. Return the range deploy allows so the
        RC resumes publishing velocity."""
        try:
            bounds = self._get_speed_bounds()  # (fmin,fmax,lmin,lmax,amin,amax)
            # fill every mode with the same valid range so the RC never picks up an
            # all-zero mode and refuses to send
            for name in ("current_speed_status", "none_speed_status",
                         "low_speed_status", "medium_speed_status",
                         "high_speed_status", "step_speed_status"):
                if hasattr(response, name):
                    self._fill_speed_status(getattr(response, name), bounds)
            response.header.code = 0
            if not self._first_speed_query_logged:
                self._first_speed_query_logged = True
                if self._log:
                    self._log.info("[RC] First GetMcLocomotionSpeedStatus received; speed range sent, RC should start publishing velocity")
        except Exception as e:
            if self._log:
                self._log.error(f"[RC] GetMcLocomotionSpeedStatus failed: {e}")
            try:
                response.header.code = 1
            except Exception:
                pass
        return response
