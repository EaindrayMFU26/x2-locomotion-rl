"""X2 joint PD gain loader. Single source of truth = joint_gains.yaml in this dir. Only depends on PyYAML."""
from pathlib import Path
import yaml


def _yaml_path() -> Path:
    return Path(__file__).resolve().parent / "joint_gains.yaml"


def _build(raw: dict) -> dict:
    rl, st = raw["rl_gains"], raw["standup_gains"]

    # damping_kd: accept both old scalar and new {default, overrides}.
    dm = raw["damping_kd"]
    if isinstance(dm, dict):
        damping_default = float(dm["default"])
        damping_overrides = {k: float(v) for k, v in (dm.get("overrides") or {}).items()}
    else:  # old scalar form, kept for backward compat
        damping_default = float(dm)
        damping_overrides = {}

    g = {
        "rl_kp": {j: float(v["kp"]) for j, v in rl.items()},
        "rl_kd": {j: float(v["kd"]) for j, v in rl.items()},
        "standup_kp": {j: float(v["kp"]) for j, v in st.items()},
        "standup_kd": {j: float(v["kd"]) for j, v in st.items()},
        "damping_kd": damping_default,  # scalar default (backward compat: sim2sim / CLI default / print)
        # resolved per-joint map = mc GetJointDampingValue: overrides.get(name, default)
        "damping_kd_map": {j: damping_overrides.get(j, damping_default) for j in rl},
        "damping_kd_overrides": damping_overrides,  # override-only joints (deploy stacks CLI baseline on top)
    }
    _validate(g)
    return g


def load_gains() -> dict:
    raw = yaml.safe_load(_yaml_path().read_text(encoding="utf-8"))
    return _build(raw)


def _validate(g: dict) -> None:
    # g must be _build output (all kp/kd already float)
    n = len(g["rl_kp"])
    for tbl in ("rl_kp", "rl_kd", "standup_kp", "standup_kd"):
        if len(g[tbl]) != n:
            raise ValueError(f"{tbl} joint count {len(g[tbl])} != {n}")
        for j, v in g[tbl].items():
            if not (isinstance(v, float) and v >= 0):
                raise ValueError(f"{tbl}[{j}] invalid gain: {v}")
    if not (g["damping_kd"] > 0):
        raise ValueError(f"damping_kd default must be positive: {g['damping_kd']}")
    rl_joints = set(g["rl_kp"])
    # override keys must be known rl joints, values positive
    for j, v in g["damping_kd_overrides"].items():
        if j not in rl_joints:
            raise ValueError(f"damping_kd.overrides unknown joint: {j} (not in rl_gains)")
        if not (v > 0):
            raise ValueError(f"damping_kd.overrides[{j}] must be positive: {v}")
    # resolved map must cover exactly the rl joints
    if set(g["damping_kd_map"]) != rl_joints:
        diff = rl_joints.symmetric_difference(set(g["damping_kd_map"]))
        raise ValueError(f"damping_kd_map joint set mismatch: {diff}")
    if set(g["standup_kp"]) != rl_joints:
        diff = rl_joints.symmetric_difference(set(g["standup_kp"]))
        raise ValueError(f"rl/standup joint name sets differ: {diff}")
