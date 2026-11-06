#!/usr/bin/env python3
"""
Batch training script for gait-refinement experiments.
Runs multiple experiments sequentially with different reward weights.
Designed for ~80h weekend unattended training (4 experiments × ~15h each).

Usage:
    conda activate isaaclab
    export ISAACLAB_PATH=~/IsaacLab
    cd ~/mc-rl
    nohup python rl/train/batch_train.py > batch_train.log 2>&1 &
"""

import re
import subprocess
import sys
import shutil
import time
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parents[2]  # repo root
CFG_FILE = SCRIPT_DIR / "rl/agibot_rl_lab/source/agibot_rl_lab/agibot_rl_lab/tasks/locomotion/robots/x2/29dof/velocity_env_cfg.py"
TRAIN_DIR = SCRIPT_DIR / "rl/agibot_rl_lab"
MAX_ITER = 40000

EXPERIMENTS = [
    {
        "name": "exp-E-gentle-arms",
        "desc": "gentle arms: arms=-0.25, track_ang=0.75, base_ang=-0.1, action_rate=-0.08",
        "joint_deviation_arms": -0.25,
        "track_ang_vel_z": 0.75,
        "base_angular_velocity": -0.1,
        "action_rate": -0.08,
    },
    {
        "name": "exp-F-moderate",
        "desc": "moderate all: arms=-0.3, track_ang=1.0, base_ang=-0.15, action_rate=-0.1",
        "joint_deviation_arms": -0.3,
        "track_ang_vel_z": 1.0,
        "base_angular_velocity": -0.15,
        "action_rate": -0.1,
    },
    {
        "name": "exp-G-strong-track",
        "desc": "strong tracking: arms=-0.25, track_ang=1.0, base_ang=-0.1, action_rate=-0.08",
        "joint_deviation_arms": -0.25,
        "track_ang_vel_z": 1.0,
        "base_angular_velocity": -0.1,
        "action_rate": -0.08,
    },
    {
        "name": "exp-H-combined-moderate",
        "desc": "combined moderate: arms=-0.3, track_ang=1.0, base_ang=-0.15, action_rate=-0.08",
        "joint_deviation_arms": -0.3,
        "track_ang_vel_z": 1.0,
        "base_angular_velocity": -0.15,
        "action_rate": -0.08,
    },
]


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def apply_config(exp: dict):
    content = CFG_FILE.read_text()

    replacements = {
        r'(track_ang_vel_z = RewTerm\(\s*func=mdp\.track_ang_vel_z_exp,\s*weight=)[\d.]+': 
            rf'\g<1>{exp["track_ang_vel_z"]}',
        r'(base_angular_velocity = RewTerm\(func=mdp\.ang_vel_xy_l2,\s*weight=)-?[\d.]+': 
            rf'\g<1>{exp["base_angular_velocity"]}',
        r'(action_rate = RewTerm\(func=mdp\.action_rate_l2,\s*weight=)-?[\d.]+': 
            rf'\g<1>{exp["action_rate"]}',
        r'(joint_deviation_arms = RewTerm\(\s*func=mdp\.joint_deviation_l1,\s*weight=)-?[\d.]+': 
            rf'\g<1>{exp["joint_deviation_arms"]}',
    }

    for pattern, replacement in replacements.items():
        content, n = re.subn(pattern, replacement, content)
        if n == 0:
            log(f"  WARNING: pattern not found: {pattern[:60]}...")

    CFG_FILE.write_text(content)


def verify_config(exp: dict):
    content = CFG_FILE.read_text()
    checks = {
        "track_ang_vel_z": str(exp["track_ang_vel_z"]),
        "base_angular_velocity": str(exp["base_angular_velocity"]),
        "action_rate": str(exp["action_rate"]),
        "joint_deviation_arms": str(exp["joint_deviation_arms"]),
    }
    for name, expected in checks.items():
        pattern = rf'{name}.*?weight={re.escape(expected)}'
        if not re.search(pattern, content, re.DOTALL):
            log(f"  VERIFY FAILED: {name} expected weight={expected}")
            return False
        log(f"  OK: {name} weight={expected}")
    return True


def run_training():
    cmd = [
        sys.executable, "scripts/rsl_rl/train.py",
        "--task", "Agibot-X2-29dof-Velocity",
        "--headless",
        "--max_iterations", str(MAX_ITER),
    ]
    log(f"  CMD: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=str(TRAIN_DIR),
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    return result.returncode


def main():
    log("=" * 60)
    log("Batch Training for Gait Refinement")
    log(f"Experiments: {len(EXPERIMENTS)}")
    log(f"Max iterations per experiment: {MAX_ITER}")
    log(f"Estimated total time: {len(EXPERIMENTS) * 15}h")
    log("=" * 60)

    backup = CFG_FILE.with_suffix(".py.bak")
    shutil.copy2(CFG_FILE, backup)
    log(f"Config backed up to {backup}")

    results = []
    for i, exp in enumerate(EXPERIMENTS):
        log("")
        log(f"{'=' * 60}")
        log(f"Experiment {i+1}/{len(EXPERIMENTS)}: {exp['name']}")
        log(f"  {exp['desc']}")
        log(f"{'=' * 60}")

        apply_config(exp)
        if not verify_config(exp):
            log(f"  SKIPPING {exp['name']} due to config verification failure!")
            results.append((exp["name"], "SKIP"))
            continue

        t0 = time.time()
        retcode = run_training()
        elapsed_h = (time.time() - t0) / 3600

        status = "OK" if retcode == 0 else f"FAIL(rc={retcode})"
        results.append((exp["name"], status, f"{elapsed_h:.1f}h"))
        log(f"  Experiment {exp['name']} finished: {status} in {elapsed_h:.1f}h")

    shutil.copy2(backup, CFG_FILE)
    log("")
    log("=" * 60)
    log("Config restored to original.")
    log("=" * 60)
    log("")
    log("Summary:")
    for r in results:
        log(f"  {r[0]}: {r[1]} {r[2] if len(r) > 2 else ''}")
    log("")
    log("Done! Use TensorBoard to compare experiments:")
    log(f"  tensorboard --logdir {TRAIN_DIR / 'logs/rsl_rl/agibot_x2_29dof_velocity'}")


if __name__ == "__main__":
    main()
