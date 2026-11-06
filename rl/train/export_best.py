#!/usr/bin/env python3
"""
Pick the checkpoint with the highest Train/mean_reward from tensorboard logs,
then export it to ONNX and TorchScript (JIT).

Usage:
    python3 export_best.py                       # auto-detect latest run, export its best checkpoint
    python3 export_best.py <log_dir>              # specify a run dir, export its best checkpoint
    python3 export_best.py <log_dir> model_44100.pt   # export a specific checkpoint (skip picking)
    python3 export_best.py --top 10 --smooth 10   # tune best-checkpoint selection
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_LOG_BASE = os.path.normpath(os.path.join(_HERE, "..", "agibot_rl_lab", "logs", "rsl_rl"))


# ---------------------------------------------------------------------------
# Checkpoint selection (formerly pick_best_checkpoint.py)
# ---------------------------------------------------------------------------

def find_latest_log_dir(base=_DEFAULT_LOG_BASE):
    candidates = []
    for task in os.listdir(base):
        task_dir = os.path.join(base, task)
        if not os.path.isdir(task_dir):
            continue
        for run in os.listdir(task_dir):
            run_dir = os.path.join(task_dir, run)
            if os.path.isdir(run_dir):
                candidates.append(run_dir)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def build_tag_map(ea):
    """Build dict: tag -> {step: value}"""
    tag_map = {}
    for tag in ea.Tags().get("scalars", []):
        tag_map[tag] = {e.step: e.value for e in ea.Scalars(tag)}
    return tag_map


def get_val(tag_map, tag, step):
    return tag_map.get(tag, {}).get(step, None)


def print_iter_summary(tag_map, step, max_iterations):
    width = 75

    def row(label, value, fmt=None):
        if value is None:
            return
        if fmt:
            val_str = fmt.format(value)
        elif isinstance(value, float):
            val_str = f"{value:.4f}"
        else:
            val_str = str(value)
        print(f"{label:>{width - len(val_str) - 2}}: {val_str}")

    print("#" * (width + 2))

    # Header
    header = f"Learning iteration {step}/{max_iterations}"
    print(f"{header:^{width + 2}}")
    print()

    # Performance
    col_time = get_val(tag_map, "Perf/collection_time", step)
    lrn_time = get_val(tag_map, "Perf/learning_time", step)
    total_time = (col_time or 0) + (lrn_time or 0)

    steps_per_sec = get_val(tag_map, "Perf/total_fps", step)
    if steps_per_sec and total_time:
        total_steps = int(steps_per_sec * step * total_time)
        row("Total steps", total_steps, fmt="{:d}")
        row("Steps per second", int(steps_per_sec), fmt="{:d}")
    if col_time is not None:
        row("Collection time", col_time, fmt="{:.3f}s")
    if lrn_time is not None:
        row("Learning time", lrn_time, fmt="{:.3f}s")

    print()

    # Losses
    row("Mean value loss",     get_val(tag_map, "Loss/value", step),     fmt="{:.4f}")
    row("Mean surrogate loss", get_val(tag_map, "Loss/surrogate", step), fmt="{:.4f}")
    row("Mean entropy loss",   get_val(tag_map, "Loss/entropy", step),   fmt="{:.4f}")

    print()

    # Main metrics
    row("Mean reward",         get_val(tag_map, "Train/mean_reward", step),         fmt="{:.2f}")
    row("Mean episode length", get_val(tag_map, "Train/mean_episode_length", step), fmt="{:.2f}")
    row("Mean action std",     get_val(tag_map, "Train/mean_action_noise_std", step), fmt="{:.2f}")

    print()

    # All Episode_Reward / Curriculum / Metrics / Episode_Termination tags
    groups = [
        "Episode_Reward",
        "Curriculum",
        "Metrics",
        "Episode_Termination",
    ]
    for group in groups:
        tags = sorted(t for t in tag_map if t.startswith(group + "/"))
        for tag in tags:
            val = get_val(tag_map, tag, step)
            row(tag, val, fmt="{:.4f}")
        if tags:
            print()

    # Footer
    print("-" * (width + 2))
    if total_time:
        row("Iteration time", total_time, fmt="{:.2f}s")
    print()


def pick_best(log_dir, top_n=5, smooth_window=5, show_iter=None):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(log_dir)
    ea.Reload()

    if "Train/mean_reward" not in ea.Tags().get("scalars", []):
        print("ERROR: 'Train/mean_reward' tag not found in log.")
        print("Available tags:", ea.Tags().get("scalars", []))
        sys.exit(1)

    tag_map = build_tag_map(ea)
    events = ea.Scalars("Train/mean_reward")
    steps = np.array([e.step for e in events])
    rewards = np.array([e.value for e in events])

    # Moving-average smoothing
    if smooth_window > 1 and len(rewards) >= smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        rewards_sm = np.convolve(rewards, kernel, mode="same")
        half = smooth_window // 2
        rewards_sm[:half] = rewards[:half]
        rewards_sm[-half:] = rewards[-half:]
    else:
        rewards_sm = rewards

    # Only consider iters that have a matching .pt file
    available = set()
    for fname in os.listdir(log_dir):
        if fname.startswith("model_") and fname.endswith(".pt"):
            try:
                it = int(fname[len("model_"):-len(".pt")])
                available.add(it)
            except ValueError:
                pass

    mask = np.isin(steps, list(available))
    if not mask.any():
        print("WARNING: No .pt checkpoints found matching log steps. Showing all steps.")
        mask = np.ones(len(steps), dtype=bool)

    valid_steps = steps[mask]
    valid_rewards_sm = rewards_sm[mask]
    valid_rewards_raw = rewards[mask]

    top_idx = np.argsort(valid_rewards_sm)[::-1][:top_n]
    max_iter = int(steps[-1]) if len(steps) > 0 else 0

    print(f"\nLog dir : {log_dir}")
    print(f"Total logged iters : {len(steps)}")
    print(f"Available checkpoints: {mask.sum()}")
    print(f"Smooth window : {smooth_window}\n")
    print(f"{'Rank':>4}  {'Iter':>8}  {'Reward(smooth)':>14}  {'Reward(raw)':>12}  {'Checkpoint'}")
    print("-" * 72)
    for rank, i in enumerate(top_idx, 1):
        it = valid_steps[i]
        ckpt = os.path.join(log_dir, f"model_{it}.pt")
        exists = "v" if os.path.exists(ckpt) else "x"
        print(f"{rank:>4}  {it:>8}  {valid_rewards_sm[i]:>14.4f}  {valid_rewards_raw[i]:>12.4f}  {exists} model_{it}.pt")

    best_iter = int(valid_steps[top_idx[0]])
    best_ckpt = os.path.join(log_dir, f"model_{best_iter}.pt")
    print(f"\nBest checkpoint: {best_ckpt}")

    # Print detailed metrics
    target_iter = show_iter if show_iter is not None else best_iter
    all_steps = sorted(tag_map.get("Train/mean_reward", {}).keys())
    if target_iter not in all_steps and all_steps:
        closest = min(all_steps, key=lambda s: abs(s - target_iter))
        print(f"\n(no exact record for iter {target_iter}, showing nearest iter {closest})")
        target_iter = closest

    print()
    print_iter_summary(tag_map, target_iter, max_iter)

    return best_ckpt


# ---------------------------------------------------------------------------
# Export (formerly export_best.py)
# ---------------------------------------------------------------------------

class PolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.mlp(x)


def export_checkpoint(ckpt_path):
    run_dir = os.path.dirname(ckpt_path)
    output_dir = os.path.join(run_dir, "exported")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    print("Checkpoint keys:", list(ckpt.keys()))

    actor_sd = ckpt["actor_state_dict"]
    print("\nActor state dict:")
    for k, v in actor_sd.items():
        print(f"  {k}: {v.shape}")

    mlp_keys = [k for k in actor_sd if k.startswith("mlp.")]
    input_dim = actor_sd["mlp.0.weight"].shape[1]
    output_dim = actor_sd[mlp_keys[-1]].shape[0]
    print(f"\nInput dim: {input_dim}, Output dim: {output_dim}")

    model = PolicyNet(input_dim, output_dim)
    state_dict = {k: v for k, v in actor_sd.items() if k.startswith("mlp.")}
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    onnx_path = os.path.join(output_dir, "policy.onnx")
    dummy = torch.randn(1, input_dim)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["obs"], output_names=["actions"],
        opset_version=18,
    )
    print(f"\nONNX exported: {onnx_path}")

    jit_path = os.path.join(output_dir, "policy.pt")
    traced = torch.jit.trace(model, dummy)
    traced.save(jit_path)
    print(f"JIT exported: {jit_path}")

    out = model(dummy)
    print(f"\nVerification - input shape: {dummy.shape}, output shape: {out.shape}")
    print(f"Output sample: {out[0, :5].detach().numpy()}")
    print("\nDone! Files:")
    print(f"  {onnx_path}")
    print(f"  {jit_path}")

    return onnx_path, jit_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pick the best checkpoint (by Train/mean_reward) and export it to ONNX + JIT."
    )
    parser.add_argument("log_dir", nargs="?", default=None, help="Training run dir (default: auto-detect latest)")
    parser.add_argument("model", nargs="?", default=None, help="Specific checkpoint file (skips picking, e.g. model_44100.pt)")
    parser.add_argument("--top",    type=int, default=5, help="Show top N checkpoints (default: 5)")
    parser.add_argument("--smooth", type=int, default=5, help="Smoothing window size (default: 5)")
    parser.add_argument("--iter",   type=int, default=None, help="Show detailed metrics for this iter (default: best)")
    args = parser.parse_args()

    log_dir = args.log_dir
    if log_dir is None:
        log_dir = find_latest_log_dir()
        if log_dir is None:
            print("ERROR: Could not find any log directory. Please specify one.")
            sys.exit(1)
        print(f"Auto-detected log dir: {log_dir}")

    if not os.path.isdir(log_dir):
        print(f"ERROR: Directory not found: {log_dir}")
        sys.exit(1)

    if args.model is not None:
        # Explicit checkpoint given: skip picking, export it directly.
        ckpt_path = os.path.join(log_dir, args.model)
        if not os.path.isfile(ckpt_path):
            print(f"ERROR: Checkpoint not found: {ckpt_path}")
            sys.exit(1)
        print(f"Using specified checkpoint: {ckpt_path}")
    else:
        ckpt_path = pick_best(log_dir, top_n=args.top, smooth_window=args.smooth, show_iter=args.iter)

    print()
    export_checkpoint(ckpt_path)


if __name__ == "__main__":
    main()
