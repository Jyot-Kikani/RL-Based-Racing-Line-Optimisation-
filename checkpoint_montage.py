import argparse
import glob
import os
import re

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from stable_baselines3 import SAC

from config import PLOT_DIR
from env.race import RacingEnv
from env.track import Track
from train import STAGES


def _checkpoint_step(path: str) -> int:
    match = re.search(r"_(\d+)_steps", os.path.basename(path))
    return int(match.group(1)) if match else 0


def _collect_checkpoints(stage: int, max_ckpts: int, include_best: bool, ckpt_dir: str = None):
    ckpt_dir = ckpt_dir or os.path.join("models", f"stage{stage}", "checkpoints")
    ckpt_files = sorted(
        glob.glob(os.path.join(ckpt_dir, f"stage{stage}_sac_*_steps.zip")),
        key=_checkpoint_step,
    )

    if include_best:
        best_base_dir = os.path.dirname(ckpt_dir)
        best_path = os.path.join(best_base_dir, "best_model.zip")
        if os.path.exists(best_path):
            ckpt_files.append(best_path)

    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")

    if len(ckpt_files) > max_ckpts:
        indices = np.linspace(0, len(ckpt_files) - 1, max_ckpts, dtype=int)
        ckpt_files = [ckpt_files[i] for i in indices]

    return ckpt_files


def _rollout(model_path: str, max_steps: int = 5000):
    import config as cfg

    cfg.HEADLESS = True
    model = SAC.load(model_path)
    env = RacingEnv(render_mode=None)

    obs, _ = env.reset()
    xs, ys, speeds = [], [], []
    terminated = truncated = False
    steps = 0

    while not (terminated or truncated) and steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        xs.append(env.car.x)
        ys.append(env.car.y)
        speeds.append(env.car.speed * 3.6)
        steps += 1

    env.close()
    return {
        "xs": np.array(xs),
        "ys": np.array(ys),
        "speeds": np.array(speeds),
        "steps": steps,
        "completed": bool(truncated and steps >= max_steps),
    }


def _prepare_runs(stage: int, max_ckpts: int, include_best: bool, ckpt_dir: str = None):
    ckpt_files = _collect_checkpoints(
        stage,
        max_ckpts=max_ckpts,
        include_best=include_best,
        ckpt_dir=ckpt_dir,
    )
    runs = []
    global_vmin = np.inf
    global_vmax = -np.inf

    for idx, path in enumerate(ckpt_files, start=1):
        step = _checkpoint_step(path)
        label = "Best Model" if "best_model" in path else f"{step // 1000}k steps"
        print(f"[{idx}/{len(ckpt_files)}] Rolling out {label} ...", end=" ", flush=True)
        run = _rollout(path)
        run["label"] = label
        run["path"] = path
        run["mean_speed"] = float(run["speeds"].mean()) if len(run["speeds"]) else 0.0
        run["max_speed"] = float(run["speeds"].max()) if len(run["speeds"]) else 0.0
        runs.append(run)
        if len(run["speeds"]):
            global_vmin = min(global_vmin, float(run["speeds"].min()))
            global_vmax = max(global_vmax, float(run["speeds"].max()))
        print(f"done ({run['steps']} steps)")

    if not np.isfinite(global_vmin):
        global_vmin = 0.0
        global_vmax = 1.0

    return runs, global_vmin, global_vmax


def _draw_frame(track, run, stage: int, frame_idx: int, reveal_frames: int,
                hold_frames: int, global_vmin: float, global_vmax: float):
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    fig.patch.set_facecolor("#08111f")
    ax = fig.add_axes([0.05, 0.10, 0.68, 0.80])
    ax.set_facecolor("#0c1527")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#27415f")
        spine.set_linewidth(1.2)

    ax.plot(track.left_bound[:, 0], track.left_bound[:, 1], color="#d8dde6", lw=1.6, alpha=0.9)
    ax.plot(track.right_bound[:, 0], track.right_bound[:, 1], color="#d8dde6", lw=1.6, alpha=0.9)
    ax.plot(track.centerline[:, 0], track.centerline[:, 1], color="#586780", lw=1.0, ls="--", alpha=0.45)

    xs = run["xs"]
    ys = run["ys"]
    speeds = run["speeds"]
    if len(xs) >= 2:
        progress_t = min(1.0, (frame_idx + 1) / max(reveal_frames, 1))
        reveal_count = max(2, int(len(xs) * progress_t))
        pts = np.array([xs[:reveal_count], ys[:reveal_count]]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        norm = plt.Normalize(global_vmin, global_vmax)
        lc = LineCollection(segs, cmap="RdYlGn", norm=norm, linewidth=3.2, alpha=0.95)
        lc.set_array(speeds[:reveal_count - 1])
        ax.add_collection(lc)
        ax.scatter([xs[0]], [ys[0]], color="white", s=26, zorder=6)
        ax.scatter([xs[reveal_count - 1]], [ys[reveal_count - 1]], color="#ffe07a", s=36, zorder=7)

        cbar = fig.colorbar(lc, ax=ax, fraction=0.032, pad=0.02)
        cbar.set_label("Speed (km/h)", color="white", fontsize=11)
        cbar.ax.yaxis.set_tick_params(color="white", labelsize=9)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
        cbar.outline.set_edgecolor("#6a84a4")

    title = "RL Racing Line Evolution"
    subtitle = f"Stage {stage} - {run['label']}"
    fig.text(0.05, 0.94, title, color="white", fontsize=24, fontweight="bold")
    fig.text(0.05, 0.905, subtitle, color="#8ecbff", fontsize=16)

    panel = fig.add_axes([0.76, 0.14, 0.20, 0.68])
    panel.set_facecolor("#0f1a2f")
    panel.set_xticks([])
    panel.set_yticks([])
    for spine in panel.spines.values():
        spine.set_edgecolor("#27415f")
        spine.set_linewidth(1.2)

    progress_pct = min(100.0, 100.0 * (frame_idx + 1) / max(reveal_frames, 1))
    status = "Completed full lap" if run["completed"] else "Crashed / ended early"
    info_lines = [
        ("Checkpoint", run["label"]),
        ("Trajectory reveal", f"{progress_pct:5.1f}%"),
        ("Episode length", f"{run['steps']} steps"),
        ("Mean speed", f"{run['mean_speed']:.1f} km/h"),
        ("Max speed", f"{run['max_speed']:.1f} km/h"),
        ("Outcome", status),
    ]

    y = 0.92
    for label, value in info_lines:
        panel.text(0.08, y, label, color="#7db6e8", fontsize=11, fontweight="bold", transform=panel.transAxes)
        panel.text(0.08, y - 0.06, value, color="white", fontsize=15, transform=panel.transAxes)
        y -= 0.15

    if frame_idx >= reveal_frames and hold_frames > 0:
        panel.text(0.08, 0.05, "Hold frame", color="#ffd36c", fontsize=10, transform=panel.transAxes)

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)


def build_montage(stage: int, max_ckpts: int = 12, fps: int = 12,
                  reveal_frames: int = 24, hold_frames: int = 12,
                  include_best: bool = True, ckpt_dir: str = None):
    import config as cfg

    os.makedirs(PLOT_DIR, exist_ok=True)
    track = Track(STAGES.get(stage, {}).get("track", cfg.TRACK_FILE))
    runs, global_vmin, global_vmax = _prepare_runs(
        stage,
        max_ckpts=max_ckpts,
        include_best=include_best,
        ckpt_dir=ckpt_dir,
    )

    out_path = os.path.join(PLOT_DIR, f"checkpoint_montage_stage{stage}.mp4")
    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (1280, 720),
    )

    if not writer.isOpened():
        raise RuntimeError("Could not open video writer for MP4 output.")

    for run in runs:
        total_frames = reveal_frames + hold_frames
        for frame_idx in range(total_frames):
            frame = _draw_frame(
                track,
                run,
                stage=stage,
                frame_idx=min(frame_idx, reveal_frames),
                reveal_frames=reveal_frames,
                hold_frames=hold_frames,
                global_vmin=global_vmin,
                global_vmax=global_vmax,
            )
            writer.write(frame)

    writer.release()
    print(f"[OK] Saved montage -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a checkpoint evolution montage video.")
    parser.add_argument("--stage", type=int, default=1, help="Training stage to visualize.")
    parser.add_argument("--max-ckpts", type=int, default=12, help="Maximum checkpoints to include.")
    parser.add_argument("--fps", type=int, default=12, help="Video frame rate.")
    parser.add_argument("--reveal-frames", type=int, default=24, help="Frames used to draw each trajectory.")
    parser.add_argument("--hold-frames", type=int, default=12, help="Frames to hold on each completed trajectory.")
    parser.add_argument("--no-best", action="store_true", help="Skip appending best_model.zip at the end.")
    parser.add_argument(
        "--ckpt-dir",
        default=None,
        help="Optional directory containing checkpoint zips for this stage.",
    )
    args = parser.parse_args()

    build_montage(
        stage=args.stage,
        max_ckpts=args.max_ckpts,
        fps=args.fps,
        reveal_frames=args.reveal_frames,
        hold_frames=args.hold_frames,
        include_best=not args.no_best,
        ckpt_dir=args.ckpt_dir,
    )
