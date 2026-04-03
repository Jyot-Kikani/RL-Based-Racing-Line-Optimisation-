# ── checkpoint_viz.py ─────────────────────────────────────────────────────────
# Visualise how the agent's racing line evolved across training checkpoints.
#
# For each checkpoint in models/stageN/checkpoints/, runs a deterministic
# rollout and plots the speed-coloured racing line on the same track axes.
#
# Usage:
#   python checkpoint_viz.py --stage 2
#   python checkpoint_viz.py --stage 2 --max-ckpts 12 --no-render
#
# Outputs:
#   outputs/plots/checkpoint_evolution_stageN.png   — grid of all checkpoints
#   outputs/plots/checkpoint_evolution_stageN.gif   — animated version (optional)

import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
import numpy as np

from stable_baselines3 import SAC
from env.race import RacingEnv
from env.track import Track
from config import PLOT_DIR
from train import STAGES


# ── Helpers ───────────────────────────────────────────────────────────────────

def _checkpoint_step(path: str) -> int:
    """Extract step count from filename: stage2_sac_50000_steps.zip → 50000."""
    m = re.search(r"_(\d+)_steps", os.path.basename(path))
    return int(m.group(1)) if m else 0


def _rollout(model_path: str, max_steps: int = 5000):
    """
    Load model, run one deterministic episode.
    Returns xs, ys, speeds (all numpy arrays).
    """
    import config as cfg
    cfg.HEADLESS = True
    model = SAC.load(model_path)
    env   = RacingEnv(render_mode=None)

    obs, _ = env.reset()
    xs, ys, speeds = [], [], []
    done = False
    step = 0

    while not done and step < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        xs.append(env.car.x)
        ys.append(env.car.y)
        speeds.append(env.car.speed * 3.6)   # km/h
        step += 1

    env.close()
    return np.array(xs), np.array(ys), np.array(speeds)


def _draw_racing_line(ax, track, xs, ys, speeds, title, global_vmin, global_vmax):
    """Draw track + speed-heatmap racing line on ax."""
    ax.set_facecolor("#0d0d1a")
    ax.set_aspect("equal")
    ax.set_title(title, color="white", fontsize=8, pad=4)
    ax.tick_params(colors="#555577", labelsize=6)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

    # Track boundaries
    ax.plot(track.left_bound[:, 0],  track.left_bound[:, 1],  color="#3a3a5c", lw=1.2)
    ax.plot(track.right_bound[:, 0], track.right_bound[:, 1], color="#3a3a5c", lw=1.2)
    # Centerline
    ax.plot(track.centerline[:, 0], track.centerline[:, 1],
            color="#4a4a6a", lw=0.8, ls="--", alpha=0.5)

    if len(xs) < 2:
        return None

    pts  = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    norm = plt.Normalize(global_vmin, global_vmax)
    lc   = LineCollection(segs, cmap="RdYlGn", norm=norm, linewidth=2.0, alpha=0.9)
    lc.set_array(speeds[:-1])
    ax.add_collection(lc)

    # Start dot
    ax.scatter([xs[0]], [ys[0]], color="white", s=20, zorder=5)
    return lc


# ── Main ──────────────────────────────────────────────────────────────────────

def build_evolution_plot(stage: int, max_ckpts: int = 16, make_gif: bool = False):
    ckpt_dir = os.path.join("models", f"stage{stage}", "checkpoints")
    ckpt_files = sorted(
        glob.glob(os.path.join(ckpt_dir, f"stage{stage}_sac_*_steps.zip")),
        key=_checkpoint_step
    )

    # Also optionally include best_model for comparison
    best_path = os.path.join("models", f"stage{stage}", "best_model.zip")
    if os.path.exists(best_path):
        ckpt_files.append(best_path)

    if not ckpt_files:
        print(f"[!] No checkpoints found in {ckpt_dir}")
        print(f"    Run training first: python train.py --stage {stage}")
        return

    # Subsample if too many
    if len(ckpt_files) > max_ckpts:
        indices = np.linspace(0, len(ckpt_files) - 1, max_ckpts, dtype=int)
        ckpt_files = [ckpt_files[i] for i in indices]

    print(f"[+] Found {len(ckpt_files)} checkpoints for Stage {stage}")

    # Load track once
    import config as cfg
    track = Track(STAGES.get(stage, {}).get("track", cfg.TRACK_FILE))

    # ── Rollout all checkpoints ───────────────────────────────────────────────
    all_xs, all_ys, all_speeds, all_titles = [], [], [], []
    global_vmin, global_vmax = np.inf, -np.inf

    for i, path in enumerate(ckpt_files):
        step = _checkpoint_step(path)
        label = f"Best Model" if "best_model" in path else f"{step//1000}k steps"
        print(f"  [{i+1}/{len(ckpt_files)}] Rolling out {label} ...", end=" ", flush=True)
        xs, ys, speeds = _rollout(path)
        all_xs.append(xs); all_ys.append(ys); all_speeds.append(speeds)
        all_titles.append(label)
        if len(speeds):
            global_vmin = min(global_vmin, speeds.min())
            global_vmax = max(global_vmax, speeds.max())
        print(f"done ({len(xs)} steps)")

    # ── Layout grid ──────────────────────────────────────────────────────────
    n = len(ckpt_files)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig = plt.figure(figsize=(ncols * 4, nrows * 3.5 + 0.8))
    fig.patch.set_facecolor("#0a0a18")
    fig.suptitle(
        f"Racing Line Evolution — Stage {stage}",
        color="white", fontsize=13, y=0.98, fontweight="bold"
    )

    gs = gridspec.GridSpec(nrows, ncols, figure=fig,
                           hspace=0.35, wspace=0.15,
                           top=0.93, bottom=0.08, left=0.04, right=0.88)

    last_lc = None
    for i in range(n):
        row, col = divmod(i, ncols)
        ax = fig.add_subplot(gs[row, col])
        lc = _draw_racing_line(ax, track,
                               all_xs[i], all_ys[i], all_speeds[i],
                               all_titles[i], global_vmin, global_vmax)
        if lc is not None:
            last_lc = lc

    # Hide unused axes
    for i in range(n, nrows * ncols):
        row, col = divmod(i, ncols)
        fig.add_subplot(gs[row, col]).set_visible(False)

    # Shared colorbar
    if last_lc is not None:
        cbar_ax = fig.add_axes([0.90, 0.12, 0.018, 0.75])
        cb = fig.colorbar(last_lc, cax=cbar_ax)
        cb.set_label("Speed (km/h)", color="white", fontsize=9)
        cb.ax.yaxis.set_tick_params(color="white", labelsize=7)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
        cb.outline.set_edgecolor("#555577")

    os.makedirs(PLOT_DIR, exist_ok=True)
    out_png = os.path.join(PLOT_DIR, f"checkpoint_evolution_stage{stage}.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n[✓] Saved grid → {out_png}")
    plt.close(fig)

    # ── Optional GIF ─────────────────────────────────────────────────────────
    if make_gif:
        try:
            import imageio.v2 as imageio
        except ImportError:
            print("[!] imageio not installed — skipping GIF. Run: pip install imageio")
            return

        frames = []
        for i in range(n):
            fig_f, ax_f = plt.subplots(figsize=(6, 5))
            fig_f.patch.set_facecolor("#0a0a18")
            lc = _draw_racing_line(ax_f, track,
                                   all_xs[i], all_ys[i], all_speeds[i],
                                   all_titles[i], global_vmin, global_vmax)
            if lc is not None:
                cb = fig_f.colorbar(lc, ax=ax_f, fraction=0.03, pad=0.02)
                cb.set_label("Speed (km/h)", color="white", fontsize=8)
                plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
            tmp = os.path.join(PLOT_DIR, f"_tmp_frame_{i:03d}.png")
            fig_f.savefig(tmp, dpi=100, bbox_inches="tight",
                          facecolor=fig_f.get_facecolor())
            plt.close(fig_f)
            frames.append(imageio.imread(tmp))
            os.remove(tmp)

        out_gif = os.path.join(PLOT_DIR, f"checkpoint_evolution_stage{stage}.gif")
        imageio.mimsave(out_gif, frames, fps=2, loop=0)
        print(f"[✓] Saved GIF  → {out_gif}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualise checkpoint-by-checkpoint racing line evolution."
    )
    parser.add_argument("--stage",     type=int, default=2,
                        help="Which training stage to visualise (default: 2)")
    parser.add_argument("--max-ckpts", type=int, default=16,
                        help="Max checkpoints to include in the grid (default: 16)")
    parser.add_argument("--gif",       action="store_true",
                        help="Also produce an animated GIF (requires imageio)")
    args = parser.parse_args()

    build_evolution_plot(args.stage, max_ckpts=args.max_ckpts, make_gif=args.gif)
