# ── rollout.py ────────────────────────────────────────────────────────────────
# Load a trained model, run N laps, export trajectory CSV.
# Run: python rollout.py --model models/best_model
#
# Output CSV columns: frame, x, y, heading, speed, reward

import argparse
import csv
import os
import numpy as np
from stable_baselines3 import SAC
from env.race import RacingEnv
from config import TRAJECTORY_DIR


def run_rollout(model_path: str, n_episodes: int = 3, render: bool = True, verbose: bool = False, no_save: bool = False):
    model  = SAC.load(model_path)
    mode   = "human" if render else None
    env    = RacingEnv(render_mode=mode)

    for ep in range(n_episodes):
        obs, _    = env.reset()
        done      = False
        frame     = 0
        rows      = []
        total_rew = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            reason = ""
            if done:
                if terminated and not info.get("on_track", True):
                    reason = "CRASH" if env._steps_since_last_wp <= 100 else "ANTI_STALL"
                elif truncated:
                    reason = "MAX_STEPS"

            rows.append([
                frame,
                round(env.car.x, 4),
                round(env.car.y, 4),
                round(env.car.heading, 4),
                round(env.car.speed * 3.6, 2),   # convert to km/h
                round(float(action[0]), 4),        # steer  (-1=left, +1=right)
                round(float(action[1]), 4),        # accel  (-1=brake, +1=throttle)
                round(reward, 4),
                reason,
            ])
            total_rew += reward
            frame += 1

            if verbose:
                print(f"  step {frame:4d} | steer={action[0]:+.3f}  accel={action[1]:+.3f} "
                      f"| speed={env.car.speed*3.6:5.1f} km/h | rew={reward:+.3f} "
                      f"| on_track={info.get('on_track', '?')} {reason}")

            if render:
                env.render()

        print(f"Episode {ep+1}: {frame} steps, total reward {total_rew:.2f}")

        if not no_save:
            os.makedirs(TRAJECTORY_DIR, exist_ok=True)
            out_path = os.path.join(TRAJECTORY_DIR, f"trajectory_ep{ep+1}.csv")
            with open(out_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["frame", "x", "y", "heading", "speed_kmh",
                                 "steer", "accel", "reward", "termination"])
                writer.writerows(rows)
            print(f"  Saved → {out_path}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default="models/best_model")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--no-save",   action="store_true", help="Skip CSV export")
    parser.add_argument("--verbose",   action="store_true", help="Print per-step actions")
    args = parser.parse_args()
    run_rollout(args.model, args.episodes, render=not args.no_render, verbose=args.verbose, no_save=args.no_save)