# ── train.py ──────────────────────────────────────────────────────────────────
import argparse
import os
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from env.race import RacingEnv


STAGES = {
    1: {
        "name":        "Survival — Drag Strip",
        "track":       "data/tracks/drag_strip.csv",
        "reward_mode": "checkpoint",
        "steps":       300_000,
        "lr":          3e-4,
        "description": "Learn to stay on track and pass checkpoints.",
    },
    2: {
        "name":        "Time Attack — Drag Strip",
        "track":       "data/tracks/drag_strip.csv",
        "reward_mode": "laptime",
        "steps":       400_000,
        "lr":          1e-4,
        "description": "Optimise racing line and speed on the same track.",
    },
    3: {
        "name":        "Transfer — Monza",
        "track":       "data/tracks/monza.csv",
        "reward_mode": "laptime",
        "steps":       600_000,
        "lr":          5e-5,
        "description": "Apply learned behaviour to a full F1 circuit.",
    },
}

MODEL_DIR = "models/"
LOG_DIR   = "logs/"


def model_path(stage):
    return os.path.join(MODEL_DIR, f"stage{stage}", "best_model")

def checkpoint_dir(stage):
    return os.path.join(MODEL_DIR, f"stage{stage}", "checkpoints")


def train(stage: int, steps_override: int = None, n_envs: int = 1, headless: bool = False):
    import config as cfg_mod
    cfg = STAGES[stage]
    steps = steps_override or cfg["steps"]

    # Patch config globals before any env is created
    cfg_mod.TRACK_FILE   = cfg["track"]
    cfg_mod.REWARD_MODE  = cfg["reward_mode"]
    cfg_mod.HEADLESS     = headless

    print(f"\n{'='*60}")
    print(f"  Stage {stage}: {cfg['name']}")
    print(f"  {cfg['description']}")
    print(f"  Track:    {cfg['track']}")
    print(f"  Reward:   {cfg['reward_mode']}")
    print(f"  Steps:    {steps:,}")
    print(f"  Envs:     {n_envs}  ({'headless' if headless else 'with render'})")
    print(f"{'='*60}\n")

    os.makedirs(os.path.join(MODEL_DIR, f"stage{stage}"), exist_ok=True)
    os.makedirs(checkpoint_dir(stage), exist_ok=True)

    # ── Build vectorised training env ─────────────────────────────────────────
    def make_env_fn():
        def _init():
            return Monitor(RacingEnv())
        return _init

    if n_envs > 1:
        # SubprocVecEnv: each env runs in its own process — best for CPU clusters
        env = SubprocVecEnv([make_env_fn() for _ in range(n_envs)])
    else:
        # DummyVecEnv: single env, same process — fine for laptop with render
        env = DummyVecEnv([make_env_fn()])

    # Eval env is always single, non-rendered
    eval_env = Monitor(RacingEnv())

    # eval_freq is per-env steps in SB3 with VecEnv, so scale it down
    # so we still evaluate every ~10k total steps regardless of n_envs
    scaled_eval_freq = max(1_000, 10_000 // n_envs)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = os.path.join(MODEL_DIR, f"stage{stage}"),
        log_path             = os.path.join(LOG_DIR, f"stage{stage}"),
        eval_freq            = scaled_eval_freq,
        n_eval_episodes      = 5,
        deterministic        = True,
        verbose              = 1,
    )
    ckpt_cb = CheckpointCallback(
        save_freq   = max(5_000, 50_000 // n_envs),
        save_path   = checkpoint_dir(stage),
        name_prefix = f"stage{stage}_sac",
    )

    # ── Load or create model ──────────────────────────────────────────────────
    prev_path = model_path(stage - 1) + ".zip"

    if stage > 1 and os.path.exists(prev_path):
        print(f"  Loading Stage {stage-1} weights from {prev_path}")
        model = SAC.load(
            prev_path,
            env             = env,
            learning_rate   = cfg["lr"],
            verbose         = 1,
            tensorboard_log = os.path.join(LOG_DIR, f"stage{stage}"),
        )
        buf_path = model_path(stage - 1) + "_replay_buffer.pkl"
        if stage == 2 and os.path.exists(buf_path):
            print(f"  Loading replay buffer from {buf_path}")
            model.load_replay_buffer(buf_path)
    else:
        if stage > 1:
            print(f"  WARNING: Stage {stage-1} model not found at {prev_path}")
            print(f"  Starting Stage {stage} from scratch.\n")
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate   = cfg["lr"],
            batch_size      = 256,
            buffer_size     = 1_000_000,
            verbose         = 1,
            tensorboard_log = os.path.join(LOG_DIR, f"stage{stage}"),
        )

    # ── Train ─────────────────────────────────────────────────────────────────
    model.learn(
        total_timesteps     = steps,
        callback            = CallbackList([eval_cb, ckpt_cb]),
        reset_num_timesteps = (stage == 1),
        tb_log_name         = f"stage{stage}_sac",
    )

    # Save final model + replay buffer
    final_path = os.path.join(MODEL_DIR, f"stage{stage}", "final_model")
    model.save(final_path)
    model.save_replay_buffer(final_path + "_replay_buffer.pkl")
    print(f"\n  Saved → {final_path}")
    print(f"  Best  → {model_path(stage)}")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Curriculum RL training")
    parser.add_argument("--stage",    type=int,  required=True, choices=[1, 2, 3])
    parser.add_argument("--steps",    type=int,  default=None,
                        help="Override timestep budget for this stage")
    parser.add_argument("--n-envs",   type=int,  default=1,
                        help="Number of parallel envs (use 8-16 on cluster)")
    parser.add_argument("--headless", action="store_true",
                        help="Disable pygame rendering (required on cluster)")
    args = parser.parse_args()
    train(args.stage, args.steps, args.n_envs, args.headless)