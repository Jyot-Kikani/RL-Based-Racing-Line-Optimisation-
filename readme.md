# RL Racing Line Optimization

Reinforcement learning project for autonomous racing-line discovery using a custom Formula-style racing simulator, Soft Actor-Critic (SAC), and curriculum learning.

The agent learns to drive a simplified race car around a closed circuit using only local geometric sensing and vehicle dynamics — without being given an optimal racing line.

---

# Overview

This project implements a complete RL training pipeline for autonomous racing behavior:

* Custom `gymnasium.Env`
* Bicycle-model vehicle dynamics
* Ray-based perception system
* SAC training with Stable-Baselines3
* Curriculum learning pipeline
* Rollout export and trajectory visualization
* PID baseline controller
* Cluster-ready training workflow

The focus is not photorealistic simulation, but building an explainable and extensible RL system for studying racing behavior, reward shaping, and continuous-control learning.

---

# Features

* 7-ray sensor observation model
* Continuous steering + throttle/brake control
* Custom F1-style track support from CSV geometry
* Curriculum learning across training stages
* Replay buffer reuse between stages
* PID baseline comparison
* Rollout trajectory export
* Racing-line and speed-profile visualization
* Checkpoint evolution visualization
* Interactive track editor
* Multi-environment parallel training support

---

# Project Structure

```text
RL/
│
├── train.py                  # Curriculum training entrypoint
├── config.py                 # Global configuration
├── rollout.py                # Deterministic rollout playback/export
├── visualize.py              # Racing-line and speed plots
├── baseline.py               # PID baseline controller
├── checkpoint_viz.py         # Checkpoint evolution visualization
├── manual_mode.py            # Manual driving/debugging
├── track_editor.py           # Interactive track editor
│
├── env/
│   ├── race.py               # Gymnasium environment
│   ├── car.py                # Vehicle model + sensors
│   ├── track.py              # Track geometry + progress logic
│   └── reward.py             # Reward functions
│
├── data/
│   └── tracks/               # Track CSV files
│
├── outputs/
│   ├── trajectories/         # Rollout CSVs
│   └── plots/                # Generated plots
│
├── models/                   # Trained models/checkpoints
├── logs/                     # TensorBoard/evaluation logs
│
└── slurm/
    └── train_stage.sbatch    # Cluster training script
```

---

# Environment Design

## Observation Space

The agent observes:

* 7 ray distances
* normalized vehicle speed

Ray angles:

```python
[-60, -40, -20, 0, 20, 40, 60]
```

The observation is intentionally compact so the policy must infer driving behavior from local geometry rather than a full map.

---

## Action Space

Continuous 2D control:

```python
[steering, acceleration]
```

Where:

* steering ∈ `[-1, 1]`
* acceleration ∈ `[-1, 1]`

Acceleration values are interpreted as:

* positive → throttle
* negative → brake

---

# Vehicle Model

The simulator uses a simplified bicycle kinematic model with:

* steering limits
* acceleration/braking limits
* aerodynamic drag
* rolling friction
* turn-induced speed scrub
* grip-limited cornering

This creates meaningful racing behavior while remaining lightweight enough for RL experimentation.

---

# Reward Design

## Stage 1 — Survival Reward

The agent learns:

* forward progress
* checkpoint completion
* staying on track

Designed to solve early exploration and survival.

---

## Stage 2/3 — Lap Time Reward

The reward shifts toward:

* speed-weighted progress
* efficient lap completion
* smoother/faster racing behavior

This staged curriculum stabilizes learning significantly compared to optimizing speed from the beginning.

---

# Training Curriculum

| Stage | Objective                 | Reward Mode  | Steps |
| ----- | ------------------------- | ------------ | ----: |
| 1     | Survival / Lap Completion | `checkpoint` |  400k |
| 2     | Time Attack Optimization  | `laptime`    |  500k |
| 3     | Fine Tuning               | `laptime`    |  300k |

All stages currently train on:

```text
data/tracks/f1.csv
```

---

# Installation

## Clone Repository

```bash
git clone <repo-url>
cd RL
```

---

## Create Environment

```bash
python -m venv venv
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

Main dependencies:

* gymnasium
* stable-baselines3
* pygame
* numpy
* scipy
* pandas
* matplotlib

---

# Training

## Stage 1

```bash
python train.py --stage 1
```

## Stage 2

```bash
python train.py --stage 2
```

## Stage 3

```bash
python train.py --stage 3
```

---

## Parallel Training

```bash
python train.py --stage 1 --headless --n-envs 8
```

---

# Rollouts

Run a trained model:

```bash
python rollout.py --model models/stage1/best_model --episodes 3
```

This supports:

* rendering
* trajectory export
* deterministic evaluation
* action inspection

---

# Visualization

## Racing Line Visualization

```bash
python visualize.py --trajectory outputs/trajectories/trajectory_ep1.csv
```

Generates:

* learned racing line heatmap
* speed profile plots

---

## Checkpoint Evolution

```bash
python checkpoint_viz.py --stage 1
```

Visualizes how the policy evolves across training checkpoints.

---

# Manual Driving

```bash
python manual_mode.py
```

Useful for:

* environment debugging
* testing physics
* validating track geometry
* verifying solvability

---

# Track Editor

```bash
python track_editor.py --file data/tracks/custom_track.csv
```

Interactive spline-based editor for creating and modifying custom tracks.

---

# Why SAC?

Soft Actor-Critic (SAC) was chosen because it:

* handles continuous control naturally
* is sample-efficient
* supports replay-buffer reuse
* encourages exploration through entropy regularization
* performs well on robotics/control-style problems

The implementation uses Stable-Baselines3 SAC with an MLP policy.

---

# Key Lessons From the Project

The project revealed several important RL engineering insights:

* Reward shaping is often more important than changing algorithms.
* Agents exploit unintended reward loopholes aggressively.
* Curriculum learning dramatically improves stability.
* Environment design strongly affects learning success.
* Small geometry or termination bugs can destabilize training.

Major fixes included:

* anti-stall termination
* rolling-start initialization
* improved on-track detection
* stronger exploration incentives

---

# Current Limitations

* Local low-dimensional observation space
* No full tire dynamics model
* No imitation learning or expert trajectories
* No true cross-track transfer curriculum currently
* Some legacy documentation/code drift

---

# Future Work

Potential extensions:

* Cross-track transfer learning
* PPO/DDPG/TD3 comparisons
* Vision-based observations
* Dynamic racing opponents
* Multi-agent racing
* Better tire and slip-angle dynamics
* Optimal-control or MPC comparison
* Real telemetry-inspired reward shaping

---

# Example Outputs

The repository supports generation of:

* trajectory CSVs
* racing-line heatmaps
* speed profiles
* checkpoint evolution plots
* baseline comparison runs

---

# Tech Stack

* Python
* Gymnasium
* Stable-Baselines3
* SAC (Soft Actor-Critic)
* NumPy
* SciPy
* Matplotlib
* Pandas
* Pygame
