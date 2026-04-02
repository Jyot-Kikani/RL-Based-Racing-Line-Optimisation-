# F1 Racing Line Optimisation — Project Status & Implementation Details

## Overview

This project trains a reinforcement learning agent to drive an F1 car around a race circuit, ultimately optimising for the fastest racing line (minimum lap time). The agent is trained using **Soft Actor-Critic (SAC)** from Stable Baselines 3 on a custom Gymnasium environment built with a **bicycle kinematic car model**, **real track geometry from TUMFTM CSV data**, and a **7-ray LiDAR-style sensor suite**.

The curriculum is split into **three progressive stages**, each building on the last.

---

## Architecture

### File Structure

```
RL/
├── train.py              # Curriculum training entrypoint
├── rollout.py            # Load saved model and render/inspect episodes
├── config.py             # Single source of truth for all constants
├── baseline.py           # PID baseline for comparison
├── visualize.py          # Post-training trajectory plotting
├── manual_mode.py        # Drive the car manually with keyboard
├── env/
│   ├── race.py           # RacingEnv (Gymnasium env)
│   ├── car.py            # Bicycle kinematic model + 7-ray sensor
│   ├── track.py          # Track loader, smoother, on-track checker
│   └── reward.py         # Reward computation (checkpoint / laptime modes)
├── data/tracks/
│   ├── drag_strip.csv    # Simple straight + hairpin track (Stage 1 & 2)
│   └── monza.csv         # Full Monza F1 circuit (Stage 3)
├── models/               # Saved SAC checkpoints
│   └── stage1/
│       ├── best_model.zip
│       └── checkpoints/
├── logs/                 # TensorBoard logs
└── outputs/trajectories/ # Trajectory CSVs from rollout.py
```

---

## Environment (`env/`)

### car.py — Bicycle Kinematic Model

| Parameter | Value | Notes |
|---|---|---|
| `WHEELBASE` | 2.5 m | Controls turning radius |
| `MAX_STEER` | 0.8 rad (~46°) | Hard clamp on steer angle |
| `MAX_ACCEL` | **15.0 m/s²** | Bumped from 4.0 — snappy acceleration |
| `MAX_BRAKE` | **15.0 m/s²** | Bumped from 8.0 — symmetric braking |
| `MAX_SPEED` | 60.0 m/s (216 km/h) | Hard cap |
| `FRICTION` | 1.5 m/s² per step | Baseline rolling resistance |
| `DRAG` | 0.001 × v² | Aerodynamic drag |
| `CORNERING_G` | 2.0 G | Limits max steer angle at high speed |
| `TURN_FRICTION` | 10.0 | Scrubs speed when turning |
| `DT` | 0.05 s | Simulation runs at 20 Hz |

**Rolling Start:** Each episode begins with `speed = 10.0 m/s` (36 km/h) to give the agent immediate forward momentum, bypassing the near-impossible "standing start from zero" exploration problem.

**Sensor Suite:** 7 rays at angles `[-60, -40, -20, 0, 20, 40, 60]` degrees, each measuring distance to the nearest track boundary up to `50 m`. Readings are normalised to `[-1, 1]`.

**Observation Space:** `(8,)` float32 — 7 sensor readings + normalised speed.  
**Action Space:** `(2,)` float32 — `[steer, accel]` both in `[-1, 1]`.

---

### track.py — Track Geometry

- Loads TUMFTM-format CSVs with `x_m, y_m, w_tr_right_m, w_tr_left_m` columns.
- Smoothed using **cubic parametric B-spline** (`scipy.splprep`) resampled to 300 waypoints.
- **`is_on_track(x, y)`** uses true **point-to-segment perpendicular distance** against the 2 nearest centerline segments, with linearly-interpolated boundary widths at the projection point. This prevents false off-track calls on parallel straights.

---

### reward.py — Reward Function

Two modes selectable via `config.REWARD_MODE`:

#### Stage 1 & 3: `"checkpoint"` mode
```
r = -STEP_PENALTY                        # -0.01 per step (urgency)
  + new_waypoints × CHECKPOINT_REWARD    # +1.00 per new waypoint reached
  - OFF_TRACK_PENALTY (if crashed)       # -10.0 on crash
```

#### Stage 2 & 3: `"laptime"` mode
```
r = -STEP_PENALTY
  + progress_delta × speed × SPEED_WEIGHT   # reward proportional to velocity
  - OFF_TRACK_PENALTY (if crashed)
```

---

### race.py — RacingEnv

Key implementation decisions:

- **Anti-stall termination:** If the agent goes `> 100` consecutive steps without reaching a new waypoint, the episode terminates with an additional `−5.0` penalty. This prevents the "coward agent" pathology where the agent discovers that sitting still for 5,000 steps scores `0`, which beats early random crashing.
- **Max episode length:** `5000` steps (250 seconds at 20 Hz). Only reached if the agent is consistently clearing waypoints.
- **Render:** pygame window with track polygon, centerline dashes, 7 sensor rays, car triangle, and HUD showing speed/waypoint/step/on_track status. Event queue is pumped every frame to prevent Windows "Not Responding" freeze.

---

## Training (`train.py`)

### Curriculum Stages

| Stage | Name | Track | Reward | Steps | LR |
|---|---|---|---|---|---|
| 1 | Survival — Drag Strip | `drag_strip.csv` | checkpoint | 300,000 | 3e-4 |
| 2 | Time Attack — Drag Strip | `drag_strip.csv` | laptime | 400,000 | 1e-4 |
| 3 | Transfer — Monza | `monza.csv` | laptime | 600,000 | 5e-5 |

**Stage handoffs:**
- Stage 2 loads the Stage 1 `best_model.zip` weights AND the replay buffer (same track, so past experience is still valid).
- Stage 3 loads Stage 2 weights only (different track — replay buffer is invalidated). Uses a very conservative LR to preserve learned behaviour.

### SAC Hyperparameters

| Parameter | Value |
|---|---|
| Policy | `MlpPolicy` (2 hidden layers × 256 units) |
| Batch size | 256 |
| Replay buffer | 1,000,000 transitions |
| Eval frequency | Every 10,000 steps |
| Eval episodes | 5 (deterministic) |
| Checkpoint freq | Every 50,000 steps |

### Running Training

```bash
python train.py --stage 1           # Stage 1 from scratch
python train.py --stage 2           # Loads Stage 1 best model
python train.py --stage 3           # Loads Stage 2 best model
python train.py --stage 1 --steps 500000   # Override timestep budget
```

### Monitoring

```bash
tensorboard --logdir logs/stage1    # Watch live metrics
```

---

## Rollout & Inspection (`rollout.py`)

```bash
# Watch a visual episode (pygame window, no recording)
python rollout.py --model models/stage1/best_model --episodes 1 --no-save

# Print per-step actions to terminal
python rollout.py --model models/stage1/best_model --episodes 1 --verbose --no-render

# Save trajectory CSVs for analysis
python rollout.py --model models/stage1/best_model --episodes 3
```

**Trajectory CSV columns:** `frame, x, y, heading, speed_kmh, steer, accel, reward, termination`

`termination` will be one of: `CRASH`, `ANTI_STALL`, or `MAX_STEPS`.

---

## Current Status (as of Apr 2, 2026)

### Stage 1 Training — ~12,000 / 300,000 steps

| Metric | Value |
|---|---|
| `ep_rew_mean` | ~100 and climbing |
| `ep_len_mean` | ~290–300 steps |
| Best eval reward | 116.40 |
| Eval ep length | 161 steps |
| Termination cause | **CRASH** at a rightward hairpin |
| Top speed reached | 216 km/h (at MAX_SPEED cap) |

**What the agent has learned so far:**
- ✅ Drive forward (not sit still)
- ✅ Accelerate consistently down the straight
- ✅ Recognise it's approaching a corner (steer starts increasing from step ~140)
- ✅ Attempt to brake when the corner arrives (accel goes to `-0.93` at step ~149)
- ❌ Anticipate the corner early enough to survive it at 216 km/h

**Diagnosis:** The agent is purely reactive. The critic hasn't yet assigned low value to the "high-speed approaching corner" state. After more crashes at this exact point, SAC will naturally learn earlier anticipatory braking.

---

## Issues Resolved This Session

| Problem | Root Cause | Fix Applied |
|---|---|---|
| Agent never moved (score: 0) | Sitting still was locally optimal vs crashing | Added `STEP_PENALTY = -0.01` per step |
| Episodes lasted 15 min | Random actions cancelled out momentum | Added anti-stall: terminate after 100 steps without a new waypoint (+ `-5.0` penalty) |
| Agent couldn't discover forward | Standing start + weak engine | Rolling start `speed = 10.0 m/s` + boosted `MAX_ACCEL` to `15.0` |
| pygame window freeze on Windows | Event queue not being pumped | Added `pygame.event.get()` loop in `render()` |
| Coward agent re-learned sitting still | `-0.01 × 5000 = -50` < `-10` crash penalty | Anti-stall forces short episodes; rolling start forces immediate movement |

---

## Future Plans

### Near Term (complete Stage 1)
- [ ] Let Stage 1 run to ~100,000 steps and check if `ep_rew_mean` has plateaued
- [ ] Verify via rollout that the agent can navigate the drag strip hairpin corner
- [ ] Confirm the `best_model.zip` checkpoint score is representative of full-lap survival

### Stage 2 — Time Attack on Drag Strip
- [ ] Switch reward to `laptime` mode: `progress × speed × SPEED_WEIGHT`
- [ ] Load Stage 1 weights + replay buffer
- [ ] Goal: agent learns to optimise speed on the same track, not just survive

### Stage 3 — Transfer to Monza
- [ ] Load Stage 2 weights (no replay buffer — different track)
- [ ] Use conservative LR (`5e-5`) to preserve cornering knowledge
- [ ] Goal: generalise learned physics to a full F1 circuit with multiple corners

### Evaluation & Comparison
- [ ] Run PID baseline (`baseline.py`) on Monza to get lap time benchmark
- [ ] Compare RL agent lap time vs PID
- [ ] Use `visualize.py` to plot trajectory overlays (RL line vs PID line vs optimal line)
- [ ] Generate report with reward curves, trajectory plots, and lap time comparison table
