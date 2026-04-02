# F1 Racing Line Optimisation — Project Progress

## Overview

Reinforcement learning agent that learns an optimal racing line on a 2D track.
The agent uses a bicycle car model with 7 raycast sensors and is trained using
Soft Actor-Critic (SAC) via Stable-Baselines3.

**Stack:** Python · Gymnasium · Stable-Baselines3 (SAC) · Pygame · NumPy · SciPy · Matplotlib

---

## What Has Been Implemented

### Project scaffold

Full folder structure created and locked before any code was written.

```
racing_line_rl/
├── env/
│   ├── __init__.py
│   ├── track.py        ✅ complete
│   ├── car.py          ✅ complete
│   ├── reward.py       ✅ complete
│   └── race.py         ✅ complete
├── data/
│   └── tracks/
│       ├── test_oval.csv   ✅ active track
│       └── monza.csv       (real TUMFTM data, too narrow for manual drive)
├── models/
│   └── checkpoints/
├── logs/
├── outputs/
│   ├── trajectories/
│   └── plots/
├── train.py            ✅ complete (ready to run)
├── manual_mode.py      ✅ complete + debugged
├── rollout.py          ✅ complete
├── visualize.py        ✅ complete
├── baseline.py         ✅ complete
├── config.py           ✅ complete
└── requirements.txt    ✅ complete
```

---

### `config.py` — centralised constants

Single source of truth for all tunable values. Key current settings:

| Parameter | Value | Note |
|---|---|---|
| `TRACK_FILE` | `test_oval.csv` | switch to `monza.csv` for training |
| `WHEELBASE` | 2.5 m | bicycle model turning radius |
| `MAX_STEER` | 0.8 rad | increased for manual drive responsiveness |
| `MAX_SPEED` | 40.0 m/s | reduced from 80 for easier control |
| `MAX_ACCEL` | 4.0 m/s² | |
| `DT` | 0.05 s | 20 Hz simulation |
| `N_SENSORS` | 7 | raycasts at −60°…+60° |
| `SENSOR_MAX_DIST` | 50 m | normalisation denominator |
| `REWARD_MODE` | `checkpoint` | switch to `laptime` after stable laps |
| `TOTAL_TIMESTEPS` | 1,000,000 | SAC training budget |
| `WINDOW_W/H` | 1200 × 900 | pygame window |

---

### `env/track.py` — track loader

Loads TUMFTM-format CSV (`x_m, y_m, w_tr_right_m, w_tr_left_m`) and exposes
clean track geometry for the environment and renderer.

**What it does:**
- Loads raw waypoints, drops duplicate closing point automatically
- Smooths and resamples centerline to 300 evenly-spaced points using a
  periodic cubic spline (`scipy.interpolate.splprep`)
- Interpolates track widths to match resampled points
- Computes left and right boundaries by offsetting centerline along
  perpendicular normal vectors
- Computes cumulative arc-length (`distances`) for progress tracking
- `nearest_waypoint(x, y)` — argmin Euclidean distance to centerline
- `is_on_track(x, y)` — distance to nearest centerline point vs track width
- `start_pos` and `start_heading` derived from first two waypoints

**Design note:** TUMFTM data is in metres, already real-world scale.
The real Monza CSV works but is visually narrow at window scale — use
`test_oval.csv` for development and manual driving.

---

### `env/car.py` — bicycle kinematic model + sensors

Implements vehicle physics and the 7-ray sensor suite.

**Physics model (kinematic bicycle model):**
```
steer_angle = steer_input × MAX_STEER
speed      += accel_input × (MAX_ACCEL or MAX_BRAKE) × DT
heading    += (speed / WHEELBASE) × tan(steer_angle) × DT
x          += speed × cos(heading) × DT
y          += speed × sin(heading) × DT
```

This is more realistic than the simple `angle += steer` pattern used in
reference repos — it enforces a physical turning radius at speed.

**Sensors:**
- 7 rays cast at `[−60°, −40°, −20°, 0°, +20°, +40°, +60°]` relative to
  heading
- Each ray tests for intersection against all left and right boundary segments
  using parametric line-segment intersection
- Returns normalised distance in `[0, 1]` (`dist / SENSOR_MAX_DIST`)
- Returns `1.0` (maximum) if no boundary hit within range

---

### `env/reward.py` — reward functions

Two modes switchable via `config.REWARD_MODE`:

| Mode | Formula | Use when |
|---|---|---|
| `checkpoint` | `+1.0` per new waypoint, `−10` off track | first — teaches forward progress |
| `laptime` | `progress_delta × speed × SPEED_WEIGHT`, `−10` off track | after agent can complete laps |

The curriculum approach (checkpoint first, laptime second) is borrowed from
the KGolemo reference project and is well-established for racing RL tasks.

---

### `env/race.py` — gymnasium environment

Wraps everything into a standard `gymnasium.Env` subclass.

**Observation space:** `Box(8,)` — 7 sensor readings + speed, all normalised
to `[−1, 1]`

**Action space:** `Box(2,)` — `[steering, acceleration]`, both in `[−1, 1]`

**Step logic:**
1. Apply action to car model
2. Check `is_on_track` → terminated if False
3. Count new waypoints passed (handles lap wrap-around)
4. Compute reward
5. Return obs, reward, terminated, truncated, info

**Renderer (pygame):**
- Track surface as filled polygon
- White boundary lines
- Dashed centerline
- 7 sensor rays in yellow with hit-point dots
- Car as red triangle pointing in heading direction, sized proportional to
  track width in pixels so it's always clearly visible
- HUD: speed, waypoint index, step count, on-track status (green/red)

**Screen transform:**
- Fits entire track into window using `(window - span×scale) / 2` centring
- Car pixel size computed as `avg_track_width_metres × scale × 1.2`
- Handles Y-axis flip (world Y-up → screen Y-down)

---

### `train.py` — SAC training (ready, not yet run)

Uses Stable-Baselines3 SAC with:
- `EvalCallback` — saves best model, logs to tensorboard every 10k steps
- `CheckpointCallback` — saves checkpoint every 50k steps
- Standard MLP policy (`MlpPolicy`)

```bash
python train.py
tensorboard --logdir logs/   # monitor in browser
```

---

### `manual_mode.py` — human play mode (working)

Arrow keys to drive, R to reset, ESC to quit.
Used to verify physics and rendering before training.
Fixed bug: pygame must be initialised inside `RacingEnv.__init__` when
`render_mode="human"`, not lazily inside `render()`.

---

### `rollout.py` — trajectory export (ready, not yet run)

Loads a trained model, runs N episodes, exports
`[frame, x, y, heading, speed, reward]` CSV per episode.

---

### `visualize.py` — racing line heatmap (ready, not yet run)

Reads trajectory CSV, plots the driven path over the track using a
`RdYlGn` speed colormap. Red = slow (braking), green = fast.
Also plots a separate speed-vs-frame profile.

---

### `baseline.py` — PID controller (ready, not yet run)

PID controller that steers proportionally to heading error toward the next
waypoint at constant throttle. Used as the comparison baseline in evaluation.
Outputs its own trajectory CSV for fair comparison with the RL agent.

---

## Bugs Fixed During Development

| Bug | Cause | Fix |
|---|---|---|
| `pygame.error: video system not initialized` | `pygame.init()` called after event loop started | moved init into `RacingEnv.__init__` when `render_mode="human"` |
| `ValueError: Invalid inputs` in splprep | TUMFTM CSV last row duplicates first row, causing zero-length segment | strip duplicate closing point on load |
| Track rendered tiny | scale calculation had asymmetric padding offsets | rewrote to use `(window - span×scale) / 2` centring |
| Car invisible | car size hardcoded at 8px independent of scale | compute `_car_px = avg_track_width_metres × scale × 1.2` |

---

## Reference Sources Used

| Source | What was taken |
|---|---|
| [KGolemo/f1-racing-line-optimization](https://github.com/KGolemo/f1-racing-line-optimization) | Raycast sensor design, checkpoint reward structure, state export format |
| [TUMFTM/racetrack-database](https://github.com/TUMFTM/racetrack-database) | Real F1 track CSVs (Monza), data format |
| [TUMFTM/global_racetrajectory_optimization](https://github.com/TUMFTM/global_racetrajectory_optimization) | Will be used as mathematical baseline for racing line comparison |
| [TDS article — rocket-meister](https://towardsdatascience.com/ultimate-guide-for-reinforced-learning-part-1-creating-a-game-956f1f2b0a91/) | gym.Env structure, Pygame render loop pattern, Euler kinematics |
| [TMRL library](https://github.com/trackmania-rl/tmrl) | SAC algorithm choice, observation/action space design |

---

## What to Implement Next

### Immediate — get training running

**1. Tune `is_on_track` for real Monza data**
The current implementation uses a simple Euclidean distance check to the nearest
centerline point. This works on the oval but can give false negatives on tight
corners of Monza where the boundary curves sharply. Consider switching to a
proper signed-distance check using the perpendicular projection onto the
centerline tangent.

**2. Switch `TRACK_FILE` back to `monza.csv` and verify**
Before training, confirm the car starts correctly and sensors hit boundaries
on the real track. Run `manual_mode.py` briefly to check.

**3. Run training**
```bash
# In config.py: set TRACK_FILE = "data/tracks/monza.csv"
# Set REWARD_MODE = "checkpoint"
python train.py
```
Expected: mean reward should increase steadily from ~0 toward 50–100+ over
500k steps. If it flatlines below 10 after 100k steps, the reward needs tuning.

**4. Monitor with tensorboard**
```bash
tensorboard --logdir logs/
```
Watch `rollout/ep_rew_mean` and `rollout/ep_len_mean`. Both should trend up.

---

### Phase 2 — reward curriculum

**5. Switch to laptime reward**
Once the agent reliably completes full laps (episode length consistently near
MAX_STEPS or lap wrap-around detected), change `REWARD_MODE = "laptime"` in
`config.py` and continue training from the checkpoint:
```python
model = SAC.load("models/best_model")
model.set_env(env)
model.learn(total_timesteps=500_000, reset_num_timesteps=False)
```

---

### Phase 3 — evaluation and visualisation

**6. Run rollout and export trajectory**
```bash
python rollout.py --model models/best_model --episodes 5
```

**7. Generate racing line heatmap**
```bash
python visualize.py --trajectory outputs/trajectories/trajectory_ep1.csv
```
This is the main deliverable visual — the speed-coloured racing line over
the track layout.

**8. Run PID baseline**
```bash
python baseline.py
```
Record mean lap time. Compare against RL agent mean lap time from rollout.

**9. Comparison table**
Produce a table in the report:

| Metric | PID baseline | RL agent |
|---|---|---|
| Mean lap time (s) | | |
| Mean speed (m/s) | | |
| Off-track rate | | |
| Lap completion rate | | |

---

### Phase 4 — stretch goals (if time allows)

**10. TUMFTM mathematical baseline**
Run the TUMFTM minimum-curvature optimizer on the same Monza track and overlay
its racing line against the RL agent's learned line. This is a strong academic
comparison — RL-learned vs mathematically optimal.

**11. Multiple tracks**
Train on Monza, test zero-shot on another TUMFTM track (e.g. Spa). Measures
how well the policy generalises vs overfitting to one circuit.

**12. Observation lookahead**
Add 3–5 curvature values of the track ahead to the observation vector. This
is the single biggest improvement to racing line quality — the agent can
anticipate corners rather than react to sensor readings.

**13. VecEnv parallel training**
Wrap the environment with SB3's `make_vec_env` to run 4–8 environments in
parallel. Cuts wall-clock training time roughly 4×.

```python
from stable_baselines3.common.env_util import make_vec_env
env = make_vec_env(RacingEnv, n_envs=4)
```

---

## Quick Command Reference

```bash
# Manual drive (verify environment)
python manual_mode.py

# Train
python train.py

# Monitor training
tensorboard --logdir logs/

# Run trained agent
python rollout.py --model models/best_model

# Visualise racing line
python visualize.py --trajectory outputs/trajectories/trajectory_ep1.csv

# Run PID baseline
python baseline.py
```

---

## Current Status

| Component | Status |
|---|---|
| Project structure | Done |
| Track loader | Done |
| Car physics + sensors | Done |
| Gym environment | Done |
| Pygame renderer | Done — working on test oval |
| Reward functions | Done — both modes implemented |
| Training script | Ready — not yet run |
| Rollout + export | Ready — not yet run |
| Visualisation | Ready — not yet run |
| PID baseline | Ready — not yet run |
| **SAC training** | **← next step** |
