# F1 Racing Line Optimization — RL

Reinforcement learning agent that learns an optimal racing line on real F1 circuit data.

## Stack
- **Environment**: custom `gymnasium.Env` with bicycle car model + 7-ray sensors
- **Algorithm**: Soft Actor-Critic (SAC) via Stable-Baselines3
- **Track data**: TUMFTM racetrack database (real F1 circuits)
- **Rendering**: Pygame

## Setup

```bash
pip install -r requirements.txt
```

Download a track CSV from [TUMFTM/racetrack-database](https://github.com/TUMFTM/racetrack-database/tree/master/tracks) and place it at `data/tracks/monza.csv`.

## Usage

```bash
# 1. Drive manually to verify the environment
python manual_mode.py

# 2. Train the RL agent
python train.py

# 3. Run the trained agent and export trajectory
python rollout.py --model models/best_model

# 4. Visualize the learned racing line
python visualize.py --trajectory outputs/trajectories/trajectory_ep1.csv

# 5. Run PID baseline for comparison
python baseline.py

# 6. Create or edit a custom track
python track_editor.py --file data/tracks/custom_track.csv
```

## Monitor training
```bash
tensorboard --logdir logs/
```

## Project structure
```
env/
  track.py      ← TUMFTM CSV loader + boundary geometry
  car.py        ← bicycle model + 7-ray sensors
  reward.py     ← checkpoint and lap-time rewards
  race.py       ← gymnasium.Env wrapper
train.py        ← SB3 SAC training loop
manual_mode.py  ← human play + pygame render
rollout.py      ← export trajectory from trained model
visualize.py    ← speed heatmap racing line plot
baseline.py     ← PID centerline follower (comparison)
config.py       ← all constants in one place
```
