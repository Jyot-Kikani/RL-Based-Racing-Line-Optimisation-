# Project Command Reference

This file collects the main commands used to run this project locally and on a cluster.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Optional virtual environment setup on Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Manual Driving

Launch the environment in manual mode:

```bash
python manual_mode.py
```

Controls:

- `Up`: accelerate
- `Down`: brake
- `Left` / `Right`: steer
- `R`: reset
- `Esc`: quit

## Training

`train.py` is the main training entrypoint. Stages are defined in `train.py`, not just `config.py`.

### Stage 1

Default Stage 1 training:

```bash
python train.py --stage 1
```

Override timestep budget:

```bash
python train.py --stage 1 --steps 100000
```

Run headless:

```bash
python train.py --stage 1 --headless
```

Run with parallel environments:

```bash
python train.py --stage 1 --headless --n-envs 8
```

### Stage 2

Default Stage 2 training:

```bash
python train.py --stage 2
```

Stage 2 with custom timesteps:

```bash
python train.py --stage 2 --steps 200000
```

Headless multi-env Stage 2:

```bash
python train.py --stage 2 --headless --n-envs 8
```

### Stage 3

Default Stage 3 training:

```bash
python train.py --stage 3
```

Stage 3 with custom timesteps:

```bash
python train.py --stage 3 --steps 150000
```

Headless multi-env Stage 3:

```bash
python train.py --stage 3 --headless --n-envs 8
```

### Training Flags

Available flags:

- `--stage {1,2,3}`: required stage selector
- `--steps <int>`: override the default timestep budget for that stage
- `--n-envs <int>`: number of parallel environments
- `--headless`: disable Pygame rendering

### Current Stage Defaults

Current built-in training curriculum:

- Stage 1: `data/tracks/f1.csv`, checkpoint reward, `400000` steps
- Stage 2: `data/tracks/f1.csv`, laptime reward, `500000` steps
- Stage 3: `data/tracks/f1.csv`, laptime reward, `300000` steps

## Rollouts

Run a trained model and render deterministic episodes:

```bash
python rollout.py --model models/stage1/best_model
```

Run multiple episodes:

```bash
python rollout.py --model models/stage2/best_model --episodes 5
```

Run faster playback:

```bash
python rollout.py --model models/stage2/best_model --speed 3
```

Run without rendering:

```bash
python rollout.py --model models/stage3/best_model --no-render
```

Run without saving trajectory CSVs:

```bash
python rollout.py --model models/stage3/best_model --no-save
```

Verbose rollout:

```bash
python rollout.py --model models/stage3/best_model --verbose
```

Combine common rollout options:

```bash
python rollout.py --model models/stage3/best_model --episodes 3 --speed 3 --no-render --verbose
```

### Rollout Flags

- `--model <path>`: model path, default is `models/best_model`
- `--episodes <int>`: number of rollout episodes
- `--speed <int>`: physics steps per rendered frame
- `--no-render`: disable live rendering
- `--no-save`: skip trajectory CSV export
- `--verbose`: print per-step action details

### Rollout Outputs

Saved to:

- `outputs/trajectories/trajectory_ep1.csv`
- `outputs/trajectories/trajectory_ep2.csv`
- `outputs/trajectories/trajectory_ep3.csv`

## Visualization

Plot the learned racing line and speed profile from a trajectory CSV:

```bash
python visualize.py --trajectory outputs/trajectories/trajectory_ep1.csv
```

Visualize a different trajectory:

```bash
python visualize.py --trajectory outputs/trajectories/trajectory_ep2.csv
```

Generated plots:

- `outputs/plots/racing_line_heatmap.png`
- `outputs/plots/speed_profile.png`

## Baseline

Run the PID baseline controller:

```bash
python baseline.py
```

Output:

- `outputs/trajectories/trajectory_baseline.csv`

## Checkpoint Visualization

Build a checkpoint evolution grid for a stage:

```bash
python checkpoint_viz.py --stage 1
```

Limit the number of checkpoints shown:

```bash
python checkpoint_viz.py --stage 2 --max-ckpts 12
```

Use a custom checkpoint directory:

```bash
python checkpoint_viz.py --stage 2 --ckpt-dir old_checkpoints/stage2/checkpoints
```

Also generate an animated GIF:

```bash
python checkpoint_viz.py --stage 2 --gif
```

Common combined example:

```bash
python checkpoint_viz.py --stage 2 --max-ckpts 16 --gif
```

Outputs:

- `outputs/plots/checkpoint_evolution_stage1.png`
- `outputs/plots/checkpoint_evolution_stage2.png`
- `outputs/plots/checkpoint_evolution_stage3.png`
- optional `outputs/plots/checkpoint_evolution_stageN.gif`

## Checkpoint Montage Video

Build an MP4 montage from checkpoints:

```bash
python checkpoint_montage.py --stage 1
```

Limit checkpoints:

```bash
python checkpoint_montage.py --stage 1 --max-ckpts 8
```

Change playback settings:

```bash
python checkpoint_montage.py --stage 1 --fps 16 --reveal-frames 30 --hold-frames 18
```

Skip appending `best_model.zip`:

```bash
python checkpoint_montage.py --stage 1 --no-best
```

Use a custom checkpoint directory:

```bash
python checkpoint_montage.py --stage 1 --ckpt-dir downloaded_stage1_f1/checkpoints
```

Output:

- `outputs/plots/checkpoint_montage_stage1.mp4`
- `outputs/plots/checkpoint_montage_stage2.mp4`
- `outputs/plots/checkpoint_montage_stage3.mp4`

## Track Editor

Open the track editor with the default output path:

```bash
python track_editor.py
```

Edit or create a specific track:

```bash
python track_editor.py --file data/tracks/custom_track.csv
```

Start with a different default width:

```bash
python track_editor.py --file data/tracks/custom_track.csv --width 15
```

Adjust preview density:

```bash
python track_editor.py --file data/tracks/custom_track.csv --preview-points 500
```

### Track Editor Flags

- `--file <path>`: CSV file to edit or create
- `--width <float>`: default width in metres for new points
- `--preview-points <int>`: spline preview resolution

## Track Generation Helper

Generate the drag-strip helper track:

```bash
python data/generate_tracks.py
```

Note: this script currently writes `tracks/drag_strip.csv` relative to the working directory.

## TensorBoard

Monitor training logs:

```bash
tensorboard --logdir logs
```

## Cluster and Slurm

Short sanity-check run:

```bash
python train.py --stage 1 --headless --n-envs 2 --steps 5000
```

Recommended cluster runs:

```bash
python train.py --stage 1 --headless --n-envs 8
python train.py --stage 2 --headless --n-envs 8
python train.py --stage 3 --headless --n-envs 8
```

Submit Stage 1 with Slurm:

```bash
sbatch --export=ALL,STAGE=1,N_ENVS=8,TIMESTEPS=400000,PYTHON_BIN=$HOME/.conda/envs/rl/bin/python slurm/train_stage.sbatch
```

Submit Stage 2 with Slurm:

```bash
sbatch --export=ALL,STAGE=2,N_ENVS=8,TIMESTEPS=500000,PYTHON_BIN=$HOME/.conda/envs/rl/bin/python slurm/train_stage.sbatch
```

Submit Stage 3 with Slurm:

```bash
sbatch --export=ALL,STAGE=3,N_ENVS=8,TIMESTEPS=300000,PYTHON_BIN=$HOME/.conda/envs/rl/bin/python slurm/train_stage.sbatch
```

Good first cluster test:

```bash
sbatch --export=ALL,STAGE=1,N_ENVS=8,TIMESTEPS=100000,PYTHON_BIN=$HOME/.conda/envs/rl/bin/python slurm/train_stage.sbatch
```

Useful Slurm monitoring commands:

```bash
squeue -u $USER
sacct -j <jobid>
tail -f train_<jobid>.log
tail -f train_err_<jobid>.log
```

Cluster TensorBoard:

```bash
tensorboard --logdir logs --host 0.0.0.0 --port 6006
ssh -L 6006:localhost:6006 <your_cluster_username>@<cluster_host>
```

## Common Workflows

### Local sanity-check workflow

```bash
python manual_mode.py
python train.py --stage 1
python rollout.py --model models/stage1/best_model --episodes 1
python visualize.py --trajectory outputs/trajectories/trajectory_ep1.csv
```

### Full curriculum workflow

```bash
python train.py --stage 1 --headless --n-envs 8
python train.py --stage 2 --headless --n-envs 8
python train.py --stage 3 --headless --n-envs 8
```

### Evaluation workflow

```bash
python rollout.py --model models/stage3/best_model --episodes 3 --no-render
python visualize.py --trajectory outputs/trajectories/trajectory_ep1.csv
python baseline.py
python checkpoint_viz.py --stage 3 --max-ckpts 16
```

## Important Path Notes

- `config.py` defaults to `data/tracks/f1.csv`
- `train.py` writes staged outputs under `models/stage1`, `models/stage2`, and `models/stage3`
- `rollout.py` examples should usually point at `models/stageN/best_model`
- plots are written to `outputs/plots/`
- trajectory CSVs are written to `outputs/trajectories/`
