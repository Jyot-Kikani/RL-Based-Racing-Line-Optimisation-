# Cluster Training Guide

This repo is prepared to train on a Slurm-based cluster using headless mode
and multiple parallel environments.

## Current Curriculum

`train.py` now uses the custom track `data/tracks/f1.csv` for all stages:

- Stage 1: checkpoint reward, survival learning
- Stage 2: laptime reward, same track
- Stage 3: low-learning-rate laptime fine-tuning, same track

Because the track stays the same across stages, later stages will also reuse
the previous stage replay buffer when it exists.

## 1. Get the repo onto the cluster

### Option A: clone from GitHub on the cluster

SSH to the cluster and clone the repository into your home or project space:

```bash
ssh <your_cluster_username>@<cluster_host>
git clone <your_repo_url>
cd RL
```

If the repo is private, use whichever authentication your university supports:

- SSH keys
- personal access token over HTTPS
- GitHub CLI if available

### Option B: push from your PC first, then clone on the cluster

On your PC:

```bash
git add .
git commit -m "Prepare custom F1 training and cluster workflow"
git push origin <branch-name>
```

Then on the cluster:

```bash
git clone <your_repo_url>
cd RL
git checkout <branch-name>
```

## 2. Create the Python environment on the cluster

Your sample job suggests this cluster commonly uses modules plus Conda, so that
should be the default approach here too.

Load the same style of modules first:

```bash
module load anaconda3
module load gcc-9.5.0
```

Then create a Conda environment:

```bash
conda create -n rl python=3.11 -y
conda activate rl
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Check the environment path:

```bash
which python
```

If the output is something like:

```bash
/home/<your_username>/.conda/envs/rl/bin/python
```

that is the path you should use in the Slurm script via `PYTHON_BIN`.

If your cluster prefers `venv`, that still works, but Conda is likely the most
natural fit given your existing jobs.

## 3. Sanity-check the install

Run a short headless test before submitting a long job:

```bash
python train.py --stage 1 --headless --n-envs 2 --steps 5000
```

You should see:

- Stage information printed to stdout
- checkpoint/log directories created
- no pygame window attempt

## 4. Recommended Slurm strategy

This project usually benefits more from **more CPU rollout workers** than from
GPU, because the custom environment simulation is substantial.

Recommended starting point:

- `--headless`
- `--n-envs 8`
- request `8` CPU cores
- moderate memory like `16G`

Try these settings first:

```bash
python train.py --stage 1 --headless --n-envs 8
python train.py --stage 2 --headless --n-envs 8
python train.py --stage 3 --headless --n-envs 8
```

If the node is strong and stable, test `--n-envs 12` or `--n-envs 16`.

## 5. Submit a job with the provided Slurm script

A template is included at `slurm/train_stage.sbatch`.

It follows the same pattern as your existing cluster jobs:

- `cd $SLURM_SUBMIT_DIR`
- `module load anaconda3`
- `module load gcc-9.5.0`
- direct execution through a configured `PYTHON_BIN`

Example submissions:

```bash
sbatch --export=ALL,STAGE=1,N_ENVS=8,TIMESTEPS=400000,PYTHON_BIN=$HOME/.conda/envs/rl/bin/python slurm/train_stage.sbatch
sbatch --export=ALL,STAGE=2,N_ENVS=8,TIMESTEPS=500000,PYTHON_BIN=$HOME/.conda/envs/rl/bin/python slurm/train_stage.sbatch
sbatch --export=ALL,STAGE=3,N_ENVS=8,TIMESTEPS=300000,PYTHON_BIN=$HOME/.conda/envs/rl/bin/python slurm/train_stage.sbatch
```

## 6. Monitor jobs

Useful Slurm commands:

```bash
squeue -u $USER
sacct -j <jobid>
tail -f logs/slurm-rl-stage-<jobid>.out
```

TensorBoard can also be launched on the cluster login node or via SSH port
forwarding if your cluster allows it:

```bash
tensorboard --logdir logs --host 0.0.0.0 --port 6006
```

Then from your PC:

```bash
ssh -L 6006:localhost:6006 <your_cluster_username>@<cluster_host>
```

Open `http://localhost:6006` locally.

## 7. Stage-by-stage workflow

### Stage 1

Goal: learn to complete laps reliably with checkpoint reward.

Run:

```bash
python train.py --stage 1 --headless --n-envs 8
```

After it finishes, inspect:

- `models/stage1/best_model.zip`
- `models/stage1/final_model.zip`
- `logs/stage1/`

### Stage 2

Goal: switch to laptime reward on the same track.

Run:

```bash
python train.py --stage 2 --headless --n-envs 8
```

This stage loads:

- Stage 1 weights
- Stage 1 replay buffer, because the track is still `data/tracks/f1.csv`

### Stage 3

Goal: low-learning-rate refinement on the same track.

Run:

```bash
python train.py --stage 3 --headless --n-envs 8
```

This stage also reuses the previous replay buffer because the track remains the
same.

## 8. Download results back to your PC

From your PC:

```bash
scp -r <your_cluster_username>@<cluster_host>:~/RL/models ./models_cluster
scp -r <your_cluster_username>@<cluster_host>:~/RL/logs ./logs_cluster
scp -r <your_cluster_username>@<cluster_host>:~/RL/outputs ./outputs_cluster
```

Adjust the remote path if your repo is stored elsewhere on the cluster.

## 9. Good first cluster run

Use this as your first real submission:

```bash
sbatch --export=ALL,STAGE=1,N_ENVS=8,TIMESTEPS=100000,PYTHON_BIN=$HOME/.conda/envs/rl/bin/python slurm/train_stage.sbatch
```

That is long enough to expose environment or scaling issues without committing
an entire overnight allocation.

## 10. Practical notes

- `config.py` is not the training source of truth for track selection; `train.py`
  stage definitions override it.
- Keep the repo, logs, and models on persistent storage, not temporary scratch,
  unless you also copy checkpoints back before job end.
- If `n_envs` is too high, throughput can get worse due to process overhead.
  Start at `8`, then benchmark `12` or `16`.
- The current Slurm template requests `--gres=gpu:1` because your sample job
  uses that pattern. If GPU queue time is high, you can consider a CPU-only
  variant later, since this project often benefits strongly from CPU env
  parallelism.
