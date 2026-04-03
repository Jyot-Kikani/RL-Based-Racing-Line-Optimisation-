#!/bin/bash
# Example usage:
#   sbatch --export=ALL,STAGE=1,N_ENVS=8,TIMESTEPS=400000 slurm/train_stage.sbatch

#SBATCH --job-name=rl-stage
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

set -euo pipefail

STAGE="${STAGE:-1}"
N_ENVS="${N_ENVS:-8}"
TIMESTEPS="${TIMESTEPS:-0}"

mkdir -p logs

# Uncomment and adapt these for your cluster if it uses environment modules.
# module purge
# module load python/3.11
# module load cuda/12.1

if [ -f "venv/bin/activate" ]; then
  source venv/bin/activate
else
  echo "venv/bin/activate not found. Create the virtual environment first."
  exit 1
fi

CMD=(python train.py --stage "${STAGE}" --headless --n-envs "${N_ENVS}")

if [ "${TIMESTEPS}" -gt 0 ]; then
  CMD+=(--steps "${TIMESTEPS}")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
