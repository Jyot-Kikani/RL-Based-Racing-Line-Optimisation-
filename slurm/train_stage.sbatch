#! /bin/bash

# Example usage:
#   sbatch --export=ALL,STAGE=1,N_ENVS=8,TIMESTEPS=400000 slurm/train_stage.sbatch

#SBATCH --job-name=rl_train
#SBATCH --output=train_%j.log
#SBATCH --error=train_err_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

module load anaconda3
module load gcc-9.5.0

STAGE="${STAGE:-1}"
N_ENVS="${N_ENVS:-8}"
TIMESTEPS="${TIMESTEPS:-0}"

# Update this once to your real environment path on the cluster.
PYTHON_BIN="${PYTHON_BIN:-$HOME/.conda/envs/rl/bin/python}"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "Python executable not found at: $PYTHON_BIN"
    echo "Create the environment first or pass PYTHON_BIN=/path/to/python via sbatch --export."
    exit 1
fi

mkdir -p logs models outputs

CMD=("$PYTHON_BIN" train.py --stage "$STAGE" --headless --n-envs "$N_ENVS")

if [ "$TIMESTEPS" -gt 0 ]; then
    CMD+=(--steps "$TIMESTEPS")
fi

echo "Running from: $SLURM_SUBMIT_DIR"
echo "Python: $PYTHON_BIN"
echo "Command: ${CMD[*]}"

"${CMD[@]}"
