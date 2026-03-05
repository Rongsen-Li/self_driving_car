#!/bin/bash
#SBATCH --job-name=cityscape_frrnA
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

echo "Starting job on $(hostname)"
echo "Time: $(date)"

module purge
module load cuda12.2/toolkit/12.2.2
module load cudnn8.9-cuda12.2/8.9.4.25

eval "$(micromamba shell hook -s bash)"
micromamba activate selfdrive_tf1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd /home/lir3/projects/self_driving_car/third_party/sssuite
mkdir -p checkpoints logs

nvidia-smi

python main.py \
  --mode train \
  --model FRRN-A \
  --dataset ../../data/cityscape \
  --crop_height 320 \
  --crop_width 640 \
  --batch_size 1 \
  --num_epochs 60
