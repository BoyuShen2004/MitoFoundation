#!/bin/bash
#SBATCH --job-name=train_Dataset001_mito2_3d_fullres
#SBATCH --output=/projects/weilab/shenb/mitoFoundation2/5model_training/slurm/logs/train/%x_%j.out
#SBATCH --error=/projects/weilab/shenb/mitoFoundation2/5model_training/slurm/logs/train/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64g
#SBATCH --time=48:00:00
#SBATCH --partition=weilab
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=shenb@bc.edu

set -euo pipefail

source /projects/weilab/shenb/miniconda3/etc/profile.d/conda.sh
conda activate nnunetv2
echo "Using Python at: $(which python)"

echo "Job started at $(date)"
START_TIME=$SECONDS

cd /projects/weilab/shenb/nnUNet/nnUNet/nnunetv2

echo ">>> Starting nnUNet planning + preprocessing... <<<"
nnUNetv2_plan_and_preprocess -d 001 --verify_dataset_integrity

echo ">>> Starting nnUNet training <<<"
nnUNetv2_train 001 3d_fullres all

DURATION=$((SECONDS - START_TIME))
echo "Job ended at $(date)"
echo "Total runtime: $((DURATION / 60)) minutes and $((DURATION % 60)) seconds"
