#!/bin/bash
#SBATCH --job-name=infer_Dataset001_mito2_3d_fullres
#SBATCH --output=/projects/weilab/shenb/mitoFoundation2/5model_training/slurm/logs/infer/%x_%j.out
#SBATCH --error=/projects/weilab/shenb/mitoFoundation2/5model_training/slurm/logs/infer/%x_%j.err
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

MITO2_PROJECT_ROOT="${MITO2_PROJECT_ROOT:-/projects/weilab/shenb/mitoFoundation2}"
export nnUNet_raw="${nnUNet_raw:-${MITO2_PROJECT_ROOT}/data/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-${MITO2_PROJECT_ROOT}/data/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-${MITO2_PROJECT_ROOT}/data/nnUNet_results}"

source "${CONDA_SH:-/projects/weilab/shenb/miniconda3/etc/profile.d/conda.sh}"
conda activate "${NNUNET_CONDA_ENV:-nnunetv2}"
echo "Using Python at: $(which python)"

echo "Inference job started at $(date)"
START_TIME=$SECONDS

cd "${MITO2_PROJECT_ROOT}/5model_training/nnUNet"

echo ">>> Starting nnUNet prediction <<<"
nnUNetv2_predict \
    -i "${MITO2_PROJECT_ROOT}/data/nnUNet_raw/Dataset001_mito2/imagesTs" \
    -o "${MITO2_PROJECT_ROOT}/data/outputs/bc" \
    -d 001 \
    -c 3d_fullres \
    -f all \
    --save_probabilities

DURATION=$((SECONDS - START_TIME))
echo "Inference job ended at $(date)"
echo "Total runtime: $((DURATION / 60)) minutes and $((DURATION % 60)) seconds"
