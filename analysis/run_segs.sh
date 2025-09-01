#!/bin/bash
#SBATCH --partition=gpus48
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/biomedic3/bglocker/mscproj/mal224/DCCL/analysis/segmentations/slurm_log.%N.%j.log
#SBATCH --job-name=segmentations

source /vol/biomedic3/bglocker/mscproj/mal224/miniforge3/bin/activate cf-contrastive-seg
export PYTHONPATH=/vol/biomedic3/bglocker/mscproj/mal224/DCCL
cd /vol/biomedic3/bglocker/mscproj/mal224/DCCL

python -m analysis.save_segs