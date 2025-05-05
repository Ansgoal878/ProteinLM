#!/bin/bash
#SBATCH --job-name=PLM-50gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50gb
#SBATCH --gres=gpu:7g.40gb:1
#SBATCH --output=plm_example_%j.log
#SBATCH --partition=COMPUTE1Q  
#SBATCH --account=changlab

image_path=/raid/ChangLab/chang_ansgoal/image/plm_cu117_virt2.sif

# Set environment variables
export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUDA_VISIBLE_DEVICES=0
export TF_GPU_ALLOCATOR=cuda_malloc_async
export TF_ENABLE_AUTO_MIXED_PRECISION=1

# Set HuggingFace cache directory to your raid space with enough storage
export HF_HOME=/raid/ChangLab/chang_ansgoal/.cache/huggingface

# Main Code: Activate environment and run the script
singularity exec --nv --bind /raid:/raid --bind /mnt/nas_1:/mnt/nas_1 $image_path bash -c "source /usr/local/miniconda3/bin/activate plm_env && python examples/basic_usage.py"