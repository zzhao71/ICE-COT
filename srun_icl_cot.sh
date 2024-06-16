#!/bin/bash
#SBATCH --job-name=test
#SBATCH --time=24:0:0
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output "slurm_logs/slurm-%j.out"
mkdir -p slurm_logs




python icl_cot.py \
    --prompt_type $1 \
    --prompt_model $2 \
    --num_shot $3 \
    --num_human_icl $4 \
    --json $5 \
    --dataset $6 \
    --out_dir $7