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

p_model=$1
num_shot=$2
num_human_icl=$3


python icl_cot.py \
    --prompt_type chatgpt_icl_by_human_icl \
    --prompt_model $p_model \
    --num_shot $num_shot \
    --num_human_icl $num_human_icl \
    --json datasets/MQuAKE-CF-3k.json \
    --dataset mquake \
    --out_dir output/mquake