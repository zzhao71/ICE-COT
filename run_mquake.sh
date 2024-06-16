# for num_shot in 1 2 3 4 5; do
#     sbatch srun_icl_cot.sh chatgpt_icl_by_human_icl gpt-4o $num_shot 5 datasets/mquake/MQuAKE-CF-3k.json mquake output/mquake 
# done

# python icl_cot.py \
#     --prompt_type zero_shot \
#     --prompt_model None \
#     --num_shot -1 \
#     --num_human_icl -1 \
#     --json datasets/mquake/MQuAKE-CF-3k.json \
#     --dataset mquake \
#     --out_dir output/mquake

# for p_model in gpt-4o gpt-3.5-turbo gpt-4-turbo; do
#     sbatch srun_mquake.sh chatgpt_icl_by_zeroshot $p_model 3 -1 datasets/mquake/MQuAKE-CF-3k.json mquake output/mquake
# done

# for num_human_icl in 1 2 3 4 5; do
#     for p_model in gpt-4o gpt-3.5-turbo gpt-4-turbo; do
#             sbatch srun_mquake.sh $p_model 3 $num_human_icl
#     done
# done

# for num_shot in 1 2 3 4 5; do
#     sbatch srun_icl_cot.sh gptj_icl_by_human_icl None $num_shot -1 datasets/mquake/MQuAKE-CF-3k.json mquake output/mquake
# done

# # traverse all json files in output/mquake
for json_file in output/mquake/MQuAKE-CF-3k_output_chatgpt-icl-by-human-icl_pmodel-gpt-4o_num-shot-3_num-human-icl-*.json; do
    python interpret_mquake.py $json_file
done

# bash srun_icl_cot.sh human_icl_only None 3 3 datasets/mquake/MQuAKE-CF-3k.json mquake output/mquake