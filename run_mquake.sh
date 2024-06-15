# for num_shot in 3 4 5; do
#     python icl_cot.py \
#         --prompt_type human_icl \
#         --prompt_model None \
#         --num_shot $num_shot \
#         --num_human_icl -1 \
#         --json datasets/mquake/MQuAKE-CF-3k.json \
#         --dataset mquake \
#         --out_dir output/mquake &
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
#     for num_shot in 1 2 3 4 5; do
#         python icl_cot.py \
#             --prompt_type chatgpt_icl_by_zeroshot \
#             --prompt_model $p_model \
#             --num_shot $num_shot \
#             --num_human_icl -1 \
#             --json datasets/mquake/MQuAKE-CF-3k.json \
#             --dataset mquake \
#             --out_dir output/mquake
#     done
# done

# for num_human_icl in 1 2 3 4 5; do
#     for p_model in gpt-4o gpt-3.5-turbo gpt-4-turbo; do
#             sbatch srun_mquake.sh $p_model 3 $num_human_icl
#     done
# done

for num_shot in 4 5; do
    python icl_cot.py \
        --prompt_type gptj_icl_by_human_icl \
        --prompt_model None \
        --num_shot $num_shot \
        --num_human_icl -1 \
        --json datasets/mquake/MQuAKE-CF-3k.json \
        --dataset mquake \
        --out_dir output/mquake &
done

# # traverse all json files in output/mquake
# for json_file in output/mquake/*.json; do
#     python interpret_mquake.py $json_file
# done