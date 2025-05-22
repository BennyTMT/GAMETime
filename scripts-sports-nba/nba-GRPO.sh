#!/bin/bash
#SBATCH -c 8
#SBATCH --mem 80G
#SBATCH -t 0-12:10:00
#SBATCH -o ./nba.out
#SBATCH -p gpu
#SBATCH --gres gpu:h200:8

monitor_gpu() {
    while true
    do
        nvidia-smi >> ./gpu.out
        sleep 180
    done
}
monitor_gpu &

export WANDB_CONSOLE=off 
export WANDB_MODE=offline
accelerate launch  --config_file recipes/accelerate_configs/zero3.yaml \
    --num_processes=7  src/open_r1/grpo_sports.py \
    --config recipes/sports/Qwen2.5-1.5B-Instruct.yaml \
    --output_dir=YOUR_OUTPUT \
    --save_strategy='steps' \
    --save_steps='500' \
    --eval_strategy='no' \
    --do_eval=0 \
    --max_prompt_length=1880 \
    --max_completion_length=2048 \
    --model_name_or_path=YOUR_SVAE_DIR \
    --dataset_name=YOUR_DATA_PATH \
    --num_generations=16