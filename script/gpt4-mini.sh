#!/bin/bash
#SBATCH -c 8
#SBATCH --mem 80G
#SBATCH -t 0-20:10:00
#SBATCH -o outs/gpt4o-mini.out
#SBATCH -p gpu
#SBATCH --gres gpu:a40:1

source activate
conda activate llm

export PYTHONPATH="${PYTHONPATH}:/scratch/wtd3gz/project_TS/llm_game_ts/text_aid_forecast/"

MNAME='gpt-4-mini'
python3 ./tsllm/main.py experiment=blank_rorder model=$MNAME
#  python3 ./tsllm/main.py experiment=blank_rorder model=gpt-4-mini