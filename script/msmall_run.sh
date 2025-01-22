#!/bin/bash
#SBATCH -c 8
#SBATCH --mem 80G
#SBATCH -t 0-15:10:00
#SBATCH -o outs/msml.out
#SBATCH -p gpu
#SBATCH --gres gpu:a100:2

source activate
conda activate llm

export PYTHONPATH="${PYTHONPATH}:/scratch/wtd3gz/project_TS/llm_game_ts/text_aid_forecast/"

MNAME='mistral_small'
python3 ./tsllm/main.py experiment=blank_rorder model=$MNAME
# python3 ./tsllm/main.py experiment=blank_rorder model=mistral_7B
#  --gres gpu:a100:2
#  -C gpupod 
