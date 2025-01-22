#!/bin/bash
#SBATCH -c 8
#SBATCH --mem 80G
#SBATCH -t 0-15:10:00
#SBATCH -o outs/m8.out
#SBATCH -p gpu
#SBATCH --gres gpu:a6000:1

# module load anaconda
source activate
conda activate llm

export PYTHONPATH="${PYTHONPATH}:/scratch/wtd3gz/project_TS/llm_game_ts/text_aid_forecast/"
    
MNAME='mistral_8B'
python3 ./tsllm/main.py experiment=blank_rorder model=$MNAME
# python3 ./tsllm/main.py experiment=blank_rorder model=mistral_7x8B