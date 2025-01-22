#!/bin/bash
#SBATCH -c 8
#SBATCH --mem 80G
#SBATCH -t 1-00:10:00
#SBATCH -o outs/l702.out
#SBATCH -p gpu
#SBATCH --gres gpu:a100:1
#SBATCH -C gpupod 

# module load anaconda
source activate
conda activate llm
export PYTHONPATH="${PYTHONPATH}:/scratch/wtd3gz/project_TS/llm_game_ts/text_aid_forecast/"

MNAME='llama3p1_70B'
python3 ./tsllm/main.py experiment=blank_rorder model=$MNAME

#  python3 ./tsllm/main.py experiment=blank_rorder model=llama3p1_70B
#  A-SBATCH --gres gpu:a100:1
#  A-SBATCH -C gpupod 
# 
#  A-SBATCH --gres gpu:a40:1
