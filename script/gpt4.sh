#!/bin/bash
#SBATCH -c 8
#SBATCH --mem 80G
#SBATCH -t 0-20:10:00
#SBATCH -o outs/gpt4o.out
#SBATCH -p gpu
#SBATCH --gres gpu:a100:1

MNAME='gpt-4'
python3 ./tsllm/main.py experiment=blank_rorder model=$MNAME
