#!/bin/bash
#SBATCH -c 8
#SBATCH --mem 80G
#SBATCH -t 1-00:10:00
#SBATCH -o outs/l702.out
#SBATCH -p gpu
#SBATCH --gres gpu:a100:1


MNAME='llama3p1_70B'
python3 ./tsllm/main.py experiment=blank_rorder model=$MNAME