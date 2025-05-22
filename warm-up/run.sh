#!/bin/bash
#SBATCH -c 8
#SBATCH --mem 80G
#SBATCH -t 0-10:10:00
#SBATCH -o ./sft_nba.out
#SBATCH -p gpu
#SBATCH --gres gpu:h200:1

conda activate open-r1
python3 sft-sports-Qwen1p5B.py