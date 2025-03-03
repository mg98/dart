#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:A100:1

module load cuda12.3/toolkit/12.3

PYTHONPATH=/home/mgregori/ltr-paper-code \
~/miniconda3/envs/decpy/bin/python ./eval_p2p.py
