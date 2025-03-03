#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0
#SBATCH --cpus-per-task=32

module load cuda12.3/toolkit/12.3

PYTHONPATH=/home/mgregori/ltr-paper-code \
~/miniconda3/envs/decpy/bin/python ./eval_context.py "$@"