#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=120:00:00
#SBATCH --cpus-per-task=4

module load cuda12.3/toolkit/12.3

PYTHONPATH=/home/mgregori/ltr-paper-code \
~/miniconda3/envs/decpy/bin/python ./main.py 
