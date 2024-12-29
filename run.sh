#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=64

PYTHONPATH=/home/mgregori/ltr-paper-code \
~/miniconda3/envs/decpy/bin/python ./main.py 
