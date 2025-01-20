#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:A6000:1
#SBATCH --chdir=/home/mgregori/ltr-paper-code/p2p_el

#module load cuda12.3/toolkit/12.3
module load cuda11.7/toolkit/11.7

./run_el-local.sh