#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:A6000:1
#SBATCH --chdir=/home/mgregori/ltr-paper-code/p2p_el

module load cuda11.7/toolkit/11.7

env_python=/var/scratch/mgregori/miniconda3/envs/decpy/bin/python
graph=fullyConnected_24.edges
run_path=./eval/data
config_file=config_EL.ini
cp $graph $config_file $run_path

machines=1 # number of machines in the runtime
iterations=80
test_after=20
eval_file=main.py
log_level=INFO # DEBUG | INFO | WARN | CRITICAL

m=0 # machine id corresponding consistent with ip.json
echo M is $m

procs_per_machine=24
echo procs per machine is $procs_per_machine

mkdir -p ./eval/data/gossip ./eval/data/local

echo "Running gossip experiment"
$env_python $eval_file -ro 0 -tea $test_after -ld ./eval/data/gossip -mid $m -ps $procs_per_machine -ms $machines -is $iterations -gf $run_path/$graph -ta $test_after -cf $run_path/$config_file -ll $log_level

echo "Running local-only experiment"
$env_python $eval_file --local-only -ro 0 -tea $test_after -ld ./eval/data/local -mid $m -ps $procs_per_machine -ms $machines -is $iterations -gf $run_path/$graph -ta $test_after -cf $run_path/$config_file -ll $log_level
