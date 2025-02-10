# Python interpreter
PYTHON = python

clean:
	rm -rf .tmp slurm-*.out *.png chronological_ndcgs.pkl p2p_el/eval

fetch:
	rsync -az mgregori@fs3.das6.tudelft.nl:~/tribler/crawl .
	$(PYTHON) fetch.py

push:
	rsync -az tribler_data/* mgregori@fs3.das6.tudelft.nl:/var/scratch/mgregori/datasets/tribler_crawl/

run: clean
	$(PYTHON) main.py
	
run-slurm:
	sbatch run.sh

all:
	sbatch run_general_eval.sh
	sbatch run_context_eval.sh
	sbatch run_context_eval.sh --ltr
	sbatch run_p2p_eval.sh
	sbatch run_p2p_eval.sh --ltr
	sbatch run_ablation_eval.sh

.PHONY: clean fetch push run run-slurm
