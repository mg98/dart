# Python interpreter
PYTHON = python

clean:
	rm -rf .tmp slurm-*.out *.png chronological_ndcgs_results.pkl

fetch:
	rsync -az mgregori@fs3.das6.tudelft.nl:~/tribler/crawl .
	$(PYTHON) fetch.py
	$(PYTHON) generate_tribler_ltr_dataset.py

push:
	rsync -az tribler_data/* mgregori@fs3.das6.tudelft.nl:/var/scratch/mgregori/datasets/tribler_crawl/

run: clean
	$(PYTHON) main.py
	
run-slurm:
	sbatch run.sh

.PHONY: clean fetch push run run-slurm
