# Python interpreter
PYTHON = python

fetch:
	rsync -az mgregori@fs3.das6.tudelft.nl:~/tribler/crawl .
	$(PYTHON) fetch.py
	$(PYTHON) generate_tribler_ltr_dataset.py

push:
	rsync -az tribler_data/* mgregori@fs3.das6.tudelft.nl:/var/scratch/mgregori/datasets/tribler_crawl/

run-local:
	$(PYTHON) main.py
	
run-slurm:
	sbatch run.sh

all: fetch run

.PHONY: fetch run all
