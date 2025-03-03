PYTHON = python

install:
	git submodule update --init --recursive
	pip install -r requirements.txt

clean:
	rm -rf .tmp slurm-*.out

fetch:
	rsync -az mgregori@fs3.das6.tudelft.nl:~/tribler/crawl .
	$(PYTHON) fetch.py

.PHONY: clean fetch
