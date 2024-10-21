# Python interpreter
PYTHON = python

fetch:
	$(PYTHON) fetch.py

run:
	$(PYTHON) main.py

all: fetch run

.PHONY: fetch run all
