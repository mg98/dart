# 🎯 DART: Decentralized Adaptive Ranking using Transformers

DART is a decentralized ranking algorithm that uses a transformer-based learning-to-rank (LTR) model to rank documents in a decentralized manner.

This repository contains the code used to produce the results in _Decentralized Adaptive Ranking using Transformers_ by Marcel Gregoriadis, Quinten Stokkink, and Johan Pouwelse, published at _EuroMLSys 2025_.

## Dataset

As part of this work, we also release a new LTR dataset based on workload from the decentralized application [Tribler](https://github.com/Tribler/tribler). The dataset and a pre-trained DART model are available in [`artifacts/`](./artifacts).

If you use this dataset, please cite the following paper:

```bibtex
@inproceedings{gregoriadis2025decentralized,
  title={Decentralized Adaptive Ranking using Transformers},
  author={Gregoriadis, Marcel and Stokkink, Quinten and Pouwelse, Johan},
  booktitle={Proceedings of the 5th Workshop on Machine Learning and Systems},
  pages={},
  year={2025}
}
```

## Installation and Usage

Note that our results were generated from a raw initial dataset, which due to privacy concerns, is not included in this repository.
It is therefore not possible to reproduce the exact results reported in the paper, and rather serves illustrative purposes.

For completeness, we still provide the installation instructions:

```
conda create -n dart python=3.9
conda activate dart
make install
```

The scripts to run the experiments are located in [`scripts/`](./scripts);
results are interpreted in [`results.ipynb`](results.ipynb).
