# EvenNet
This is the official repository of NIPS 2022 paper *EvenNet: Ignoring Odd-Hop Neighbors Improves Robustness of Graph Neural Networks*.

## Setup
The implementation is based on python 3.
You can simply run

`pip install -r requirements.txt`

## Repository structure
```
|-- src
  |--attack.py          # Attack methods
  |--cSBM_dataset.py    # Generate cSBM datasets
  |--dataset_utils.py   # Dataloader
  |--main_atk.py        # Main code against graph attacks
  |--main_common.py     # Main code on common datasets
  |--main_inductive.py  # Main code on cSBM datasets
  |--models.py          # Models
  |--parse.py           # Parser & model loader
  |--propagate.py       # Propagation/Convolutional layers
  |--train_atk.py       # Training code against graph attacks
  |--train_common.py    # Training code on common datasets
  |--train_inductive.py # Training code on cSBM datasets
  |--utils.py           # Other used functions
  ##### scripts #####
  |--exp_common.sh      # EvenNet for node classification
  |--exp_csbm.sh        # EvenNet on cSBM datasets
  |--exp_dice.sh        # EvenNet against DICE attacks
  |--exp_heter.sh       # EvenNet for defense on heterophilic datasets.
  |--exp_matk.sh        # EvenNet against poison attacks
  |--exp_random.sh      # EvenNet against random attacks
  
|-- atk-data            # Dataset used for graph attacks
|-- data                # Dataset on clean datasets (ogbn-arxiv is not provided)
|-- logs                # Empty repo for logs
|-- GIA-HAO             # Experiments against graph injection attacks
```

## Run pipeline
1. cd the src/ directory
2. Running corresponding scripts
For example, to run expeiermnts on real-world datasets, try:
`sh exp_common.sh`

## Attribution
[GPRGNN](https://github.com/jianhao2016/GPRGNN)

[LINKX](https://github.com/CUAI/Non-Homophily-Large-Scale)

[DeepRobust](https://github.com/DSE-MSU/DeepRobust)

[GIA-HAO](https://github.com/LFhase/GIA-HAO)
