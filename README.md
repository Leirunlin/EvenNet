# EvenNet
This is the official repository of NIPS 2022 paper *EvenNet: Ignoring Odd-Hop Neighbors Improves Robustness of Graph Neural Networks*.

## Setup
The implementation is based on python 3, and
* deeprobust==0.2.4
* dgl==0.6.0
* numpy==1.18.1
* ogb==1.3.3
* torch==1.8.1
* torch_geometric==1.6.3

You can simply run
`pip install -r requirements.txt`.

## Dataset
We provide generated cSBM datasets, real-world datasets in "./data", and pertubed graphs in "./atk_data/atk_adj/".
Ogbn-arxiv is not provided, you can download it via [ogb official](https://ogb.stanford.edu/docs/nodeprop/)

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
|-- data                # Dataset on clean datasets
|-- logs                # Empty repo for logs
|-- GIA-HAO             # Experiments against graph injection attacks
```

## Run pipeline
1. Create empty directory ./logs/ (To save the experiment results.)
2. cd the ./src/ directory
3. Running corresponding scripts
For example, to run experiments on real-world datasets, try:

   `sh exp_common.sh` 

   To run experiments against Metattack / MinMax attack, try:
  
  `sh exp_matk.sh`

## Attribution
[GPRGNN](https://github.com/jianhao2016/GPRGNN)

[LINKX](https://github.com/CUAI/Non-Homophily-Large-Scale)

[DeepRobust](https://github.com/DSE-MSU/DeepRobust)

[GIA-HAO](https://github.com/LFhase/GIA-HAO)

## Citation
```
@inproceedings{Lei2022evennet,
  title={EvenNet: Ignoring Odd-Hop Neighbors Improves Robustness of Graph Neural Networks},
  author={Lei, Runlin and Wang, Zhen and Li, Yaliang and Ding, Bolin and Wei, Zhewei},
  booktitle={NeurIPS},
  year={2022}
}
```
