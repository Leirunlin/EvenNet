#!/bin/sh

train_rate=0.1
val_rate=0.1
runs=10

for dataset in cora citeseer acm pubmed
do
for ratio in 0.4 0.8 1.2 1.6
do
python main_atk.py --dataset $dataset --attack_type DICE --attack_ratio $ratio --runs $runs --net EvenNet --K 4 \
--epochs 200 --early_stopping 30 --train_rate $train_rate --val_rate $val_rate
done
done

