#!/bin/sh

train_rate=0.1
val_rate=0.1
runs=10
hid=64

# To achieve better performance, try tunning alpha over [0.1, 0.2, 0.5, 0.9]
#   A default alpha=0.1 is usually good enough.

for dataset in cora citeseer 
do
for ratio in 0.2 0.4 0.6
do
for alpha in 0.1 0.2 0.5 0.9
do
python main_atk.py --alpha $alpha --dataset $dataset --attack_type Random --attack_ratio $ratio --runs $runs \
--net EvenNet --hidden $hid --epochs 200 --early_stopping 30 --K 10
done
done
done