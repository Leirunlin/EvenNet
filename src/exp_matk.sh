#!/bin/sh

train_rate=0.1
val_rate=0.1
runs=5

# To achieve better performance, you can try tunning alpha in [0.1, 0.2, 0.5, 0.9]
#   A default alpha=0.1 is usually good enough.

for atk in Meta MinMax
do
for dataset in cora citeseer acm 
do
for alpha in 0.1 0.2 0.5 0.9
do
python main_atk.py --dataset $dataset --net EvenNet --train_rate $train_rate --val_rate $val_rate --attack_ratio 0.2 --attack_type $atk --lr 0.01 --hidden 64 \
                    --alpha $alpha --runs $runs --K 10 --epochs 200 --early_stopping 30
done
done
done
