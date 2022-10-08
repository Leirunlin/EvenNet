#!/bin/sh

for dataset in cora
do
python main_common.py --dataset $dataset --net EvenNet --early_stopping 200 --epochs 1000 --hidden 64 --weight_decay 0.0005  --runs 10 --lr 0.01 --dropout 0.5 --dprate 0.5 --K 10 
done

for dataset in citeseer 
do
python main_common.py --dataset $dataset --net EvenNet --early_stopping 200 --epochs 1000 --hidden 64 --weight_decay 0.0005  --runs 10 --lr 0.05 --dropout 0.1 --dprate 0.3 --K 10 
done

for dataset in pubmed 
do
python main_common.py --dataset $dataset --net EvenNet --early_stopping 200 --epochs 1000 --hidden 64 --weight_decay 0.0005  --runs 10 --lr 0.05 --dropout 0.1 --dprate 0.3 --K 10 
done

for dataset in chameleon
do
python main_common.py --dataset $dataset --net EvenNet --early_stopping 200 --epochs 1000 --hidden 64 --weight_decay 0.0  --runs 10 --lr 0.05 --dropout 0.5 --dprate 0.7 --K 10 
done

for dataset in squirrel
do
python main_common.py --dataset $dataset --net EvenNet --early_stopping 200 --epochs 1000 --hidden 64 --weight_decay 0.0  --runs 10 --lr 0.05 --dropout 0.5 --dprate 0.7 --K 10 
done

for dataset in actor
do
python main_common.py --dataset $dataset --net EvenNet --early_stopping 200 --epochs 1000 --hidden 64 --weight_decay 0.0005  --runs 10 --lr 0.05 --dropout 0.3 --dprate 0.7 --K 10 
done

for dataset in cornell texas 
do
python main_common.py --dataset $dataset --net EvenNet --early_stopping 200 --epochs 1000 --hidden 64 --weight_decay 0.0005  --runs 10 --lr 0.05 --dropout 0.5 --dprate 0.1 --K 10 
done

