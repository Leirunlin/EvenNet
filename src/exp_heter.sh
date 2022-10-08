atk_ratio=0.2
atk_type=MinMax
runs=5

for dataset in chameleon
do
python main_atk.py --dataset $dataset --net EvenNet --attack_ratio $atk_ratio --attack_type $atk_type --lr 0.01 --weight_decay 0.0 --runs $runs
done

for dataset in squirrel
do
python main_atk.py --dataset $dataset --net EvenNet --attack_ratio $atk_ratio --attack_type $atk_type --lr 0.05 --weight_decay 0.0 --runs $runs
done




