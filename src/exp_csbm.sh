p=0.50
phi=50

# Generate cSBM datasets, we have provided generated datasets already.
# python cSBM_dataset.py --train_nodes 3000 --val_nodes 3000 --test_nodes 3000 --phi_train $p --phi_test -$p --name csbm_lh_$phi
# python cSBM_dataset.py --train_nodes 3000 --val_nodes 3000 --test_nodes 3000 --phi_train $p --phi_test $p --name csbm_ll_$phi
# python cSBM_dataset.py --train_nodes 3000 --val_nodes 3000 --test_nodes 3000 --phi_train -$p --phi_test -$p --name csbm_hh_$phi
# python cSBM_dataset.py --train_nodes 3000 --val_nodes 3000 --test_nodes 3000 --phi_train -$p --phi_test $p --name csbm_hl_$phi

for dataset in csbm_lh_$phi \
               csbm_ll_$phi \
               csbm_hl_$phi \
               csbm_hh_$phi 
do 
python main_inductive.py --dataset $dataset --device $device --net EvenNet --runs 10 --epochs 1000 --early_stopping 200 --hidden 64 --lr 0.01 --dropout 0.5 --weight_decay 0.0005
done

p=0.75
phi=75

# Generate cSBM datasets, we have provided generated datasets already.
# python cSBM_dataset.py --train_nodes 3000 --val_nodes 3000 --test_nodes 3000 --phi_train $p --phi_test -$p --name csbm_lh_$phi
# python cSBM_dataset.py --train_nodes 3000 --val_nodes 3000 --test_nodes 3000 --phi_train $p --phi_test $p --name csbm_ll_$phi
# python cSBM_dataset.py --train_nodes 3000 --val_nodes 3000 --test_nodes 3000 --phi_train -$p --phi_test -$p --name csbm_hh_$phi
# python cSBM_dataset.py --train_nodes 3000 --val_nodes 3000 --test_nodes 3000 --phi_train -$p --phi_test $p --name csbm_hl_$phi

for dataset in csbm_lh_$phi \
               csbm_ll_$phi \
               csbm_hl_$phi \
               csbm_hh_$phi 
do 
python main_inductive.py --dataset $dataset --device $device --net EvenNet --runs 10 --epochs 1000 --early_stopping 200 --hidden 64 --lr 0.01 --dropout 0.7 --weight_decay 0.0005
done
