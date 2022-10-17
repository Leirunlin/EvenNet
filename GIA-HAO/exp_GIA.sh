for dataset in 'grb-cora' 'grb-citeseer'
do
for atk in gia seqgia agia
do
for alpha in 0.1 0.2 0.5 0.9
do
python gnn_misg.py --dataset $dataset  --inductive --eval_robo --model 'evennet' --eval_robo_blk --use_ln 1 --K 2 --eval_attack $atk --alpha $alpha
done
done
done

