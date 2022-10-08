from deeprobust.graph.data import Dpr2Pyg
from deeprobust.graph.data import Dataset
from dataset_utils import DatasetLoader
from train_atk import Exp, deepcopy
import argparse
from parse import parse_method, parser_add_main_args
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import to_undirected
from utils import *
from attack import *


### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
device = torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu')
args.device = device
print(args)

# load dataset
SEEDS = [15, 26, 37, 48, 59, 70, 81, 92, 103, 114]
results = []
time_list = []

for RP in range(args.runs):
    if args.dataset not in ["arxiv", 'chameleon', 'squirrel']:
        dataset = Dataset(root='../atk_data/data', name=args.dataset, setting='nettack', seed=SEEDS[RP])
        dataset = Dpr2Pyg(dataset)
        data = dataset[0]
    elif args.dataset in ['arxiv']:
        # Arxiv
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='../data')
        split_idx = dataset.get_idx_split()
        data = dataset[0]
        n = data.num_nodes
        data.train_mask = index_to_mask(split_idx['train'], size=n)
        data.val_mask = index_to_mask(split_idx['valid'], size=n)
        data.test_mask = index_to_mask(split_idx['test'], size=n)
        data.edge_index = to_undirected(data.edge_index)
        data.y = torch.ravel(data.y)
    else:
        # Chamelon & Squirrel
        dataset = DatasetLoader(name=args.dataset)
        data = dataset[0]
        args.train_rate = 0.6
        args.val_rate = 0.2
        percls_trn = int(round(args.train_rate * len(data.y) / dataset.num_classes))
        val_lb = int(round(args.val_rate * len(data.y)))
        data = random_planetoid_splits(dataset[0], dataset.num_classes, percls_trn, val_lb, seed=SEEDS[RP])

    model = parse_method(dataset=dataset, args=args)
    Atk_model = Attacker(args, data)
    # attack
    adj_attack = Atk_model.attack(RP)

    if args.attack_type in ['Meta', 'MinMax', 'Random']:
        exp = Exp(model, args, device)
        atk_data = deepcopy(data)
        atk_data.edge_index = adj_attack
        best_val_acc, avg_time = exp.fit(atk_data)
        test_acc_before = 1  # Only test once
        test_acc_after = exp.test(atk_data)
    else:
        # DICE Attack
        model = parse_method(dataset=dataset, args=args)
        exp = Exp(model, args, device)
        best_val_acc, avg_time = exp.fit(data)
        test_acc_before = exp.test(data)
        # Avoid adj cache
        if args.net in ['GCN_SVD', 'GCN_Jaccard', "RGCN", "H2GCN"]:
            model.reset_adj = True
        atk_data = deepcopy(data)
        atk_data.edge_index = adj_attack
        test_acc_after = exp.test(atk_data)
    acc_drop = (test_acc_before - test_acc_after) / test_acc_before
    results.append([best_val_acc, test_acc_before, test_acc_after, acc_drop])
    # record time
    time_list.append(avg_time * 1000)

# ######################
# #### Summary part ####
# ######################

results = np.array(results, dtype=object)
avg_time_mean = round(np.mean(time_list), 3)
# Accuracy summary
val_acc_mean, test_acc_before_mean, test_acc_after_mean, acc_drop_mean = np.mean(results, axis=0) * 100

# Uncertainty summary
val_values = np.asarray(results)[:, 0]
test_before_values = np.asarray(results)[:, 1]
test_after_values = np.asarray(results)[:, 2]
acc_drop_values = np.asarray(results)[:, 3]

val_uncertainty = cal_uncertainty(val_values)
test_before_uncertainty = cal_uncertainty(test_before_values)
test_after_uncertainty = cal_uncertainty(test_after_values)
acc_drop_uncertainty = cal_uncertainty(acc_drop_values)
# Total summary
gnn_name = args.net
exp_name = f'{gnn_name} on dataset {args.dataset}, atk_type {args.attack_type}, ' \
           f'atk_pct {args.attack_ratio}, in {avg_time_mean} time and {args.runs} repeated experiment:'
acc_res = f'Val acc mean = {val_acc_mean:.4f} ± ' f'{val_uncertainty * 100:.4f}  \t '\
          f'Before atk acc mean = {test_acc_before_mean:.4f} ± ' f'{test_before_uncertainty * 100:.4f}  \t ' \
          f'After atk acc mean = {test_acc_after_mean:.4f} ± ' f'{test_after_uncertainty * 100:.4f}  \t ' \
          f'Acc drop mean = {acc_drop_mean:.4f} ± ' f'{acc_drop_uncertainty * 100:.4f}  \t ' \


with open("../logs/log_atk_{}.txt".format(args.attack_type), 'a+') as f:
    f.write("-------------------------------------------------\n")
    f.write(exp_name + "\n")
    f.write(acc_res + "\n")
    f.write("-------------------------------------------------\n\n")
    f.close()
