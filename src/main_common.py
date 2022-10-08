from dataset_utils import DatasetLoader
from train_common import Exp
from utils import *
from attack import *
import argparse
from parse import parse_method, parser_add_main_args


### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()

# load dataset
dataset = DatasetLoader(name=args.dataset)
data = dataset[0]

# Same split as in BernNet
SEEDS = [1941488137, 4198936517, 983997847, 4023022221, 4019585660, 2108550661, 1648766618, 629014539, 3212139042,
        2424918363]
device = torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu')
args.device = device
results = []

for RP in range(args.runs):
    args.train_rate = 0.6
    args.val_rate = 0.2
    percls_trn = int(round(args.train_rate * len(data.y) / dataset.num_classes))
    val_lb = int(round(args.val_rate * len(data.y)))
    data = random_planetoid_splits(dataset[0], dataset.num_classes, percls_trn, val_lb, seed=SEEDS[RP])
    model = parse_method(dataset=dataset, args=args)
    exp = Exp(model, args, device)
    # fit model
    best_val_acc = exp.fit(data)
    test_acc = exp.test(data)
    results.append([best_val_acc, test_acc])


# ######################
# #### Summary part ####
# ######################

results = np.array(results, dtype=object)
# Accuracy summary
val_acc_mean, test_acc_mean = np.mean(results, axis=0) * 100
# Uncertainty summary
val_values = np.asarray(results)[:, 0]
test_values = np.asarray(results)[:, 1]

val_uncertainty = cal_uncertainty(val_values)
test_uncertainty = cal_uncertainty(test_values)

# Total summary
gnn_name = args.net
exp_name = f'{gnn_name} on dataset {args.dataset}'
acc_res = f'Val acc mean = {val_acc_mean:.4f} ± ' f'{val_uncertainty * 100:.4f}  \t '\
          f'Test acc mean = {test_acc_mean:.4f} ± ' f'{test_uncertainty * 100:.4f}  \t ' \

with open("../logs/log_common_" + gnn_name + ".txt", 'a+') as f:
    f.write("-------------------------------------------------\n")
    f.write(exp_name + "\n")
    f.write(acc_res + "\n")
    f.write("-------------------------------------------------\n\n")
    f.close()
