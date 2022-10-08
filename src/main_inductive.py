import argparse
from dataset_utils import DatasetLoader
from train_inductive import Exp
from utils import *
from parse import parse_method, parser_add_main_args


### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()

device = torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu')
args.device = device

# load dataset
train_dataset, val_dataset, test_dataset = DatasetLoader(args.dataset)

results = []
type_results = []

for RP in range(args.runs):
    model = parse_method(dataset=train_dataset, args=args)
    # Random split
    exp = Exp(model, args, device)
    # fit model
    train_acc, best_val_acc = exp.fit(train_dataset, val_dataset)
    test_acc, test_loss = exp.test(test_dataset)
    results.append([train_acc, test_acc, best_val_acc])


# ######################
# #### Summary part ####
# ######################

# Accuracy summary
results = np.array(results, dtype=object)
train_acc_mean, test_acc_mean, val_acc_mean = np.mean(results, axis=0) * 100

# Uncertainty summary
train_values = np.asarray(results)[:, 0]
test_values = np.asarray(results)[:, 1]
val_values = np.asarray(results[:, 2])

train_uncertainty = cal_uncertainty(train_values)
test_uncertainty = cal_uncertainty(test_values)
val_uncertainty = cal_uncertainty(val_values)

# Total summary
gnn_name = args.net
exp_name = f'{gnn_name} on dataset {args.dataset}, ' \
           f'in {args.runs} repeated experiment:'
acc_res = f'train acc mean = {train_acc_mean:.4f}± ' f'{train_uncertainty * 100:.4f}  \t ' \
          f'test acc mean = {test_acc_mean:.4f} ± ' f'{test_uncertainty * 100:.4f}  \t ' \
          f'val acc mean = {val_acc_mean:.4f} ± ' f'{val_uncertainty * 100:.4f}  \t '

with open("../logs/log_ind.txt", 'a+') as f:
    f.write("-------------------------------------------------\n")
    f.write(exp_name + "\n\n")
    f.write(acc_res + "\n")
    f.write("-------------------------------------------------\n")
    f.close()
