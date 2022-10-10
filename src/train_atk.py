import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from copy import deepcopy
from deeprobust.graph.defense import GCN
import timeit


class Exp(nn.Module):
    def __init__(self, model, args, device=None):
        super(Exp, self).__init__()
        assert device is not None, "Please specify 'device'!"
        self.model = model.to(device)
        self.device = device
        self.dropout = args.dropout
        self.weight_decay = args.weight_decay
        self.k = args.K
        self.lr = args.lr
        self.pro_lr = args.pro_lr
        self.train_iters = args.epochs
        self.patience = args.early_stopping
        self.net = args.net
        self.dataset_name = args.dataset
        self.attack_type = args.attack_type
        self.output = None
        self.best_model = None
        self.best_output = None

    def fit(self, data):
        return self.train_with_early_stopping(data.to(self.device), self.train_iters, self.patience)

    def train_with_early_stopping(self, data, train_iters, patience):
        if self.net == "BernNet" or self.net == "GPRGNN" or self.net == "EvenNet":
            optimizer = torch.optim.Adam(
                [{'params': self.model.lin1.parameters(), 'weight_decay': self.weight_decay, 'lr': self.lr},
                 {'params': self.model.lin2.parameters(), 'weight_decay': self.weight_decay, 'lr': self.lr},
                 {'params': self.model.prop1.parameters(), 'weight_decay': 0.00, 'lr': self.pro_lr}])
        elif self.net == 'GCNII':
            optimizer = torch.optim.Adam([
             dict(params=self.model.reg_params, weight_decay=0.01),
             dict(params=self.model.non_reg_params, weight_decay=5e-4)
         ], lr=self.lr)
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        labels = data.y
        train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask
        early_stopping = patience
        best_loss_val = 100

        for i in range(train_iters):
            start = timeit.default_timer()
            self.model.train()
            optimizer.zero_grad()
            output = self.model(data)
            if self.net == "RGCN":
                loss_train = self.model.loss(output[train_mask], labels[train_mask])
            elif self.net == "GPRGNN_reg":
                loss_train = F.nll_loss(output[train_mask], labels[train_mask])
                theta = self.model.prop1.temp
                for j in range(1, self.k, 2):
                    loss_train += 0.05*torch.abs(theta[j])
            else:
                loss_train = F.nll_loss(output[train_mask], labels[train_mask])
            loss_train.backward()
            optimizer.step()
            self.model.eval()
            output = self.model(data)
            loss_val = F.nll_loss(output[val_mask], labels[val_mask])
            preds = output[val_mask].max(1)[1].type_as(labels)
            correct = preds.eq(labels[val_mask]).double()
            correct = correct.sum()
            val_acc = correct / len(labels[val_mask])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.model.state_dict())
                patience = early_stopping
            else:
                patience -= 1
            if i > early_stopping and patience <= 0:
                break
            if i % 10 == 0:
                print("Train set results:",
                      "epoch= {:.1f}".format(i), '\n', \
                      "train loss={}".format(loss_train), "val_acc={:.3f}".format(val_acc))
        stop = timeit.default_timer()
        tot_time = stop - start
        avg_time = tot_time / i

        self.model.load_state_dict(weights)
        output = self.model(data)
        preds = output[val_mask].max(1)[1].type_as(labels)
        correct = preds.eq(labels[val_mask]).double()
        correct = correct.sum()
        best_val_acc = correct / len(labels[val_mask])
        print("Train set results:",
              "epoch= {:.1f}".format(i))
        return best_val_acc.item(), avg_time

    def test(self, data):
        self.model.eval()
        data = data.to(self.device)
        val_mask, test_mask = data.val_mask, data.test_mask
        labels = data.y
        output = self.model(data)

        loss_test = F.nll_loss(output[test_mask], labels[test_mask])
        preds = output[test_mask].max(1)[1].type_as(labels)
        correct = preds.eq(labels[test_mask]).double()
        correct = correct.sum()
        acc_test = correct / len(labels[test_mask])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))

        return acc_test.item()

    def predict(self, data):
        self.model.eval()
        return self.model(data)

