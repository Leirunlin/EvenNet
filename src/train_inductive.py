import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch_geometric.data import DataLoader
from copy import deepcopy


class Exp(nn.Module):
    def __init__(self, model, args, device=None):
        super(Exp, self).__init__()
        assert device is not None, "Please specify 'device'!"
        self.model = model.to(device)
        self.device = device
        self.k = args.K
        self.dropout = args.dropout
        self.weight_decay = args.weight_decay
        self.lr = args.lr
        self.pro_lr = args.pro_lr
        self.train_iters = args.epochs
        self.patience = args.early_stopping
        self.net = args.net
        self.dataset_name = args.dataset
        self.output = None
        self.best_model = None
        self.best_output = None

    def fit(self, train_dataset, val_dataset):
        return self.train_with_early_stopping(train_dataset, val_dataset, self.train_iters, self.patience)

    def train_with_early_stopping(self, train_dataset, val_dataset, train_iters, patience):
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

        best_val_acc = 0.0
        early_stopping = patience
        best_loss_val = 100
        train_loader = DataLoader(train_dataset)
        val_loader = DataLoader(val_dataset)
        for i in range(train_iters):
            train_loss = val_loss = train_acc = val_acc = 0.0
            self.model.train()
            if self.net == 'H2GCN':
                self.model.reset_adj = True
            for step, batch in enumerate(train_loader):
                optimizer.zero_grad()
                batch.to(self.device)
                out = self.model(batch)
                if self.net == "GPRGNN_reg":
                    loss = F.nll_loss(out, batch.y)
                    theta = self.model.prop1.temp
                    for j in range(1, self.k, 2):
                        loss += 0.05 * torch.abs(theta[j])
                else:
                    loss = F.nll_loss(out, batch.y)

                train_loss += loss.item()
                loss.backward()
                optimizer.step()

                self.model.eval()
                logits = self.model(batch)
                pred = logits.max(1)[1]
                num_nodes = batch.y.shape[0]
                train_acc += pred.eq(batch.y).sum().item() / num_nodes

            n_graphs = train_dataset.__len__()
            train_acc /= n_graphs
            train_loss /= n_graphs

            # validation
            self.model.eval()
            if self.net == 'H2GCN':
                self.model.reset_adj = True
            for step, batch in enumerate(val_loader):
                batch.to(self.device)
                logits = self.model(batch)
                val_loss += F.nll_loss(logits, batch.y)
                pred = logits.max(1)[1]
                num_nodes = batch.y.shape[0]
                val_acc += pred.eq(batch.y).sum().item() / num_nodes

            n_graphs = val_dataset.__len__()
            val_acc /= n_graphs
            if best_loss_val > val_loss:
                best_loss_val = val_loss
                best_val_acc = val_acc
                weights = deepcopy(self.model.state_dict())
                patience = early_stopping
            else:
                patience -= 1
            if i > early_stopping and patience <= 0:
                break
            if i % 10 == 0:
                print("Train set results:",
                      "epoch= {:.1f}".format(i), '\n', \
                      "train loss={}".format(train_loss), "val_acc={:.3f}".format(val_acc))

        self.model.load_state_dict(weights)
        print("Train set results:",
              "epoch= {:.1f}".format(i))
        return train_acc, best_val_acc

    def test(self, test_dataset):
        self.model.eval()
        acc = 0
        test_loss = 0
        loader = DataLoader(test_dataset)
        for step, batch in enumerate(loader):
            batch.to(self.device)
            logits = self.model(batch)
            test_loss += F.nll_loss(logits, batch.y)
            pred = logits.max(1)[1]
            num_nodes = batch.y.shape[0]
            acc += pred.eq(batch.y).sum().item() / num_nodes

        n_graphs = test_dataset.__len__()
        acc /= n_graphs
        test_loss /= n_graphs
        print("Test set results:",
              "accuracy= {:.4f}".format(acc))
        return acc, test_loss

