import scipy as sp
from torch_geometric.data import Data
from torch_geometric.utils import get_laplacian
import pickle
from datetime import datetime
from tqdm import tqdm
from utils import *
import os.path as osp
import os
import argparse
import torch
import matplotlib.pyplot as plt
from torch_geometric.data import InMemoryDataset


def ContextualSBM(n, n_total, d, Lambda, p, mu, phase, name, seed, save_figs=False):
    np.random.seed()
    c_in = d + np.sqrt(d) * Lambda
    c_out = d - np.sqrt(d) * Lambda
    y = np.ones(n)
    chosen_index = np.random.choice(a=n, size=int(n/2), replace=False)
    y[chosen_index] = -1
    y = np.asarray(y, dtype=int)
    # creating edge_index
    edge_index = [[], []]
    for i in range(n - 1):
        for j in range(i + 1, n):
            if y[i] * y[j] > 0:
                Flip = np.random.binomial(1, c_in / n)
            else:
                Flip = np.random.binomial(1, c_out / n)
            if Flip > 0.5:
                edge_index[0].append(i)
                edge_index[1].append(j)
                edge_index[0].append(j)
                edge_index[1].append(i)

    edge_index = torch.tensor(edge_index, dtype=torch.long)
    # creating node features
    x = np.zeros([n, p])
    np.random.seed(seed)
    u = np.random.normal(0, 1 / np.sqrt(p), [1, p])
    # Refactor constant, balcance information from features and graph structure.
    m = 4.5
    for i in range(n):
        Z = np.random.normal(0, 1, [1, p])
        x[i] = np.sqrt(mu / n /m) * y[i] * u + Z / np.sqrt(p)

    data = Data(x=torch.tensor(x, dtype=torch.float32),
                edge_index=edge_index,
                y=torch.tensor((y + 1) // 2, dtype=torch.int64))

    # order edge list and remove duplicates if any.
    data.coalesce()

    # add parameters to attribute
    data.Lambda = Lambda
    data.mu = mu
    data.n = n
    data.p = p
    data.d = d
    return data


def parameterized_Lambda_and_mu(theta, p, n, epsilon=0.1):
    '''
    based on claim 3 in the paper, 

        lambda^2 + mu^2/gamma = 1 + epsilon.

    1/gamma = p/n
    longer axis: 1
    shorter axis: 1/gamma.
    =>
        lambda = sqrt(1 + epsilon) * sin(theta * pi / 2)
        mu = sqrt(gamma * (1 + epsilon)) * cos(theta * pi / 2)
    '''
    from math import pi
    gamma = n / p
    assert (theta >= -1) and (theta <= 1)
    Lambda = np.sqrt(1 + epsilon) * np.sin(theta * pi / 2)
    mu = np.sqrt(gamma * (1 + epsilon)) * np.cos(theta * pi / 2)
    return Lambda, mu


def save_data_to_pickle(data, p2root='../data/', file_name=None):
    '''
    if file name not specified, use time stamp.
    '''
    now = datetime.now()
    surfix = now.strftime('%b_%d_%Y-%H:%M')
    if file_name is None:
        tmp_data_name = '_'.join(['cSBM_data', surfix])
    else:
        tmp_data_name = file_name
    p2cSBM_data = osp.join(p2root, tmp_data_name)
    if not osp.isdir(p2root):
        os.makedirs(p2root)
    with open(p2cSBM_data, 'bw') as f:
        pickle.dump(data, f)
    return p2cSBM_data


class dataset_ContextualSBM(InMemoryDataset):
    r"""Create synthetic dataset based on the contextual SBM from the paper:
    https://arxiv.org/pdf/1807.09596.pdf

    Use the similar class as InMemoryDataset, but not requiring the root folder.

       See `here <https://pytorch-geometric.readthedocs.io/en/latest/notes/
    create_dataset.html#creating-in-memory-datasets>`__ for the accompanying
    tutorial.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset if not specified use time stamp.

        for {n, d, p, Lambda, mu}, with '_' as prefix: intial/feed in argument.
        without '_' as prefix: loaded from data information

        n: number nodes
        d: avg degree of nodes
        p: dimenstion of feature vector.

        Lambda, mu: parameters balancing the mixture of information, 
                    if not specified, use parameterized method to generate.

        epsilon, theta: gap between boundary and chosen ellipsoid. theta is 
                        angle of between the selected parameter and x-axis.
                        choosen between [0, 1] => 0 = 0, 1 = pi/2

        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    def __init__(self, root, name=None,
                 d=5, p=1000, Lambda=None, mu=None,
                 epsilon=0.1, theta_train=0.5, theta_test=0.5,
                 train_graphs=[1, 0],
                 val_graphs=[1, 0],
                 test_graphs=[1, 0],
                 train_nodes=3000,
                 val_nodes=1000,
                 test_nodes=1000,
                 split='train',
                 transform=None, pre_transform=None):

        now = datetime.now()
        surfix = now.strftime('%b_%d_%Y-%H:%M')
        if name is None:
            # not specifing the dataset name, create one with time stamp.
            self.name = '_'.join(['cSBM_data', surfix])
        else:
            self.name = name

        self._n_graphs = train_graphs + val_graphs + test_graphs
        self._n_nodes = train_nodes
        self._d = d
        self._p = p

        self._Lambda = Lambda
        self._mu = mu
        self._epsilon = epsilon
        self._theta_train = theta_train
        self._theta_test = theta_test

        self._train_graphs = train_graphs
        self._val_graphs = val_graphs
        self._test_graphs = test_graphs
        self._train_nodes = train_nodes
        self._val_nodes = val_nodes
        self._test_nodes = test_nodes

        root = osp.join(root, self.name)
        if not osp.isdir(root):
            os.makedirs(root)
        super(dataset_ContextualSBM, self).__init__(
            root, transform, pre_transform)
        if split == 'train':
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif split == 'val':
            self.data, self.slices = torch.load(self.processed_paths[1])
        elif split == 'test':
            self.data, self.slices = torch.load(self.processed_paths[2])
        # overwrite the dataset attribute n, p, d, Lambda, mu
        self.Lambda = self.data.Lambda[0].item()
        self.mu = self.data.mu[0].item()
        self.n = self.data.n[0].item()
        self.p = self.data.p[0].item()
        self.d = self.data.d[0].item()

    @property
    def raw_file_names(self):
        file_names = [self.name]
        return file_names

    @property
    def processed_file_names(self):
        return ['data_train.pt', 'data_val.pt', 'data_test.pt']

    def download(self):
        pass

    def process(self):
        n_total = sum(self._train_graphs) * self._train_nodes + \
                  sum(self._val_graphs) * self._val_nodes + \
                  sum(self._test_graphs) * self._test_nodes
        seed = 12321
        for i, phase in enumerate([('train', self._train_graphs, self._train_nodes),
                                   ('val', self._val_graphs, self._val_nodes),
                                   ('test', self._test_graphs, self._test_nodes)]):
            phase_name = phase[0]
            num_graphs_1, num_graphs_2 = phase[1]
            num_nodes = phase[2]
            data_list = []
            print("Generating " + str(num_graphs_1 + num_graphs_2) + " " + phase_name + " graphs...")
            for g in tqdm(range(num_graphs_1 + num_graphs_2)):
                name = self.raw_file_names[0]
                p2f = osp.join(self.raw_dir, name)
                if not osp.isfile(p2f):
                    if phase_name == "train" and g < num_graphs_1:
                        self._Lambda, self._mu = parameterized_Lambda_and_mu(self._theta_train,
                                                                             self._p,
                                                                             n_total,
                                                                             self._epsilon)
                    else:
                        self._Lambda, self._mu = parameterized_Lambda_and_mu(self._theta_test,
                                                                             self._p,
                                                                             n_total,
                                                                             self._epsilon)
                    tmp_data = ContextualSBM(num_nodes, n_total, self._d, self._Lambda, self._p, self._mu,
                                                    phase_name, self.name, seed)
                    data_list.append(tmp_data)
            torch.save(self.collate(data_list), self.processed_paths[i])

    def __repr__(self):
        return '{}()'.format(self.name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phi_train', type=float, default=0.5)
    parser.add_argument('--phi_test', type=float, default=-0.5)
    parser.add_argument('--epsilon', type=float, default=3.25)
    parser.add_argument('--root', default='../data/')
    parser.add_argument('--name', default='csbm1')
    parser.add_argument('--train_nodes', type=int, default=3000)
    parser.add_argument('--val_nodes', type=int, default=3000)
    parser.add_argument('--test_nodes', type=int, default=3000)
    parser.add_argument('--num_features', type=int, default=2000)
    parser.add_argument('--avg_degree', type=float, default=5)
    parser.add_argument('--train_graphs_1', type=int, default=1)
    parser.add_argument('--train_graphs_2', type=int, default=0)
    parser.add_argument('--val_graphs_1', type=int, default=1)
    parser.add_argument('--val_graphs_2', type=int, default=0)
    parser.add_argument('--test_graphs_1', type=int, default=1)
    parser.add_argument('--test_graphs_2', type=int, default=0)

    args = parser.parse_args()

    dataset_ContextualSBM(root=args.root,
                          name=args.name,
                          theta_train=args.phi_train,
                          theta_test=args.phi_test,
                          epsilon=args.epsilon,
                          train_nodes=args.train_nodes,
                          val_nodes=args.val_nodes,
                          test_nodes=args.test_nodes,
                          d=args.avg_degree,
                          p=args.num_features,
                          train_graphs=[args.train_graphs_1, args.train_graphs_2],
                          val_graphs=[args.val_graphs_1, args.val_graphs_2],
                          test_graphs=[args.test_graphs_1, args.test_graphs_2],
                          split='train')

