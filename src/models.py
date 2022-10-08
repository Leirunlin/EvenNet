import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch.distributions.multivariate_normal import MultivariateNormal
from torch_sparse import SparseTensor, matmul
from torch_scatter.scatter import scatter_add

import scipy.sparse as sp
from scipy.sparse import csr_matrix, lil_matrix
from torch_geometric.nn import GATConv, GCNConv, JumpingKnowledge
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import to_scipy_sparse_matrix
from torch.nn.parameter import Parameter
from propagate import GPR_prop, GPR_prop_reg, Bern_prop, Even_prop, \
    FALayer, GCN_Guard_Conv, GCNIIdenseConv, pGNNConv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from deeprobust.graph.defense.gcn import GraphConvolution as GCNConv_drb
from deeprobust.graph.defense.gcn_preprocess import dropedge_jaccard
from deeprobust.graph.defense.r_gcn import GGCL_D, GGCL_F
from deeprobust.graph.utils import normalize_adj_tensor, is_sparse_tensor


class GPRGNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size, K, alpha, Init, dprate, dropout):
        super(GPRGNN, self).__init__()
        self.lin1 = Linear(in_channels, hidden_size)
        self.lin2 = Linear(hidden_size, out_channels)
        self.prop1 = GPR_prop(K, alpha, Init)

        self.Init = Init
        self.dprate = dprate
        self.dropout = dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)


class EvenNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size, K, alpha, Init, dprate, dropout):
        super(EvenNet, self).__init__()
        self.lin1 = Linear(in_channels, hidden_size)
        self.lin2 = Linear(hidden_size, out_channels)
        self.prop1 = Even_prop(K, alpha, Init)

        self.Init = Init
        self.dprate = dprate
        self.dropout = dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop1.reset_parameters()
        # self.bn.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)


class GPRGNN_reg(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size, K, alpha, Init, dprate, dropout):
        super(GPRGNN_reg
              , self).__init__()
        self.lin1 = Linear(in_channels, hidden_size)
        self.lin2 = Linear(hidden_size, out_channels)
        self.prop1 = GPR_prop_reg(K, alpha, Init)

        self.Init = Init
        self.dprate = dprate
        self.dropout = dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)


class BernNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size, k, dprate, dropout):
        super(BernNet, self).__init__()
        self.lin1 = Linear(in_channels, hidden_size)
        self.lin2 = Linear(hidden_size, out_channels)
        self.prop1 = Bern_prop(k)
        self.dprate = dprate
        self.dropout = dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        return F.log_softmax(x, dim=1)


class FAGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size, eps=0.3, layer_num=2, dropout=0.5):
        super(FAGCN, self).__init__()
        self.eps = eps
        self.layer_num = layer_num
        self.dropout = dropout

        self.layers = nn.ModuleList()
        for i in range(self.layer_num):
            self.layers.append(FALayer(self.dropout, hidden_size))

        self.t1 = nn.Linear(in_channels, hidden_size)
        self.t2 = nn.Linear(hidden_size, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.t1.weight, gain=1.414)
        nn.init.xavier_normal_(self.t2.weight, gain=1.414)

    def forward(self, data):
        h = data.x
        edge_index = data.edge_index
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.relu(self.t1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        raw = h
        for i in range(self.layer_num):
            h = self.layers[i](h, edge_index)
            h = self.eps * raw + h
        h = self.t2(h)
        return F.log_softmax(h, 1)


class H2GCNConv(nn.Module):
    """ Neighborhood aggregation step """

    def __init__(self):
        super(H2GCNConv, self).__init__()

    def reset_parameters(self):
        pass

    def forward(self, x, adj_t, adj_t2):
        x1 = matmul(adj_t, x)
        x2 = matmul(adj_t2, x)
        return torch.cat([x1, x2], dim=1)


class H2GCN(torch.nn.Module):
    """ implementation from
    [*Large Scale Learning on Non-Homophilous Graphs: New Benchmarks and Strong Simple Methods*]
    (https://arxiv.org/abs/2110.14446)
    """

    def __init__(self, in_channels, hidden_size, out_channels, num_nodes,
                 num_layers=2, dropout=0.5, save_mem=False, use_bn=False, conv_dropout=True):
        super(H2GCN, self).__init__()

        self.feature_embed = Linear(in_channels, hidden_size)

        self.convs = nn.ModuleList()
        self.convs.append(H2GCNConv())

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_size * 2 * len(self.convs)))

        for l in range(num_layers - 1):
            self.convs.append(H2GCNConv())
            if l != num_layers - 2:
                self.bns.append(nn.BatchNorm1d(hidden_size * 2 * len(self.convs)))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.conv_dropout = conv_dropout  # dropout neighborhood aggregation steps

        self.jump = JumpingKnowledge('cat')
        last_dim = hidden_size * (2 ** (num_layers + 1) - 1)
        self.final_project = nn.Linear(last_dim, out_channels)

        self.num_nodes = num_nodes
        self.reset_adj = True

    def reset_parameters(self):
        self.feature_embed.reset_parameters()
        self.final_project.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def init_adj(self, edge_index):
        """ cache normalized adjacency and normalized strict two-hop adjacency,
        neither has self loops
        """
        n = self.num_nodes
        if isinstance(edge_index, SparseTensor):
            dev = edge_index.device
            adj_t = edge_index
            adj_t = csr_matrix(adj_t.to_scipy())
            adj_t[adj_t > 0] = 1
            adj_t[adj_t < 0] = 0
            adj_t = SparseTensor.from_scipy(adj_t).to(dev)
        elif isinstance(edge_index, torch.Tensor):
            row, col = edge_index
            adj_t = SparseTensor(row=col, col=row, value=None, sparse_sizes=(n, n))
        adj_t.remove_diag(0)
        adj_t2 = matmul(adj_t, adj_t)
        adj_t2.remove_diag(0)
        adj_t = csr_matrix(adj_t.to_scipy())
        adj_t2 = csr_matrix(adj_t2.to_scipy())
        adj_t2 = adj_t2 - adj_t
        adj_t2[adj_t2 > 0] = 1
        adj_t2[adj_t2 < 0] = 0

        adj_t = SparseTensor.from_scipy(adj_t)
        adj_t2 = SparseTensor.from_scipy(adj_t2)

        adj_t = gcn_norm(adj_t, None, n, add_self_loops=False)
        adj_t2 = gcn_norm(adj_t2, None, n, add_self_loops=False)

        self.adj_t = adj_t.to(edge_index.device)
        self.adj_t2 = adj_t2.to(edge_index.device)

    def forward(self, data):
        if self.reset_adj:
            self.init_adj(data.edge_index)
            self.reset_adj = False

        adj_t = self.adj_t
        adj_t2 = self.adj_t2
        x = data.x
        x = self.feature_embed(x)
        x = self.activation(x)
        xs = [x]
        if self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t, adj_t2)
            if self.use_bn:
                x = self.bns[i](x)
            xs.append(x)
            if self.conv_dropout:
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t, adj_t2)
        if self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        xs.append(x)

        x = self.jump(xs)
        if not self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.final_project(x)
        return F.log_softmax(x, dim=1)


class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size, dropout):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_size)
        self.conv2 = GCNConv(hidden_size, out_channels)
        self.dropout = dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GCN_SVD(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size, k, dropout, device):
        super(GCN_SVD, self).__init__()
        self.conv1 = GCNConv_drb(in_channels, hidden_size)
        self.conv2 = GCNConv_drb(hidden_size, out_channels)
        self.dropout = dropout
        self.device = device
        self.k = k
        self.modified_adj = None
        self.reset_adj = True

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def truncatedSVD(self, data, k=50):
        if sp.issparse(data):
            data = data.asfptype()
            U, S, V = sp.linalg.svds(data, k=k)
            diag_S = np.diag(S)
        else:
            U, S, V = np.linalg.svd(data)
            U = U[:, :k]
            S = S[:k]
            V = V[:k, :]
            diag_S = np.diag(S)
        return U @ diag_S @ V

    def forward(self, data):
        if self.reset_adj:
            rows, cols = data.edge_index.cpu().numpy()
            adj = csr_matrix((np.ones(data.edge_index.shape[1]), (rows, cols)))
            tmp_adj = self.truncatedSVD(adj, self.k)
            modified_adj = torch.from_numpy(tmp_adj)
            if is_sparse_tensor(modified_adj):
                adj_norm = normalize_adj_tensor(modified_adj, sparse=True)
            else:
                adj_norm = normalize_adj_tensor(modified_adj)
            self.modified_adj = adj_norm.type(torch.float32).to(self.device)
            self.reset_adj = False

        x = data.x
        x = F.relu(self.conv1(x, self.modified_adj))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, self.modified_adj)
        return F.log_softmax(x, dim=1)


class GCN_Jaccard(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size, dropout, device):
        super(GCN_Jaccard, self).__init__()
        self.conv1 = GCNConv_drb(in_channels, hidden_size)
        self.conv2 = GCNConv_drb(hidden_size, out_channels)
        self.dropout = dropout
        self.device = device
        self.modified_adj = None
        self.reset_adj = True

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        if self.reset_adj:
            features = data.x.cpu().numpy()
            rows, cols = data.edge_index.cpu().numpy()
            np_adj = csr_matrix((np.ones(data.edge_index.shape[1]), (rows, cols)))
            threshold = 0.01
            if not sp.issparse(np_adj):
                np_adj = sp.csr_matrix(np_adj)
            adj_triu = sp.triu(np_adj, format='csr')
            if sp.issparse(features):
                features = features.todense().A  # make it easier for njit processing
            removed_cnt = dropedge_jaccard(adj_triu.data, adj_triu.indptr, adj_triu.indices, features,
                                           threshold=threshold)
            modified_adj = torch.from_numpy((adj_triu + adj_triu.transpose()).todense())
            if is_sparse_tensor(modified_adj):
                adj_norm = normalize_adj_tensor(modified_adj, sparse=True)
            else:
                adj_norm = normalize_adj_tensor(modified_adj)
            self.modified_adj = adj_norm.type(torch.float32).to(self.device)
            self.reset_adj = False

        x = data.x
        x = F.relu(self.conv1(x, self.modified_adj))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, self.modified_adj)
        return F.log_softmax(x, dim=1)


class GCN_Guard(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size, dropout=0.5, drop=False, n_edge=1, attention=True,
                 with_relu=True, with_bias=True, device=None):
        super(GCN_Guard, self).__init__()
        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = in_channels
        self.hidden_sizes = [hidden_size]
        self.nclass = out_channels
        self.dropout = dropout

        self.with_relu = with_relu
        self.with_bias = with_bias
        self.n_edge = n_edge
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None
        self.gate = Parameter(torch.rand(1)) # creat a generator between [0,1]
        self.test_value = Parameter(torch.rand(1))
        self.drop_learn_1 = Linear(2, 1)
        self.drop_learn_2 = Linear(2, 1)
        self.drop = drop
        self.attention = attention
        self.bn1 = torch.nn.BatchNorm1d(hidden_size)
        self.bn2 = torch.nn.BatchNorm1d(hidden_size)
        self.gc1 = GCN_Guard_Conv(in_channels, hidden_size, bias=True,)
        self.gc2 = GCN_Guard_Conv(hidden_size, out_channels, bias=True, )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        rows, cols = edge_index.cpu().numpy()
        adj_values = torch.ones(rows.shape[0])
        adj = torch.sparse_coo_tensor(indices=torch.tensor([rows, cols]), values=adj_values)

        if self.attention:
            adj = self.att_coef(x, adj, i=0).to(self.device)
        edge_index = adj._indices()
        x = self.gc1(x, edge_index, edge_weight=adj._values())
        x = F.relu(x)
        if self.attention:  # if attention=True, use attention mechanism
            adj_2 = self.att_coef(x, adj, i=1).to(self.device)
            adj_memory = adj_2.to_dense()  # without memory
            row, col = adj_memory.nonzero()[:, 0], adj_memory.nonzero()[:, 1]
            edge_index = torch.stack((row, col), dim=0)
            adj_values = adj_memory[row, col]
        else:
            edge_index = adj._indices()
            adj_values = adj._values()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index, edge_weight=adj_values)
        return F.log_softmax(x, dim=1)

    def initialize(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()
        self.drop_learn_1.reset_parameters()
        self.drop_learn_2.reset_parameters()
        try:
            self.gate.reset_parameters()
            self.fc2.reset_parameters()
        except:
            pass

    def att_coef(self, fea, edge_index, is_lil=False, i=0):
        if is_lil == False:
            edge_index = edge_index._indices()
        else:
            edge_index = edge_index.tocoo()

        n_node = fea.shape[0]
        row, col = edge_index[0].cpu().data.numpy()[:], edge_index[1].cpu().data.numpy()[:]

        fea_copy = fea.cpu().data.numpy()
        sim_matrix = cosine_similarity(X=fea_copy, Y=fea_copy)  # try cosine similarity
        sim = sim_matrix[row, col]
        sim[sim < 0.1] = 0
        """build a attention matrix"""
        att_dense = lil_matrix((n_node, n_node), dtype=np.float32)
        att_dense[row, col] = sim
        if att_dense[0, 0] == 1:
            att_dense = att_dense - sp.diags(att_dense.diagonal(), offsets=0, format="lil")
        # normalization, make the sum of each row is 1
        att_dense_norm = normalize(att_dense, axis=1, norm='l1')

        """add learnable dropout, make character vector"""
        if self.drop:
            character = np.vstack((att_dense_norm[row, col].A1,
                                   att_dense_norm[col, row].A1))
            character = torch.from_numpy(character.T)
            drop_score = self.drop_learn_1(character)
            drop_score = torch.sigmoid(drop_score)  # do not use softmax since we only have one element
            mm = torch.nn.Threshold(0.5, 0)
            drop_score = mm(drop_score)
            mm_2 = torch.nn.Threshold(-0.49, 1)
            drop_score = mm_2(-drop_score)
            drop_decision = drop_score.clone().requires_grad_()
            drop_matrix = lil_matrix((n_node, n_node), dtype=np.float32)
            drop_matrix[row, col] = drop_decision.cpu().data.numpy().squeeze(-1)
            att_dense_norm = att_dense_norm.multiply(drop_matrix.tocsr())  # update, remove the 0 edges

        if att_dense_norm[0, 0] == 0:  # add the weights of self-loop only add self-loop at the first layer
            degree = (att_dense_norm != 0).sum(1).A1
            lam = 1 / (degree + 1)  # degree +1 is to add itself
            self_weight = sp.diags(np.array(lam), offsets=0, format="lil")
            att = att_dense_norm + self_weight  # add the self loop
        else:
            att = att_dense_norm

        row, col = att.nonzero()
        att_adj = np.vstack((row, col))
        att_edge_weight = att[row, col]
        att_edge_weight = np.exp(att_edge_weight)  # exponent, kind of softmax
        att_edge_weight = torch.tensor(np.array(att_edge_weight)[0], dtype=torch.float32)  # .cuda()
        att_adj = torch.tensor(att_adj, dtype=torch.int64)  # .cuda()

        shape = (n_node, n_node)
        new_adj = torch.sparse.FloatTensor(att_adj, att_edge_weight, shape)
        return new_adj

    def add_loop_sparse(self, adj, fill_value=1):
        # make identify sparse tensor
        row = torch.range(0, int(adj.shape[0] - 1), dtype=torch.int64)
        i = torch.stack((row, row), dim=0)
        v = torch.ones(adj.shape[0], dtype=torch.float32)
        shape = adj.shape
        I_n = torch.sparse.FloatTensor(i, v, shape)
        return adj + I_n.to(self.device)


class EGCNGuard(nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels, num_layers=3,
                 dropout=0.5, layer_norm_first=False, use_ln=True, attention_drop=True, threshold=0.3):
        super(EGCNGuard, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_size, add_self_loops=False))
        self.lns = torch.nn.ModuleList()
        self.lns.append(torch.nn.LayerNorm(in_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_size, hidden_size, add_self_loops=False))
            self.lns.append(torch.nn.LayerNorm(hidden_size))
        self.lns.append(torch.nn.LayerNorm(hidden_size))
        self.convs.append(GCNConv(hidden_size, out_channels, add_self_loops=False))

        self.dropout = dropout
        self.layer_norm_first = layer_norm_first
        self.use_ln = use_ln

        self.attention_drop = attention_drop
        self.gate = 0.
        self.prune_edge = True
        self.threshold = threshold

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        row, col = edge_index
        adj_values = torch.ones(row.shape[0]).to(row.device)
        n_total = data.num_nodes
        graph_size = torch.Size((n_total, n_total))
        adj = SparseTensor(row=row, col=col, value=adj_values, sparse_sizes=graph_size)
        if self.layer_norm_first:
            x = self.lns[0](x)
        new_adj = adj
        for i, conv in enumerate(self.convs[:-1]):
            new_adj = self.att_coef(x, new_adj)
            x = conv(x, new_adj)
            if self.use_ln:
                x = self.lns[i + 1](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        new_adj = self.att_coef(x, new_adj)
        x = self.convs[-1](x, new_adj)
        return x.log_softmax(dim=-1)

    def att_coef(self, features, adj):
        with torch.no_grad():
            row, col = adj.coo()[:2]
            n_total = features.size(0)
            if features.size(1) > 512 or row.size(0) > 5e5:
                batch_size = int(1e8 // features.size(1))
                bepoch = row.size(0) // batch_size + (row.size(0) % batch_size > 0)
                sims = []
                for i in range(bepoch):
                    st = i * batch_size
                    ed = min((i + 1) * batch_size, row.size(0))
                    sims.append(F.cosine_similarity(features[row[st:ed]], features[col[st:ed]]))
                sims = torch.cat(sims, dim=0)
            else:
                sims = F.cosine_similarity(features[row], features[col])
            mask = torch.logical_or(sims >= self.threshold, row == col)
            row = row[mask]
            col = col[mask]
            sims = sims[mask]
            has_self_loop = (row == col).sum().item()
            if has_self_loop:
                sims[row == col] = 0

            # normalize sims
            deg = scatter_add(sims, row, dim=0, dim_size=n_total)
            deg_inv_sqrt = deg.pow_(-0.5)
            deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
            sims = deg_inv_sqrt[row] * sims * deg_inv_sqrt[col]

            # add self-loops
            deg_new = scatter_add(torch.ones(sims.size(), device=sims.device), col, dim=0, dim_size=n_total) + 1
            deg_inv_sqrt_new = deg_new.float().pow_(-1.0)
            deg_inv_sqrt_new.masked_fill_(deg_inv_sqrt == float('inf'), 0)

            if has_self_loop == 0:
                new_idx = torch.arange(n_total, device=row.device)
                row = torch.cat((row, new_idx), dim=0)
                col = torch.cat((col, new_idx), dim=0)
                sims = torch.cat((sims, deg_inv_sqrt_new), dim=0)
            elif has_self_loop < n_total:
                print(f"add {n_total - has_self_loop} remaining self-loops")
                new_idx = torch.ones(n_total, device=row.device).bool()
                new_idx[row[row == col]] = False
                new_idx = torch.nonzero(new_idx, as_tuple=True)[0]
                row = torch.cat((row, new_idx), dim=0)
                col = torch.cat((col, new_idx), dim=0)
                sims = torch.cat((sims, deg_inv_sqrt_new[new_idx]), dim=0)
                sims[row == col] = deg_inv_sqrt_new
            else:
                sims[row == col] = deg_inv_sqrt_new
            sims = sims.exp()
            graph_size = torch.Size((n_total, n_total))
            new_adj = SparseTensor(row=row, col=col, value=sims, sparse_sizes=graph_size)
        return new_adj


class RGCN(torch.nn.Module):
    """
    Robust Graph Convolutional Networks Against Adversarial Attacks. KDD 2019.
    Modified from deeprobust.graph.defense.r_gcn
    """
    def _normalize_adj(self, adj, power=-1/2):
        A = adj + torch.eye(len(adj)).to(self.device)
        D_power = (A.sum(1)).pow(power)
        D_power[torch.isinf(D_power)] = 0.
        D_power = torch.diag(D_power)
        return D_power @ A @ D_power

    def loss(self, input, labels):
        loss = F.nll_loss(input, labels)
        miu1 = self.gc1.miu
        sigma1 = self.gc1.sigma
        kl_loss = 0.5 * (miu1.pow(2) + sigma1 - torch.log(1e-8 + sigma1)).mean(1)
        kl_loss = kl_loss.sum()
        norm2 = torch.norm(self.gc1.weight_miu, 2).pow(2) + \
                torch.norm(self.gc1.weight_sigma, 2).pow(2)
        return loss + self.beta1 * kl_loss + self.beta2 * norm2

    def __init__(self, in_channels, out_channels, hidden_size,
                 device, n_nodes, dropout, gamma, beta1, beta2):
        super(RGCN, self).__init__()
        self.gamma = gamma
        self.beta1 = beta1
        self.beta2 = beta2
        self.device = device
        self.nclass = out_channels
        self.n_feats = in_channels
        self.nhid = hidden_size
        self.dropout = dropout
        self.n_nodes = n_nodes
        self.gc1 = GGCL_F(self.n_feats, self.nhid, dropout=self.dropout)
        self.gc2 = GGCL_D(self.nhid, self.nclass, dropout=self.dropout)
        self.gaussian = MultivariateNormal(torch.zeros(self.n_nodes, self.nclass),
                torch.diag_embed(torch.ones(self.n_nodes, self.nclass)))
        self.adj_norm1, self.adj_norm2 = None, None
        self.reset_adj = True

    def reset_parameters(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def forward(self, data):
        features = data.x
        if self.reset_adj:
            adj = torch.from_numpy(to_scipy_sparse_matrix(data.edge_index).todense()).to(self.device)
            self.adj_norm1, self.adj_norm2 = self._normalize_adj(adj, power=-1 / 2), self._normalize_adj(adj, power=-1)
            self.reset_adj = False
        miu, sigma = self.gc1(features, self.adj_norm1, self.adj_norm2, self.gamma)
        miu, sigma = self.gc2(miu, sigma, self.adj_norm1, self.adj_norm2, self.gamma)
        output = miu + self.gaussian.sample().to(self.device) * torch.sqrt(sigma + 1e-8)
        return F.log_softmax(output, dim=1)


class GAT(torch.nn.Module):
    def __init__(self,  in_channels, out_channels, hidden_size, dropout=0.5, heads=2):
        super(GAT, self).__init__()
        self.conv1 = GATConv(
            in_channels,
            hidden_size,
            heads=heads)
        self.conv2 = GATConv(
            hidden_size * heads,
            out_channels,
            heads=1,
            concat=False)
        self.dropout = dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GCNII(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size,
                      num_layers, alpha, lamda,
                      dropout):
        super(GCNII, self).__init__()
        GConv = GCNIIdenseConv
        self.lamda = lamda
        self.alpha = alpha
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.convs.append(torch.nn.Linear(in_channels, hidden_size))
        for _ in range(num_layers):
            self.convs.append(GConv(hidden_size, hidden_size, cached=False))
        self.convs.append(torch.nn.Linear(hidden_size,out_channels))
        self.reg_params = list(self.convs[1:-1].parameters())
        self.non_reg_params = list(self.convs[0:1].parameters())+list(self.convs[-1:].parameters())

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        _hidden = []
        x = F.dropout(x, self.dropout ,training=self.training)
        x = F.relu(self.convs[0](x))
        _hidden.append(x)
        for i,con in enumerate(self.convs[1:-1]):
            x = F.dropout(x, self.dropout ,training=self.training)
            beta = np.log(self.lamda/(i+1)+1)
            x = F.relu(con(x, edge_index,self.alpha, _hidden[0],beta,edge_weight))
        x = F.dropout(x, self.dropout ,training=self.training)
        x = self.convs[-1](x)
        return F.log_softmax(x, dim=1)


class pGNNNet(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_hid=16,
                 mu=0.1,
                 p=2,
                 K=2,
                 dropout=0.5,
                 cached=False):
        super(pGNNNet, self).__init__()
        self.dropout = dropout
        self.lin1 = torch.nn.Linear(in_channels, num_hid)
        self.conv1 = pGNNConv(num_hid, out_channels, mu, p, K, cached=cached)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index, None)
        return F.log_softmax(x, dim=1)


class MLP(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size, dropout):
        super(MLP, self).__init__()
        self.lin1 = Linear(in_channels, hidden_size)
        self.lin2 = Linear(hidden_size, out_channels)
        self.dropout = dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        return F.log_softmax(x, dim=1)
