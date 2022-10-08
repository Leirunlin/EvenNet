from models import *
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix, homophily_ratio
import seaborn as sns
from deeprobust.graph.utils import get_train_val_test


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask


def mask_to_index(mask, size):
    all_idx = np.arange(size)
    return all_idx[mask.cpu().numpy()]


def random_planetoid_splits(data, num_classes, percls_trn=20, val_lb=500, seed=12134):
    index = [i for i in range(0, data.y.shape[0])]
    train_idx = []
    rnd_state = np.random.RandomState(seed)
    for c in range(num_classes):
        class_idx = np.where(data.y.cpu() == c)[0]
        if len(class_idx) < percls_trn:
            train_idx.extend(class_idx)
        else:
            train_idx.extend(rnd_state.choice(class_idx, percls_trn, replace=False))
    rest_index = [i for i in index if i not in train_idx]
    val_idx = rnd_state.choice(rest_index, val_lb, replace=False)
    test_idx = [i for i in rest_index if i not in val_idx]

    data.train_mask = index_to_mask(train_idx, size=data.num_nodes)
    data.val_mask = index_to_mask(val_idx, size=data.num_nodes)
    data.test_mask = index_to_mask(test_idx, size=data.num_nodes)
    return data


def cal_uncertainty(values):
    return np.max(np.abs(sns.utils.ci(sns.algorithms.bootstrap(values,
                                                                       func=np.mean,
                                                                       n_boot=1000), 95) - values.mean()))

