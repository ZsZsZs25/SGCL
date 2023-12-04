import os
import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_sparse import SparseTensor

def set_random_seeds(random_seed=0):
    r"""Sets the seed for generating random numbers."""
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def load_mask(dataset, mask_dir='./mask'):
    r"""Load preset train/val/test mask."""
    path = "{}/{}.pth".format(mask_dir, dataset)
    return torch.load(path)


def create_mask(dataset, dataset_name='WikiCS', data_seed=0, mask_path=None):
    r"""Create train/val/test mask for each dataset."""
    data = dataset[0]
    if dataset_name == 'WikiCS':
        train_mask = data.train_mask.t()
        val_mask = data.val_mask.t()
        test_mask = data.test_mask.repeat(20, 1)
    elif dataset_name in ['Computers', 'Photos', 'CS', 'Physics']:
        idx = np.arange(len(data.y))
    
        train_mask = torch.zeros((20, data.y.size(0)), dtype=torch.bool)
        val_mask = torch.zeros((20, data.y.size(0)), dtype=torch.bool)
        test_mask = torch.zeros((20, data.y.size(0)), dtype=torch.bool)

        for i in range(20):
            train_idx, test_idx = train_test_split(idx, test_size=0.8, random_state=data_seed + i)
            train_idx, val_idx = train_test_split(train_idx, test_size=0.5, random_state=data_seed + i)

            train_mask[i,train_idx] = True
            val_mask[i, val_idx] = True
            test_mask[i, test_idx] = True
    elif dataset_name in ['Cora', 'Citeseer', 'Pubmed']:
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask
    elif dataset_name in ['mag']:
        split_idx = dataset.get_idx_split()
        train_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
        val_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
        test_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
        train_mask[split_idx['train']['paper']] = True
        val_mask[split_idx['valid']['paper']] = True
        test_mask[split_idx['test']['paper']] = True        
    else:
        split_idx = dataset.get_idx_split()
        train_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
        val_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
        test_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
        train_mask[split_idx['train']] = True
        val_mask[split_idx['valid']] = True
        test_mask[split_idx['test']] = True


    # save preset mask
    if mask_path is not None:
        torch.save([train_mask, val_mask, test_mask], mask_path)

    return train_mask, val_mask, test_mask
    
def edgeidx2sparse(edge_index, num_nodes):
    return SparseTensor.from_edge_index(
        edge_index, sparse_sizes=(num_nodes, num_nodes)
    ).to(edge_index.device)
