import numpy as np
import torch

from torch_geometric import datasets
from torch_geometric.datasets import Planetoid
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.transforms import NormalizeFeatures, ToUndirected, Compose
from torch_geometric.utils import to_undirected


def get_dataset(root, name, transform=NormalizeFeatures()):
    pyg_dataset_dict = {
        'Cora': (Planetoid, 'Cora'),
        'Citeseer': (Planetoid, 'Citeseer'),
        'Pubmed': (Planetoid, 'Pubmed'),
        'CS': (datasets.Coauthor, 'CS'),
        'Physics': (datasets.Coauthor, 'physics'),
        'Computers': (datasets.Amazon, 'Computers'),
        'Photos': (datasets.Amazon, 'Photo'),
        'arxiv': (PygNodePropPredDataset, 'ogbn-arxiv'),
        'mag': (PygNodePropPredDataset, 'ogbn-mag'),
        'products': (PygNodePropPredDataset, 'ogbn-products')
    }

    assert name in pyg_dataset_dict, "Dataset must be in {}".format(list(pyg_dataset_dict.keys()))

    dataset_class, name = pyg_dataset_dict[name]
    if name in ['ogbn-arxiv']:
        dataset = dataset_class(name=name, 
                                root=root, 
                                transform=Compose([
                                    ToUndirected(),
                                    # NormalizeFeatures()
                                ]))
    elif name in ['ogbn-products']:
        dataset = dataset_class(name=name, 
                                root=root,)
    elif name in ['ogbn-mag']:
        dataset = dataset_class(name=name, root=root,
                                transform=Compose([
                                    ToUndirected(),
                                    NormalizeFeatures()
                                ]))
        rel_data = dataset[0]
        # We are only interested in paper <-> paper relations.
        data = Data(
                x=rel_data.x_dict['paper'],
                edge_index=rel_data.edge_index_dict[('paper', 'cites', 'paper')],
                y=rel_data.y_dict['paper'])
        data = transform(data)
        dataset.data = data
    elif name in ['Cora', 'Citeseer', 'Pubmed']:
        dataset = dataset_class(root, name=name, transform=Compose([
            ToUndirected()
        ]))
    else:
        dataset = dataset_class(root, name=name, transform=transform)
    return dataset        


def get_wiki_cs(root, transform=NormalizeFeatures()):
    dataset = datasets.WikiCS(root, transform=transform)
    data = dataset[0]
    std, mean = torch.std_mean(data.x, dim=0, unbiased=False)
    data.x = (data.x - mean) / std
    data.edge_index = to_undirected(data.edge_index)
    return [data], dataset.num_classes


class ConcatDataset(InMemoryDataset):
    r"""
    PyG Dataset class for merging multiple Dataset objects into one.
    """
    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        self.__indices__ = None
        self.__data_list__ = []
        for dataset in datasets:
            self.__data_list__.extend(list(dataset))
        self.data, self.slices = self.collate(self.__data_list__)