import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_scipy_sparse_matrix
import torch
from torch_geometric.data import Data


def connected_components(g, return_subgraphs=False, directed=False):
    """
    Compute connected components of graph g.
    :parameter g : graph to use as Data object.
    :parameter return_subgraphs : whether to return a list of Data subgraphs.
    """
    adj = to_scipy_sparse_matrix(g.edge_index, num_nodes=g.num_nodes)
    
    if return_subgraphs:
        num_components, component = sp.csgraph.connected_components(adj, directed=directed, return_labels=True)
        _, count = np.unique(component, return_counts=True)
        subgraphs = []
        for subset_idx in count.argsort(): 
            subset = np.in1d(component, subset_idx)
            subgraph = g.subgraph(torch.from_numpy(subset).to(torch.bool))
            subgraphs.append(subgraph)
        return num_components, subgraphs
    
    num_components, component = sp.csgraph.connected_components(adj, directed=directed)
    return num_components


class NumberOfSubgraphs(BaseTransform):
    """
    Compute connected components of graph g as a Transform.
    :parameter g : graph to use as Data object.
    :parameter return_subgraphs : whether to return a list of Data subgraphs.
    """
    def __init__(self, return_subgraphs=False, directed=False):
        self.directed = directed
        self.return_subgraphs = return_subgraphs

    def __call__(self, data: Data):
        ret_value = connected_components(data, self.return_subgraphs, self.directed)
        if self.return_subgraphs:
            data.num_subgraphs = ret_value[0]
            data.subgraphs = ret_value[1]
        else:
            data.num_subgraphs = ret_value
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.return_subgraphs})'

class ConnectedComponentsRemoveLargest(BaseTransform):
    """
    Compute connected components of graph g as a Transform, removing the largest one.
    :parameter g : graph to use as Data object.
    Returns a list of Data objects for each graph. Each data object is a connected 
    component of the initial graph. The largest connected component 
    is not included in the returned list.
    """
    def __init__(self, directed=False):
        self.directed = directed
        
    def __call__(self, data: Data) -> list:
        _, _subgraphs = connected_components(data, return_subgraphs=True, directed=self.directed)
        subgraphs = sorted(_subgraphs, key=lambda g : g.x.shape[0])[:-1]
        return subgraphs

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'


class NumLayers(BaseTransform):
    """
    Compute number of layers in a jet graph.
    """
    def __call__(self, data: Data) -> Data:
        data.num_layers = data.x[:, 2].unique().size()[0]
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'