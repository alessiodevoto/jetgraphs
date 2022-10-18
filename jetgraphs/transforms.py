from dis import dis
import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_scipy_sparse_matrix, to_undirected
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

class BuildEdges(BaseTransform):
    """
    Add edges to a graph without any edges, by connecting nodes according to following parameters.
    :parameter directed:  whether the graph should be directed or not.
    :parameter same_layer_threshold: nodes in same layer with distance <= same_layer_threshold will be connected
    :parameter consecutive_layer_threshold: nodes in same layer with distance <= consecutive_layer_threshold will be connected
    :parameter distance_p : p value for computing p-norm, i.e. the distance among nodes. p=2 is Euclides, p=1 Manhattan
    """
    def __init__(
                self, 
                directed=True, 
                same_layer_threshold = 0.6, 
                consecutive_layer_threshold = 0.6,
                self_loops = False,
                distance_p = 2
                ):
        self.directed = directed
        self.same_layer_threshold = same_layer_threshold 
        self.consecutive_layer_threshold = consecutive_layer_threshold
        self.self_loops = self_loops
        self.distance_p = distance_p

    def __call__(self, data: Data) -> Data:
        nodes = data.x
        # Compute distances for each pair of nodes, using eta and phi as coordinates.
        distances = torch.cdist(nodes[:,:2], nodes[:,:2], p=self.distance_p) # (num_nodes, num_nodes), distances[i][j] = distance between node i and node j
        
        # Initialize array with all possible *directed* edges, going from
        # lower to higher layers of graph (0 -> 1 -> 2 -> 3).
        nodes_idx = torch.arange(nodes.shape[0])
        edges = torch.combinations(nodes_idx, with_replacement=self.self_loops) 

        # Loop over all directed edges and filter out not compliant with parameters.
        valid_edges = []
        edge_attributes = []

        for edge in edges:
            src = edge[0].item()
            dst = edge[1].item()
            
            # Check if nodes are in same or consecutive layer.
            layer_gap = nodes[src][2] - nodes[dst][2]
            same_layer = layer_gap == 0
            consecutive_layer = layer_gap == 1

            # Consider distance between src and dst and add if below threshold.
            edge_len = distances[src][dst]
            if edge_len <= self.same_layer_threshold and same_layer:
                valid_edges.append(edge)
                edge_attributes.append(edge_len)
            elif edge_len <= self.consecutive_layer_threshold and consecutive_layer:
                valid_edges.append(edge)
                edge_attributes.append(edge_len)
        
        if len(valid_edges) == 0:
            return data

        valid_edges = torch.stack(valid_edges)
        valid_edges = valid_edges.permute(1,0)
        edge_attributes = torch.tensor(edge_attributes)

        # If requested, make graph undirected.
        if not self.directed:
            valid_edges, edge_attributes = to_undirected(valid_edges, edge_attributes)

        # Update Data object.
        data.edge_attr = edge_attributes
        data.edge_index = valid_edges

        return data
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__} directed_{self.directed}, same_{self.same_layer_threshold}, consecutive_{self.consecutive_layer_threshold}'


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

class LayersNum(BaseTransform):
    """
    Compute layers' numbers in a jet graph.
    """
    def __call__(self, data: Data) -> Data:
        data.layers_num = data.x[:, 2].int().unique()
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'