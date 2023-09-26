import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_scipy_sparse_matrix, to_undirected
import torch
from torch_geometric.data import Data
from torch.nn.functional import one_hot as one_hot_encode


"""
This file contains transforms for jetgraph Data objects. 
"""


def _connected_components(g, return_subgraphs=False, directed=False):
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
    All nodes will have self loops.
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
                distance_p = 2,
                self_loop_weight = 1
                ):
        self.directed = directed
        self.same_layer_threshold = same_layer_threshold 
        self.consecutive_layer_threshold = consecutive_layer_threshold
        self.self_loops = True # so far we only support building graphs with self loops
        self.distance_p = distance_p
        self.self_loop_weight = self_loop_weight

    def __call__(self, data: Data) -> Data:
        nodes = data.x
        # Compute distances for each pair of nodes, using eta and phi as coordinates.
        distances = torch.cdist(abs(nodes[:,:2]), abs(nodes[:,:2]), p=self.distance_p) # (num_nodes, num_nodes), distances[i][j] = distance between node i and node j
        
        # Initialize array with all possible *directed* edges, going from
        # lower to higher layers of graph (0 -> 1 -> 2 -> 3).
        nodes_idx = torch.arange(nodes.shape[0])
        edges = torch.combinations(nodes_idx, with_replacement=self.self_loops)
        
        # Remark: edges is a list of tuples, in which the first element of each tuple
        # is smaller than the second. This way, when considering a pair of nodes,
        # we always consider first the edge going from lower to higher indexed nodes.
        # The above property ensures that edge direction is from lower to higher levels.

        # Loop over all directed edges and filter out not compliant with parameters.
        valid_edges = []
        edge_attributes = []

        for edge in edges:
            src = edge[0].item()
            dst = edge[1].item()
            
            # Check if nodes are in same or consecutive layer.
            layer_gap = nodes[src][2] - nodes[dst][2]
            same_layer = layer_gap == 0
            consecutive_layer = layer_gap == -1

            # Consider distance between src and dst and add if below threshold.
            edge_len = distances[src][dst]
            
            if src == dst: 
                # self loop
                valid_edges.append(edge)
                edge_attributes.append(self.self_loop_weight)
            elif edge_len <= self.same_layer_threshold and same_layer:
                valid_edges.append(edge)
                edge_attributes.append(edge_len)
            elif edge_len <= self.consecutive_layer_threshold and consecutive_layer:
                valid_edges.append(edge)
                edge_attributes.append(edge_len)
        
        # In case we end up with no edges we just return the initial nodes.
        if len(valid_edges) == 0:
            return data

        # Create tensors for storing edges and edge attributes.
        valid_edges = torch.stack(valid_edges)
        valid_edges = valid_edges.permute(1,0)
        edge_attributes = torch.tensor(edge_attributes).unsqueeze(1)

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
    :parameter return_subgraphs : whether to return a list of Data subgraphs.
    :return a Data object with the new field num_subgraphs, that equals the number of subgraphs 
    in graph. 
    If return subgraphs is True, the returned data object will also have another field "subgrpahs"
    containing a list of the subgraphs.
    """
    def __init__(self, return_subgraphs=False, directed=False):
        self.directed = directed
        self.return_subgraphs = return_subgraphs

    def __call__(self, data: Data):
        ret_value = _connected_components(data, self.return_subgraphs, self.directed)
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
        _, _subgraphs = _connected_components(data, return_subgraphs=True, directed=self.directed)
        subgraphs = sorted(_subgraphs, key=lambda g : g.x.shape[0])[:-1]
        return subgraphs

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'


class NumLayers(BaseTransform):
    """
    Compute number of layers in a jet graph.
    :return a Data object with the field "num_layers" containing how 
    many unique layers exist in the original graph. 
    """
    def __call__(self, data: Data) -> Data:
        data.num_layers = data.x[:, 2].unique().size()[0]
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'


class LayersNum(BaseTransform):
    """
    Compute layers' numbers in a jet graph.
    :return a Data object with the field "num_layers" containing the list of 
    unique layers existing in the original graph. 
    """
    def __call__(self, data: Data) -> Data:
        layers_num = data.x[:, 2].int().unique()
        data.layers_num = str(layers_num.tolist())
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'


class OneHotEncodeLayer(BaseTransform):
  """
  One hot encode the third attribute of each node, which is a integer from 0 to 3.
  """
  def __call__(self, data):
    one_hot_encoding = one_hot_encode(data.x[:,2].to(torch.int64), num_classes=4)
    data.x = torch.cat([data.x[:,:2], one_hot_encoding, data.x[:,-1].unsqueeze(1)], dim=-1)
    return data


class OneHotDecodeLayer(BaseTransform):
  """
  One hot encode the third attribute of each node, which is a integer from 0 to 3.
  """
  def __call__(self, data):
    decoded_layer = torch.argmax(data.x[:, 2:6], dim=1)
    data.x = torch.cat([data.x[:,:2], decoded_layer.view(-1, 1), data.x[:,-1].unsqueeze(1)], dim=-1)
    return data


class GraphFilter(BaseTransform):
    """
    A filter that returns True if graph has at least min_num_nodes nodes.
    """
    def __init__(self, min_num_nodes : int):
        self.min_num_nodes = min_num_nodes

    def __call__(self, data: Data):
        if data.x.shape[0]>=self.min_num_nodes:
            return True
        return False

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.min_num_nodes})'