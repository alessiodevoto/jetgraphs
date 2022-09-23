import re
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.utils import to_networkx
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_scipy_sparse_matrix
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D

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


class ConnectedComponents(BaseTransform):
    """
    Compute connected components of graph g as a Transform.
    :parameter g : graph to use as Data object.
    :parameter return_subgraphs : whether to return a list of Data subgraphs.
    """
    def __init__(self, return_subgraphs=False, directed=False):
        self.directed = directed
        self.return_subgraphs = return_subgraphs

    def __call__(self, data: Data) -> Data:
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
    Compute connected components of graph g as a Transform.
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

def plot_jet_graph(g, node_distance=0.3, display_energy_as='colors', ax=None, figsize=(5, 5), elev=30, angle=0):
    """
    Display graph g, assuming 4 attributes per node and optimal distance between node and node size.
    TODO complete this comment
    :parameter g: graph to plot Data object
    :parameter node_distance :
    :parameter diplay_energy_as : how the energy should be displayed options are ['colors', 'size', 'colors_and_size']
    :parameter ax : matplotlib axis
    """
    plt.style.use('ggplot')
    # The graph to visualize
    G = to_networkx(g, node_attrs=['x'])

    # Remove last coordinate, i.e. energy, and store it in other attribute field
    for node_idx in G.nodes():
        G.nodes[node_idx]['energy'] = G.nodes[node_idx]['x'][-1]
        G.nodes[node_idx]['x'] = G.nodes[node_idx]['x'][0:3]

    # 3d spring layout
    pos = nx.spring_layout(G, k=node_distance, dim=3, seed=779)
    # Extract node and edge positions from the layout
    node_xyz = np.array([pos[v] for v in sorted(G)])
    edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])

    energies = np.array([G.nodes[node_idx]['energy'] for node_idx in G.nodes])

    # Create the 3D figure
    if not ax:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    # Plot the nodes - alpha is scaled by "depth" automatically
    if display_energy_as == 'colors':
        ax.scatter(*node_xyz.T, c=energies, s=50, ec="w")
    elif display_energy_as == 'size':
        ax.scatter(*node_xyz.T, s=energies / np.linalg.norm(energies) * 1000, ec="w")
    elif display_energy_as == 'colors_and_size':
        ax.scatter(*node_xyz.T, c=energies, s=energies / np.linalg.norm(energies) * 1000, ec="w")
    else:
        ax.scatter(*node_xyz.T, s=50, ec="w")

    # Plot the edges
    for vizedge in edge_xyz:
        ax.plot(*vizedge.T, color="tab:gray")

    def _format_axes(ax):
        # Turn gridlines off
        ax.grid(False)
        # Suppress tick labels
        for dim in (ax.xaxis, ax.yaxis):
            dim.set_ticks([])
        ax.zaxis.set_ticks([1,2,3,4])
        # Set axes labels
        ax.set_xlabel("η")
        ax.set_ylabel("φ")
        ax.set_zlabel("l")

    _format_axes(ax)

    # Set the initial view
    ax.view_init(elev, angle)

    plt.savefig(f'/Users/alessiodevoto/projects/graph/{angle}.png')
    plt.close('all')
    


def network_plot(g, angle, save=False, elev=10, s=50):

    n = g.x.shape[0]
    print(n)

    # 3D network plot
    with plt.style.context(('ggplot')):
        
        fig = plt.figure(figsize=(10,7))
        ax = Axes3D(fig)
        
        # Loop on the pos dictionary to extract the x,y,z coordinates of each node
        for idx in range(n):
            xi = g.x[idx, 0]
            yi = g.x[idx, 1]
            zi = g.x[idx, 2]
            
            # Scatter plot
            ax.scatter(xi, yi, zi, s=s, edgecolors='k', alpha=0.7)
        
        # Loop on the list of edges to get the x,y,z, coordinates of the connected nodes
        # Those two points are the extrema of the line to be plotted
        for i in range(g.edge_index.shape[1]):
        
            src = g.edge_index[0, i]
            dst = g.edge_index[1, i]

            x = np.array((g.x[src, 0], g.x[dst, 0]))
            y = np.array((g.x[src, 1], g.x[dst, 1]))
            z = np.array((g.x[src, 2], g.x[dst, 2]))
        
            # Plot the connecting lines
            ax.plot(x, y, z, c='black', alpha=0.5)
    
    # Set the initial view
    ax.view_init(elev, angle)

    #plt.savefig(f'/Users/alessiodevoto/projects/graph/{angle}.png')
    #plt.close('all')

    # Hide the axes
    #ax.set_axis_off()
    
    

# Export a Jetgraph dataset to Pandas Dataframe.
def stats_to_pandas(dataset, additional_col_names=[]):
    """
    Export a Jetgraph dataset to Pandas Dataframe. If no additional col_names are provided, 
    then the pandas dataframe will have a row for each graph in dataset,
    with ['y', 'num_nodes', 'num_edges'] columns. 
    
    Returns:
    - a pandas dataframe 
    - a dataset name (str)
    """
    data = []
    col_names = ['y', 'num_nodes', 'num_edges']
    col_names.extend(additional_col_names)

    for elem in dataset:
        g = [elem.y.item(), elem.num_nodes, elem.num_edges]
        g.extend([elem.get(col) for col in additional_col_names])
        data.append(g)

    df = pd.DataFrame(data)
    df.columns = col_names

    return df, dataset.dataset_name
