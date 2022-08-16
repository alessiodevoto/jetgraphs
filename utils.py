import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.utils import to_networkx
from torch_geometric.utils import to_scipy_sparse_matrix
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

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

def plot_jet_graph(g, node_distance=0.3, display_energy_as='colors', ax=None, figsize=(5, 5)):
    """
    Display graph g, assuming 4 attributes per node and optimal distance between node and node size.
    TODO complete this comment
    :parameter g: graph to plot Data object
    :parameter node_distance :
    :parameter diplay_energy_as : options are []
    :parameter ax : matplotlib axis
    """

    # The graph to visualize
    G = to_networkx(g, node_attrs=['x'])

    # Remove last coordinate, i.e. energy,  and store it in other attribute field
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
        for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
            dim.set_ticks([])
        # Set axes labels
        ax.set_xlabel("η")
        ax.set_ylabel("φ")
        ax.set_zlabel("l")

    _format_axes(ax)
