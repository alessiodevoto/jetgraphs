import networkx as nx
import numpy as np
from torch_geometric.utils import to_networkx
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D


def plot_jet_graph(g, node_distance=0.3, display_energy_as='colors', ax=None, figsize=(5, 5), elev=30, angle=0):
    """
    Display graph g, assuming 4 attributes per node and optimal distance between node and node size.
    TODO complete this comment
    :parameter g: graph to plot Data object
    :parameter node_distance :
    :parameter diplay_energy_as : how the energy should be displayed options are ['colors', 'size', 'colors_and_size']
    :parameter ax : matplotlib axis
    """

    print("WARNING: this function will be removed soon")

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
        #ax.grid(False)
        # Suppress tick labels
        #for dim in (ax.xaxis, ax.yaxis):
            #dim.set_ticks([])
        ax.zaxis.set_ticks([1,2,3,4])
        # Set axes labels
        ax.set_xlabel("η")
        ax.set_ylabel("φ")
        ax.set_zlabel("l")

    _format_axes(ax)

    # Set the initial view
    ax.view_init(elev, angle)

    #plt.savefig(f'/Users/alessiodevoto/projects/graph/{angle}.png')
    #plt.close('all')


    

def plot_jet_graph2(g, save=False, angle=30, elev=10, ax=None, color_layers=True, energy_is_size=True, figsize=(5,5), save_to_path=False, **kwargs):
    """
    Display graph g, assuming 4 attributes (eta, phi, layer, energy) per node and optimal distance between node and node size.
    :parameter g: Data object containing graph to plot.
    :parameter elev stores the elevation angle in the z plane. 
    :parameter angle stores the azimuth angle in the x,y plane.
    :parameter color_layers whether to color nodes on different layers with different colors
    :parameter energy_is_size: whether to make nodes with higher energy bigger
    :parameter ax : matplotlib axis
    """

    def layers_colormap(idx):
        if int(idx) > 4:
            raise AttributeError(f'Only 4 colors available for layer. {idx} is out of bound.')
        colors = ['r', 'b', 'g', 'c']
        return colors[int(idx)]

    if energy_is_size and g.x.shape[1] < 4:
        raise AttributeError(f'Cannot plot energy as size of nodes if provided graph has only {g.x.shape[1]} attributes. Energy should be the fourth attribute.')

    num_nodes = g.x.shape[0]

    # 3D network plot
    with plt.style.context(('ggplot')):
        
        # Create the 3D figure
        if not ax:
            fig = plt.figure(figsize=figsize)
            ax = ax = Axes3D(fig)
        else:
            fig = ax.get_figure()
        
        # Loop on the adjacency matrix to extract the x,y,z coordinates of each node 
        for idx in range(num_nodes):
            xi = g.x[idx, 0]
            yi = g.x[idx, 1]
            zi = g.x[idx, 2]
            ci = g.x[idx, 2]            # layer is represented as color
            ei = g.x[idx, 3] * 500     # energy is represented as size
            
            # Scatter plot
            size = ei.item() if energy_is_size else matplotlib.rcParams['lines.markersize'] ** 2
            color = layers_colormap(ci.item()) if color_layers else 'b'
            ax.scatter(xi, yi, zi, color=color, s=size, edgecolors='k', alpha=0.7, **kwargs)
        
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

    # Set axis labels
    ax.set_xlabel("η")
    ax.set_ylabel("φ")
    ax.set_zlabel("l")

    # Save or display right away
    if save_to_path is not False:
        plt.savefig(save_to_path)
        plt.close('all')
    else:
        plt.show()
    
    

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
