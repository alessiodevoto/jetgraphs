import networkx as nx
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import numpy 
from pandas import DataFrame
from collections.abc import Iterable
import seaborn as sns
import os
import re


"""
Here we have some utility functions to plot a single graph and analyse a jetgraph dataset. 
Please use functions with signature ending in "v2" as much as possible.
"""


def plot_jet_graph(g, angle=30, elev=10, ax=None, color_layers=True, energy_is_size=True, figsize=(5,5), save_to_path=False, **kwargs):
    """
    Display graph g, assuming 4 node features (eta, phi, layer, energy) per node and optimal distance between node and node size.
    If g has 7 node features, then the function assumes the layer was one hot encoded and transforms node features from 2 to 6 
    into labels. 
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
    

    if g.x.shape[1] == 7:
      from .transforms import OneHotDecodeLayer
      g = OneHotDecodeLayer()(g)
    
    assert g.x.shape[1] == 4, "The provided graph must have either 7 or 4 node features."


    num_nodes = g.x.shape[0]

    # 3D network plot
    with plt.style.context(('ggplot')):
        
        # Create the 3D figure
        if not ax:
            fig = plt.figure(figsize=figsize)
            ax = Axes3D(fig)
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
    ax.zaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xticks(rotation=-45)
    plt.yticks(rotation=45)

    fig.subplots_adjust(wspace=0)
    # Save or display right away
    if save_to_path is not False:
        plt.savefig(save_to_path)
        plt.close('all')
    
    
    
def stats_to_pandas(dataset : Iterable, additional_col_names=[]):
    """
    Export an iterable of graphs to a Pandas Dataframe. If no additional col_names are provided, 
    then the pandas dataframe will have a row for each graph in dataset,
    with ['y', 'num_nodes', 'num_edges'] columns. 
    
    Returns:
    - a pandas dataframe 
    - a dataset name (str)
    """
    data = []
    col_names = ['y', 'num_nodes', 'num_edges']
    col_names.extend(additional_col_names)

    # Horrible:
    inferred_col_names = [x for x in dataset[0].__dict__['_store'].keys() if x.startswith('num') and x not in col_names]
    col_names.extend(inferred_col_names)

    for elem in dataset:
        g = [elem.y.item(), elem.num_nodes, elem.num_edges]
        g.extend([elem.get(col) for col in additional_col_names])
        g.extend([elem.get(col) for col in inferred_col_names])
        data.append(g)

    df = pd.DataFrame(data)
    df.columns = col_names

    # Rename "y" to "class".
    df.rename(columns={'y': 'class'}, inplace=True)

    return df


def plot_dataset_info(df: DataFrame, title: str, include_cols : Iterable = False, exclude_cols: Iterable = False, separate_classes: bool = False, save_to_path=False, format='pdf'):
  """
  Print statistical info about Pandas dataframe of graphs.
  - df : a pandas dataframe where each row is a graph and each column a property of that graph
  - title : name to give to plot
  - include_cols : cols to be included in plots
  - exclude_cols : cols to be excluded from plots
  - separate_classes : whether to make different plots for class = 1 and class = 0
  - save_to_path : path where to save image. If False, image will just be displayed.
  """
  plt.style.use('ggplot') 
  # Select list of columns to plot.
  df_cols = list(df.columns)
  if include_cols and exclude_cols:
    raise ValueError('Yuo can either specify columns to include or to exclude, not both.')
  if include_cols:
    cols_to_plot = [col for col in df_cols if col in include_cols]
  elif exclude_cols:
    cols_to_plot = [col for col in df_cols if col not in exclude_cols]
  else:
    cols_to_plot = df_cols

  # Prepare plots structure.
  print(f"Creating plots for columns: {cols_to_plot} from dataset with columns: {df_cols})")
  num_plots = len(cols_to_plot) 
  if not separate_classes:
    num_plots += 1      # +1 for correlation matrix
  fig, axs = plt.subplots(num_plots, figsize= (6, num_plots*5)) 
  fig.subplots_adjust(hspace =.5, wspace=.5)
  # Set title.
  title = f'{title} (NOISE vs SIGNAL)' if separate_classes else f'{title} (ALL)' 
  fig.suptitle(title, fontsize=16)
  # Just in case we are plotting only one column.
  if not isinstance(axs, numpy.ndarray):
    axs = [axs]

  # Distributions of column fields.
  for i, col in enumerate(cols_to_plot): 
    axs[i].set_xlabel(col)
    axs[i].set_ylabel('# graphs')

    if separate_classes:
      df0 = df.groupby('class')[col].value_counts().unstack(0).sort_index()
      df0 = df0.rename(columns={0:'noise', 1:'signal'})
      x_ticks = None if len(df0.index) < 100 else np.arange(0, df0.index[-1], 30)
      df0.plot.bar(ax=axs[i], xticks=x_ticks, color={'signal':'tab:orange', 'noise':'tab:blue'})
      axs[i].legend()
    else:
      df0 = df[col].value_counts().sort_index()
      x_ticks = None if len(df0.index) < 100 else np.arange(0, df0.index[-1], 30)
      df0.plot.bar(ax=axs[i], xticks=x_ticks, color='c')
      

  # Correlation matrix.
  if not separate_classes:
    df_corr = df.corr()
    mask = np.triu(np.ones_like(df_corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(df_corr, mask=mask, cmap=cmap, vmax=.3, center=0,
              square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .5}, ax=axs[i+1])
  
  # Save or display right away
  if save_to_path is not False:
      plt.savefig(os.path.join(save_to_path, title+'.'+format), dpi=300)
      plt.close('all')
  else:
      plt.show()

def _repr(obj) -> str:
    if obj is None:
        return 'None'
    return re.sub('(<.*?)\\s.*(>)', r'\1\2', obj.__repr__())