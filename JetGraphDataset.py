import os
import os.path as osp
import random
from tqdm import tqdm

import torch
import networkx as nx
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.utils.convert import from_networkx


class JetGraphDatasetInMemory(InMemoryDataset):
  """
  Dataset containing samples of jetgraphs. 
  """

  def __init__(self, url, root, subset=False, transform=None, pre_transform=None, pre_filter=None):
    self.url = url
    self.subset = subset
    super().__init__(root, transform, pre_transform, pre_filter)
    
    self.data, self.slices = torch.load(self.processed_paths[0])


  def download(self):
    # Download to `self.raw_dir`.
    url = self.url
    download_url(url, self.raw_dir)
    # Due to zip file structure, extraction will be in directory self.raw_dir/raw_dir
    extract_zip(osp.join(self.raw_dir, 'download'), self.raw_dir)
    

  def process(self):

    subdirs = [x for x in os.listdir(self.raw_dir) if osp.isdir(osp.join(self.raw_dir,x))]
    if len(subdirs) > 1:
      raise RuntimeError(f'More than one subdirectories have been found, but just one is needed: {subdir}')


    old_subdir = subdirs[0]
    subdir = 'jetgraphs'
    self.dataset_name = old_subdir.split('/')[-1]
    os.rename(osp.join(self.raw_dir, old_subdir), osp.join(self.raw_dir, subdir))
    
    

    raw_dir = osp.join(self.raw_dir, subdir)
    if not osp.exists(raw_dir):
      raise FileNotFoundError(f'{raw_dir} not found. Maybe there was an inconsistency between different versions of the same dataset. Make sure the directory tree is correctly organized and delete it if necessary.')
    filenames = [osp.join(raw_dir, f) for f in os.listdir(raw_dir) if f.startswith('jet') and f.endswith('.gml')]

    # Select subset of filenames, if necessary.
    if self.subset:
      initial_num_graphs = len(filenames)
      num_graphs = int((int(self.subset[:-1]) / 100 ) *  initial_num_graphs)
      print(f'Extracting {num_graphs} graphs from the initial {initial_num_graphs}.')
      # First 50% is signal, second 50% is noise
      ignore = initial_num_graphs - num_graphs
      filenames = filenames[ignore//2 : -ignore//2]
      print(f'Extracted {len(filenames)} graphs.')
    
    # Read all graphs into data list, converting one by one (they should fit in memory).
    graphs_without_edges, signal_graphs_without_edges = 0, 0
    graphs_without_nodes, signal_graphs_without_nodes  = 0, 0

    data_list = []
    for graph_file in tqdm(filenames):
      # Read graph into networkx object.
      new_graph = nx.read_gml(graph_file)
      new_graph = nx.MultiGraph(new_graph)
      
      # Check that graph structure makes sense, if not, exclude it.
      if len(new_graph.edges) > 0 and len(new_graph.nodes) > 0:
        data = from_networkx(new_graph, group_node_attrs = all, group_edge_attrs = all)
        data.y = int(new_graph.graph['y'])
        data_list.append(data)
      elif len(new_graph.edges) == 0:
        graphs_without_edges += 1
        signal_graphs_without_edges += int(new_graph.graph['y'])
      elif len(new_graph.nodes) == 0:
        graphs_without_nodes += 1
        signal_graphs_without_nodes += int(new_graph.graph['y'])
    
    
    print(f'Filtered out {graphs_without_nodes} graphs without nodes, of which {signal_graphs_without_nodes} were signal.')
    print(f'Filtered out {graphs_without_edges} graphs without edges, of which {signal_graphs_without_edges} were signal.')
    print(f'Processing finished!')

    if self.pre_filter is not None:
        data_list = [data for data in data_list if self.pre_filter(data)]

    if self.pre_transform is not None:
        data_list = [self.pre_transform(data) for data in data_list]

    # Save obtained torch tensor for later use.
    data, slices = self.collate(data_list)
    torch.save((data, slices), self.processed_paths[0])

  
  # AUXILIARY FUNCTIONS
  def stats(self):
    print('\n*** JetGraph Dataset ***\n')
    print(f'Number of classes: {self.num_classes}')
    print(f'Number of graphs: {len(self)}')
    print(f'Dataset is undirected: {self.is_undirected}')
    print(f'Number of node features: {self.num_features}')
    print(f'Number of edge features: {self.num_edge_features}')
    
    print(f'Average number of nodes per graph: {self.avg_nodes_per_graph:.2f}')
    print(f'Average number of edges per graph: {self.avg_edges_per_graph:.2f}')
    print(f'Average number of layers per graph: {self.avg_layers_per_graph:.2f}')
    
  
  # PROPERTIES
  @property
  def is_undirected(self):
    # for now it is undirected and we know it
    return True
    # return all(g.is_undirected() for g in self)
  

  @property
  def is_directed(self):
    # for now it is undirected and we know it
    return False
    # return all(g.is_directed() for g in self)
  

  @property
  def avg_nodes_per_graph(self):
    total_nodes = 0
    for g in self:
        total_nodes += g.num_nodes
    avg_assets_per_graph = total_nodes / self.len()
    return avg_assets_per_graph


  @property
  def avg_edges_per_graph(self):
    total_edges = 0
    for g in self:
        total_edges += g.num_edges
    avg_edges_per_graph = total_edges / self.len()
    return avg_edges_per_graph
  

  @property
  def avg_layers_per_graph(self):
    def num_layers(g):
      return g.x[:,2].unique().size()[0]
    total_layers = 0
    for g in self:
        total_layers += num_layers(g)
    avg_edges_per_graph = total_layers / self.len()
    return avg_edges_per_graph


  @property
  def num_node_features(self) -> int:
    return self.get(0).x.size(1)


  @property
  def num_edge_features(self) -> int:
    sample = self.get(0).edge_attr
    return sample.size(1) if sample is not None else None
  

  @property
  def raw_file_names(self):
    return ['jetgraphs']


  @property
  def processed_file_names(self):
    return ['jet_graph_processed_.pt']

    