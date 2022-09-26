import os
import os.path as osp
import random
from tqdm import tqdm
import warnings
import re

import torch
import networkx as nx
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.utils.convert import from_networkx
from jetgraphs.utils import connected_components

def extract_dataset_name(ugly_name: str):
    return re.findall('[0-9].[0-9]dR[0-9].[0-9]', ugly_name.split('/')[-1])[0]

class JetGraphDatasetInMemory(InMemoryDataset):
    """
    Dataset containing samples of jetgraphs.
    :parameter url: a String containing the url to the dataset.
    :parameter root: where to download (or look for) the dataset.
    :parameter subset: a String defining the percentage of dataset to be used. e.g. '10.5%'.
    :parameter min_num_nodes: minimum number of nodes that a graph must have not to be excluded.
    """

    def __init__(self, url, root, min_num_nodes=1, subset=False, transform=None, pre_transform=None, pre_filter=None):
        self.url = url
        self.subset = subset
        self.min_num_nodes = min_num_nodes 
        super().__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices, dataset_name, subset = torch.load(self.processed_paths[0])
       
        if subset != self.subset:
            print('A preprocessed version exits, but contains a different number of nodes. Processing graphs again.')
            self.process(dataset_name=dataset_name)

        self.dataset_name = dataset_name
        

    def download(self):
        # Download to `self.raw_dir`.
        url = self.url
        download_url(url, self.raw_dir)
        extract_zip(osp.join(self.raw_dir, 'download'), self.raw_dir)

    def process(self, dataset_name=None):

        # Just a bit messy with all dirs/subdirs names.
        """# TODO we should find a clean way to extract this, regardless of the confused organization.
        subdirs = [x for x in os.listdir(self.raw_dir) if osp.isdir(osp.join(self.raw_dir, x))]
        if len(subdirs) > 1:
            raise RuntimeError(f'More than one subdirectories found, but just one is needed: {subdirs}')

        old_subdir = subdirs[0]
        subdir = 'jetgraph_files'
        self.dataset_name = old_subdir.split('/')[-1]
        os.rename(osp.join(self.raw_dir, old_subdir), osp.join(self.raw_dir, subdir))

        raw_dir = osp.join(self.raw_dir, subdir)
        if not osp.exists(raw_dir):
            raise FileNotFoundError(
                f'{raw_dir} not found. Maybe there was an inconsistency between different versions of the same dataset. Make sure the directory tree is correctly organized and delete it if necessary.')
        filenames = [osp.join(raw_dir, f) for f in os.listdir(raw_dir) if f.startswith('jet') and f.endswith('.gml')]"""
        
        
        # The directory tree is a bit messy, so we have to 'walk' to locate the right subdirectory.
        gen = os.walk(self.raw_dir)
        t = next(gen)
        while(len(t[-1]) < 2 or not t[-1][0].endswith('gml')):
            t = next(gen)
        
        # Found correct subdir, move it to self.raw_dir/jetgraph_files
        old_subdir = t[0]
        self.dataset_name = extract_dataset_name(old_subdir) if not dataset_name else dataset_name
        os.rename(old_subdir, osp.join(self.raw_dir,'jetgraph_files'))

        jetgraph_files = osp.join(self.raw_dir, 'jetgraph_files')
        if not osp.exists(jetgraph_files):
            raise FileNotFoundError(
                f'{jetgraph_files} not found. Maybe there was an inconsistency between different versions of the same dataset. Make sure the directory tree is correctly organized and delete it if necessary.')
        filenames = [osp.join(jetgraph_files, f) for f in os.listdir(jetgraph_files) if f.startswith('jet') and f.endswith('.gml')]


        # Select subset of filenames, if necessary.
        if self.subset:
            initial_num_graphs = len(filenames)
            num_graphs = int((float(self.subset[:-1]) / 100) * initial_num_graphs)
            print(f'Selecting {num_graphs} graphs from the initial {initial_num_graphs}.')
            # First 50% is signal, second 50% is noise, so said Joe.
            ignore = initial_num_graphs - num_graphs
            filenames = filenames[ignore // 2: -ignore // 2]
            print(f'Selected {len(filenames)} graphs.')

        # Read all graphs into data list, converting one by one (they should fit in memory).
        graphs_without_edges, signal_graphs_without_edges = 0, 0
        graphs_without_nodes, signal_graphs_without_nodes = 0, 0

        data_list = []
        for graph_file in tqdm(filenames):
            # Read graph into networkx object.
            new_graph = nx.read_gml(graph_file)
            new_graph = nx.MultiGraph(new_graph)

            # Check that graph structure makes sense, if not, exclude it.
            if len(new_graph.edges) > 0 and len(new_graph.nodes) >= self.min_num_nodes:
                data = from_networkx(new_graph, group_node_attrs=all, group_edge_attrs=all)
                data.y = int(new_graph.graph['y'])
                data_list.append(data)
            if len(new_graph.edges) == 0:
                graphs_without_edges += 1
                signal_graphs_without_edges += int(new_graph.graph['y'])
            if len(new_graph.nodes) < self.min_num_nodes:
                graphs_without_nodes += 1
                signal_graphs_without_nodes += int(new_graph.graph['y'])
            
            
        print(
            f'Filtered out {graphs_without_nodes} graphs with less than {self.min_num_nodes} nodes, of which {signal_graphs_without_nodes} were signal.')
        print(
            f'Filtered out {graphs_without_edges} graphs without edges, of which {signal_graphs_without_edges} were signal.')
        print(f'Processing finished!')

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Save obtained torch tensor for later use.
        data, slices = self.collate(data_list)
        torch.save((data, slices, self.dataset_name, self.subset), self.processed_paths[0])

    # AUXILIARY FUNCTIONS
    def stats(self):
        print(f'\n*** JetGraph Dataset version:{self.dataset_name} ***\n')

        # Static features, should be the same for all JetGraph datasets.
        print(f'Number of classes: {self.num_classes}')
        print(f'Number of graphs: {len(self)}')
        print(f'Dataset is undirected: {self.is_undirected}')
        print(f'Number of node features: {self.num_features}')
        print(f'Number of edge features: {self.num_edge_features}')

        # Dynamic features, may change from version to version.
        print(f'Number of positive samples:{self.num_positive_samples:.2f}' )
        m, s = self.edges_per_graph
        print(f'Nodes per graph -> mean: {m:.2f} ,  std: {s:.2f}')
        m, s = self.nodes_per_graph
        print(f'Edges per graph -> mean: {m:.2f},  std: {s:.2f}')
        m, s = self.layers_per_graph
        print(f'Layers per graph -> mean: {m:.2f}, std: {s:.2f}')
        m,s = self.subgraphs_stats
        print(f'Subgraphs per graph-> mean: {m:.2f},  std: {s:.2f}')
        
        # TODO Add plotting options.

    # PROPERTIES
    @property
    def is_undirected(self):
        # For now it is undirected and we know it.
        return True
        # return all(g.is_undirected() for g in self)

    @property
    def is_directed(self):
        # For now it is undirected and we know it.
        return False
        # return all(g.is_directed() for g in self)

    @property
    def nodes_per_graph(self):
        nodes = torch.tensor([g.num_nodes for g in self]).float()
        return nodes.mean(), nodes.std()

    @property
    def edges_per_graph(self):
        edges = torch.tensor([g.num_edges for g in self]).float()
        return edges.mean(), edges.std()

    @property
    def layers_per_graph(self):
        def num_layers(g):
            return g.x[:, 2].unique().size()[0]
        layers = torch.tensor([num_layers(g) for g in self]).float()
        return layers.mean(), layers.std()

    @property
    def num_node_features(self) -> int:
        return self.get(0).x.size(1)

    @property
    def num_edge_features(self) -> int:
        sample = self.get(0).edge_attr
        return sample.size(1) if sample is not None else None

    @property
    def raw_file_names(self):
        return ['jetgraph_files']

    @property
    def processed_file_names(self):
        return ['jet_graph_processed_.pt']

    @property
    def num_positive_samples(self):
        return sum([x.y.item() for x in self])
    
    @property
    def subgraphs_stats(self):
        # average number and standard deviation of subgraphs per graph. 
        subgraphs = torch.tensor([connected_components(g) for g in self]).float()
        return subgraphs.mean(), subgraphs.std()
