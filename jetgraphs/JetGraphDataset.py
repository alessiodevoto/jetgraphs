import os
import os.path as osp
from tqdm import tqdm
import tarfile
import re

import torch
import networkx as nx
from torch_geometric.data import InMemoryDataset, download_url, extract_zip, Data
from torch_geometric.utils.convert import from_networkx
from .transforms import connected_components


"""
VERSION 1.
This version build a datasets from a directory containing gml files. This is deprecated in favor
of version 2.
"""

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
        print("This class will be soon removed in favor of JetGraphDatasetInMemory_v2")
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

"""
VERSION 2.
While version 1 was used to build a dataset from preprocessed gml files, version 2 builds graphs 
from raw data directly, allowing for a deeper control on the graph building pipeline.
"""
import os
import os.path as osp
from tqdm import tqdm
import tarfile
import re

import torch
import networkx as nx
from torch_geometric.data import InMemoryDataset, download_url, extract_zip, Data
from torch_geometric.utils.convert import from_networkx

import deepdish as dd
import glob

TOTAL_GRAPHS = 100000 
GRAPHS_IN_SIGNAL_SUBDIR = 50000
GRAPHS_IN_NOISE_SUBDIR = 25000

def percentage(part, whole):
  return float(whole) * float(part) / 100

def is_noise_subdir(subdir):
    return "background" in subdir.lower()

class JetGraphDatasetInMemory_v2(InMemoryDataset):
    """
    Dataset containing samples of jetgraphs. 
    The downloaded dataset will be built without graph edges. To build graph edges, see
    jetgraphs.transforms.BuildEdges.
    :parameter url: a String containing the url to the dataset.
    :parameter root: where to download (or look for) the dataset.
    :parameter subset: a String defining the percentage of dataset to be used. e.g. '10.5%'.
    :parameter min_num_nodes: minimum number of nodes that a graph must have not to be excluded.
    :parameter verbose: whether to display more info while processing graphs.
    """
    def __init__(
                self, 
                url, 
                root,
                subset='100%', 
                min_num_nodes=1,
                verbose = False, 
                transform=None, 
                pre_transform=None, 
                pre_filter=None):
        
        self.url = url
        self.subset = subset
        self.min_num_nodes = min_num_nodes
        self.verbose = verbose 
        
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices, dataset_name, subset = torch.load(self.processed_paths[0])

        print(f"Loaded dataset with name {dataset_name}, containing subset of {subset}")
        if subset != self.subset:
            print('This dataset contains a different number of nodes or has different settings. Processing graphs again.')
            self.process(dataset_name=dataset_name)
            self.data, self.slices, dataset_name, subset = torch.load(self.processed_paths[0])


        self.dataset_name = dataset_name
        

    def download(self):
        """
        Download tar file to raw_dir, extract directories to raw_dir and rename files. 
        Finally cleanup all unused files.
        """
        # Download self.url to self.raw_dir.
        download_url(self.url, self.raw_dir)
        os.rename(osp.join(self.raw_dir, 'download'), osp.join(self.raw_dir, 'download.tar'))

        # Extract everything in self.raw_dir.
        tar = tarfile.open(osp.join(self.raw_dir, 'download.tar'))
        tar.extractall(self.raw_dir)
        tar.close()

        # Clean.
        os.remove(osp.join(self.raw_dir, 'download.tar'))

        # Rename files.
        self._rename_filenames()

    
    def _rename_filenames(self):
        """
        Filenames are poorly named and so we have to rename them for easier processing.
        This function just renames all downloaded files to make the processing clear.
        """
        print("Renaming files...")

        # Add the 'a0' prefix to files for layer 0.
        for c in ['s', 'b']:     # signal, background
            # match any .h5 file in self.raw_dir that does not start with 'a0'.
            pattern = os.path.join(self.raw_dir, "*", f"[{c}]*.h5")
            result = glob.glob(pattern)
            for filename in result:
                # add a0 which is missing in all directories
                os.rename(filename, filename.replace(f'{c}6', f'a0_{c}6').replace(f'{c}6', ''))
        
        # Remove useless version number and make file names prettier.
        for c in ['s', 'b']: # signal, background
            # match any .h5 file in self.raw_dir and remove useless version number.
            pattern = os.path.join(self.raw_dir, "*", "*.h5")
            result = glob.glob(pattern)
            for filename in result:
                os.rename(filename, filename.replace(f'{c}6', ''))
        print("Done renaming files!")
    
    def _preprocess(self):
        print("[Preprocessing] Preprocessing data to build dataset of graphs without edges.")
        
        data_list = []
        # Attributes to retrieve for each graph.
        attributes = ["eta", "phi", "energy", "energyabs"] 
    
        # There should be 3 subdirectories: Signal, Background 1 and Background 2.
        
        # Process each subdirectory separately. 
        subdirs = os.listdir(self.raw_dir)
        for subdir in subdirs:
            print(f"[Preprocessing] Reading files in {subdir}...")
            is_noise = is_noise_subdir(subdir) 
            num_graphs = GRAPHS_IN_NOISE_SUBDIR if is_noise else GRAPHS_IN_SIGNAL_SUBDIR

            # Build dictionary of raw data from each subdirectory indpendently.
            dataset_by_layer = {}
            for layer in range(0,4):
                dataset_by_layer[layer] = {}
                for attribute in attributes:
                    filepath = os.path.join(self.raw_dir, subdir, f"a{layer}_tupleGraph_bar{attribute}.h5")
                    if self.verbose:
                        print(f"[Preprocessing] Reading: {filepath}")
                    dataset_by_layer[layer][attribute] = dd.io.load(filepath)

            # Process raw tuples contained in dictionary.
            print("[Preprocessing] Building graphs...")
            for gid in tqdm(range(1, num_graphs)):
                # We are going to put in _nodes a tensor for each layer, of size (num_nodes_in_layer, num_attributes + 1).
                # The +1 stems from the column that points out the node layer.
                _nodes = []
                for layer in range(0,4):
                    # We are going to put in _layer_nodes a tensor for each attribute, of size (num_nodes_in_layer, 1).
                    _layer_nodes = []
                    for attribute in attributes:
                        _layer_nodes.append(torch.tensor(list(dataset_by_layer[layer][attribute][gid].values())))
                    # Add layer id, it must be in position 3 of list to comply with older schema.
                    _layer_nodes.insert(2, torch.ones_like(_layer_nodes[-1])*layer)
                    # Build graph layer attributes, of size (num_nodes_in_layer, num_attributes).
                    layer_nodes = torch.cat(_layer_nodes, dim=-1)
                    _nodes.append(layer_nodes)
                # Stack 4 layers, to get graph of size (num_nodes_in_graph, num_attributes).
                nodes = torch.cat(_nodes, dim=0)
                
                # Filter out nodes based on conditions.
                # So far the only condition is energyabs <= 400.
                invalid_nodes_mask = (nodes[:, -1] <= 400) 
                nodes = nodes[~invalid_nodes_mask]    
                
                # If no nodes are left after deleting unwanted, just skip this graph.
                if nodes.shape[0] == 0:
                    continue
                
                # Finally create and append graph to list.
                graph_class = 0 if is_noise else 1
                # Last column is absolute energy, not useful from now, so we delete it.
                nodes = nodes[:,:-1]
                graph = Data(x=nodes, edge_attr=None, edge_index=None, y=graph_class)
                data_list.append(graph)
                
            print(f"[Preprocessing] Done preprocessing files in {subdir}")

        print("[Preprocessing] Done preprocessing all subdirectories!")
        
        # Save obtained torch tensor for later.
        torch.save(data_list, self.pre_processed_path)
    

    def process(self, dataset_name=None):
        # If raw data was not preprocessed, do it now.
        # This should be done only once after download.
        if not os.path.exists(self.pre_processed_path):
            self._preprocess()
        
        # Dataset name so that we can keep track of amount of nodes in each processed version.
        if dataset_name:
            self.dataset_name = dataset_name
        else:
            self.dataset_name = f"min_nodes_{self.min_num_nodes}" if not dataset_name else dataset_name
            

        # Load preprocessed graphs.
        print("[Processing] Loading preprocessed data...")
        data_list = torch.load(self.pre_processed_path) 
        
        # Work out amount of graphs to process.
        signal_graphs, noise_graphs = GRAPHS_IN_SIGNAL_SUBDIR, GRAPHS_IN_NOISE_SUBDIR
        if self.subset:
            requested_graphs = int(percentage(self.subset[:-1], TOTAL_GRAPHS))
            print(f'[Processing] Selecting {requested_graphs} graphs from the initial {TOTAL_GRAPHS}.')
            noise_graphs = int(percentage(25, requested_graphs))
            signal_graphs = int(percentage(50, requested_graphs))
            print(f'[Processing] Graphs will be {noise_graphs} from Background 0, {noise_graphs} from Background 1, and {signal_graphs} from Signal subdir.')
        
        # Build (possibly reduced) list of graphs.
        signal_subset = data_list[0:signal_graphs]
        noise_subset_0 = data_list[GRAPHS_IN_SIGNAL_SUBDIR:GRAPHS_IN_SIGNAL_SUBDIR+noise_graphs]
        noise_subset_1 = data_list[-noise_graphs:]
        processed_data_list = signal_subset + noise_subset_0 + noise_subset_1

        # Apply transforms and filter.
        if self.pre_filter is not None:
            print("[Processing] Filtering out unwanted graphs...")  
            processed_data_list = [data for data in tqdm(processed_data_list) if data.x.shape[0]>=self.min_num_nodes]

        if self.pre_transform is not None:
            print("[Processing] Applying pre transform...")
            processed_data_list = [self.pre_transform(data) for data in tqdm(processed_data_list)]

        # Save obtained torch tensor.
        data, slices = self.collate(processed_data_list)
        torch.save((data, slices, self.dataset_name, self.subset), self.processed_paths[0])

    # AUXILIARY FUNCTIONS
    def stats(self):
        print(f'\n*** JetGraph Dataset version:{self.dataset_name} ***\n')
        print(f'Number of classes: {self.num_classes}')
        print(f'Number of graphs: {len(self)}')
        print(f'Dataset is directed: {self.is_directed}')
        print(f'Number of node features: {self.num_node_features}')
        print(f'Number of edge features: {self.num_edge_features}')
        print(f'Number of positive samples:{self.num_positive_samples:.2f}' )
        
    
    # PROPERTIES
    @property
    def is_directed(self):
        return self.get(0).is_directed()

    @property
    def num_node_features(self) -> int:
        return self.get(0).x.size(1)

    @property
    def num_edge_features(self) -> int:
        sample = self.get(0).edge_attr
        return sample.size(1) if sample is not None else None

    @property
    def num_positive_samples(self):
        return sum([x.y.item() for x in self])
    
    @property
    def raw_file_names(self):
        return ['Background2_v6', 'Background3_v6', 'Signal_v6']

    @property
    def processed_file_names(self):
        return ['jet_graph_processed.pt']
    
    @property
    def pre_processed_path(self):
        return os.path.join(self.raw_dir, 'preprocessed_no_edges.list')