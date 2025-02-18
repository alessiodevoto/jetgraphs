import os, shutil
import os.path as osp
from tqdm import tqdm
import tarfile
import glob
import torch
from torch_geometric.data import InMemoryDataset, download_url, Data
import re
import deepdish as dd


def _repr(obj) -> str:
    if obj is None:
        return 'None'
    return re.sub('(<.*?)\\s.*(>)', r'\1\2', obj.__repr__())


TOTAL_GRAPHS = 199867 
GRAPHS_IN_SIGNAL_SUBDIR =   50000
GRAPHS_IN_NOISE_SUBDIR =    25000

def percentage(part, whole):
  return float(whole) * float(part) / 100

def is_noise_subdir(subdir):
    return "background" in subdir.lower()

class DarkPhotonDataset(InMemoryDataset):
    """
    Dataset containing samples of jetgraphs. 
    The downloaded dataset will be built without graph edges. To build graph edges, see
    jetgraphs.transforms.BuildEdges.
    :parameter url: a String containing the url to the dataset.
    :parameter root: where to download (or look for) the dataset.
    :parameter subset: a String defining the percentage of dataset to be used. e.g. '10.5%'.
    :parameter verbose: whether to display more info while processing graphs.
    """
    def __init__(
                self, 
                root,
                url="https://cernbox.cern.ch/s/PYurUUzcNdXEGpz/download", 
                subset='100%', 
                verbose = False, 
                transform = None, 
                pre_transform = None, 
                pre_filter = None,
                post_filter = None,
                remove_download = False
                ):
        
        self.url = url
        self.subset = subset
        self.verbose = verbose 
        self.post_filter = post_filter
        self.remove_download = remove_download
        
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices, subset, post_filter = torch.load(self.processed_paths[0], weights_only=False)
        
        print(f"Loaded dataset containing subset of {subset}")
        if subset != self.subset or post_filter != _repr(self.post_filter):
            print('The loaded dataset has different settings from the ones requested. Processing graphs again.')
            self.process()
            self.data, self.slices, subset, post_filter = torch.load(self.processed_paths[0], weights_only=False)
            
 

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
        if self.remove_download:
            print("Removing compressed download...")
            os.remove(osp.join(self.raw_dir, 'download.tar'))

        # Rename files.
        self._rename_filenames()

    
    def _rename_filenames(self):
        """
        Filenames are poorly named and so we have to rename them for easier processing.
        This function just renames all downloaded files to make the processing clear.
        """
        
        print("Moving Directories to raw directory...")
        pattern = os.path.join(self.raw_dir, "**", "*_v6")
        data_dirs = glob.glob(pattern, recursive=True) # should return Signal_v6, Background2_v6, Background3_v6 
        for subdir in data_dirs:
            shutil.move(subdir, self.raw_dir)
        

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

        if not os.path.exists(os.path.join(self.raw_dir, "Signal_v6")):
            print("Probably something went wrong during download. Trying to rename files again...")
            self._rename_filenames()
        
        data_list = []
        # Attributes to retrieve for each graph.
        attributes = ["eta", "phi", "energy", "energyabs"] 
    
        # There should be 3 subdirectories: Signal, Background 1 and Background 2.
        
        # Process each subdirectory separately. 
        pattern = os.path.join(self.raw_dir, "*_v6")
        data_dirs = glob.glob(pattern)
        print(data_dirs)
        for subdir in data_dirs:
            print(f"[Preprocessing] Reading files in {subdir}...")
            is_noise = is_noise_subdir(subdir) 
            num_graphs = GRAPHS_IN_NOISE_SUBDIR if is_noise else GRAPHS_IN_SIGNAL_SUBDIR

            # Build dictionary of raw data from each subdirectory indpendently.
            dataset_by_layer = {}
            for layer in range(0,4):
                dataset_by_layer[layer] = {}
                for attribute in attributes:
                    filepath = os.path.join(subdir, f"a{layer}_tupleGraph_bar{attribute}.h5")
                    if self.verbose:
                        print(f"[Preprocessing] Reading: {filepath}")
                    dataset_by_layer[layer][attribute] = dd.io.load(filepath)

            # Process raw tuples contained in dictionary.
            print("[Preprocessing] Building graphs...")
            for gid in tqdm(range(1, num_graphs)):
                # We are going to put in _nodes a tensor for each layer, of size (num_nodes_in_layer, num_attributes + 1).
                # The +1 stems from the column that points out the node layer.
                _nodes = []
                # _nodes2 = []
                for layer in range(0,4):
                    # We are going to put in _layer_nodes a tensor for each attribute, of size (num_nodes_in_layer, 1).
                    _layer_nodes = []
                    # _layer_nodes2 = []
                    for attribute in attributes:
                        #if attribute != "CNNscores":
                        _layer_nodes.append(torch.tensor(list(dataset_by_layer[layer][attribute][gid].values())))
                        #else:
                        #    _layer_nodes2.append(torch.tensor(list(dataset_by_layer[layer][attribute][gid])))
                    # Add layer id, it must be in position 3 of list to comply with older schema.
                    _layer_nodes.insert(2, torch.ones_like(_layer_nodes[-1])*layer)
                    # Build graph layer attributes, of size (num_nodes_in_layer, num_attributes).
                    # layer_nodes = []
                    # for i, node in enumerate(_layer_nodes):
                    #     if i == 2:
                    #         new_node = torch.ones_like(_layer_nodes[-1]) * layer
                    #         layer_nodes.append(new_node)
                    #     layer_nodes.append(node)

                    layer_nodes = torch.cat(_layer_nodes, dim=-1)
                    # layer_nodes2 = torch.cat(_layer_nodes2, dim=-1)
                    _nodes.append(layer_nodes)
                    # _nodes2.append(layer_nodes2)
                    #layer_nodes = torch.cat(_layer_nodes, dim=-1)
                    #_nodes.append(layer_nodes)
                # Stack 4 layers, to get graph of size (num_nodes_in_graph, num_attributes).
                nodes = torch.cat(_nodes, dim=0)
                # nodes2 = torch.cat(_nodes2, dim=0)
                
                # Filter out nodes based on conditions.
                # So far the only condition is energyabs <= 400.
                invalid_nodes_mask = (nodes[:, -1] <= 400) 
                nodes = nodes[~invalid_nodes_mask] 
                # nodes2 = nodes2[~invalid_nodes_mask] 

                
                # If no nodes are left after deleting unwanted, just skip this graph.
                if nodes.shape[0] == 0:
                    continue
                
                # Finally create and append graph to list.
                graph_class = 0 if is_noise else 1
                # Attach CNN scores
                #CNNscores = nodes2[0].item()
                # Last column is absolute energy, not useful from now, so we delete it.
                nodes = nodes[:,:-1]
                graph = Data(x=nodes, edge_attr=None, edge_index=None, y=graph_class) #, CNNscores = CNNscores)
                data_list.append(graph)
                
            print(f"[Preprocessing] Done preprocessing files in {subdir}")

        print("[Preprocessing] Done preprocessing all subdirectories!")
        
        # Save obtained torch tensor for later.
        torch.save(data_list, self.pre_processed_path)
    

    def process(self):
        # If raw data was not preprocessed, do it now.
        # This should be done only once after download.
        if not os.path.exists(self.pre_processed_path):
            self._preprocess()
        
        # Load preprocessed graphs.
        print("[Processing] Loading preprocessed data...")
        data_list = torch.load(self.pre_processed_path, weights_only=False) 
        
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

        # self.pre_filter = lambda x : data.x.shape[0] >= self.min_num_nodes
        # print(f"[Processing] Filtering out graphs with less than {self.min_num_nodes} nodes...")  
        # processed_data_list = [data for data in tqdm(processed_data_list) if data.x.shape[0]>=self.min_num_nodes]

        if self.pre_filter is not None:
            print("[Processing] Pre-filtering out unwanted graphs...")  
            processed_data_list = [data for data in tqdm(processed_data_list) if self.pre_filter(data)]

        if self.pre_transform is not None:
            print("[Processing] Applying pre transform...")
            processed_data_list = [self.pre_transform(data) for data in tqdm(processed_data_list)]
        
        if self.post_filter is not None:
            print("[Processing] Post-filtering out unwanted graphs...")  
            processed_data_list = [data for data in tqdm(processed_data_list) if self.post_filter(data)]

        # Save obtained torch tensor.
        data, slices = self.collate(processed_data_list)
        torch.save((data, slices, self.subset, _repr(self.post_filter)), self.processed_paths[0])

    # AUXILIARY FUNCTIONS
    def stats(self):
        print(f'\n*** JetGraph Dataset ***\n')
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