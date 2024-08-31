from JetGraphDataset import JetGraphDatasetInMemory_v2
from transforms import BuildEdges, OneHotEncodeLayer
from torch_geometric.transforms import Compose, LargestConnectedComponents, RemoveIsolatedNodes
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

from utils import plot_jet_graph, plot_metrics
import numpy as np
from matplotlib import pyplot as plt
import torch

import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-GUI environments
import h5py
from collections import Counter
import matplotlib.pyplot as plt
torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_float32_matmul_precision('high')
# Where data is to be downloaded and stored.
datasets_root = "/user/jcarmignani/jgtorch/jetgraphs_workspace/datasets" 
# Secret url to dataset. 
raw_data_url = "https://cernbox.cern.ch/s/PYurUUzcNdXEGpz/download"

# In the next lines we define settings to build the dataset's edges and graphs.

# As discussed, we switch from 0.6x0.6 thresholds for the edges (Baseline model0) to 0.6x0.3 models 1 & 2.
edge_builder = BuildEdges(
    directed=False,
    #directed=False,#Alessio
    self_loop_weight=0,
    same_layer_threshold=0.6, 
    consecutive_layer_threshold=0.3,#test related threshold
    distance_p=2)


from transforms import GraphFilter 

graph_filter = GraphFilter(min_num_nodes=2) # only graphs with at least 3 nodes will be accepted

from torch_geometric.transforms import Compose 
from transforms import NumLayers, LayersNum

optional_transforms = Compose([NumLayers(), LayersNum()])

custom_transforms = Compose([
    edge_builder,   # build edges same as before
    LargestConnectedComponents() # extract main subgraph
    ])


# Finally download the dataset.
jet_graph_dataset = JetGraphDatasetInMemory_v2(
    root = datasets_root,           # directory where to download data 
    url = raw_data_url,             # url to raw data
    #subset = "100%",                # which subset of the intial 100k graph to consider, default is 100%
    pre_filter = graph_filter,      # filter graphs with less than 3 nodes 
    pre_transform = custom_transforms,    # edge_builder should be passed as pre_transform to keep data on disk.
    post_filter = graph_filter,     # filter graphs with less than 3 nodes
    transform = optional_transforms,
    ) 
# Extract y labels
labels = [data.y[0].item() for data in jet_graph_dataset]

# Count label occurrences
label_counter = Counter(labels)

# Print the distribution
print("Label Distribution:")
for label, count in label_counter.items():
    print(f"Label {label}: {count} occurrences")

# # Optional: Plot the distribution
# plt.bar(label_counter.keys(), label_counter.values())
# plt.xlabel('Labels')
# plt.ylabel('Occurrences')
# plt.title('Label Distribution in jet_graph_dataset')
# plt.show()

# Extract labels for the entire dataset
all_labels = [m.y[0].item() for m in jet_graph_dataset]
# Create the dataloaders.
train_idx, testing_idx = train_test_split(
    range(len(jet_graph_dataset)), 
    stratify=all_labels, 
    test_size=0.4, 
    random_state=43
)
# Extract labels for the testing_idx subset
testing_labels = [m.y[0].item() for m in jet_graph_dataset[testing_idx]]

# Check class distribution in testing_labels
label_counter = Counter(testing_labels)
print("Label distribution in testing_idx:", label_counter)

# Ensure there are more than one class present in testing_labels
if len(label_counter) <= 1:
    raise ValueError("Only one class present in testing_idx after the first split. Adjust your split or check your dataset.")

# Perform the second stratified train-test split
pred_idx, test_idx = train_test_split(
    range(len(jet_graph_dataset[testing_idx])), 
    stratify=testing_labels, 
    test_size=0.5, 
    random_state=43
)

# Check class distribution in pred_idx and test_idx subsets
pred_labels = [testing_labels[i] for i in pred_idx]
test_labels = [testing_labels[i] for i in test_idx]
print("Label distribution in pred_idx:", Counter(pred_labels))
print("Label distribution in test_idx:", Counter(test_labels))


#from JetGraphDataset import stats
jet_graph_dataset.stats()
# Instantiate a model. 
# We take it from jetgraphs.models, but it can be any model for pytorch lightning.
from models import Shallow_GCN
from models import Residual_Arma, GAT, Arma, TGAT

# Instantiate models between 16-512 channels
#model = TGAT(hidden_channels=512, node_feat_size=jet_graph_dataset[0].x.shape[1], use_edge_attr=True)
model = Arma(hidden_channels=256,node_feat_size=jet_graph_dataset[0].x.shape[1], use_edge_attr=True)



print(model)
train_dataset = jet_graph_dataset[train_idx]
testing_dataset = jet_graph_dataset[testing_idx]
test_dataset = testing_dataset[test_idx]
pred_dataset = testing_dataset[pred_idx]
train_loader = DataLoader(train_dataset, batch_size=512, num_workers=0, shuffle=True)#, pin_memory=False
test_loader = DataLoader(test_dataset, batch_size=512, num_workers=0)
pred_loader = DataLoader(pred_dataset, batch_size=512, num_workers=0)



#Produce sets
train_idx_set = set(train_idx)
test_idx_set = set(testing_idx)


# Grab the GI indices needed for selections
# Get outliers
outlier_indicesK3 = np.load('Appendix_allPrunPlots_model2Corr/outlier_indicesK3corr_test2.npy')
print(len(outlier_indicesK3))



outlier_indicesK5 = np.load('Appendix_allPrunPlots_model2Corr/outlier_indicesK5corr_test2.npy')
print(len(outlier_indicesK5))



outlier_indicesK10 = np.load('Appendix_allPrunPlots_model2Corr/outlier_indicesK10corr_test2.npy')
print(len(outlier_indicesK10))



# Grab the NI indices needed for selections
# Read the list from the HDF5 file
with h5py.File('Prop_Opp_dataAnalysis/model2/train_idx_mod60_test2_fpProps.h5', 'r') as f:
    train_idx_modfp = list(f['train_idx_mod60_test2_fpProps'])

# Print the list to verify the data
print(len(train_idx_modfp))
print(len(train_idx))

# Read the list from the HDF5 file
with h5py.File('Prop_Opp_dataAnalysis/model2/train_idx_mod60_test2_fnProps.h5', 'r') as f:
    train_idx_modfn = list(f['train_idx_mod60_test2_fnProps'])

# Print the list to verify the data
print(len(train_idx_modfn))
print(len(train_idx))

# Read the list from the HDF5 file
with h5py.File('Prop_Opp_dataAnalysis/model2/train_idx_mod60_test2_tpOpps.h5', 'r') as f:
    train_idx_modtp = list(f['train_idx_mod60_test2_tpOpps'])

# Print the list to verify the data
print(len(train_idx_modtp))
print(len(train_idx))

# Read the list from the HDF5 file
with h5py.File('Prop_Opp_dataAnalysis/model2/train_idx_mod60_test2_tnOpps.h5', 'r') as f:
    train_idx_modtn = list(f['train_idx_mod60_test2_tnOpps'])

# Print the list to verify the data
print(len(train_idx_modtn))
print(len(train_idx))

train_idx_modallni = list(set(train_idx_modfp)&set(train_idx_modfn)&set(train_idx_modtp)&set(train_idx_modtn))

print(len(train_idx_modallni))
print(len(train_idx))



#Produce total and filtered indices for datasets
outlier_indices_setfp = set(train_idx_modfp)
filtered_train_idx_setfp = outlier_indices_setfp
filtered_test_idx_setfp = test_idx_set 
filtered_idx_setfp = (filtered_train_idx_setfp | filtered_test_idx_setfp) #- outlier_indices_set
idx_modfp = list(filtered_idx_setfp)

outlier_indices_setfn = set(train_idx_modfn)
filtered_train_idx_setfn = outlier_indices_setfn
filtered_test_idx_setfn = test_idx_set 
filtered_idx_setfn = (filtered_train_idx_setfn | filtered_test_idx_setfn) #- outlier_indices_set
idx_modfn = list(filtered_idx_setfn)

outlier_indices_settp = set(train_idx_modtp)
filtered_train_idx_settp = outlier_indices_settp
filtered_test_idx_settp = test_idx_set 
filtered_idx_settp = (filtered_train_idx_settp | filtered_test_idx_settp) #- outlier_indices_set
idx_modtp = list(filtered_idx_settp)

outlier_indices_settn = set(train_idx_modtn)
filtered_train_idx_settn = outlier_indices_settn
filtered_test_idx_settn = test_idx_set 
filtered_idx_settn = (filtered_train_idx_settn | filtered_test_idx_settn) #- outlier_indices_set
idx_modtn = list(filtered_idx_settn)

outlier_indices_setallni = set(train_idx_modallni)
filtered_train_idx_setallni = outlier_indices_setallni
filtered_test_idx_setallni = test_idx_set 
filtered_idx_setallni = (filtered_train_idx_setallni | filtered_test_idx_setallni) #- outlier_indices_set
idx_modallni = list(filtered_idx_setallni)

# Filter the dataset by excluding outlier indices
outlier_indices_setK5 = set(outlier_indicesK5)
#Produce total and filtered indices for datasets
filtered_train_idx_setK5 = train_idx_set - outlier_indices_setK5
filtered_test_idx_setK5 = test_idx_set - outlier_indices_setK5
filtered_idx_setK5 = (filtered_train_idx_setK5 | filtered_test_idx_setK5) #- outlier_indices_set
idx_modK5 = list(filtered_idx_setK5)

outlier_indices_setK3 = set(outlier_indicesK3)
filtered_train_idx_setK3 = train_idx_set - outlier_indices_setK3
filtered_test_idx_setK3 = test_idx_set - outlier_indices_setK3
filtered_idx_setK3 = (filtered_train_idx_setK3 | filtered_test_idx_setK3) #- outlier_indices_set
idx_modK3 = list(filtered_idx_setK3)

outlier_indices_setK10 = set(outlier_indicesK10)
filtered_train_idx_setK10 = train_idx_set - outlier_indices_setK10
filtered_test_idx_setK10 = test_idx_set - outlier_indices_setK10
filtered_idx_setK10 = (filtered_train_idx_setK10 | filtered_test_idx_setK10) #- outlier_indices_set
idx_modK10 = list(filtered_idx_setK10)


# import itertools
# id_listfp = list(itertools.chain.from_iterable(idx_modfp))
# id_listfn = list(itertools.chain.from_iterable(idx_modfn))
# id_listtp = list(itertools.chain.from_iterable(idx_modtp))
# id_listtn = list(itertools.chain.from_iterable(idx_modtn))
# id_listallni = list(itertools.chain.from_iterable(idx_modallni))
# id_listK3 = list(itertools.chain.from_iterable(idx_modK3))
# id_listK5 = list(itertools.chain.from_iterable(idx_modK5))
# id_listK10 = list(itertools.chain.from_iterable(idx_modK10))

#print(id_list1)
#print(id_list2)

import matplotlib.pyplot as plt
import pandas as pd
# Create a pandas DataFrame from the ID list
dffp = pd.DataFrame(idx_modfp, columns=['IDs'])
dffn = pd.DataFrame(idx_modfn, columns=['IDs'])
dftp = pd.DataFrame(idx_modtp, columns=['IDs'])
dftn = pd.DataFrame(idx_modtn, columns=['IDs'])
dfallni = pd.DataFrame(idx_modallni, columns=['IDs'])
dfk3 = pd.DataFrame(idx_modK3, columns=['IDs'])
dfk5 = pd.DataFrame(idx_modK5, columns=['IDs'])
dfk10 = pd.DataFrame(idx_modK10, columns=['IDs'])


# Group the DataFrame by IDs and count the frequency
id_countsfp = dffp['IDs'].value_counts().sort_index()
id_countsfn = dffn['IDs'].value_counts().sort_index()
id_countstp = dftp['IDs'].value_counts().sort_index()
id_countstn = dftn['IDs'].value_counts().sort_index()
id_countsallni = dfallni['IDs'].value_counts().sort_index()
id_countsk3 = dfk3['IDs'].value_counts().sort_index()
id_countsk5 = dfk5['IDs'].value_counts().sort_index()
id_countsk10 = dfk10['IDs'].value_counts().sort_index()


ids_fp = id_countsfp.index.tolist()
ids_fn = id_countsfn.index.tolist()
ids_tp = id_countstp.index.tolist()
ids_tn = id_countstn.index.tolist()
ids_allni = id_countsallni.index.tolist()
ids_k3 = id_countsk3.index.tolist()
ids_k5 = id_countsk5.index.tolist()
ids_k10 = id_countsk10.index.tolist()


from utils import dataset_to_pandas


#####Important to modify dataset to correspond modified indices! 
dfBaseline = dataset_to_pandas(jet_graph_dataset)
dffp = dataset_to_pandas(jet_graph_dataset[ids_fp])
dffn = dataset_to_pandas(jet_graph_dataset[ids_fn])
dftp = dataset_to_pandas(jet_graph_dataset[ids_tp])
dftn = dataset_to_pandas(jet_graph_dataset[ids_tn])
dfallni = dataset_to_pandas(jet_graph_dataset[ids_allni])
dfk3 = dataset_to_pandas(jet_graph_dataset[ids_k3])
dfk5 = dataset_to_pandas(jet_graph_dataset[ids_k5])
dfk10 = dataset_to_pandas(jet_graph_dataset[ids_k10])

dfBaseline.to_hdf('Prop_Opp_dataAnalysis/model2/dfBaseline_model2.h5',key='data',mode='w')
dffp.to_hdf('Prop_Opp_dataAnalysis/model2/dffp_model2.h5',key='data',mode='w')
dffn.to_hdf('Prop_Opp_dataAnalysis/model2/dffn_model2.h5',key='data',mode='w')
dftp.to_hdf('Prop_Opp_dataAnalysis/model2/dftp_model2.h5',key='data',mode='w')
dftn.to_hdf('Prop_Opp_dataAnalysis/model2/dftn_model2.h5',key='data',mode='w')
dfallni.to_hdf('Prop_Opp_dataAnalysis/model2/dfallni_model2.h5',key='data',mode='w')
dfk3.to_hdf('Prop_Opp_dataAnalysis/model2/dfk3_model2.h5',key='data',mode='w')
dfk5.to_hdf('Prop_Opp_dataAnalysis/model2/dfk5_model2.h5',key='data',mode='w')
dfk10.to_hdf('Prop_Opp_dataAnalysis/model2/dfk10_model2.h5',key='data',mode='w')