# For reproducibility.
import torch
torch.manual_seed(12345) 
from utils import plot_jet_graph, plot_metrics
# For warnings when running Captum on graphs.
import warnings
warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)
from JetGraphDataset import JetGraphDatasetInMemory_v2
from transforms import BuildEdges, OneHotEncodeLayer
from torch_geometric.transforms import Compose, LargestConnectedComponents, RemoveIsolatedNodes
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

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

print(len(jet_graph_dataset))

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

# # Enable gradients for input features
# for data in jet_graph_dataset:
#     data.x.requires_grad_(True)
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

# Create the dataloaders.
train_dataset = jet_graph_dataset[train_idx]
testing_dataset = jet_graph_dataset[testing_idx]
test_dataset = testing_dataset[test_idx]
pred_dataset = testing_dataset[pred_idx]
train_loader = DataLoader(train_dataset, batch_size=512, num_workers=0, shuffle=True)#, pin_memory=False
test_loader = DataLoader(test_dataset, batch_size=512, num_workers=0)
pred_loader = DataLoader(pred_dataset, batch_size=512, num_workers=0)

#from JetGraphDataset import stats
jet_graph_dataset.stats()
# Instantiate a model. 
# We take it from jetgraphs.models, but it can be any model for pytorch lightning.
from models import Shallow_GCN
from models import Residual_Arma, GAT, Arma
model = Arma(hidden_channels=256,node_feat_size=jet_graph_dataset[0].x.shape[1], use_edge_attr= False)
#model = Shallow_GCN(hidden_channels=256, node_feat_size=jet_graph_dataset[0].x.shape[1], use_edge_attr= False)
print(model) 

import pytorch_lightning as ptlight
from pytorch_lightning import loggers as pl_loggers

import numpy as np


# Provide directory to store checkpoints and train.
chkpt_dir = './checkpoints/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Add LearningRateMonitor
lr_monitor = ptlight.callbacks.LearningRateMonitor(logging_interval='step')

# We save checkpoints every epoch 
checkpoint_callback = ptlight.callbacks.ModelCheckpoint(
    dirpath=chkpt_dir,
    filename='gcn-{epoch:02d}',
    every_n_epochs=5,
    save_top_k=-1)

EarlyStop = ptlight.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.00, patience=7, verbose=True, mode="min")

# Define trainer.
epochs = 200
trainer = ptlight.Trainer(
    devices=1 if str(device).startswith("cuda") else 0,
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    default_root_dir='./', 
    max_epochs=epochs, 
    callbacks=[checkpoint_callback, lr_monitor, EarlyStop],
    enable_progress_bar=True,
    log_every_n_steps=5,
    check_val_every_n_epoch=5)

# Predict. Full arma
pretrained_filename = f'checkpoints/gcn-epoch=best.ckpt'
model = Shallow_GCN.load_from_checkpoint(pretrained_filename)

weights = trainer.predict(model, pred_loader)
truths = []
CNNscores = []
for data in pred_dataset:
    truths.append((data['y'].unsqueeze(1).float()).tolist())


weights = np.array(sum(weights, []))
truths = np.array(sum(truths, []))

print(weights.shape)
np.save('GNNweights.npy', weights)

print(truths.shape)
np.save('GNNtruths.npy', truths)

CNNweights = np.load('CNNweights.npy')
CNNtruths = np.load('CNNtruths.npy')

#plot metrics
plot_metrics(weights, truths, CNNweights, CNNtruths, odd_th=0.5, outname='Baseline_GNN_CNN.pdf')