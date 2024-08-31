from JetGraphDataset import JetGraphDatasetInMemory_v2
from transforms import BuildEdges, OneHotEncodeLayer
from torch_geometric.transforms import Compose, LargestConnectedComponents, RemoveIsolatedNodes
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import torch
from collections import Counter
import matplotlib.pyplot as plt
torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_float32_matmul_precision('high')
from lightning.pytorch.loggers import WandbLogger
import wandb
wandb.login()

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

# Initialize wandb logger
wandb_logger = WandbLogger(project='DarkPhotonJets', job_type='train')


print(model)
train_dataset = jet_graph_dataset[train_idx]
testing_dataset = jet_graph_dataset[testing_idx]
test_dataset = testing_dataset[test_idx]
pred_dataset = testing_dataset[pred_idx]
train_loader = DataLoader(train_dataset, batch_size=512, num_workers=0, shuffle=True)#, pin_memory=False
test_loader = DataLoader(test_dataset, batch_size=512, num_workers=0)
pred_loader = DataLoader(pred_dataset, batch_size=512, num_workers=0)

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from utils import plot_jet_graph, plot_metrics
import numpy as np
from matplotlib import pyplot as plt
# For reproducibility.
import torch
#torch.manual_seed(12345) 
# Provide directory to store checkpoints and train.
chkpt_dir = './checkpoints/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Add LearningRateMonitor
lr_monitor = LearningRateMonitor(logging_interval='step')

# We save checkpoints every epoch 
checkpoint_callback = ModelCheckpoint(
    dirpath=chkpt_dir,
    filename='gcn-{epoch:02d}',
    every_n_epochs=1,
    save_top_k=-1)

EarlyStop = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=15, verbose=True, mode="min")

# Define trainer.
epochs = 150
trainer = Trainer(
    devices=1 if str(device).startswith("cuda") else 0,
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    default_root_dir='./', 
    max_epochs=epochs, 
    logger=wandb_logger,
    callbacks=[checkpoint_callback, lr_monitor, EarlyStop],
    enable_progress_bar=True,
    log_every_n_steps=1,
    check_val_every_n_epoch=1)

# Train model.
trainer.fit(model, train_loader, test_loader)#, ckpt_path="./checkpoints/gcn-epoch=last.ckpt")

# Evaluate the model on the held-out test set ⚡⚡
trainer.test(model, dataloaders = pred_loader)

# Close wandb run
wandb.finish()



# Check collected cached gradients
print(f"Number of gradient entries collected: {len(model.best_epoch_gradients)}")



import numpy as np
import matplotlib.pyplot as plt

# Extract and flatten gradients
gradient_data = []
for grad in model.best_epoch_gradients:
    if grad is not None and len(grad) > 0:
        # Ensure grad is a numpy array and flatten it
        # Compute the norm of each node's gradient across its features
        node_norms = np.linalg.norm(grad, axis=1)
        # Compute the mean of these norms
        mean_norm = np.mean(node_norms)
        # Flatten the result to ensure it's a 1-dimensional array
        grad_array = np.array([mean_norm]).flatten()
        gradient_data.append(grad_array)

if gradient_data:
    # Concatenate all non-empty, flattened gradients into one array
    concatenated_gradients = np.concatenate(gradient_data)

    # # Calculate the mean and standard deviation for the threshold calculation
    # mean = np.mean(concatenated_gradients)
    # std_dev = np.std(concatenated_gradients)
    # threshold1 = mean - 2 * std_dev
    # threshold2 = mean + 20 * std_dev
    
    # Calculate the median and interquartile range for threshold calculation
    median_grad = np.median(concatenated_gradients)
    Q1 = np.percentile(concatenated_gradients, 25)
    Q3 = np.percentile(concatenated_gradients, 75)
    IQR = Q3 - Q1

    # Set k to a higher value to target the tail of the distribution
    k = 5  # Adjust k from 3 to 10 for highly skewed data

    # Calculate the upper threshold based on the IQR method
    threshold1 = median_grad + k * IQR
   



    print("Threshold:", threshold1)
   

    # Save all gradient magnitudes to a file
    np.save('gradient_magnitudes_test.npy', concatenated_gradients)
    
    # Determine outlier indices based on the predefined threshold
    outlier_indices = [i for i, magnitude in enumerate(concatenated_gradients) if magnitude > threshold1]

    # Save the outlier indices to a file
    np.save('outlier_indices_test.npy', outlier_indices)
    
    # Filter the dataset by excluding outlier indices
    filtered_dataset = [jet_graph_dataset[i] for i in range(len(jet_graph_dataset)) if i not in outlier_indices]

    # Output the results
    print("Filtered Dataset Length:", len(filtered_dataset))
    #print("Outlier Indices:", outlier_indices)

    # Plot histogram of all magnitudes
    plt.hist(concatenated_gradients, bins=50, edgecolor='black')
    plt.axvline(threshold1, color='r', linestyle='dashed', linewidth=1, label=f'Threshold: {threshold1:.2f}')
    
    plt.title('Histogram of Gradient Magnitudes')
    plt.xlabel('Magnitude')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('gradient_magnitudes_histogram_test.pdf')
    #plt.show()

else:
    print("No valid gradient data available for analysis.")
