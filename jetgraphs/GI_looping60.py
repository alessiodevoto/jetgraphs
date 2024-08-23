from JetGraphDataset import JetGraphDatasetInMemory_v2
from transforms import BuildEdges, OneHotEncodeLayer
from torch_geometric.transforms import Compose, LargestConnectedComponents, RemoveIsolatedNodes
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import pytorch_lightning as ptlight
from pytorch_lightning import loggers as pl_loggers
from utils import plot_jet_graph, plot_metrics
import numpy as np
from matplotlib import pyplot as plt
import torch
from models import Residual_Arma, GAT, Arma
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-GUI environments

from collections import Counter

torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_float32_matmul_precision('high')
# Where data is to be downloaded and stored.
datasets_root = "/user/jcarmignani/jgtorch/jetgraphs_workspace/datasets" 
# Secret url to dataset. 
raw_data_url = "https://cernbox.cern.ch/s/PYurUUzcNdXEGpz/download"

# In the next lines we define settings to build the dataset's edges and graphs.

# As discussed, we stick to 0.6x0.6 thresholds for the edges.
edge_builder = BuildEdges(
    directed=False,
    #directed=False,#Alessio
    self_loop_weight=0,
    same_layer_threshold=0.6, 
    consecutive_layer_threshold=0.6,#test related threshold
    distance_p=2)


from transforms import GraphFilter 

graph_filter = GraphFilter(min_num_nodes=2) # only graphs with at least 3 nodes will be accepted

from torch_geometric.transforms import Compose 
from transforms import NumLayers, LayersNum

optional_transforms = Compose([NumLayers(), LayersNum()])

custom_transforms = Compose([
    edge_builder,   # build edges same as before
    #LargestConnectedComponents() # extract main subgraph
    ])


# Finally download the dataset.
jet_graph_dataset = JetGraphDatasetInMemory_v2(
    root = datasets_root,           # directory where to download data 
    url = raw_data_url,             # url to raw data
    pre_filter = graph_filter,      # filter graphs with less than 3 nodes 
    pre_transform = custom_transforms,    # edge_builder should be passed as pre_transform to keep data on disk.
    post_filter = graph_filter,     # filter graphs with less than 3 nodes
    transform = optional_transforms,
    ) 
# Create the dataloaders.
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
train_loader = DataLoader(train_dataset, batch_size=512, num_workers=10)#, shuffle=True)#, pin_memory=False
test_loader = DataLoader(test_dataset, batch_size=512, num_workers=10)
pred_loader = DataLoader(pred_dataset, batch_size=512, num_workers=10)

# Provide directory to store checkpoints and train.
chkpt_dir = './checkpoints/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Add LearningRateMonitor
lr_monitor = ptlight.callbacks.LearningRateMonitor(logging_interval='step')

# We save checkpoints every epoch 
checkpoint_callback = ptlight.callbacks.ModelCheckpoint(
    dirpath=chkpt_dir,
    filename='gcn-{epoch:02d}',
    every_n_epochs=1,
    save_top_k=-1)

EarlyStop = ptlight.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.00, patience=15, verbose=True, mode="min")

# Define trainer.
epochs = 150
trainer = ptlight.Trainer(
    devices=1 if str(device).startswith("cuda") else 0,
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    default_root_dir='./', 
    max_epochs=epochs, 
    callbacks=[checkpoint_callback, lr_monitor, EarlyStop],
    enable_progress_bar=True,
    log_every_n_steps=1,
    check_val_every_n_epoch=1)


# Predict. Full arma
pretrained_filename = f'./checkpoints/gcn-epoch=best.ckpt'
model = Arma.load_from_checkpoint(pretrained_filename)
weights = trainer.predict(model, train_loader) #we're here interested in analysisng the training set
truths = []

for data in train_dataset:
    truths.append((data['y'].unsqueeze(1).float()).tolist())
    
weights = np.array(sum(weights, []))
truths = np.array(sum(truths, []))


idx_tpWP99 = [] # TP
idx_tnWP99 = [] # TN
idx_fnWP99 = [] # FN
idx_fpWP99 = [] # FP

idx_tpWP99t = [] # TP
idx_tnWP99t = [] # TN
idx_fnWP99t = [] # FN
idx_fpWP99t = [] # FP

#model1 cuts for top 5000 graphs

for i in range(len(weights)):
    if truths[i]==0 and weights[i] < 0.00275:
        idx_tnWP99.append(i)
       
       
for i in range(len(weights)):
    if truths[i]==1 and weights[i] > 0.9996:
        idx_tpWP99.append(i)
        

for i in range(len(weights)):
    if truths[i]==0 and weights[i] > 0.62:
        idx_fpWP99.append(i)
        

for i in range(len(weights)):
    if truths[i]==1 and weights[i] < 0.415:
        idx_fnWP99.append(i)
        
#Totals
for i in range(len(weights)):
    if truths[i]==0 and weights[i] < 0.01:
        idx_tnWP99t.append(i)
       
       
for i in range(len(weights)):
    if truths[i]==1 and weights[i] > 0.99:
        idx_tpWP99t.append(i)
        

for i in range(len(weights)):
    if truths[i]==0 and weights[i] > 0.99:
        idx_fpWP99t.append(i)
        

for i in range(len(weights)):
    if truths[i]==1 and weights[i] < 0.01:
        idx_fnWP99t.append(i)

# model2 cuts for top 5000 graphs
# for i in range(len(weights)):
#     if truths[i]==0 and weights[i] < 0.0025:
#         idx_tnWP99.append(i)
       
       
# for i in range(len(weights)):
#     if truths[i]==1 and weights[i] > 0.99935:
#         idx_tpWP99.append(i)
        

# for i in range(len(weights)):
#     if truths[i]==0 and weights[i] > 0.62:
#         idx_fpWP99.append(i)
        

# for i in range(len(weights)):
#     if truths[i]==1 and weights[i] < 0.45:
#         idx_fnWP99.append(i)

#model0 cuts for top 5000 graphs

# for i in range(len(weights)):
#     if truths[i]==0 and weights[i] < 0.00223:
#         idx_tnWP99.append(i)
       
       
# for i in range(len(weights)):
#     if truths[i]==1 and weights[i] > 0.99975:
#         idx_tpWP99.append(i)
        

# for i in range(len(weights)):
#     if truths[i]==0 and weights[i] > 0.585:
#         idx_fpWP99.append(i)
        

# for i in range(len(weights)):
#     if truths[i]==1 and weights[i] < 0.4:
#         idx_fnWP99.append(i)
        
print("TPs: ",len(idx_tpWP99))
print("TNs: ",len(idx_tnWP99))
print("FNs: ",len(idx_fnWP99))
print("FPs: ",len(idx_fpWP99))

print("TPs tot: ",len(idx_tpWP99t))
print("TNs tot: ",len(idx_tnWP99t))
print("FNs tot: ",len(idx_fnWP99t))
print("FPs tot: ",len(idx_fpWP99t))


# !!! Important to change the GI denominators down according to each models samplings as in the following lists: !!!
#model0
# TPs:  5118
# TNs:  5198
# FNs:  5050
# FPs:  5109
# TPs tot:  22024
# TNs tot:  12778
# FNs tot:  67
# FPs tot:  33

#model2
# TPs:  5223
# TNs:  5381
# FNs:  5271
# FPs:  5187
# TPs tot:  18679
# TNs tot:  12351
# FNs tot:  65
# FPs tot:  43

#model1
# TPs:  5027
# TNs:  5188
# FNs:  5059
# FPs:  5130
# TPs tot:  21590
# TNs tot:  11549
# FNs tot:  71
# FPs tot:  57


from explainability import CaptumPipeline



#FP
# Initialize high_frequency_ids_prop as an empty list
high_frequency_ids_opp = []

# Initialize your loop counter
i = 0

# Define your stopping condition here (e.g., after a certain number of iterations)
max_iterations = 5000
train_idx_mod = train_idx
gi = []
while i < max_iterations:
    global_influencers = high_frequency_ids_opp
    print("GIs: ", global_influencers)
    if len(sum(gi, [])) == 0:
        gi_list = [item for item in high_frequency_ids_opp]
    else:
        gi_list = [item + len(sum(gi, [])) for item in high_frequency_ids_opp]

    gi.append(gi_list)
    ####Start Loop Proponnents
    # Modify train_idx_mod based on global_influencers
    train_idx_mod = [x for i, x in enumerate(train_idx_mod) if i not in global_influencers]

    # Instanitate the Captum pipeline.
    pipeline = CaptumPipeline(
        model=model,
        checkpoint_dir=chkpt_dir,
        dataset=jet_graph_dataset,
        train_idx=train_idx_mod,
        captum_impl='fast', # or 'base'
        epochs=epochs
    )
    
    test_influence_indices=idx_fpWP99
    #test_influence_indices
    pipeline.run_captum(test_influence_indices=test_influence_indices)
    _ , idx_opponents = pipeline.display_results(angle=30, elev=10, save_to_dir=False)

    import itertools
    id_list1 = list(itertools.chain.from_iterable(idx_opponents.tolist()))
    

    import matplotlib.pyplot as plt
    import pandas as pd
    
    # Create a pandas DataFrame from the ID list
    df1 = pd.DataFrame(id_list1, columns=['IDs'])
    

    # Group the DataFrame by IDs and count the frequency
    id_counts1 = df1['IDs'].value_counts().sort_index()
   
    

    high_frequency_ids_opp = id_counts1[(id_counts1/5130)*100 > 60].index.tolist()
    print("opps: ",high_frequency_ids_opp)

    
    i += 1
    
    # Add your stopping condition here
    if high_frequency_ids_opp == []:
        # Create the frequency plot
        ((id_counts1/5130) * 100).plot(kind='bar', figsize=(20, 10))

        # Add labels and title
        plt.xlabel('IDs')
        plt.ylabel('Frequency')
        plt.title('ID Frequency Plot')

        # Save the plot to a file
        plt.savefig("fpOppsdist60GNN1.pdf")

        # Close the plot to free up memory
        plt.close()
        break



# Initialize high_frequency_ids_prop as an empty list
high_frequency_ids_prop = []

# Initialize your loop counter
i = 0

# Define your stopping condition here (e.g., after a certain number of iterations)
max_iterations = 5000
#train_idx_mod = train_idx

while i < max_iterations:
    global_influencers = high_frequency_ids_prop
    print("GIs: ", global_influencers)
    #if len(sum(gi, [])) == 0:
    gi_list = [item for item in high_frequency_ids_prop]
    # else:
    #     gi_list = [item + len(sum(gi, [])) for item in high_frequency_ids_prop]
        
    gi.append(gi_list)
    ####Start Loop Proponnents
    # Modify train_idx_mod based on global_influencers
    train_idx_mod = [x for i, x in enumerate(train_idx_mod) if i not in global_influencers]

    # Instanitate the Captum pipeline.
    pipeline = CaptumPipeline(
        model=model,
        checkpoint_dir=chkpt_dir,
        dataset=jet_graph_dataset,
        train_idx=train_idx_mod,
        captum_impl='fast', # or 'base'
        epochs=epochs
    )
    
    test_influence_indices=idx_fpWP99
    #test_influence_indices
    pipeline.run_captum(test_influence_indices=test_influence_indices)
    idx_proponents, _ = pipeline.display_results(angle=30, elev=10, save_to_dir=False)

    
    id_list2 = list(itertools.chain.from_iterable(idx_proponents.tolist()))
    
    # Create a pandas DataFrame from the ID list
    df2 = pd.DataFrame(id_list2, columns=['IDs'])
    

    # Group the DataFrame by IDs and count the frequency
    id_counts2 = df2['IDs'].value_counts().sort_index()
    

    

    high_frequency_ids_prop = id_counts2[(id_counts2/5130)*100 > 60].index.tolist()
    print("props: ",high_frequency_ids_prop)
    
    i += 1
    #gi.append(high_frequency_ids_prop)
    # Add your stopping condition here
    if high_frequency_ids_prop == []:
        # Create the frequency plot
        ((id_counts2/5130) * 100).plot(kind='bar', figsize=(20, 10))

        # Add labels and title
        plt.xlabel('IDs')
        plt.ylabel('Frequency')
        plt.title('ID Frequency Plot')

        # Save the plot to a file
        plt.savefig("fpPropsdist60GNN1.pdf")

        # Close the plot to free up memory
        plt.close()
        break


ids_prop = id_counts2.index.tolist()
# print("proponents indices: ", ids_prop)
#ids_opp = id_counts1.index.tolist()
#print("opponents indices: ", ids_opp)
np.save('fpPropstest1.npy', ids_prop)
print('fpPropstest1 len: ', len(ids_prop))


train_idx_mod = [x for i, x in enumerate(train_idx_mod) if i not in ids_prop]
#train_idx_mod = [x for i, x in enumerate(train_idx_mod) if i not in ids_opp]

import h5py

#writing train_idx_mod list
# Create an HDF5 file and open it in write mode
with h5py.File('train_idx_mod60_test1_fpProps.h5', 'w') as f:
    # Create a dataset in the HDF5 file and store the list
    f.create_dataset('train_idx_mod60_test1_fpProps', data=train_idx_mod)

print("len of modified indices without GIs and top fp proponents: ", len(train_idx_mod))
print("len of training indices: ", len(train_idx))
gi = sum(gi, [])
np.save('GIsfpTest1corr60.npy', gi)
print('GIs 60 fp len: ', len(gi))




#FN
# Initialize high_frequency_ids_prop as an empty list
high_frequency_ids_opp = []

# Initialize your loop counter
i = 0

# Define your stopping condition here (e.g., after a certain number of iterations)
max_iterations = 5000
train_idx_mod = train_idx
gi = []
while i < max_iterations:
    global_influencers = high_frequency_ids_opp
    print("GIs: ", global_influencers)
    if len(sum(gi, [])) == 0:
        gi_list = [item for item in high_frequency_ids_opp]
    else:
        gi_list = [item + len(sum(gi, [])) for item in high_frequency_ids_opp]

    gi.append(gi_list)
    ####Start Loop Proponnents
    # Modify train_idx_mod based on global_influencers
    train_idx_mod = [x for i, x in enumerate(train_idx_mod) if i not in global_influencers]

    # Instanitate the Captum pipeline.
    pipeline = CaptumPipeline(
        model=model,
        checkpoint_dir=chkpt_dir,
        dataset=jet_graph_dataset,
        train_idx=train_idx_mod,
        captum_impl='fast', # or 'base'
        epochs=epochs
    )
    
    test_influence_indices=idx_fnWP99
    #test_influence_indices
    pipeline.run_captum(test_influence_indices=test_influence_indices)
    _ , idx_opponents = pipeline.display_results(angle=30, elev=10, save_to_dir=False)

    
    id_list1 = list(itertools.chain.from_iterable(idx_opponents.tolist()))
    
    # Create a pandas DataFrame from the ID list
    df1 = pd.DataFrame(id_list1, columns=['IDs'])
    

    # Group the DataFrame by IDs and count the frequency
    id_counts1 = df1['IDs'].value_counts().sort_index()
   
    

    high_frequency_ids_opp = id_counts1[(id_counts1/5059)*100 > 60].index.tolist()
    print("opps: ",high_frequency_ids_opp)

    
    i += 1
    
    # Add your stopping condition here
    if high_frequency_ids_opp == []:
        # Create the frequency plot
        ((id_counts1/5059) * 100).plot(kind='bar', figsize=(20, 10))

        # Add labels and title
        plt.xlabel('IDs')
        plt.ylabel('Frequency')
        plt.title('ID Frequency Plot')

        # Save the plot to a file
        plt.savefig("fnOppsdist60GNN1.pdf")

        # Close the plot to free up memory
        plt.close()
        break



# Initialize high_frequency_ids_prop as an empty list
high_frequency_ids_prop = []

# Initialize your loop counter
i = 0

# Define your stopping condition here (e.g., after a certain number of iterations)
max_iterations = 5000
#train_idx_mod = train_idx

while i < max_iterations:
    global_influencers = high_frequency_ids_prop
    print("GIs: ", global_influencers)
    #if len(sum(gi, [])) == 0:
    gi_list = [item for item in high_frequency_ids_prop]
    # else:
    #     gi_list = [item + len(sum(gi, [])) for item in high_frequency_ids_prop]
        
    gi.append(gi_list)
    ####Start Loop Proponnents
    # Modify train_idx_mod based on global_influencers
    train_idx_mod = [x for i, x in enumerate(train_idx_mod) if i not in global_influencers]

    # Instanitate the Captum pipeline.
    pipeline = CaptumPipeline(
        model=model,
        checkpoint_dir=chkpt_dir,
        dataset=jet_graph_dataset,
        train_idx=train_idx_mod,
        captum_impl='fast', # or 'base'
        epochs=epochs
    )
    
    test_influence_indices=idx_fnWP99
    #test_influence_indices
    pipeline.run_captum(test_influence_indices=test_influence_indices)
    idx_proponents, _ = pipeline.display_results(angle=30, elev=10, save_to_dir=False)

    
    id_list2 = list(itertools.chain.from_iterable(idx_proponents.tolist()))
    
    # Create a pandas DataFrame from the ID list
    df2 = pd.DataFrame(id_list2, columns=['IDs'])
    

    # Group the DataFrame by IDs and count the frequency
    id_counts2 = df2['IDs'].value_counts().sort_index()
    

    

    high_frequency_ids_prop = id_counts2[(id_counts2/5059)*100 > 60].index.tolist()
    print("props: ",high_frequency_ids_prop)
    
    i += 1
    #gi.append(high_frequency_ids_prop)
    # Add your stopping condition here
    if high_frequency_ids_prop == []:
        # Create the frequency plot
        ((id_counts2/5059) * 100).plot(kind='bar', figsize=(20, 10))

        # Add labels and title
        plt.xlabel('IDs')
        plt.ylabel('Frequency')
        plt.title('ID Frequency Plot')

        # Save the plot to a file
        plt.savefig("fnPropsdist60GNN1.pdf")

        # Close the plot to free up memory
        plt.close()
        break


ids_prop = id_counts2.index.tolist()
# print("proponents indices: ", ids_prop)
#ids_opp = id_counts1.index.tolist()
#print("opponents indices: ", ids_opp)
np.save('fnPropstest1.npy', ids_prop)
print('fnPropstest1 len: ', len(ids_prop))


train_idx_mod = [x for i, x in enumerate(train_idx_mod) if i not in ids_prop]
#train_idx_mod = [x for i, x in enumerate(train_idx_mod) if i not in ids_opp]



#writing train_idx_mod list
# Create an HDF5 file and open it in write mode
with h5py.File('train_idx_mod60_test1_fnProps.h5', 'w') as f:
    # Create a dataset in the HDF5 file and store the list
    f.create_dataset('train_idx_mod60_test1_fnProps', data=train_idx_mod)

print("len of modified indices without GIs and top fn proponents: ", len(train_idx_mod))
print("len of training indices: ", len(train_idx))
gi = sum(gi, [])
np.save('GIsfnTest1corr60.npy', gi)
print('GIs 60 fn len: ', len(gi))



#TN
# Initialize high_frequency_ids_prop as an empty list
high_frequency_ids_opp = []

# Initialize your loop counter
i = 0

# Define your stopping condition here (e.g., after a certain number of iterations)
max_iterations = 5000
train_idx_mod = train_idx
gi = []
while i < max_iterations:
    global_influencers = high_frequency_ids_opp
    print("GIs: ", global_influencers)
    if len(sum(gi, [])) == 0:
        gi_list = [item for item in high_frequency_ids_opp]
    else:
        gi_list = [item + len(sum(gi, [])) for item in high_frequency_ids_opp]

    gi.append(gi_list)
    ####Start Loop Proponnents
    # Modify train_idx_mod based on global_influencers
    train_idx_mod = [x for i, x in enumerate(train_idx_mod) if i not in global_influencers]

    # Instanitate the Captum pipeline.
    pipeline = CaptumPipeline(
        model=model,
        checkpoint_dir=chkpt_dir,
        dataset=jet_graph_dataset,
        train_idx=train_idx_mod,
        captum_impl='fast', # or 'base'
        epochs=epochs
    )
    
    test_influence_indices=idx_tnWP99
    #test_influence_indices
    pipeline.run_captum(test_influence_indices=test_influence_indices)
    _ , idx_opponents = pipeline.display_results(angle=30, elev=10, save_to_dir=False)

    
    id_list1 = list(itertools.chain.from_iterable(idx_opponents.tolist()))
    
    # Create a pandas DataFrame from the ID list
    df1 = pd.DataFrame(id_list1, columns=['IDs'])
    

    # Group the DataFrame by IDs and count the frequency
    id_counts1 = df1['IDs'].value_counts().sort_index()
   
    

    high_frequency_ids_opp = id_counts1[(id_counts1/5188)*100 > 60].index.tolist()
    print("opps: ",high_frequency_ids_opp)

    
    i += 1
    
    # Add your stopping condition here
    if high_frequency_ids_opp == []:
        # Create the frequency plot
        ((id_counts1/5188) * 100).plot(kind='bar', figsize=(20, 10))

        # Add labels and title
        plt.xlabel('IDs')
        plt.ylabel('Frequency')
        plt.title('ID Frequency Plot')

        # Save the plot to a file
        plt.savefig("tnOppsdist60GNN1.pdf")

        # Close the plot to free up memory
        plt.close()
        break



# Initialize high_frequency_ids_prop as an empty list
high_frequency_ids_prop = []

# Initialize your loop counter
i = 0

# Define your stopping condition here (e.g., after a certain number of iterations)
max_iterations = 5000
#train_idx_mod = train_idx

while i < max_iterations:
    global_influencers = high_frequency_ids_prop
    print("GIs: ", global_influencers)
    #if len(sum(gi, [])) == 0:
    gi_list = [item for item in high_frequency_ids_prop]
    # else:
    #     gi_list = [item + len(sum(gi, [])) for item in high_frequency_ids_prop]
        
    gi.append(gi_list)
    ####Start Loop Proponnents
    # Modify train_idx_mod based on global_influencers
    train_idx_mod = [x for i, x in enumerate(train_idx_mod) if i not in global_influencers]

    # Instanitate the Captum pipeline.
    pipeline = CaptumPipeline(
        model=model,
        checkpoint_dir=chkpt_dir,
        dataset=jet_graph_dataset,
        train_idx=train_idx_mod,
        captum_impl='fast', # or 'base'
        epochs=epochs
    )
    
    test_influence_indices=idx_tnWP99
    #test_influence_indices
    pipeline.run_captum(test_influence_indices=test_influence_indices)
    idx_proponents, _ = pipeline.display_results(angle=30, elev=10, save_to_dir=False)

    
    id_list2 = list(itertools.chain.from_iterable(idx_proponents.tolist()))
    
    # Create a pandas DataFrame from the ID list
    df2 = pd.DataFrame(id_list2, columns=['IDs'])
    

    # Group the DataFrame by IDs and count the frequency
    id_counts2 = df2['IDs'].value_counts().sort_index()
    

    

    high_frequency_ids_prop = id_counts2[(id_counts2/5188)*100 > 60].index.tolist()
    print("props: ",high_frequency_ids_prop)
    
    i += 1
    #gi.append(high_frequency_ids_prop)
    # Add your stopping condition here
    if high_frequency_ids_prop == []:
        # Create the frequency plot
        ((id_counts2/5188) * 100).plot(kind='bar', figsize=(20, 10))

        # Add labels and title
        plt.xlabel('IDs')
        plt.ylabel('Frequency')
        plt.title('ID Frequency Plot')

        # Save the plot to a file
        plt.savefig("tnPropsdist60GNN1.pdf")

        # Close the plot to free up memory
        plt.close()
        break


#ids_prop = id_counts2.index.tolist()
# print("proponents indices: ", ids_prop)
ids_opp = id_counts1.index.tolist()
#print("opponents indices: ", ids_opp)
np.save('tnOppstest1.npy', ids_opp)
print('tnOppstest1 len: ', len(ids_opp))


#train_idx_mod = [x for i, x in enumerate(train_idx_mod) if i not in ids_prop]
train_idx_mod = [x for i, x in enumerate(train_idx_mod) if i not in ids_opp]



#writing train_idx_mod list
# Create an HDF5 file and open it in write mode
with h5py.File('train_idx_mod60_test1_tnOpps.h5', 'w') as f:
    # Create a dataset in the HDF5 file and store the list
    f.create_dataset('train_idx_mod60_test1_tnOpps', data=train_idx_mod)

print("len of modified indices without GIs and top tn opponents: ", len(train_idx_mod))
print("len of training indices: ", len(train_idx))
gi = sum(gi, [])
np.save('GIstnTest1corr60.npy', gi)
print('GIs 60 tn len: ', len(gi))



#TP
# Initialize high_frequency_ids_prop as an empty list
high_frequency_ids_opp = []

# Initialize your loop counter
i = 0

# Define your stopping condition here (e.g., after a certain number of iterations)
max_iterations = 5000
train_idx_mod = train_idx
gi = []
while i < max_iterations:
    global_influencers = high_frequency_ids_opp
    print("GIs: ", global_influencers)
    if len(sum(gi, [])) == 0:
        gi_list = [item for item in high_frequency_ids_opp]
    else:
        gi_list = [item + len(sum(gi, [])) for item in high_frequency_ids_opp]

    gi.append(gi_list)
    ####Start Loop Proponnents
    # Modify train_idx_mod based on global_influencers
    train_idx_mod = [x for i, x in enumerate(train_idx_mod) if i not in global_influencers]

    # Instanitate the Captum pipeline.
    pipeline = CaptumPipeline(
        model=model,
        checkpoint_dir=chkpt_dir,
        dataset=jet_graph_dataset,
        train_idx=train_idx_mod,
        captum_impl='fast', # or 'base'
        epochs=epochs
    )
    
    test_influence_indices=idx_tpWP99
    #test_influence_indices
    pipeline.run_captum(test_influence_indices=test_influence_indices)
    _ , idx_opponents = pipeline.display_results(angle=30, elev=10, save_to_dir=False)

    
    id_list1 = list(itertools.chain.from_iterable(idx_opponents.tolist()))
    
    # Create a pandas DataFrame from the ID list
    df1 = pd.DataFrame(id_list1, columns=['IDs'])
    

    # Group the DataFrame by IDs and count the frequency
    id_counts1 = df1['IDs'].value_counts().sort_index()
   
    

    high_frequency_ids_opp = id_counts1[(id_counts1/5027)*100 > 60].index.tolist()
    print("opps: ",high_frequency_ids_opp)

    
    i += 1
    
    # Add your stopping condition here
    if high_frequency_ids_opp == []:
        # Create the frequency plot
        ((id_counts1/5027) * 100).plot(kind='bar', figsize=(20, 10))

        # Add labels and title
        plt.xlabel('IDs')
        plt.ylabel('Frequency')
        plt.title('ID Frequency Plot')

        # Save the plot to a file
        plt.savefig("tpOppsdist60GNN1.pdf")

        # Close the plot to free up memory
        plt.close()
        break



# Initialize high_frequency_ids_prop as an empty list
high_frequency_ids_prop = []

# Initialize your loop counter
i = 0

# Define your stopping condition here (e.g., after a certain number of iterations)
max_iterations = 5000
#train_idx_mod = train_idx

while i < max_iterations:
    global_influencers = high_frequency_ids_prop
    print("GIs: ", global_influencers)
    #if len(sum(gi, [])) == 0:
    gi_list = [item for item in high_frequency_ids_prop]
    # else:
    #     gi_list = [item + len(sum(gi, [])) for item in high_frequency_ids_prop]
        
    gi.append(gi_list)
    ####Start Loop Proponnents
    # Modify train_idx_mod based on global_influencers
    train_idx_mod = [x for i, x in enumerate(train_idx_mod) if i not in global_influencers]

    # Instanitate the Captum pipeline.
    pipeline = CaptumPipeline(
        model=model,
        checkpoint_dir=chkpt_dir,
        dataset=jet_graph_dataset,
        train_idx=train_idx_mod,
        captum_impl='fast', # or 'base'
        epochs=epochs
    )
    
    test_influence_indices=idx_tpWP99
    #test_influence_indices
    pipeline.run_captum(test_influence_indices=test_influence_indices)
    idx_proponents, _ = pipeline.display_results(angle=30, elev=10, save_to_dir=False)

    
    id_list2 = list(itertools.chain.from_iterable(idx_proponents.tolist()))
    
    # Create a pandas DataFrame from the ID list
    df2 = pd.DataFrame(id_list2, columns=['IDs'])
    

    # Group the DataFrame by IDs and count the frequency
    id_counts2 = df2['IDs'].value_counts().sort_index()
    

    

    high_frequency_ids_prop = id_counts2[(id_counts2/5027)*100 > 60].index.tolist()
    print("props: ",high_frequency_ids_prop)
    
    i += 1
    #gi.append(high_frequency_ids_prop)
    # Add your stopping condition here
    if high_frequency_ids_prop == []:
        # Create the frequency plot
        ((id_counts2/5027) * 100).plot(kind='bar', figsize=(20, 10))

        # Add labels and title
        plt.xlabel('IDs')
        plt.ylabel('Frequency')
        plt.title('ID Frequency Plot')

        # Save the plot to a file
        plt.savefig("tpPropsdist60GNN1.pdf")

        # Close the plot to free up memory
        plt.close()
        break


#ids_prop = id_counts2.index.tolist()
# print("proponents indices: ", ids_prop)
ids_opp = id_counts1.index.tolist()
#print("opponents indices: ", ids_opp)
np.save('tpOppstest1.npy', ids_opp)
print('tpOppstest1 len: ', len(ids_opp))


#train_idx_mod = [x for i, x in enumerate(train_idx_mod) if i not in ids_prop]
train_idx_mod = [x for i, x in enumerate(train_idx_mod) if i not in ids_opp]



#writing train_idx_mod list
# Create an HDF5 file and open it in write mode
with h5py.File('train_idx_mod60_test1_tpOpps.h5', 'w') as f:
    # Create a dataset in the HDF5 file and store the list
    f.create_dataset('train_idx_mod60_test1_tpOpps', data=train_idx_mod)

print("len of modified indices without GIs and top tp opponents: ", len(train_idx_mod))
print("len of training indices: ", len(train_idx))
gi = sum(gi, [])
np.save('GIstpTest1corr60.npy', gi)
print('GIs 60 tp len: ', len(gi))


print("End of pipeline!!!")