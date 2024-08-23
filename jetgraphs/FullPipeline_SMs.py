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
import matplotlib.pyplot as plt
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
    consecutive_layer_threshold=0.3,#test related threshold
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



class SMDataset:
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data



# Instanitate the Captum pipeline.
from explainability import CaptumPipeline
pipeline = CaptumPipeline(
    model=model,
    checkpoint_dir=chkpt_dir,
    dataset=jet_graph_dataset,
    train_idx=train_idx,
    #train_dataset=DataLoader(jet_graph_dataset[train_idx], batch_size=512, num_workers= 8, shuffle=False),
    captum_impl='fast', # or 'base'
    epochs=epochs
)

# FNs
test_influence_indices=idx_fnWP99
test_influence_indices
pipeline.run_captum(test_influence_indices=test_influence_indices)
idx_proponents, idx_opponents = pipeline.display_results(angle=30, elev=10, save_to_dir=False)
import itertools
id_list1 = list(itertools.chain.from_iterable(idx_proponents.tolist()))
id_list2 = list(itertools.chain.from_iterable(idx_opponents.tolist()))
#print(id_list1)
#print(id_list2)

import matplotlib.pyplot as plt
import pandas as pd
# Create a pandas DataFrame from the ID list
df1 = pd.DataFrame(id_list1, columns=['IDs'])
df2 = pd.DataFrame(id_list2, columns=['IDs'])

# Group the DataFrame by IDs and count the frequency
id_counts1 = df1['IDs'].value_counts().sort_index()
id_counts2 = df2['IDs'].value_counts().sort_index()

ids_prop = id_counts1.index.tolist()
#print(ids_prop)
ids_opp = id_counts2.index.tolist()
#print(ids_opp)

from utils import dataset_to_pandas
from models_sm import ArmaSM




# Predict. Full arma
#pretrained_filename = f'./ARMA/test2d/NoPruning/gcn-epoch=144.ckpt'
modell = ArmaSM.load_from_checkpoint(pretrained_filename)
modell.to(device)
#weightss = trainer.predict(modell, loader)


#####Important to modify dataset to correspond modified indices! BUGG
jetdatasetm = jet_graph_dataset[train_idx]
df1 = dataset_to_pandas(jetdatasetm[ids_prop])
df2 = dataset_to_pandas(jetdatasetm[ids_opp])
df1.to_hdf('Prop_Opp_dataAnalysis/model1/FN_propopp_test1/prop_set_FNwoGIs60_fullDataset_test1CorrNew.h5',key='data',mode='w')
df2.to_hdf('Prop_Opp_dataAnalysis/model1/FN_propopp_test1/opp_set_FNwoGIs60_fullDataset_test1CorrNew.h5', key='data',mode='w')

#Explainer for proponents indices FNs 
from torch_geometric.explain import Explainer, GNNExplainer
explainer = Explainer(
        model=modell, # model to explain
        algorithm=GNNExplainer(epochs=50), # explainer type with number of epochs
        #explainer_config=dict( # parameters associated to the explainer
        explanation_type='model',
        node_mask_type=None,
        edge_mask_type='object',#None,
        #),
        model_config=dict( # charachteristics of the model we explain
            mode='multiclass_classification',
            task_level='graph',
            return_type='log_probs',
        ),
        threshold_config=dict( # treshold strategy
            threshold_type='topk',
            value=7,
        )
    )
list_SMexpl=[]


plot_counter = 0  # Initialize a counter for plotting
#####Important to modify train dataset to correspond modified indices! BUGG
train_datasetm = jet_graph_dataset[train_idx]
for i in ids_prop:
    
    g = train_datasetm[i]
    g = g.to(device)
    #print(f'\nJetGraph {i}')
    #print(g)
    #print(g.x)
    #print(g.y)
    #plot_jet_graph(g)
    #plt.show()
    # Check if your node features tensor requires gradients
    g.x.requires_grad_(True)

    explanation = explainer(g.x, g.edge_index, mini_batch=g) # train on the single instance
    g_exp = explanation.get_explanation_subgraph() # explain the instance in which all nodes and edges with zero attribution are masked out
    

    #plot the explanations
    # Plot and save the graph for the first three indices
    # if plot_counter < 3:
    #     plt.figure()
    #     plot_jet_graph(
    #         g , 
    #         angle=30, 
    #         elev=30, 
    #         color_layers=True, 
    #         energy_is_size=True,  
    #         save_to_path=f'Prop_Opp_dataAnalysis/model0/FN_propopp_test0/FNProp_graph_{i}.png')  # Save the graph with a unique filename
    #     #plt.close()
    #     plt.figure()
    #     plot_jet_graph(
    #         g_exp , 
    #         angle=30, 
    #         elev=30, 
    #         color_layers=True, 
    #         energy_is_size=True,  
    #         save_to_path=f'Prop_Opp_dataAnalysis/model0/FN_propopp_test0/FNProp_graphExpl_{i}.png')  # Save the graph with a unique filename
    #     #plt.close()
    #     plot_counter += 1  # Increment the plotting counter
    # plot_jet_graph(
    #     g_compl, 
    #     angle=30, 
    #     elev=30, 
    #     color_layers=True, 
    #     energy_is_size=True,  
    #     save_to_path=False)
    # #g_exp2
    # #g_exp3
    # plt.show()
    list_SMexpl.append(g_exp)


    
from utils import stats_to_pandasSM
dataset_SMexpl = SMDataset(data_list=list_SMexpl)



print(dataset_SMexpl[0])

#store SM datasets
dfa = stats_to_pandasSM(dataset_SMexpl)



dfa.to_hdf('Prop_Opp_dataAnalysis/model1/FN_propopp_test1/prop_dataset_SMexpl_FNwoGIsCorrNew.h5',key='data',mode='w')

print("Done with FN props!")



# FPs
test_influence_indices=idx_fpWP99
test_influence_indices
pipeline.run_captum(test_influence_indices=test_influence_indices)
idx_proponents, idx_opponents = pipeline.display_results(angle=30, elev=10, save_to_dir=False)
import itertools
id_list1 = list(itertools.chain.from_iterable(idx_proponents.tolist()))
id_list2 = list(itertools.chain.from_iterable(idx_opponents.tolist()))
#print(id_list1)
#print(id_list2)

import matplotlib.pyplot as plt
import pandas as pd
# Create a pandas DataFrame from the ID list
df1 = pd.DataFrame(id_list1, columns=['IDs'])
df2 = pd.DataFrame(id_list2, columns=['IDs'])

# Group the DataFrame by IDs and count the frequency
id_counts1 = df1['IDs'].value_counts().sort_index()
id_counts2 = df2['IDs'].value_counts().sort_index()

ids_prop = id_counts1.index.tolist()
##print(ids_prop)
ids_opp = id_counts2.index.tolist()
#print(ids_opp)

from utils import dataset_to_pandas
#####Important to modify dataset to correspond modified indices! BUGG
jetdatasetm = jet_graph_dataset[train_idx]
df1 = dataset_to_pandas(jetdatasetm[ids_prop])
df2 = dataset_to_pandas(jetdatasetm[ids_opp])
df1.to_hdf('Prop_Opp_dataAnalysis/model1/FP_propopp_test1/prop_set_FPwoGIs60_fullDataset_test1CorrNew.h5',key='data',mode='w')
df2.to_hdf('Prop_Opp_dataAnalysis/model1/FP_propopp_test1/opp_set_FPwoGIs60_fullDataset_test1CorrNew.h5', key='data',mode='w')

#Explainer for proponents indices FNs 
from torch_geometric.explain import Explainer, GNNExplainer
explainer = Explainer(
        model=modell, # model to explain
        algorithm=GNNExplainer(epochs=50), # explainer type with number of epochs
        #explainer_config=dict( # parameters associated to the explainer
        explanation_type='model',
        node_mask_type=None,
        edge_mask_type='object',#None,
        #),
        model_config=dict( # charachteristics of the model we explain
            mode='multiclass_classification',
            task_level='graph',
            return_type='log_probs',
        ),
        threshold_config=dict( # treshold strategy
            threshold_type='topk',
            value=7,
        )
    )
list_SMexpl=[]



#####Important to modify train dataset to correspond modified indices! BUGG
train_datasetm = jet_graph_dataset[train_idx]
for i in ids_prop:
    
    g = train_datasetm[i]
    g = g.to(device)
    #print(f'\nJetGraph {i}')
    #print(g)
    #print(g.x)
    #print(g.y)
    #plot_jet_graph(g)
    #plt.show()
    # Check if your node features tensor requires gradients
    g.x.requires_grad_(True)
    

    explanation = explainer(g.x, g.edge_index, mini_batch=g) # train on the single instance
    g_exp = explanation.get_explanation_subgraph() # explain the instance in which all nodes and edges with zero attribution are masked out
    

    #plot the explanations
    # Plot and save the graph for the first three indices
    # if plot_counter < 3:
    #     plt.figure()
    #     plot_jet_graph(
    #         g , 
    #         angle=30, 
    #         elev=30, 
    #         color_layers=True, 
    #         energy_is_size=True,  
    #         save_to_path=f'Prop_Opp_dataAnalysis/model0/FP_propopp_test0/FPProp_graph_{i}.png')  # Save the graph with a unique filename
    #     #plt.close()
    #     plt.figure()
    #     plot_jet_graph(
    #         g_exp , 
    #         angle=30, 
    #         elev=30, 
    #         color_layers=True, 
    #         energy_is_size=True,  
    #         save_to_path=f'Prop_Opp_dataAnalysis/model0/FP_propopp_test0/FPProp_graphExpl_{i}.png')  # Save the graph with a unique filename
    #     #plt.close()
    #     plot_counter += 1  # Increment the plotting counter
    list_SMexpl.append(g_exp)


    
from utils import stats_to_pandasSM
dataset_SMexpl = SMDataset(data_list=list_SMexpl)



print(dataset_SMexpl[0])

#store SM datasets
dfa = stats_to_pandasSM(dataset_SMexpl)



dfa.to_hdf('Prop_Opp_dataAnalysis/model1/FP_propopp_test1/prop_dataset_SMexpl_FPwoGIsCorrNew.h5',key='data',mode='w')

print("Done with FP props!")


# TPs
test_influence_indices=idx_tpWP99
test_influence_indices
pipeline.run_captum(test_influence_indices=test_influence_indices)
idx_proponents, idx_opponents = pipeline.display_results(angle=30, elev=10, save_to_dir=False)
import itertools
id_list1 = list(itertools.chain.from_iterable(idx_proponents.tolist()))
id_list2 = list(itertools.chain.from_iterable(idx_opponents.tolist()))
#print(id_list1)
#print(id_list2)

import matplotlib.pyplot as plt
import pandas as pd
# Create a pandas DataFrame from the ID list
df1 = pd.DataFrame(id_list1, columns=['IDs'])
df2 = pd.DataFrame(id_list2, columns=['IDs'])

# Group the DataFrame by IDs and count the frequency
id_counts1 = df1['IDs'].value_counts().sort_index()
id_counts2 = df2['IDs'].value_counts().sort_index()

ids_prop = id_counts1.index.tolist()
#print(ids_prop)
ids_opp = id_counts2.index.tolist()
#print(ids_opp)

from utils import dataset_to_pandas
#####Important to modify dataset to correspond modified indices! BUGG
jetdatasetm = jet_graph_dataset[train_idx]
df1 = dataset_to_pandas(jetdatasetm[ids_prop])
df2 = dataset_to_pandas(jetdatasetm[ids_opp])
df1.to_hdf('Prop_Opp_dataAnalysis/model1/TP_propopp_test1/prop_set_TPwoGIs60_fullDataset_test1CorrNew.h5',key='data',mode='w')
df2.to_hdf('Prop_Opp_dataAnalysis/model1/TP_propopp_test1/opp_set_TPwoGIs60_fullDataset_test1CorrNew.h5', key='data',mode='w')

#Explainer for proponents indices FNs 
from torch_geometric.explain import Explainer, GNNExplainer
explainer = Explainer(
        model=modell, # model to explain
        algorithm=GNNExplainer(epochs=50), # explainer type with number of epochs
        #explainer_config=dict( # parameters associated to the explainer
        explanation_type='model',
        node_mask_type=None,
        edge_mask_type='object',#None,
        #),
        model_config=dict( # charachteristics of the model we explain
            mode='multiclass_classification',
            task_level='graph',
            return_type='log_probs',
        ),
        threshold_config=dict( # treshold strategy
            threshold_type='topk',
            value=7,
        )
    )
list_SMexpl=[]



#####Important to modify train dataset to correspond modified indices! BUGG
train_datasetm = jet_graph_dataset[train_idx]
for i in ids_opp:
    
    g = train_datasetm[i]
    g = g.to(device)
    #print(f'\nJetGraph {i}')
    #print(g)
    #print(g.x)
    #print(g.y)
    #plot_jet_graph(g)
    #plt.show()
    # Check if your node features tensor requires gradients
    g.x.requires_grad_(True)
    

    explanation = explainer(g.x, g.edge_index, mini_batch=g) # train on the single instance
    g_exp = explanation.get_explanation_subgraph() # explain the instance in which all nodes and edges with zero attribution are masked out
    

    #plot the explanations
    # Plot and save the graph for the first three indices
    # if plot_counter < 3:
    #     plt.figure()
    #     plot_jet_graph(
    #         g , 
    #         angle=30, 
    #         elev=30, 
    #         color_layers=True, 
    #         energy_is_size=True,  
    #         save_to_path=f'Prop_Opp_dataAnalysis/model0/TP_propopp_test0/TPOpp_graph_{i}.png')  # Save the graph with a unique filename
    #     #plt.close()
    #     plt.figure()
    #     plot_jet_graph(
    #         g_exp , 
    #         angle=30, 
    #         elev=30, 
    #         color_layers=True, 
    #         energy_is_size=True,  
    #         save_to_path=f'Prop_Opp_dataAnalysis/model0/TP_propopp_test0/TPOpp_graphExpl_{i}.png')  # Save the graph with a unique filename
    #     #plt.close()
    #     plot_counter += 1  # Increment the plotting counter
    list_SMexpl.append(g_exp)


    
from utils import stats_to_pandasSM
dataset_SMexpl = SMDataset(data_list=list_SMexpl)



print(dataset_SMexpl[0])

#store SM datasets
dfa = stats_to_pandasSM(dataset_SMexpl)



dfa.to_hdf('Prop_Opp_dataAnalysis/model1/TP_propopp_test1/opp_dataset_SMexpl_TPwoGIsCorrNew.h5',key='data',mode='w')
print("Done with TP opps!")

# TNs
test_influence_indices=idx_tnWP99
test_influence_indices
pipeline.run_captum(test_influence_indices=test_influence_indices)
idx_proponents, idx_opponents = pipeline.display_results(angle=30, elev=10, save_to_dir=False)
import itertools
id_list1 = list(itertools.chain.from_iterable(idx_proponents.tolist()))
id_list2 = list(itertools.chain.from_iterable(idx_opponents.tolist()))
#print(id_list1)
#print(id_list2)

import matplotlib.pyplot as plt
import pandas as pd
# Create a pandas DataFrame from the ID list
df1 = pd.DataFrame(id_list1, columns=['IDs'])
df2 = pd.DataFrame(id_list2, columns=['IDs'])

# Group the DataFrame by IDs and count the frequency
id_counts1 = df1['IDs'].value_counts().sort_index()
id_counts2 = df2['IDs'].value_counts().sort_index()

ids_prop = id_counts1.index.tolist()
#print(ids_prop)
ids_opp = id_counts2.index.tolist()
#print(ids_opp)

from utils import dataset_to_pandas
#####Important to modify dataset to correspond modified indices! BUGG
jetdatasetm = jet_graph_dataset[train_idx]
df1 = dataset_to_pandas(jetdatasetm[ids_prop])
df2 = dataset_to_pandas(jetdatasetm[ids_opp])
df1.to_hdf('Prop_Opp_dataAnalysis/model1/TN_propopp_test1/prop_set_TNwoGIs60_fullDataset_test1CorrNew.h5',key='data',mode='w')
df2.to_hdf('Prop_Opp_dataAnalysis/model1/TN_propopp_test1/opp_set_TNwoGIs60_fullDataset_test1CorrNew.h5', key='data',mode='w')

#Explainer for proponents indices FNs 
from torch_geometric.explain import Explainer, GNNExplainer
explainer = Explainer(
        model=modell, # model to explain
        algorithm=GNNExplainer(epochs=50), # explainer type with number of epochs
        #explainer_config=dict( # parameters associated to the explainer
        explanation_type='model',
        node_mask_type=None,
        edge_mask_type='object',#None,
        #),
        model_config=dict( # charachteristics of the model we explain
            mode='multiclass_classification',
            task_level='graph',
            return_type='log_probs',
        ),
        threshold_config=dict( # treshold strategy
            threshold_type='topk',
            value=7,
        )
    )
list_SMexpl=[]



#####Important to modify train dataset to correspond modified indices! BUGG
train_datasetm = jet_graph_dataset[train_idx]
for i in ids_opp:
    
    g = train_datasetm[i]
    g = g.to(device)
    #print(f'\nJetGraph {i}')
    #print(g)
    #print(g.x)
    #print(g.y)
    #plot_jet_graph(g)
    #plt.show()
    # Check if your node features tensor requires gradients
    g.x.requires_grad_(True)

    explanation = explainer(g.x, g.edge_index, mini_batch=g) # train on the single instance
    g_exp = explanation.get_explanation_subgraph() # explain the instance in which all nodes and edges with zero attribution are masked out
    

    #plot the explanations
    # Plot and save the graph for the first three indices
    # if plot_counter < 3:
    #     plt.figure()
    #     plot_jet_graph(
    #         g , 
    #         angle=30, 
    #         elev=30, 
    #         color_layers=True, 
    #         energy_is_size=True,  
    #         save_to_path=f'Prop_Opp_dataAnalysis/model0/TN_propopp_test0/TNOpp_graph_{i}.png')  # Save the graph with a unique filename
    #     #plt.close()
    #     plt.figure()
    #     plot_jet_graph(
    #         g_exp , 
    #         angle=30, 
    #         elev=30, 
    #         color_layers=True, 
    #         energy_is_size=True,  
    #         save_to_path=f'Prop_Opp_dataAnalysis/model0/TN_propopp_test0/TNOpp_graphExpl_{i}.png')  # Save the graph with a unique filename
    #     #plt.close()
    #     plot_counter += 1  # Increment the plotting counter
    list_SMexpl.append(g_exp)


    
from utils import stats_to_pandasSM
dataset_SMexpl = SMDataset(data_list=list_SMexpl)



print(dataset_SMexpl[0])

#store SM datasets
dfa = stats_to_pandasSM(dataset_SMexpl)



dfa.to_hdf('Prop_Opp_dataAnalysis/model1/TN_propopp_test1/opp_dataset_SMexpl_TNwoGIsCorrNew.h5',key='data',mode='w')

print("Done with TN opps!")

print("Full Pipline's end.")



