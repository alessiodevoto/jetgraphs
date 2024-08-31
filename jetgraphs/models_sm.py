import torch
from torch.nn import Dropout, Linear, BCEWithLogitsLoss, functional as F
from torchmetrics.functional import accuracy
#from pytorch_lightning import LightningModule
from lightning import LightningModule
from sklearn import metrics
import numpy as np

from torch.optim import Adam

from torch_geometric.nn import LayerNorm, BatchNorm
from torch_geometric.nn import GCNConv, GATv2Conv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

from abc import abstractmethod

import torch
from torch_geometric.nn import LayerNorm, BatchNorm, ARMAConv, GCNConv, GATv2Conv, ChebConv, global_mean_pool, global_max_pool, global_add_pool, PositionalEncoding

from torch.optim.lr_scheduler import ReduceLROnPlateau




### saves grads at best epoch
default_node_features = 4

class BaseJetGraphGCN(LightningModule):
    def __init__(self, hidden_channels, node_feat_size=None, use_edge_attr=False, learning_rate=0.001, loss_func=BCEWithLogitsLoss()):
        super(BaseJetGraphGCN, self).__init__()
        #torch.manual_seed(12345)

        #self.automatic_optimization = False  # Disable automatic optimization
        self.hidden_channels = hidden_channels
        self.node_features_size = node_feat_size if node_feat_size else default_node_features
        self.use_edge_attr = use_edge_attr
        self.loss = loss_func
        self.lr = learning_rate
        

        # Registering model parameters and hyperparameters for checkpointing
        self.save_hyperparameters(ignore=['loss_func'])

    @abstractmethod
    def forward(self, x, edge_index, mini_batch):#switch for GNNExplainer #(self, mini_batch):#
        pass
    def compute_loss(self, out, target):
        l = torch.nn.BCEWithLogitsLoss()
        return l(out, target)

    def training_step(self, batch, batch_idx):
        out = self(batch)  # Perform a single forward pass
        loss = self.compute_loss(out, batch.y.unsqueeze(1).float())  # Compute loss

        

        self.log('train_loss', loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch.y.unsqueeze(1).float().size(0))
        return loss

    

    def validation_step(self, batch, batch_idx):
        out = self(batch)  # Perform a single forward pass.
        predictions = torch.sigmoid(out).detach().cpu().numpy()
        labels = batch.y.unsqueeze(1).float()


        # Loss.
        loss = self.compute_loss(out, labels)  # Compute the loss.
        # Accuracy.
        acc = metrics.accuracy_score(labels.detach().cpu().numpy(), np.round(predictions))
        # F1 score.
        f1_score = metrics.f1_score(labels.detach().cpu().numpy(), np.round(predictions))
        # AUC.
        roc_auc = metrics.roc_auc_score(labels.detach().cpu().numpy(), predictions)

        # Log loss and metric
        self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=labels.size(0))
        self.log('val_accuracy', acc, on_step=False, on_epoch=True, sync_dist=True, batch_size=labels.size(0))
        self.log('val_f1_score', f1_score, on_step=False, on_epoch=True, sync_dist=True, batch_size=labels.size(0))
        self.log('val_roc_auc', roc_auc, on_step=False,  on_epoch=True, sync_dist=True, batch_size=labels.size(0))

        

        return loss

    def predict_step(self, batch, batch_idx):
        out = self(batch)  # Perform a single forward pass.
        predictions = torch.sigmoid(out).detach().cpu().numpy()
        #labels = batch.y.unsqueeze(1).float()
        return predictions.tolist()
    
    

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            }
        }

class ArmaSM(BaseJetGraphGCN):
    def __init__(self,
                 hidden_channels,
                 #batch_size=64,
                 node_feat_size=None,
                 learning_rate=0.001,
                 use_edge_attr=False,
                 loss_func=torch.nn.BCEWithLogitsLoss()):
        super(ArmaSM, self).__init__(hidden_channels=hidden_channels, node_feat_size=node_feat_size,
                                   learning_rate=learning_rate, use_edge_attr=use_edge_attr, loss_func=loss_func)
        torch.manual_seed(12345)
        self.norm = BatchNorm(self.node_features_size)
        self.conv1 = ARMAConv(self.node_features_size, self.hidden_channels, num_stacks= 3)
        self.conv2 = ARMAConv(self.hidden_channels, self.hidden_channels, num_stacks= 3)
        self.conv3 = ARMAConv(self.hidden_channels, self.hidden_channels, num_stacks= 3)
        self.lin0 = Linear(self.hidden_channels, self.hidden_channels)
        self.lin = Linear(self.hidden_channels, 1)
        #self.all_graph_gradients = []  # Initialize the list to store gradients
        

    def forward(self, x, edge_index, mini_batch):#switch for GNNExplainer #(self, mini_batch):#
        # 0. Unbatch elements in mini batch
        x, edge_index, batch = mini_batch.x, mini_batch.edge_index, mini_batch.batch
        edge_attr = mini_batch.edge_attr if self.use_edge_attr else None
        #x.requires_grad_(True)  # Ensure gradients can be computed

        # 1. Apply Batch normalization
        x = self.norm(x)

        # 2. Obtain node embeddings
        x = self.conv1(x, edge_index, edge_weight=edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight=edge_attr)
        x = F.relu(x)
        x = self.conv3(x, edge_index, edge_weight=edge_attr)
        x = F.relu(x)

        # 3. Readout layer
        y1 = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        y2 = global_max_pool(x, batch)  # [batch_size, hidden_channels]
        y3 = global_add_pool(x, batch)  # [batch_size, hidden_channels]
        z = self.lin0(y1 + y2 + y3).relu()
        
        # 4. Apply a final classifier
        z = F.dropout(z, p=0.5, training=self.training)
        z = self.lin(z)
        

        return z

# ### saves grads at best epoch
# default_node_features = 4

# class BaseJetGraphGCN(LightningModule):
#     def __init__(self, hidden_channels, node_feat_size=None, use_edge_attr=False, learning_rate=0.001, loss_func=BCEWithLogitsLoss()):
#         super(BaseJetGraphGCN, self).__init__()
#         self.hidden_channels = hidden_channels
#         self.node_features_size = node_feat_size if node_feat_size else default_node_features
#         self.use_edge_attr = use_edge_attr
#         self.loss = loss_func
#         self.lr = learning_rate
#         self.current_epoch_gradients = []
#         self.best_epoch_gradients = []
#         self.best_epoch = -1
#         self.best_val_loss = float('inf')
#         self.validation_outputs = []  # Initialize validation_outputs

#         # Registering model parameters and hyperparameters for checkpointing
#         self.save_hyperparameters(ignore=['loss_func'])

#     @abstractmethod
#     def forward(self, x, edge_index, mini_batch):#switch for GNNExplainer #(self, mini_batch):#
#         pass
    
#     def compute_loss(self, out, target):
#         l = torch.nn.BCEWithLogitsLoss()
#         return l(out, target)

#     def training_step(self, batch, batch_idx):
#         out = self(batch)  # Perform a single forward pass
#         loss = self.compute_loss(out, batch.y.unsqueeze(1).float())  # Compute loss

#         # Register hook on the input batch.x
#         batch.x.retain_grad()  # Ensure gradients are retained
#         batch.x.register_hook(lambda grad: self.save_gradient(grad, batch.batch))

#         self.log('train_loss', loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch.y.unsqueeze(1).float().size(0))
#         return loss

#     def save_gradient(self, grad, batch_index):
#         # Ensure current_epoch_gradients is initialized
#         if not hasattr(self, 'current_epoch_gradients'):
#             self.current_epoch_gradients = []

#         # Loop through each entry in the batch
#         for idx in range(batch_index.max().item() + 1):
#             # Extract the gradients for the current graph entry
#             entry_gradients = grad[batch_index == idx]

#             # Append the gradients for the current entry to the list
#             self.current_epoch_gradients.append(entry_gradients.detach().cpu().numpy())

#     def validation_step(self, batch, batch_idx):
#         out = self(batch)  # Perform a single forward pass.
#         predictions = torch.sigmoid(out).detach().cpu().numpy()
#         labels = batch.y.unsqueeze(1).float()

#         # Loss.
#         loss = self.compute_loss(out, labels)  # Compute the loss.
#         # Accuracy.
#         acc = metrics.accuracy_score(labels.detach().cpu().numpy(), np.round(predictions))
#         # F1 score.
#         f1_score = metrics.f1_score(labels.detach().cpu().numpy(), np.round(predictions))
#         # AUC.
#         roc_auc = metrics.roc_auc_score(labels.detach().cpu().numpy(), predictions)

#         # Log loss and metric
#         self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=labels.size(0))
#         self.log('val_accuracy', acc, on_step=False, on_epoch=True, sync_dist=True, batch_size=labels.size(0))
#         self.log('val_f1_score', f1_score, on_step=False, on_epoch=True, sync_dist=True, batch_size=labels.size(0))
#         self.log('val_roc_auc', roc_auc, on_step=False,  on_epoch=True, sync_dist=True, batch_size=labels.size(0))

#         self.validation_outputs.append({'val_loss': loss})  # Store validation outputs

#         return {'val_loss': loss}

#     def on_validation_epoch_end(self):
#         # Get the average validation loss for the epoch
#         avg_val_loss = torch.stack([x['val_loss'] for x in self.validation_outputs]).mean()

#         # Check if the current epoch is the best
#         if avg_val_loss < self.best_val_loss:
#             self.best_val_loss = avg_val_loss
#             self.best_epoch = self.current_epoch
#             self.best_epoch_gradients = self.current_epoch_gradients.copy()

#         # Clear the current gradients and validation outputs
#         self.current_epoch_gradients = []
#         self.validation_outputs = []

#     def on_train_epoch_end(self):
#         # Clear the current gradients at the end of training epoch
#         self.current_epoch_gradients = []

#     def predict_step(self, batch, batch_idx):
#         out = self(batch)  # Perform a single forward pass.
#         predictions = torch.sigmoid(out).detach().cpu().numpy()
#         return predictions.tolist()
    
#     def configure_optimizers(self):
#         optimizer = Adam(self.parameters(), lr=self.lr)
#         scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
#         return {
#             'optimizer': optimizer,
#             'lr_scheduler': {
#                 'scheduler': scheduler,
#                 'monitor': 'val_loss',
#             }
#         }

# class ArmaSM(BaseJetGraphGCN):
#     def __init__(self, hidden_channels, node_feat_size=None, learning_rate=0.001, use_edge_attr=False, loss_func=torch.nn.BCEWithLogitsLoss()):
#         super(ArmaSM, self).__init__(hidden_channels=hidden_channels, node_feat_size=node_feat_size,
#                                    learning_rate=learning_rate, use_edge_attr=use_edge_attr, loss_func=loss_func)
#         torch.manual_seed(12345)
#         self.norm = BatchNorm(self.node_features_size)
#         self.conv1 = ARMAConv(self.node_features_size, self.hidden_channels, num_stacks= 3)
#         self.conv2 = ARMAConv(self.hidden_channels, self.hidden_channels, num_stacks= 3)
#         self.conv3 = ARMAConv(self.hidden_channels, self.hidden_channels, num_stacks= 3)
#         self.lin0 = Linear(self.hidden_channels, self.hidden_channels)
#         self.lin = Linear(self.hidden_channels, 1)

#     def forward(self, x, edge_index, mini_batch):#switch for GNNExplainer #(self, mini_batch):#
#         x, edge_index, batch = mini_batch.x, mini_batch.edge_index, mini_batch.batch
#         edge_attr = mini_batch.edge_attr if self.use_edge_attr else None
#         x.requires_grad_(True)  # Ensure gradients can be computed

#         # 1. Apply Batch normalization
#         x = self.norm(x)

#         # 2. Obtain node embeddings
#         x = self.conv1(x, edge_index, edge_weight=edge_attr)
#         x = F.relu(x)
#         x = self.conv2(x, edge_index, edge_weight=edge_attr)
#         x = F.relu(x)
#         x = self.conv3(x, edge_index, edge_weight=edge_attr)
#         x = F.relu(x)

#         # 3. Readout layer
#         y1 = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
#         y2 = global_max_pool(x, batch)  # [batch_size, hidden_channels]
#         y3 = global_add_pool(x, batch)  # [batch_size, hidden_channels]
#         z = self.lin0(y1 + y2 + y3).relu()
        
#         # 4. Apply a final classifier
#         z = F.dropout(z, p=0.5, training=self.training)
#         z = self.lin(z)
        
#         return z

