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
        self.hidden_channels = hidden_channels
        self.node_features_size = node_feat_size if node_feat_size else default_node_features
        self.use_edge_attr = use_edge_attr
        self.loss = loss_func
        self.lr = learning_rate
        self.current_epoch_gradients = []
        self.best_epoch_gradients = []
        self.best_epoch = -1
        self.best_val_loss = float('inf')
        self.validation_outputs = []  # Initialize validation_outputs

        # Registering model parameters and hyperparameters for checkpointing
        self.save_hyperparameters(ignore=['loss_func'])

    @abstractmethod
    def forward(self, mini_batch):#, x, edge_index, mini_batch):#switch for GNNExplainer #(self, mini_batch):#
        pass
    
    def compute_loss(self, out, target):
        l = torch.nn.BCEWithLogitsLoss()
        return l(out, target)

    def training_step(self, batch, batch_idx):
        out = self(batch)  # Perform a single forward pass
        loss = self.compute_loss(out, batch.y.unsqueeze(1).float())  # Compute loss

        # Register hook on the input batch.x
        batch.x.retain_grad()  # Ensure gradients are retained
        batch.x.register_hook(lambda grad: self.save_gradient(grad, batch.batch))

        self.log('train_loss', loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch.y.unsqueeze(1).float().size(0))
        return loss

    def save_gradient(self, grad, batch_index):
        # Ensure current_epoch_gradients is initialized
        if not hasattr(self, 'current_epoch_gradients'):
            self.current_epoch_gradients = []

        # Loop through each entry in the batch
        for idx in range(batch_index.max().item() + 1):
            # Extract the gradients for the current graph entry
            entry_gradients = grad[batch_index == idx]

            # Append the gradients for the current entry to the list
            self.current_epoch_gradients.append(entry_gradients.detach().cpu().numpy())

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
        try:
         roc_auc = metrics.roc_auc_score(labels.detach().cpu().numpy(), predictions)
         self.log('val_roc_auc', roc_auc, on_step=False,  on_epoch=True, sync_dist=True, batch_size=labels.size(0), logger=True)
        except ValueError:
            pass
        # Log loss and metric
        self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=labels.size(0), logger=True)
        self.log('val_accuracy', acc, on_step=False, on_epoch=True, sync_dist=True, batch_size=labels.size(0), logger=True)
        self.log('val_f1_score', f1_score, on_step=False, on_epoch=True, sync_dist=True, batch_size=labels.size(0), logger=True)
        

        self.validation_outputs.append({'val_loss': loss})  # Store validation outputs

        return {'val_loss': loss}
    
    def test_step(self, batch, batch_idx):
        out = self(batch)
        predictions = torch.sigmoid(out).detach().cpu().numpy()
        labels = batch.y.unsqueeze(1).float()

        loss = self.compute_loss(out, labels, batch.totweight.unsqueeze(1).float())

        acc = metrics.accuracy_score(labels.detach().cpu().numpy(), np.round(predictions), sample_weight=batch.totweight.float().detach().cpu().numpy())
                                      
        f1_score = metrics.f1_score(labels.detach().cpu().numpy(), np.round(predictions), sample_weight=batch.totweight.float().detach().cpu().numpy())
                                    
        roc_auc = metrics.roc_auc_score(labels.detach().cpu().numpy(), predictions, sample_weight=batch.totweight.float().detach().cpu().numpy())
        
        batch_size = labels.size(0)  # Get the batch size

        self.log('test_loss', loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size, logger=True)
        self.log('test_accuracy', acc, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size, logger=True)
        self.log('test_f1_score', f1_score, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size, logger=True)
        self.log('test_roc_auc', roc_auc, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size, logger=True)

        return {'test_loss': loss}

    def on_validation_epoch_end(self):
        # Get the average validation loss for the epoch
        avg_val_loss = torch.stack([x['val_loss'] for x in self.validation_outputs]).mean()

        # Check if the current epoch is the best
        if avg_val_loss < self.best_val_loss:
            self.best_val_loss = avg_val_loss
            self.best_epoch = self.current_epoch
            self.best_epoch_gradients = self.current_epoch_gradients.copy()

        # Clear the current gradients and validation outputs
        self.current_epoch_gradients = []
        self.validation_outputs = []

    def on_train_epoch_end(self):
        # Clear the current gradients at the end of training epoch
        self.current_epoch_gradients = []

    def predict_step(self, batch, batch_idx):
        out = self(batch)  # Perform a single forward pass.
        predictions = torch.sigmoid(out).detach().cpu().numpy()
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

class Arma(BaseJetGraphGCN):
    def __init__(self, hidden_channels, node_feat_size=None, learning_rate=0.001, use_edge_attr=False, loss_func=torch.nn.BCEWithLogitsLoss()):
        super(Arma, self).__init__(hidden_channels=hidden_channels, node_feat_size=node_feat_size,
                                   learning_rate=learning_rate, use_edge_attr=use_edge_attr, loss_func=loss_func)
        torch.manual_seed(12345)
        self.norm = BatchNorm(self.node_features_size)
        self.conv1 = ARMAConv(self.node_features_size, self.hidden_channels, num_stacks= 3)
        self.conv2 = ARMAConv(self.hidden_channels, self.hidden_channels, num_stacks= 3)
        self.conv3 = ARMAConv(self.hidden_channels, self.hidden_channels, num_stacks= 3)
        self.lin0 = Linear(self.hidden_channels, self.hidden_channels)
        self.lin = Linear(self.hidden_channels, 1)

    def forward(self, mini_batch):#, x, edge_index, mini_batch):#switch for GNNExplainer #(self, mini_batch):#
        x, edge_index, batch = mini_batch.x, mini_batch.edge_index, mini_batch.batch
        edge_attr = mini_batch.edge_attr if self.use_edge_attr else None
        x.requires_grad_(True)  # Ensure gradients can be computed

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





class Shallow_GCN(BaseJetGraphGCN):
    def __init__(self,
                 hidden_channels,
                 #batch_size=64,
                 node_feat_size=None,
                 learning_rate=0.001,
                 use_edge_attr=False,
                 loss_func=torch.nn.BCEWithLogitsLoss()):
        super(Shallow_GCN, self).__init__(hidden_channels=hidden_channels, node_feat_size=node_feat_size,
                                   learning_rate=learning_rate, use_edge_attr=use_edge_attr, loss_func=loss_func)
        torch.manual_seed(12345)

        self.norm = BatchNorm(self.node_features_size)
        self.conv1 = GCNConv(self.node_features_size, self.hidden_channels)
        self.conv2 = GCNConv(self.hidden_channels, self.hidden_channels)
        self.conv3 = GCNConv(self.hidden_channels, self.hidden_channels)
        self.conv4 = GCNConv(self.hidden_channels, self.hidden_channels)
        self.lin = Linear(self.hidden_channels, 1)

    def forward(self, mini_batch):#,x, edge_index, mini_batch):
        # 0. Unbatch elements in mini batch
        x, edge_index, batch = mini_batch.x, mini_batch.edge_index, mini_batch.batch

        edge_attr = mini_batch.edge_attr if self.use_edge_attr else None

        x.requires_grad_(True)  # Ensure gradients can be computed
        
        # 1. Apply Batch normalization
        x = self.norm(x)

        # 2. Obtain node embeddings
        x = self.conv1(x, edge_index, edge_weight=edge_attr)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weight=edge_attr)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_weight=edge_attr)
        x = x.relu()
        x = self.conv4(x, edge_index, edge_weight=edge_attr)

        # 3. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 4. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


class Residual_GCN(BaseJetGraphGCN):
    def __init__(self,
                 hidden_channels,
                 #batch_size=64,
                 node_feat_size=None,
                 learning_rate=0.001,
                 use_edge_attr=False,
                 loss_func=torch.nn.BCEWithLogitsLoss()):
        super(Residual_GCN, self).__init__(hidden_channels=hidden_channels, node_feat_size=node_feat_size,
                                   learning_rate=learning_rate, use_edge_attr=use_edge_attr, loss_func=loss_func)
        torch.manual_seed(12345)
        self.norm = BatchNorm(self.node_features_size)
        self.norm_residual = BatchNorm(self.hidden_channels)
        self.conv1 = GCNConv(self.node_features_size, self.hidden_channels)
        self.conv2 = GCNConv(self.hidden_channels, self.hidden_channels)
        self.conv3 = GCNConv(self.hidden_channels, self.hidden_channels)
        self.conv4 = GCNConv(self.hidden_channels, self.hidden_channels)
        self.lin = Linear(self.hidden_channels, 1)

    def forward(self, mini_batch):
        # 0. Unbatch elements in mini batch
        x, edge_index, batch = mini_batch.x, mini_batch.edge_index, mini_batch.batch

        edge_attr = mini_batch.edge_attr if self.use_edge_attr else None

        # 1. Apply Batch normalization
        x = self.norm(x)

        # 2. Obtain node embeddings
        x = self.conv1(x, edge_index, edge_weight=edge_attr)
        x = self.norm_residual(x.relu() + x)
        x = self.conv2(x, edge_index, edge_weight=edge_attr)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_weight=edge_attr)
        x = self.norm_residual(x.relu() + x)
        x = self.conv4(x, edge_index, edge_weight=edge_attr)

        # 3. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 4. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


class Residual_Arma(BaseJetGraphGCN):
    def __init__(self,
                 hidden_channels,
                 #batch_size=64,
                 node_feat_size=None,
                 learning_rate=0.001,
                 use_edge_attr=False,
                 loss_func=torch.nn.BCEWithLogitsLoss()):
        super(Residual_Arma, self).__init__(hidden_channels=hidden_channels, node_feat_size=node_feat_size,
                                   learning_rate=learning_rate, use_edge_attr=use_edge_attr, loss_func=loss_func)
        torch.manual_seed(12345)
        self.norm = BatchNorm(self.node_features_size)
        self.norm_residual = BatchNorm(self.hidden_channels)
        self.conv1 = ARMAConv(self.node_features_size, self.hidden_channels, num_stacks= 3)
        self.conv2 = ARMAConv(self.hidden_channels, self.hidden_channels, num_stacks= 3)
        self.conv3 = ARMAConv(self.hidden_channels, self.hidden_channels, num_stacks= 3)
        self.conv4 = ARMAConv(self.hidden_channels, self.hidden_channels, num_stacks= 3)
        self.lin0 = Linear(self.hidden_channels, self.hidden_channels)
        self.lin = Linear(self.hidden_channels, 1)

    def forward(self, mini_batch):
        # 0. Unbatch elements in mini batch
        x, edge_index, batch = mini_batch.x, mini_batch.edge_index, mini_batch.batch

        edge_attr = mini_batch.edge_attr if self.use_edge_attr else None

        # 1. Apply Batch normalization
        x = self.norm(x)

        # 2. Obtain node embeddings
        x1 = self.conv1(x, edge_index, edge_weight=edge_attr)
        x11 = F.relu(x1)#, negative_slope=0.1)
        x2 = self.conv2(x11, edge_index, edge_weight=edge_attr)
        x22 = F.relu(x2)#, negative_slope=0.1)
        x3 = self.conv3(x22, edge_index, edge_weight=edge_attr)
        x33 = F.relu(x3)#, negative_slope=0.1)
        x4 = self.conv4(x33, edge_index, edge_weight=edge_attr)
        x44 = F.relu(x4)#, negative_slope=0.1)
        x = self.norm_residual(x44 + x33 + x22 + x11)

        # 3. Readout layer
        y1 = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        y2 = global_max_pool(x, batch)  # [batch_size, hidden_channels]
        y3 = global_add_pool(x, batch)  # [batch_size, hidden_channels]
        z = self.lin0(y1 + y2 + y3).relu()
        
        # 4. Apply a final classifier
        z = F.dropout(z, p=0.5, training=self.training)
        z = self.lin(z)
        

        return z


class Cheb(BaseJetGraphGCN):
    def __init__(self,
                 hidden_channels,
                 #batch_size=64,
                 node_feat_size=None,
                 learning_rate=0.001,
                 use_edge_attr=False,
                 loss_func=torch.nn.BCEWithLogitsLoss()):
        super(Cheb, self).__init__(hidden_channels=hidden_channels, node_feat_size=node_feat_size,
                                   learning_rate=learning_rate, use_edge_attr=use_edge_attr, loss_func=loss_func)
        torch.manual_seed(12345)
        self.norm = BatchNorm(self.node_features_size)
        self.conv1 = ChebConv(self.node_features_size, self.hidden_channels, K=4)
        self.conv2 = ChebConv(self.hidden_channels, self.hidden_channels, K=4)
        self.conv3 = ChebConv(self.hidden_channels, self.hidden_channels, K=4)
        self.conv4 = ChebConv(self.hidden_channels, self.hidden_channels, K=4)
        self.lin = Linear(self.hidden_channels, 1)

    def forward(self, mini_batch):
        # 0. Unbatch elements in mini batch
        x, edge_index, batch = mini_batch.x, mini_batch.edge_index, mini_batch.batch
        edge_attr = mini_batch.edge_attr if self.use_edge_attr else None

        # 1. Apply Batch normalization
        x = self.norm(x)

        # 2. Obtain node embeddings
        x = self.conv1(x, edge_index, edge_weight=edge_attr)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weight=edge_attr)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_weight=edge_attr)
        x = x.relu()
        x = self.conv4(x, edge_index, edge_weight=edge_attr)

        # 3. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 4. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x




class Residual_GAT(BaseJetGraphGCN):
    def __init__(self,
                 hidden_channels,
                 #batch_size=64,
                 node_feat_size=None,
                 learning_rate=0.001,
                 use_edge_attr=False,
                 loss_func=torch.nn.BCEWithLogitsLoss(), heads=1):
        super(Residual_GAT, self).__init__(hidden_channels=hidden_channels, node_feat_size=node_feat_size,
                                   learning_rate=learning_rate, use_edge_attr=use_edge_attr, loss_func=loss_func)
        torch.manual_seed(12345)
        self.norm = BatchNorm(self.node_features_size)
        self.norm_residual = BatchNorm(self.hidden_channels)
        self.conv1 = GATv2Conv(self.node_features_size, self.hidden_channels, edge_dim=1, concat=False, heads=heads)
        self.conv2 = GATv2Conv(self.hidden_channels, self.hidden_channels,edge_dim=1, concat=False, heads=heads)
        self.conv3 = GATv2Conv(self.hidden_channels, self.hidden_channels, edge_dim=1,concat=False, heads=heads)
        self.conv4 = GATv2Conv(self.hidden_channels, self.hidden_channels,edge_dim=1, concat=False, heads=heads)
        self.lin = Linear(self.hidden_channels, 1)

    def forward(self, mini_batch):
        # 0. Unbatch elements in mini batch
        x, edge_index, batch = mini_batch.x, mini_batch.edge_index, mini_batch.batch
        edge_attr = mini_batch.edge_attr if self.use_edge_attr else None

        # 1. Apply Batch normalization
        x = self.norm(x)

        # 2. Obtain node embeddings
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.norm_residual(x.relu() + x)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_attr=edge_attr)
        x = self.norm_residual(x.relu() + x)
        x = self.conv4(x, edge_index, edge_attr=edge_attr)

        # 3. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 4. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


class GAT(BaseJetGraphGCN):
    def __init__(self,
                 hidden_channels,
                 #batch_size=64,
                 node_feat_size=None,
                 learning_rate=0.001,
                 use_edge_attr=False,
                 loss_func=torch.nn.BCEWithLogitsLoss(), heads=7):
        super(GAT, self).__init__(hidden_channels=hidden_channels, node_feat_size=node_feat_size,
                                   learning_rate=learning_rate, use_edge_attr=use_edge_attr, loss_func=loss_func)
        torch.manual_seed(12345)
        self.norm = BatchNorm(self.node_features_size)
        self.conv1 = GATv2Conv(self.node_features_size, self.hidden_channels,edge_dim=1, concat=False, heads=heads)
        self.conv2 = GATv2Conv(self.hidden_channels, self.hidden_channels,edge_dim=1, concat=False, heads=heads)
        self.conv3 = GATv2Conv(self.hidden_channels, self.hidden_channels,edge_dim=1, concat=False, heads=heads)
        self.conv4 = GATv2Conv(self.hidden_channels, self.hidden_channels, edge_dim=1,concat=False, heads=heads)
        self.lin0 = Linear(self.hidden_channels, self.hidden_channels)
        self.lin = Linear(self.hidden_channels, 1)

    def forward(self, mini_batch):
        # 0. Unbatch elements in mini batch
        x, edge_index, batch = mini_batch.x, mini_batch.edge_index, mini_batch.batch
        edge_attr = mini_batch.edge_attr if self.use_edge_attr else None

        # 1. Apply Batch normalization
        x = self.norm(x)

        # 2. Obtain node embeddings
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_attr=edge_attr)
        x = x.relu()
        x = self.conv4(x, edge_index, edge_attr=edge_attr)
        x = x.relu()

        # 3. Readout layer
        y1 = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        y2 = global_max_pool(x, batch)  # [batch_size, hidden_channels]
        y3 = global_add_pool(x, batch)  # [batch_size, hidden_channels]
        z = self.lin0(y1 + y2 + y3).relu()
        
        # 4. Apply a final classifier
        z = F.dropout(z, p=0.5, training=self.training)
        z = self.lin(z)
        

        return z
    



class TGAT(BaseJetGraphGCN):
    def __init__(self,
                 hidden_channels,
                 node_feat_size=None,
                 learning_rate=0.001,
                 use_edge_attr=False,
                 loss_func=torch.nn.BCEWithLogitsLoss(weight=None),
                 dropout_prob=0.1 , loss_weight=None):
        super(TGAT, self).__init__(hidden_channels=hidden_channels, node_feat_size=node_feat_size,
                                        learning_rate=learning_rate, use_edge_attr=use_edge_attr,
                                        loss_func=loss_func, dropout_prob=dropout_prob, loss_weight=loss_weight)
        # Define positional embeddings
        self.positional_embeddings = PositionalEncoding(out_channels=hidden_channels)#WIP...

        # Define the network architecture
        self.norm = BatchNorm(self.node_features_size)#.cuda()
        self.Lnorm1 = BatchNorm(64)#.cuda()
        self.Lnorm2 = BatchNorm(64 + hidden_channels)#.cuda()
        self.Lnorm3 = LayerNorm(2 * (64 + hidden_channels))
        #self.fc_input = Linear(self.node_features_size + hidden_channels*self.node_features_size, hidden_channels)
        self.fc_input = Linear(self.node_features_size , 64)
        self.gat1 = GATv2Conv(64, hidden_channels, heads=3, edge_dim=1)#.cuda()
        self.gat2 = GATv2Conv(3 * hidden_channels, hidden_channels, heads=1, edge_dim=1)#.cuda()
        self.gat3 = GATv2Conv(hidden_channels, hidden_channels, heads=3, edge_dim=1)#.cuda()
        self.gat4 = GATv2Conv(3 * hidden_channels, hidden_channels, heads=1, edge_dim=1)#.cuda()
        self.gat5 = GATv2Conv(hidden_channels, hidden_channels, heads=3, edge_dim=1)#.cuda()
        self.gat6 = GATv2Conv(3 * hidden_channels, hidden_channels)#.cuda()
        self.fc1 = Linear(64 + hidden_channels, 2 * hidden_channels)#.cuda()
        self.fc2 = Linear(2 * hidden_channels, hidden_channels)#.cuda()
        #self.fc3 = Linear(7 * hidden_channels, hidden_channels)#.cuda()
        self.fc3 = Linear(2 * (64 + hidden_channels), hidden_channels)
        self.fc4 = Linear(hidden_channels, 1)#.cuda()
        self.dropout = Dropout(dropout_prob)#.cuda()

    def forward(self, data):
        # 1. Apply Batch normalization
        x = self.norm(data.x)

        # Apply positional encoding
        #positional_encodings = self.positional_embeddings(x)

        # Ensure positional encodings match the batch size and input dimension
        #positional_encodings = positional_encodings.view(x.size(0), x.size(1), -1)

        # Print shapes before concatenation
        #print(f"Shape of x: {x.shape}")
        #print(f"Shape of positional_encodings: {positional_encodings.shape}")

        # Adjust positional_encodings to match the shape of x
        #positional_encodings = positional_encodings.view(x.size(0), -1)

        # Check the new shape of positional_encodings
        #print(f"Adjusted shape of positional_encodings: {positional_encodings.shape}")


        #x = torch.cat([x, positional_encodings], dim=1) # Concatenate along the feature dimension
        # Transform input features
        #x = F.relu(self.fc_input(torch.cat([x, positional_encodings(x)], dim=-1)))# Work in progress for positional embeddings...
        #x = F.relu(self.fc_input(x))

        # Check new shape after concatenation
        #print(f"Shape after concatenation: {x.shape}")

        x = F.relu(self.fc_input(x))

        # Apply GAT Transformer Encoder block 
        y = self.Lnorm1(x)
        
        y = self.dropout(F.relu(self.gat1(y, data.edge_index, data.edge_attr)))
        y = self.dropout(F.relu(self.gat2(y, data.edge_index, data.edge_attr)))
        #Can be commented for low parameters model:
        y = self.dropout(F.relu(self.gat3(y, data.edge_index, data.edge_attr)))
        y = self.dropout(F.relu(self.gat4(y, data.edge_index, data.edge_attr)))
        y = self.dropout(F.relu(self.gat5(y, data.edge_index, data.edge_attr)))
        y = self.dropout(F.relu(self.gat6(y, data.edge_index)))
        
        y = torch.cat([x, y], dim=1)
        yy = self.Lnorm2(y)
        yy = F.relu(self.fc1(yy))
        yy = self.fc2(yy)
        y = torch.cat([y, yy], dim=1)

    
        
        # Concatenate the outputs of the Tokenizer with blocks
        y = torch.cat([x, y], dim=1)

        # Apply fully connected layers
        y = self.Lnorm3(y)
        y = F.relu(self.fc3(y))
        y = global_max_pool(y, data.batch)

        #Apply final classifier 
        y = self.dropout(y)
        y = self.fc4(y)

        return y 


# Function to instantiate one of the available models.
def make_model(cfg):
    m = cfg['model']
    use_edge_attr = bool(int(cfg['use_edge_attributes']))

    assert m in ['gnn', 'gat', 'cheb', 'res_gat', 'res_gnn', 'arma', 'tgat'], 'Model not available.'
    if m == 'gnn':
        return Shallow_GCN(hidden_channels=cfg['hidden_layers'], use_edge_attr=use_edge_attr)
    elif m == 'gat':
        return GAT(hidden_channels=cfg['hidden_layers'], heads=cfg['attention_heads'], use_edge_attr=use_edge_attr)
    elif m == 'cheb':
        return Cheb(hidden_channels=cfg['hidden_layers'], use_edge_attr=use_edge_attr)
    elif m == 'res_gat':
        return Residual_GAT(hidden_channels=cfg['hidden_layers'], use_edge_attr=use_edge_attr)
    elif m == 'res_gnn':
        return Residual_GCN(hidden_channels=cfg['hidden_layers'], use_edge_attr=use_edge_attr)
    elif m == 'arma':
        return Arma(hidden_channels=cfg['hidden_layers'], use_edge_attr=use_edge_attr)
    elif m == 'tgat':
        return TGAT(hidden_channels=cfg['hidden_layers'], use_edge_attr=use_edge_attr)
