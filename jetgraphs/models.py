import torch
from torch.nn import Linear, CrossEntropyLoss, functional as F
from torchmetrics.functional import accuracy
import torch.nn.functional as F
from torch_geometric.nn import LayerNorm, BatchNorm
from torch_geometric.nn import GCNConv, GATv2Conv, ChebConv, ARMAConv
from torch_geometric.nn import global_mean_pool
from pytorch_lightning import LightningModule

from sklearn import metrics
import numpy as np

from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import LayerNorm, BatchNorm
from torch_geometric.nn import GCNConv, GATv2Conv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

from abc import abstractmethod

default_node_features = 4


class BaseJetGraphGCN(LightningModule):
    def __init__(self,
                 #batch_size,
                 hidden_channels,
                 #num_stacks,
                 node_feat_size=None,
                 use_edge_attr=False,
                 learning_rate=0.001,
                 loss_func=torch.nn.BCEWithLogitsLoss()):
        super(BaseJetGraphGCN, self).__init__()
        torch.manual_seed(12345)

        # Network structure.
        #self.batch_size = batch_size
        self.hidden_channels = hidden_channels
        #self.num_stacks = num_stacks
        self.node_features_size = node_feat_size if node_feat_size else default_node_features
        self.use_edge_attr = use_edge_attr

        # Loss.
        self.loss = loss_func
        self.lr = learning_rate

        # Save hyper-parameters to self.hparams (auto-logged by W&B).
        self.save_hyperparameters()

    @abstractmethod
    def forward(self, x, edge_index, mini_batch):#mini_batch):#x, edge_index, mini_batch):#switch for GNNExplainer
        pass

    def training_step(self, batch, batch_idx):
        out = self(batch)  # Perform a single forward pass.
        loss = self.loss(out, batch.y.unsqueeze(1).float())  # Compute the loss.

        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch)  # Perform a single forward pass.
        predictions = torch.sigmoid(out).detach().cpu().numpy()
        labels = batch.y.unsqueeze(1).float()


        # Loss.
        loss = self.loss(out, labels)  # Compute the loss.
        # Accuracy.
        acc = metrics.accuracy_score(labels.detach().cpu().numpy(), np.round(predictions))
        # F1 score.
        f1_score = metrics.f1_score(labels.detach().cpu().numpy(), np.round(predictions))
        # AUC.
        roc_auc = metrics.roc_auc_score(labels.detach().cpu().numpy(), predictions)

        # Log loss and metric
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_accuracy', acc, on_step=False, on_epoch=True)
        self.log('val_f1_score', f1_score, on_step=False, on_epoch=True)
        self.log('val_roc_auc', roc_auc, on_step=False,  on_epoch=True)

        return loss

    def predict_step(self, batch, batch_idx):
        out = self(batch)  # Perform a single forward pass.
        predictions = torch.sigmoid(out).detach().cpu().numpy()
        #labels = batch.y.unsqueeze(1).float()
        return predictions.tolist()


    # def __init__(self,
    #              hidden_channels,
    #              node_feat_size=None,
    #              use_edge_attr=False,
    #              learning_rate=0.001,
    #              loss_func=torch.nn.BCEWithLogitsLoss()):
    #     super(BaseJetGraphGCN, self).__init__()
    #     torch.manual_seed(12345)

    #     # Network structure.
    #     self.hidden_channels = hidden_channels
    #     self.node_features_size = node_feat_size if node_feat_size else default_node_features
    #     self.use_edge_attr = use_edge_attr

    #     # Loss.
    #     self.loss = loss_func
    #     self.lr = learning_rate

    #     # Save hyper-parameters to self.hparams (auto-logged by W&B).
    #     self.save_hyperparameters()

    # @abstractmethod
    # def forward(self, mini_batch):
    #     pass

    # def training_step(self, batch, batch_idx):
    #     out = self(batch)  # Perform a single forward pass.
    #     loss = self.loss(out, batch.y.unsqueeze(1).float())  # Compute the loss.

    #     self.log('train_loss', loss, on_step=False, on_epoch=True)
    #     return loss

    # def validation_step(self, batch, batch_idx):
    #     out = self(batch)  # Perform a single forward pass.
    #     predictions = torch.sigmoid(out).detach().cpu().numpy()
    #     labels = batch.y.unsqueeze(1).float()


    #     # Loss.
    #     loss = self.loss(out, labels)  # Compute the loss.
    #     # Accuracy.
    #     acc = metrics.accuracy_score(labels.detach().cpu().numpy(), np.round(predictions))
    #     # F1 score.
    #     f1_score = metrics.f1_score(labels.detach().cpu().numpy(), np.round(predictions))
    #     # AUC.
    #     roc_auc = metrics.roc_auc_score(labels.detach().cpu().numpy(), predictions)

    #     # Log loss and metric
    #     self.log('val_loss', loss, on_step=False, on_epoch=True)
    #     self.log('val_accuracy', acc, on_step=False, on_epoch=True)
    #     self.log('val_f1_score', f1_score, on_step=False, on_epoch=True)
    #     self.log('val_roc_auc', roc_auc, on_step=False,  on_epoch=True)

    #     return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min = 0.00001)
        return [optimizer], [scheduler]


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

    def forward(self,x, edge_index, mini_batch):
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


class Arma(BaseJetGraphGCN):
    def __init__(self,
                 hidden_channels,
                 #batch_size=64,
                 node_feat_size=None,
                 learning_rate=0.001,
                 use_edge_attr=False,
                 loss_func=torch.nn.BCEWithLogitsLoss()):
        super(Arma, self).__init__(hidden_channels=hidden_channels, node_feat_size=node_feat_size,
                                   learning_rate=learning_rate, use_edge_attr=use_edge_attr, loss_func=loss_func)
        torch.manual_seed(12345)
        self.norm = BatchNorm(self.node_features_size)
        self.conv1 = ARMAConv(self.node_features_size, self.hidden_channels, num_stacks= 3)
        self.conv2 = ARMAConv(self.hidden_channels, self.hidden_channels, num_stacks= 3)
        self.conv3 = ARMAConv(self.hidden_channels, self.hidden_channels, num_stacks= 3)
        self.lin0 = Linear(self.hidden_channels, self.hidden_channels)
        self.lin = Linear(self.hidden_channels, 1)

    def forward(self,  x, edge_index, mini_batch):#mini_batch):#x, edge_index, mini_batch):#switch for GNNExplainer
        # 0. Unbatch elements in mini batch
        x, edge_index, batch = mini_batch.x, mini_batch.edge_index, mini_batch.batch
        edge_attr = mini_batch.edge_attr if self.use_edge_attr else None

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


# Function to instantiate one of the available models.
def make_model(cfg):
    m = cfg['model']
    use_edge_attr = bool(int(cfg['use_edge_attributes']))

    assert m in ['gnn', 'gat', 'cheb', 'res_gat', 'res_gnn', 'arma'], 'Model not available.'
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
