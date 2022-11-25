import os
import shutil
from typing import Any, Optional, Tuple
import os.path as osp
import datetime
from captum.influence import TracInCP, TracInCPFast
from torch_geometric.loader import DataLoader

import torch
from torch import Tensor
from torch.nn import Module
import matplotlib.pyplot as plt
from .utils import plot_jet_graph

from captum.influence._utils.common import (
    _jacobian_loss_wrt_inputs,
    _tensor_batch_dot,
    _gradient_dot_product,
)

from captum._utils.gradient import (
    _compute_jacobian_wrt_params,
    _compute_jacobian_wrt_params_with_sample_wise_trick,
)


def _capture_inputs(layer: Module, input: Tensor, output: Tensor) -> None:
    r"""Save activations into layer.activations in forward pass"""

    layer_inputs.append(input[0].detach())

def _basic_computation_tracincp_fast(
  influence_instance: TracInCPFast,
  inputs: Tuple[Any, ...],
  targets: Tensor,):
  
  """
  For instances of TracInCPFast and children classes, computation of influence scores
  or self influence scores repeatedly calls this function for different checkpoints
  and batches.

  Args:
      influence_instance (TracInCPFast): A instance of TracInCPFast or its children.
      inputs (Tuple of Any): A batch of examples, which could be a training batch
              or test batch, depending which method is the caller. Does not
              represent labels, which are passed as `targets`. The assumption is
              that `self.model(*inputs)` produces the predictions for the batch.
      targets (tensor): If computing influence scores on a loss function,
              these are the labels corresponding to the batch `inputs`.
  """
  global layer_inputs
  layer_inputs = []
  assert isinstance(influence_instance.final_fc_layer, Module)
  handle = influence_instance.final_fc_layer.register_forward_hook(_capture_inputs)
  
  out = influence_instance.model(inputs) # instead of *inputs

  assert influence_instance.loss_fn is not None
  input_jacobians = _jacobian_loss_wrt_inputs(
      influence_instance.loss_fn, out.float(), targets.float(), influence_instance.vectorize # added .float here
  ) 
  handle.remove()
  _layer_inputs = layer_inputs[0]

  return input_jacobians, _layer_inputs

class TracInCPFastGNN(TracInCPFast):

  def _influence_batch_tracincp_fast(
        self,
        inputs: Tuple[Any, ...],
        targets: Tensor,
        batch: Tuple[Any, ...],
    ):
        """
        computes influence scores for a single training batch
        """

        def get_checkpoint_contribution(checkpoint):

            assert (
                checkpoint is not None
            ), "None returned from `checkpoints`, cannot load."

            learning_rate = self.checkpoints_load_func(self.model, checkpoint)

            input_jacobians, input_layer_inputs =_basic_computation_tracincp_fast(
                self,
                inputs[0], # instead of inputs
                targets,
            )

            src_jacobian, src_layer_input = _basic_computation_tracincp_fast(

                self, batch, batch.y.unsqueeze(1) # instead of batch[0:-1], batch[-1]
            )
            return (
                _tensor_batch_dot(input_jacobians, src_jacobian)
                * _tensor_batch_dot(input_layer_inputs, src_layer_input)
                * learning_rate
            )

        batch_tracin_scores = get_checkpoint_contribution(self.checkpoints[0])

        for checkpoint in self.checkpoints[1:]:
            batch_tracin_scores += get_checkpoint_contribution(checkpoint)

        return batch_tracin_scores
  
  def _self_influence_batch_tracincp_fast(self, batch: Tuple[Any, ...]):
    """
    Computes self influence scores for a single batch
    """

    def get_checkpoint_contribution(checkpoint):

        assert (
            checkpoint is not None
        ), "None returned from `checkpoints`, cannot load."

        learning_rate = self.checkpoints_load_func(self.model, checkpoint)

        batch_jacobian, batch_layer_input = _basic_computation_tracincp_fast(

            self, batch, batch.y.unsqueeze(1) # instead of batch[0:-1], batch[-1]
        )

        return (
            torch.sum(batch_jacobian ** 2, dim=1)
            * torch.sum(batch_layer_input ** 2, dim=1)
            * learning_rate
        )

    batch_self_tracin_scores = get_checkpoint_contribution(self.checkpoints[0])

    for checkpoint in self.checkpoints[1:]:
        batch_self_tracin_scores += get_checkpoint_contribution(checkpoint)

    return batch_self_tracin_scores

class TracInCPGNN(TracInCP):

  def _basic_computation_tracincp(
    self,
    inputs: Tuple[Any, ...],
    targets: Optional[Tensor] = None,
) -> Tuple[Tensor, ...]:
    """
    For instances of TracInCP, computation of influence scores or self influence
    scores repeatedly calls this function for different checkpoints
    and batches.
    Args:
        inputs (Tuple of Any): A batch of examples, which could be a training batch
                or test batch, depending which method is the caller. Does not
                represent labels, which are passed as `targets`. The assumption is
                that `self.model(*inputs)` produces the predictions for the batch.
        targets (tensor or None): If computing influence scores on a loss function,
                these are the labels corresponding to the batch `inputs`.
    """
    """print('BASIC INPUTS:\n', inputs)
    print('BASIC INPUTS[0]:\n', inputs[0])
    print('GNN(inputs[0])\n', gnn(inputs[0]) )"""
    if self.sample_wise_grads_per_batch:
        return _compute_jacobian_wrt_params_with_sample_wise_trick(
            self.model,
            [inputs], # Instead of *inputs
            targets,
            self.loss_fn,
            self.reduction_type,
        )
    return _compute_jacobian_wrt_params(
        self.model,
        [inputs],# Instead of *inputs
        targets.float(),
        self.loss_fn,
    )
   
  def _influence_batch_tracincp(
      self,
      inputs: Tuple[Any, ...],
      targets: Optional[Tensor],
      batch: Tuple[Any, ...],
  ):
      """
      computes influence scores for a single training batch
      """

      def get_checkpoint_contribution(checkpoint):

        assert (
            checkpoint is not None
        ), "None returned from `checkpoints`, cannot load."

        learning_rate = self.checkpoints_load_func(self.model, checkpoint)

        input_jacobians = self._basic_computation_tracincp(
            inputs[0],
            targets.float(),
        )


        return (
            _gradient_dot_product(
                input_jacobians,
                self._basic_computation_tracincp(
                    batch, batch.y.unsqueeze(1) # instead of batch[0:-1], batch[-1]
                ),
            )
            * learning_rate
        )

      batch_tracin_scores = get_checkpoint_contribution(self.checkpoints[0])

      for checkpoint in self.checkpoints[1:]:
          batch_tracin_scores += get_checkpoint_contribution(checkpoint)

      return batch_tracin_scores
    
  def _self_influence_batch_tracincp(self, batch: Tuple[Any, ...]):
    """
    Computes self influence scores for a single batch
    """

    def get_checkpoint_contribution(checkpoint):

        assert (
            checkpoint is not None
        ), "None returned from `checkpoints`, cannot load."

        learning_rate = self.checkpoints_load_func(self.model, checkpoint)

        layer_jacobians = self._basic_computation_tracincp(
             batch, batch.y.unsqueeze(1) # instead of batch[0:-1], batch[-1]
            )

        # note that all variables in this function are for an entire batch.
        # each `layer_jacobian` in `layer_jacobians` corresponds to a different
        # layer. `layer_jacobian` is the jacobian w.r.t to a given layer's
        # parameters. if the given layer's parameters are of shape *, then
        # `layer_jacobian` is of shape (batch_size, *). for each layer, we need
        # the squared jacobian for each example. so we square the jacobian and
        # sum over all dimensions except the 0-th (the batch dimension). We then
        # sum the contribution over all layers.
        return (
            torch.sum(
                torch.stack(
                    [
                        torch.sum(layer_jacobian.flatten(start_dim=1) ** 2, dim=1)
                        for layer_jacobian in layer_jacobians
                    ],
                    dim=0,
                ),
                dim=0,
            )
            * learning_rate
        )

    batch_self_tracin_scores = get_checkpoint_contribution(self.checkpoints[0])

    for checkpoint in self.checkpoints[1:]:
        batch_self_tracin_scores += get_checkpoint_contribution(checkpoint)

    return batch_self_tracin_scores

def checkpoints_load_func(model, path):
    """
    Load a chekpoint into model. Side effect on model.
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    # add other except clauses to load from different kinds of checkpoint
    try:
        ckpt = torch.load(path)
        lr = ckpt['optimizer_states'][0]['param_groups'][0]['lr']
    except: 
        lr = 1.
    return lr

def display_proponents_and_opponents(
    test_examples_batch, 
    correct_dataset, 
    test_examples_true_labels, 
    test_examples_predicted_labels, 
    test_examples_predicted_probs, 
    proponents_indices, 
    opponents_indices,
    save_to_dir = False,
    **kwargs):
    """
    Plot results from captum tracin.
    """
    sample_num = 0
    for (
        test_example,
        test_example_true_label,
        test_example_predicted_label,
        test_example_predicted_prob,
        test_example_proponents,
        test_example_opponents,
    ) in zip(
        test_examples_batch,
        test_examples_true_labels,
        test_examples_predicted_labels,
        test_examples_predicted_probs,
        proponents_indices,
        opponents_indices,
    ):

        num_examples = len(proponents_indices) # num of prop and opp for each example
        graph_size = 5 # plot size for a single graph.
        cols =  2 # columns to display graphs in plot.
        rows = num_examples + 1  # rows to display graphs in plot.
        figsize = (cols*graph_size, (rows+2)*graph_size)
        fig = plt.figure(figsize=figsize)
         
        # Plot test example at top middle location.
        true_label = test_example_true_label.item()
        predicted_label = int(test_example_predicted_label.item())
        predicted_prob = test_example_predicted_prob.item()
        title = f"True label: {true_label}, predicted label: {predicted_label},  predicted prob: {predicted_prob:.2f}"
        ax = fig.add_subplot(rows, 2, 1, projection="3d")
        ax.set_title(title)
        plot_jet_graph(g=test_example, ax=ax, **kwargs)
        
        # Plot proponents on left column.
        idx = 3
        test_example_graphs = [correct_dataset[i] for i in test_example_proponents]
        for i in range(len(test_example_proponents)):
            ax = fig.add_subplot(rows, cols, idx, projection="3d") 
            ax.set_title(f'[Proponent] Label:{test_example_graphs[i].y.item()}')
            ax.title.set_color('green') 
            plot_jet_graph(g=test_example_graphs[i], ax = ax, **kwargs) 
            idx += 2
        
        # Plot opponents on right column.
        idx = 4
        test_example_graphs = [correct_dataset[i] for i in test_example_opponents]
        for i in range(len(test_example_opponents)):
            ax = fig.add_subplot(rows, cols, idx, projection="3d")
            ax.set_title(f'[Opponent] Label:{test_example_graphs[i].y.item()}') 
            ax.title.set_color('red') 
            plot_jet_graph(g=test_example_graphs[i], ax = ax, **kwargs)
            idx += 2
        
        plt.subplots_adjust( wspace=.1)
        
        # Save to dir if necessary.
        if save_to_dir is not False:
            os.makedirs(save_to_dir, exist_ok=True)
            date = datetime.datetime.today()
            form_date = f'{date.year}-{date.month}-{date.day}-{date.hour}:{date.minute}'
            plt.savefig(os.path.join(save_to_dir, f"{form_date}_{sample_num}.pdf"))
            sample_num += 1
            plt.close('all')
    
    plt.show()


class CaptumPipeline:

    def __init__(self, model, dataset, train_idx, checkpoint_dir, epochs, captum_impl='fast'):
        
        print("Initializing Captum pipeline...")
        self.model = model
        self.checkpoint_dir = checkpoint_dir
        self.epochs = epochs
        self.dataset = dataset

        # If lightning was used, we should clean the directory from non-checkpoint files.
        if os.path.exists(os.path.join(checkpoint_dir, 'lightning_logs')):
            shutil.rmtree(os.path.join(checkpoint_dir, 'lightning_logs'))

        # We first load the model with the last checkpoint so that the predictions we make in the next cell will be for the trained model.
        print("Loading checkpoint...")
        correct_dataset_final_checkpoint = osp.join(self.checkpoint_dir, f'gnn-epoch={epochs-1}.ckpt')
        checkpoints_load_func(model, correct_dataset_final_checkpoint)

        # Dataloader for Captum.
        print("Initializing loader...")
        self.influence_src_dataloader = DataLoader(dataset[train_idx], batch_size=64, shuffle=False)

        print(f"Initializing Captum {captum_impl}...")
        # Prepare Captum
        if captum_impl == 'fast':
            self.tracin_impl = TracInCPFastGNN(
                model=model,
                final_fc_layer=list(model.children())[-1],
                influence_src_dataset=self.influence_src_dataloader,
                checkpoints=checkpoint_dir,
                checkpoints_load_func=checkpoints_load_func,
                loss_fn=torch.nn.functional.binary_cross_entropy_with_logits,
                batch_size=2048,
                vectorize=False
            )
        elif captum_impl == 'base':
            self.tracin_impl = TracInCPGNN(
                model=model,
                influence_src_dataset=self.influence_src_dataloader,
                checkpoints=checkpoint_dir,
                checkpoints_load_func=checkpoints_load_func,
                loss_fn=torch.nn.functional.binary_cross_entropy_with_logits,
                batch_size=2048,
                vectorize=False
            )

    def run_captum(self, test_influence_indices):
        
        print("Initializing Dataloaders for Captum Pipeline...")

        # Prepare samples.
        test_influence_loader = DataLoader(self.dataset[test_influence_indices], batch_size=len(test_influence_indices), shuffle=False)
        self.test_examples_batch = next(iter(test_influence_loader))
        self.test_examples_predicted_probs = torch.sigmoid(self.model(self.test_examples_batch)) 
        self.test_examples_predicted_labels = (self.test_examples_predicted_probs > 0.5).float()
        self.test_examples_true_labels = self.test_examples_batch.y.unsqueeze(1)

        print(f"Going to compute proponents and opponents for {len(test_influence_loader)} indices...")
        
        start_time = datetime.datetime.now()

        # Compute proponents.
        p_idx, p_scores = self.tracin_impl.influence(
            inputs = self.test_examples_batch, 
            targets = self.test_examples_true_labels, 
            k=len(test_influence_indices), 
            proponents=True, 
            unpack_inputs=False
        )
        print("Compute proponents done!")

        # Compute opponents.
        o_idx, o_scores = self.tracin_impl.influence(
            inputs = self.test_examples_batch, 
            targets = self.test_examples_true_labels, 
            k=len(test_influence_indices), 
            proponents=False, 
            unpack_inputs=False
        )
        print("Compute opponents done!")


        total_minutes = (datetime.datetime.now() - start_time).total_seconds() / 60.0
        print(
            "Computed proponents / opponents over a dataset of %d examples in %.2f minutes"
            % (len(self.influence_src_dataloader)*self.influence_src_dataloader.batch_size, total_minutes)
        )

        self.proponents_idx, self.proponents_scores = p_idx, p_scores
        self.opponents_idx, self.opponents_scores = o_idx, o_scores
    
    def display_results(self, save_to_dir=False, **kwargs):
        print("Rebuilding dataset for displaying results...")
        src_dataset = []
        for x in self.influence_src_dataloader:
            src_dataset.extend(x.to_data_list())

        print("Displaying Captum results...")
        display_proponents_and_opponents(
            self.test_examples_batch.to_data_list(), 
            src_dataset, 
            self.test_examples_true_labels, 
            self.test_examples_predicted_labels, 
            self.test_examples_predicted_probs, 
            self.proponents_idx, 
            self.opponents_idx,
            save_to_dir, 
            **kwargs)
