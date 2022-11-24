from typing import Any, Optional, Tuple

from captum.influence._utils.common import _load_flexible_state_dict
from captum.influence import TracInCP, TracInCPFast

import torch
from captum._utils.common import _get_module_from_name
from captum.influence._core.tracincp import TracInCPBase, KMostInfluentialResults
from captum.influence._utils.common import (
    _jacobian_loss_wrt_inputs,
    _load_flexible_state_dict,
    _tensor_batch_dot,
    _gradient_dot_product,
)

from math import ceil

from torch import Tensor
from torch.nn import Module

import matplotlib.pyplot as plt

from .utils import plot_jet_graph



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


from captum._utils.gradient import (
    _compute_jacobian_wrt_params,
    _compute_jacobian_wrt_params_with_sample_wise_trick,
)


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
    # _load_flexible_state_dict(net, path, keyname='state_dict')
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
    **kwargs):
  
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

    num_examples = len(proponents_indices)
    true_label = test_example_true_label.item()
    predicted_label = int(test_example_predicted_label.item())
    predicted_prob = test_example_predicted_prob.item()

    
    title = f"True label: {true_label}, predicted label: {predicted_label},  predicted prob: {predicted_prob:.2f}"
    graph_size = 5
    cols =  2
    rows = num_examples + 1
    figsize = (cols*graph_size, rows*graph_size,)

    
    fig = plt.figure(figsize=figsize)
    
    # Plot test example.
    ax = fig.add_subplot(rows, 3, 2, projection="3d")
    ax.set_title(title)
    plot_jet_graph(g=test_example, ax=ax, **kwargs)
    
    idx = 3
    test_example_graphs = [correct_dataset[i] for i in test_example_proponents]
    for i in range(len(test_example_proponents)):
        ax = fig.add_subplot(rows, cols, idx, projection="3d") 
        ax.set_title(f'[Proponent] Label:{test_example_graphs[i].y.item()}')
        ax.title.set_color('green') 
        plot_jet_graph(g=test_example_graphs[i], ax = ax, **kwargs) 
        idx += 2
    
    idx = 4
    test_example_graphs = [correct_dataset[i] for i in test_example_opponents]
    for i in range(len(test_example_opponents)):
        ax = fig.add_subplot(rows, cols, idx, projection="3d")
        ax.set_title(f'[Opponent] Label:{test_example_graphs[i].y.item()}') 
        ax.title.set_color('red') 
        plot_jet_graph(g=test_example_graphs[i], ax = ax, **kwargs)
        idx += 2
    
    plt.subplots_adjust(hspace=.6, wspace=.2)
    plt.show()