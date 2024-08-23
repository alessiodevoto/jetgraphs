# Jetgraphs
Tools for manipulating Jetgraphs. Jetgraphs are graphs built on top of physical processes. 

:grey_exclamation: this package is pip-installable! See `examples/JetGraphDataset_v5.1.ipynb` for a tutorial to download a dataset and start working with jetgraphs.

- `JetgraphDataset` lets you download a dataset of jetgraphs and manipulate them with a bunch of trasforms.
- `jetgraphs.utils` offers, among the others, some functions to pretty-plot Jetgraphs and visualize what you are doing.
- `jetgraphs.explainability` offers explainable AI tools. For now, we have Captum TracIn adjusted to work with pytorch geometric.

## Branch specifics
- Pilot scripts are added and to be followed in order to produce full results in paper. Pipeline steps include everything from training to evaluation then explainability analyses. (*WIP to add full pipeline recipe A to Z*)
- @carmigna: A Tiny _Transformer-inspired_ Graph Attention model __TGAT__ (with commented lines for potential embeddings or auto-encodings _still wip_) is added to list of models. I originally designed and fully tested this model while working as ML liason within the TauCP ATLAS group. The context of the project is an adaptation of @alessiodevoto _jetgraphs_ (_with many thanks_) to create a GNN based code that unifies TauID (_Signal/Background_) and Tau Decay Modes Discrimination (_multi-classifier_). __TauJetGraphs__ ATLAS internal project showed excellent and several orders of magnitude higher level performance using TGAT as core model when benchmarked against legacy RNN while taking advantage of all cutting edge techniques presented here and 74 mostly high-level input variables of a huge _not yet public_ TauID Dataset (20 Million events and much more for validation). TGAT model was tested also on other HEP use cases and produced excellent ROC curve performance in comparison with other Graph Convolution models including the ARMA-based one used here within the context of Dark Photon Jets project.     
- Wandb logging added to scripts to register fully the model's parameters with transparent Performance and Computational metrics while running the training and/or evaluations.
- Differential Programming functions are implemented within optimised PyTorch Lightning modules allowing the Kappa _gradient-based_ pruning technique. (*See paper*)
- _Paper ref. to be added here_ 

## Jetgraphs

Some jetgraphs samples are displayed here:

| | |
|:-------------------------:|:-------------------------:|
|<img width="400" src="https://github.com/alessiodevoto/jetgraphs/blob/main/images/g0.png?raw=true">  |  <img width="400" src="https://github.com/alessiodevoto/jetgraphs/blob/main/images/g1.png?raw=true">|
|<img width="400"  src="https://github.com/alessiodevoto/jetgraphs/blob/main/images/g2.png?raw=true">  |  <img width="400" src="https://github.com/alessiodevoto/jetgraphs/blob/main/images/g4.png?raw=true">|

