# Jetgraphs
Tools for manipulating Jetgraphs. Jetgraphs are graphs built on top of physical processes. 

:grey_exclamation: this package is pip-installable! See `examples/JetGraphDataset_v5.1.ipynb` for a tutorial to download a dataset and start working with jetgraphs.

- `JetgraphDataset` lets you download a dataset of jetgraphs and manipulate them with a bunch of trasforms.
- `jetgraphs.utils` offers, among the others, some functions to pretty-plot Jetgraphs and visualize what you are doing.
- `jetgraphs.explainability` offers explainable AI tools. For now, we have Captum TracIn adjusted to work with pytorch geometric.

## Branch specifics
- Pilot scripts are added and to be followed in order to produce full results. Pipeline steps include everything from training to evaluation then explainability analyses. (*WIP to add full pipeline recipe A to Z*)
- Wandb logging added to scripts to register fully the model's parameters with transparent Performance and Computational metrics while running the training and/or evaluations.
- Differential Programming functions are implemented within optimised PyTorch Lightning modules allowing the Kappa _gradient-based_ pruning technique. (*See paper*)
- _Paper ref. to be added here_ 

## Jetgraphs

Some jetgraphs samples are displayed here:

| | |
|:-------------------------:|:-------------------------:|
|<img width="400" src="https://github.com/alessiodevoto/jetgraphs/blob/main/images/g0.png?raw=true">  |  <img width="400" src="https://github.com/alessiodevoto/jetgraphs/blob/main/images/g1.png?raw=true">|
|<img width="400"  src="https://github.com/alessiodevoto/jetgraphs/blob/main/images/g2.png?raw=true">  |  <img width="400" src="https://github.com/alessiodevoto/jetgraphs/blob/main/images/g4.png?raw=true">|

