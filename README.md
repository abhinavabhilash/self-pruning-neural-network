# Self-Pruning Neural Network

A self-pruning feedforward neural network trained on CIFAR-10.
The network learns to automatically prune its own weights during training using learnable sigmoid gates and an L1 sparsity penalty.

## Features
- **Custom Prunable Linear Layers**: Each weight is augmented with a continuously learnable gate.
- **Dynamic Sparse Training**: Unstructured pruning achieved dynamically during the standard backward pass.
- **Adjustable Compression**: Use the `lambda` hyperparameter to tune the balance between accuracy and network sparsity.

## Installation
Install the project dependencies:
```bash
python3 -m pip install -r requirements.txt
```

## Usage
Run the training script which will automatically execute 3 separate lambda experiments, print logs, generate gate distribution histograms, and output a final results table.
```bash
python3 self_pruning_nn_final.py
```
