# Hybrid Quantum-Classical Neural Network using Haar Wavelet for Feature Extraction 
This repository contains a code for implementing an integrated Haar-Quanvolutional Neural Network on the Oral Cancer dataset.

The code relies on the `Pytorch-Quantum` Library. The data needs to be downloaded separately from: [A histopathological image repository of normal epithelium of Oral Cavity and Oral Squamous Cell Carcinoma](https://data.mendeley.com/datasets/ftmp4cvtmb/1)  

# Installation

## Cloning and handling dependencies 
Clone the repo:
```
 git clone https://github.com/Next-di-mension/qnn-qhw.git &&bcd qnn-qhw
```
The `PyTorch-Quantum` module requires the older version of qiskit, therefore, activate the `.venv` environment which contains all the necessary modules.
```
 conda activate .venv
```
# Running the code 
Run `hcnn_modified.py` file to train and test the model. The labels for corresponding train and test data are in `labels.csv`. Specify the path to this file before setting model to train. 

# Overview of the work 
[perm gates](https://github.com/Next-di-mension/qnn-qhw/files/12542128/perm.gates.pdf)
