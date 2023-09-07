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
Run `hcnn_modified.py` file to train and test the model. The labels for corresponding train and test data are in `labels.csv`. Specify the path to this file before setting the model to train. 

# Overview of the work 
This proposed model is a hybrid model. It consists of two sections: a quanvolutional filter part and then a classical layer part. First, we create a quanvolutional filter. a transformational layer called a ”quanvolutional layer,” which operates on input data by locally transforming the data using random quantum circuits, similar to the transformations performed by random convolutional filter layers. Quanvolutional layers apply random quantum circuits to input data, enabling local transformations and feature extraction.

![QNN](https://github.com/Next-di-mension/qnn-qhw/assets/98448938/86de7b81-1f94-4635-8c15-566e00fbff36)

The quantum Haar wavelet is a quantum state that shares similarities with the classical Haar wavelet. It is localized both in the time and energy domains, allowing for the analysis of quantum states at different energy scales. By decomposing a quantum state into quantum Haar wavelet coefficients, it is possible to extract information about its energy distribution at different scales. In the quantum domain, the input vector is the multi-qubit state with a $2^n$ dimension. Where $n$ is the number of qubits. To describe the QHWT effectively with quantum circuits, there are two main elements that need to be considered. first is Hadamard gate and second is the permutation matrix $\Pi_{2^n}$ applied to $n$ qubit quantum register.

![QHW circuit](https://github.com/Next-di-mension/qnn-qhw/assets/98448938/b2d3c1bd-3ad6-41ee-a82d-eeaf087e6c01)


