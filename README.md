# Hybrid Quantum-Classical Neural Network using Haar Wavelet for Feature Extraction

This repository presents an innovative approach to integrating a Haar-Quanvolutional Neural Network with the Oral Cancer dataset, leveraging the PyTorch-Quantum Library for its implementation.

## Repository Structure
```
.
├── data
├── Images
├── hcnn.py
├── hcnn_cross_fold.py
├── LICENSE
├── kernel.py
├── labels.csv
├── requirements.txt
├── results.yml
├── README.md

```

## Data Source

The Oral Cancer dataset is essential for this project and needs to be downloaded separately from: [A histopathological image repository of normal epithelium of Oral Cavity and Oral Squamous Cell Carcinoma](https://data.mendeley.com/datasets/ftmp4cvtmb/1).

## Installation

### Cloning and Handling Dependencies

First, clone the repository:

```bash
git clone https://github.com/Next-di-mension/qnn-qhw.git && cd qnn-qhw
```

Due to specific version requirements for the `PyTorch-Quantum` module, it's recommended to create a new virtual environment:

```bash
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the Code

- Use `hcnn.py` for the PyTorch implementation to train and test the model.
- Use `hcnn_cross_fold.py` for the TensorFlow implementation.
- Before training, specify the path to the `labels.csv` file which contains labels for the train and test data.

## Model Overview
![QNN](https://github.com/Next-di-mension/qnn-qhw/assets/98448938/86de7b81-1f94-4635-8c15-566e00fbff36)

The proposed model is a hybrid, consisting of a quanvolutional filter and classical layer sections. The quanvolutional filter uses random quantum circuits, similar to convolutional filters in classical neural networks, for local data transformation and feature extraction.

### Quantum Haar Wavelet

The quantum Haar wavelet (QHW) is a state localized in both time and energy domains, suitable for analyzing quantum states at various energy scales. The QHWT decomposes a quantum state into coefficients, revealing its energy distribution. The effective description of QHWT in quantum circuits involves Hadamard gates and a permutation matrix $\Pi_{2^n}$ applied to an $n$ qubit quantum register.

![QHW circuit](https://github.com/Next-di-mension/qnn-qhw/assets/98448938/b2d3c1bd-3ad6-41ee-a82d-eeaf087e6c01)

## Sample Images

Sample convolutions generated from the model are shown below:

![non_oral_can_conv_img (1)](https://github.com/Next-di-mension/qnn-qhw/assets/98448938/ef4eff8f-168f-4b81-aa38-171acae47502)


