

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

import torchquantum as tq

from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import cv2

from skimage import io
from PIL import Image
from time import time
import pickle as pkl    

# define controlled hadamard gate
sq2 = 1/np.sqrt(2)
def controlled_H(qdev, target,control):
      qdev.apply(tq.QubitUnitary(
      has_params=True,init_params=([[1,0,0,0],[0,sq2,0,sq2],[0,0,1,0],[0,sq2,0,-sq2]]),wires=[target,control]))

class QuanvolutionFilter(tq.QuantumModule):
  # the __init__ method initializes the quantum device, the general encoder,
  # a random quantum layer, and a measurement operator.
    def __init__(self):
        super().__init__()
        self.n_wires = 4  # two ancillas
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        # encoding the input data
        self.encoder = tq.GeneralEncoder(
        [   {'input_idx': [0], 'func': 'ry', 'wires': [0]},
            {'input_idx': [1], 'func': 'ry', 'wires': [1]},
            {'input_idx': [2], 'func': 'ry', 'wires': [2]},
            {'input_idx': [3], 'func': 'ry', 'wires': [3]},])
        


        # random circuit layer 
        #self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)
        #self.expval = tq.expval()
    
# x has the dimension of (batch_size, 28, 28) representing a batch of greyscale images
#The method first reshapes the input data into a 2D array of shape 
#(batch_size, 784) by concatenating adjacent 2x2 blocks of pixels.
# data is the new reshaped tensor 
    def forward(self, x, use_qiskit=False):
        bsz = x.shape[0] # batch size
        size = 256 # height and width of an image
        x = x.view(bsz, size, size) # reshape the data 

        data_list = []

        for c in range(0, size, 2):
            for r in range(0, size, 2):
                data = torch.transpose(torch.cat((x[:, c, r], x[:, c, r+1], x[:, c+1, r], x[:, c+1, r+1])).view(4, bsz), 0, 1)
                if use_qiskit:
                    data = self.qiskit_processor.process_parameterized(
                        self.q_device, self.encoder, self.q_layer, self.measure, data)
                else:
                    self.encoder(self.q_device, data)
                    
                    #haar wavelet
                    # level 1 
                    self.q_device.h(wires = 3) 
                    self.q_device.swap([3,2])
                    self.q_device.swap([2,1])
                    self.q_device.swap([1,0])

                    # level 2 
                    controlled_H(self.q_device, target=2, control= 3)
                    self.q_device.cswap([3,2,1])
                    self.q_device.cswap([3,1,0])

                    # level 3
                    
                    #self.q_device.ccx([2,3,4])
                    #controlled_H(self.q_device, target=1, control= 4)
                    #self.q_device.ccx([2,3,4])
                    #perm
                    #self.q_device.ccx([2,3,4])
                    #self.q_device.cswap([4,1,0])
                    #self.q_device.ccx([2,3,4])

                    #level 4
                    #self.q_device.ccx([2,3,4])
                    #self.q_device.ccx([1,4,5])
                    #controlled_H(self.q_device, target=0, control= 5)
                    #self.q_device.ccx([1,4,5])
                    #self.q_device.ccx([2,3,4])



                    #self.q_layer(self.q_device)
                    data = self.measure(self.q_device)
                    
                    #for i in range(4):
                    #    measure_result = []
                    #    measure_result.append(tq.expval(self.q_device,wires=i, observables= tq.PauliZ(wires=i)))

                    #    data = torch.tensor([measure_result])

                    
                data_list.append(data.view(bsz, 4)) # only keep the first 4 qubits
        
        result = torch.cat(data_list, dim=1).float()
        
        return result



class HybridModel(torch.nn.Module): 
    def __init__(self): 
        super().__init__() 
        self.qf = QuanvolutionFilter()
        self.linear = torch.nn.Linear(4*128*128, 2) 
    def forward(self, x, use_qiskit=False):
        with torch.no_grad():
          x = self.qf(x, use_qiskit)
        x = self.linear(x)
        return F.softmax(x, -1) 
    # F.log_softmax is the log of the softmax function, which is a common choice for the output of a classification model.
    
class HybridModel_without_qf(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(256*256, 2)
    
    def forward(self, x, use_qiskit=False):
        x = x.view(-1, 256*256)
        x = self.linear(x)
        return F.log_softmax(x, -1)
    

# importing the dataset 
class Oral_Can_Data(Dataset):
    '''Oral cancer dataset'''
    def __init__(self, csv_file, root_dir,transform = None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations) # returns the number of samples in the dataset
    
    def __getitem__(self, index):
        if torch.is_tensor(index): #Returns True if obj is a PyTorch tensor.
            index = index.tolist() # returns the list of the tensor or converts the tensor to a list

        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0]) 
        image = io.imread(img_path) # read the iamge
        image = np.array(image) # convert it to array 256 x 256
        image = torch.tensor(image) # convert it to tensor
        can_type = self.annotations.iloc[index, 1] # labels:  1 for non-cancerous, 2 for cancerous
        can_type = torch.tensor([can_type])
        

    
        sample = {'image': image, 'can_type': can_type}
        
        for index in range(10):
            sample

        if self.transform:
            image = self.transform(image)
        
        return sample # returns the image and the label


batch_size = 1
num_workers = 0
pin_memory = True
num_epochs = 10
learning_rate = 0.00001
train_CNN = False
pin_memory = True

# loading the dataset
dataset = Oral_Can_Data(csv_file= 'D:\Github\qnn-qhw\labels.csv',root_dir='D:\Github\qnn-qhw\Combined_data_resized')

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_set, val_set,  test_set = torch.utils.data.random_split(dataset, [train_size,val_size,test_size])

#dataflow = dict({'train' : train_set, 'valid': val_set , 'test' : test_set})
train_loader = DataLoader(dataset=train_set, batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory)
validation_loader = DataLoader(dataset=val_set,  batch_size=batch_size,num_workers=num_workers, pin_memory=pin_memory)
test_loader = DataLoader(dataset=test_set,  batch_size=batch_size,num_workers=num_workers, pin_memory=pin_memory)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    
    for batch, data in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(data['image'])
        #print(pred, pred.shape)
        #print(data['can_type'], data['can_type'].shape)
        #print(pred[0][0], data['can_type'][0][0])
        loss = loss_fn(pred, torch.max(data['can_type'], 1)[1])

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(data['image'])
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        if not os.path.isdir('.'+'/latest'):
            os.makedirs('.'+'/latest',exist_ok=True)

            
            pkl.dump((pred), open('.'+"/latest/state","wb"))


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for data in dataloader:
            pred = model(data['image'])
            test_loss += loss_fn(pred, torch.max(data['can_type'], 1)[1]).item()
            correct += (pred.argmax(1) == torch.max(data['can_type'], 1)[1]).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = HybridModel().to(device)
model_without_qf = HybridModel_without_qf().to(device)
n_epochs = 5
#optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
#criterion = torch.nn.CrossEntropyLoss()
criterion = torch.nn.BCELoss()


loss_fn = torch.nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for t in range(n_epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    start_time = time()
    train_loop(train_loader, model, loss_fn, optimizer)
    end_time = time()
    print(f"Training time: {end_time - start_time}s")
    start_time_1 = time()
    test_loop(test_loader, model, loss_fn)
    end_time_1 = time()
    print(f"Test time: {end_time_1 - start_time_1}s")
print("Done!")


