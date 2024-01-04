import pennylane as qml
import tensorflow as tf
import numpy as np
from tqdm import tqdm 
import numpy as np


dev = qml.device('default.qubit', wires=6)

@qml.qnode(dev, interface='tf')
def circuit(input):   # input is a 2 x 2 array which encodes the pixel values of the image
    qml.RY(input[0, 0]/255, wires=0)  # 1st element
    qml.RY(input[1, 0]/255, wires=1)
    qml.RY(input[1, 1]/255, wires=2)
    qml.RY(input[0, 1]/255, wires=3)

    # level 1
    qml.Hadamard(wires = 3)
    qml.SWAP([3,2])
    qml.SWAP([2,1])
    qml.SWAP([1,0])

    # level 2
    qml.ctrl(qml.Hadamard, control=3)(wires=2)
    qml.ctrl(qml.SWAP, control=3)(wires=[2,1])
    qml.ctrl(qml.SWAP, control=3)(wires=[1,0])

    # level 3
    qml.MultiControlledX(wires = [2,3,4] )
    qml.ctrl(qml.Hadamard, control=4)(wires=1)
    qml.MultiControlledX(wires = [2,3,4] )
    # perm
    qml.MultiControlledX(wires = [2,3,4] )
    qml.ctrl(qml.SWAP, control=4)(wires=[1,0])
    qml.MultiControlledX(wires = [2,3,4] )

    # level 4
    qml.MultiControlledX(wires = [2,3,4] )
    qml.MultiControlledX(wires = [1,4,5] )
    qml.ctrl(qml.Hadamard, control=5)(wires=0)
    qml.MultiControlledX(wires = [1,4,5] )
    qml.MultiControlledX(wires = [2,3,4] )

    qml.RY(3.44, wires=2)
    qml.RY(4.39, wires=3)
    qml.RY(0.58, wires=3)
    qml.RY(1.63, wires=3)
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]




imgs_all = np.load(r'D:\Github\qnn-qhw\batch_267_oral.npy')

all_convs =[]

for i in tqdm(range(20)):
    conv_generated = np.zeros((128,128, 4))
    img_array = imgs_all[i]

    for j in range(0, 256, 2):
        for k in range(0, 256, 2):
            window = img_array[j:j+2, k:k+2]
            output = circuit(window)
            conv_generated[j//2, k//2, 0] = output[0]
            conv_generated[j//2, k//2, 1] = output[1]
            conv_generated[j//2, k//2, 2] = output[2]
            conv_generated[j//2, k//2, 3] = output[3]

    all_convs.append(conv_generated)
    
np.save(r'D:\Github\qnn-qhw\conv_batch_267_non_oral_1_20_imgs.npy', all_convs)