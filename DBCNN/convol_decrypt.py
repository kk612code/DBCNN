import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import sys
import os
import random

def deconvolution(data, weight):
    num = data.size
    data = data.reshape((1, len(data)))
    shape = weight.shape
    wt_num = shape[1]
    n_opt = wt_num - 1

    ###disorder
    tmp = data[0].copy()
    t_n = (num // wt_num) * wt_num
    for i in range(1, t_n):
        j = (i % (num // wt_num)) * wt_num + (i // (num // wt_num))
        data[0][j] = tmp[i]
    for i in range(1, t_n):
        data[0][t_n-i] = data[0][t_n-i] ^ data[0][t_n-i-1]

    ###opt
    for i in range(0, num):
        for j in range(0, n_opt):
            if (i-n_opt+j) < 0:
                data[0][i] = int(data[0][i]) - int(weight[0][j])
            else:
                data[0][i] = int(data[0][i]) - int(weight[0][j] ^ data[0][i-n_opt+j])
        data[0][i] = data[0][i] ^ weight[0][n_opt]

    return data[0]

def layerNoWeight(weight, data):
    data = deconvolution(data, weight)
    return data


def readImages(npy_path): #direct = './train/'
    data = np.load(npy_path)
    return data

def allImageToLayer(data, wt): #0(r), 1(g), 2(b), 3(grey)
    I0 = data
    I0 = I0.reshape(I0.size)  
    xordt0 = layerNoWeight(wt, I0) 
    xordt0 = xordt0.reshape(data.shape[0], data.shape[1])
    data = xordt0
    return data

def execute(data, n_layer, wt):
    ###test
    data = allImageToLayer(data, wt)

    print('layer: %2d/%2d' % (n_layer, num_layers))
    return data
    ###test


path = './data_stack/3x10/encrypt0.npy'
data = readImages(path)
weights = np.load('./data_stack/3x10/weights.npy')
num_layers = len(weights)
for i in range(0, num_layers):
    data = execute(data, i+1, weights[num_layers-1-i])
np.save('./data_stack/3x10/decrypt0.npy', data)
plt.imshow(data, cmap='gray')
plt.axis('off')
plt.show()





