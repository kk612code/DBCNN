import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
import random


def convolution_weight(n1, n2):
    mt = np.random.randint(0, 255, size=(n1, n2))
    return np.array(mt, dtype=np.uint8)


def convolution(data, weight):
    shape = weight.shape
    wt_num = shape[1]
    data = data.reshape((1, len(data)))
    dt_num = data.size

    for i in range(0, wt_num-1):
        tmp = np.insert(data[i], 0, 0)
        tmp = np.delete(tmp, -1)
        data = np.insert(data, i + 1, tmp, axis=0)

    for i in range(0, wt_num):
        data[i] = weight[0][wt_num-1-i] ^ data[i]

    ###opt
    for i in range(1, wt_num):
            data[0] = np.add(data[0], data[i])

    ###disorder
    tmp = data[0].copy()
    t_n = (dt_num // wt_num) * wt_num
    for i in range(1, t_n):
        j = (i % wt_num) * (dt_num // wt_num) + (i // wt_num)
        tmp[i] = tmp[i] ^ tmp[i-1]
        data[0][j] = tmp[i]

    return data[0]


def layerNoWeight(weight, data):
    data = convolution(data, weight)
    return data


def readImages(direct): #direct = './train/'
    data = []
    for root, dirs, file in os.walk(direct):
        for fl in file:
            if os.path.splitext(fl)[1] == '.png':
                # image = np.load(direct+fl)
                image = plt.imread(direct+fl)
                image = np.array(image[:, :, 0] * 255, np.uint8)
                data.append(image)
    return data

#Count how often each value occurs
def distribute(data, maxv=256):
    leng = len(data)
    uni, count = np.unique(data, return_counts=True)
    return np.pad(count, (0, 256 - len(count)), mode='constant')/leng #(1,256)

#Calculates the KL divergence value of a uniform distribution over a given data distribution
uniform = [1./256. for i in range(0, 256)]
def computUniform(data, pre=uniform):
    data = data + np.spacing(1)
    div = np.dot(pre, np.log(pre / data)) 
    return div

def allImageToLayer(data, n1=1, n2=2): #0(r), 1(g), 2(b), 3(grey)
    wt = convolution_weight(n1, n2) #get initialized weights
    num = len(data)
    xordata = []
    div0 = 0
    for i in range(0, num):
        I0 = data[i] 
        I0 = I0.reshape(I0.size)
        xordt0 = layerNoWeight(wt, I0) 
        dis0 = distribute(xordt0)
        div0 += computUniform(dis0)
        xordt0 = xordt0.reshape(data[i].shape[0], data[i].shape[1])
        xordata.append(xordt0)
    return wt, xordata, div0

def execute(data, n_layer):
    ###test
    wt, data, div0 = allImageToLayer(data, n1=1, n2=5)

    print('layer %d/%d div %f' % (n_layer, num_layers, div0))
    return wt, data
    ###test


data = readImages('./database/')
weights = []
num_layers = 4
for i in range(0, num_layers):
    wt, data = execute(data, i+1)
    weights.append(wt)
np.save('./data_stack/3x10/weights.npy', weights)
plt.imshow(data[0], cmap='gray')
plt.show()

for i in range(0, len(data)):
    np.save('./data_stack/3x10/encrypt'+str(i)+'.npy', data[i])
plt.imsave('./data_stack/3x10/encrypt0.png', data[0], cmap='gray');








