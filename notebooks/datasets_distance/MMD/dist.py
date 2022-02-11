
# coding: utf-8

# In[135]:


# Implementation from https://github.com/dougalsutherland/opt-mmd

import sys, os
import numpy as np
from math import sqrt

CHANNEL_MEANS = (33.791240975260735/255,)
CHANNEL_STDS = (79.17246803641319/255,)


# In[136]:


def kernelwidth_old(x1, x2):
    '''Function to estimate the sigma parameter
    
       The RBF kernel width sigma is computed according to a rule of thumb: 

       Pick sigma such that the exponent of exp(- ||x-y|| / (2*sigma2)),
       in other words ||x-y|| / (2*sigma2),  equals 1 for the median distance x
       and y of all distances between points from both data sets X and Y.
    '''
    n, nfeatures = x1.shape
    m, mfeatures = x2.shape
    
    k1 = np.sum((x1*x1), 1)
    q = np.tile(k1, (m, 1)).transpose()
    del k1
    
    k2 = np.sum((x2*x2), 1)
    r = np.tile(k2, (n, 1))
    del k2
    
    h= q + r
    del q,r
    
    # The norm
    h = h - 2*np.dot(x1,x2.transpose())
    h = np.array(h, dtype=float)
    
    mdist = np.median([i for i in h.flat if i])
    
    sigma = sqrt(mdist/2.0)
    if not sigma: sigma = 1
    
    return sigma


# ## Compare all MNIST datasest

# In[137]:


# Add Bayesian-and-novelty directory to the PYTHONPATH
import sys
import os
sys.path.append(os.path.realpath('../../../..'))

# Autoreload changes in utils, etc.
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import torch
from torchvision import datasets, transforms
import numpy as np

from novelty.utils.datasets import GaussianNoiseDataset
from novelty.utils.datasets import UniformNoiseDataset
from novelty.utils import DatasetSubset


torch.manual_seed(1)


# # MNIST 0-9

# In[138]:


def get_mnist_test_data(mnist_dir):
    """
    Return flattened and scaled MNIST test data as a numpy array.
    
    Saves/loads dataset from mnist_dir.
    """
    print("Loading MNIST test")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CHANNEL_MEANS, CHANNEL_STDS)
    ])

    dataset = datasets.MNIST(mnist_dir, train=False, download=True, transform=transform)
    dataset = np.array([a[0].numpy() for a in dataset])
    dataset = dataset.astype('float32')
    return dataset.reshape(dataset.shape[0], 784)

mnistTestX = get_mnist_test_data('/media/tadenoud/DATADisk/datasets/mnist/')


# In[139]:


def get_fashion_mnist_test_data(fashion_mnist_dir):
    """
    Return flattened and scaled Fashion MNIST test data as a numpy array.
    
    Saves/loads dataset from fashion_mnist_dir.
    """
    print("Loading Fashion MNIST")
      
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CHANNEL_MEANS, CHANNEL_STDS)
    ])
    
    dataset = datasets.FashionMNIST(fashion_mnist_dir, train=False, download=True, transform=transform)
    dataset = np.array([a[0].numpy() for a in dataset])
    dataset = dataset.astype('float32')
    return dataset.reshape(dataset.shape[0], 784)

fashionTestX = get_fashion_mnist_test_data('/media/tadenoud/DATADisk/datasets/fashion_mnist/')


# # Distance function

# In[166]:


from sklearn.metrics.pairwise import euclidean_distances

a = np.array([
    [1, 2],
    [2, 3],
    [5, 6],
])


def kernelwidth(X, Y, zeros=True):
    X = np.concatenate((X, Y), axis=0)
    res = euclidean_distances(X, X)

    # Get only upper triangle values
    # Removes distances between elements and their self from median calc
    if not zeros:
        res = res[np.triu_indices(len(res), 1)]
    
    return np.median(res)


# In[167]:


def pairwise_distance(X, Y):   
    XX = np.dot(X, X.T)
    
    if X is Y:
        YY = XX.T
    else:
        YY = np.dot(Y, Y.T)
    
    distances = np.dot(X, Y.T)
    
    distances *= -2
    distances += np.diag(XX)[:, np.newaxis]
    distances += np.diag(YY)[np.newaxis, :]
 
    return np.sqrt(distances)

def kernelwidth_new(X, Y):
    X = np.concatenate((X, Y), axis=0)
    return np.median(pairwise_distance(X, X))

res1 = kernelwidth(a, a)
res2 = kernelwidth_new(a, a)

