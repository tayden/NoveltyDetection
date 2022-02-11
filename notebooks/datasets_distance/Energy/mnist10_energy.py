
# coding: utf-8

# In[1]:


# Implementation from https://github.com/dougalsutherland/opt-mmd

import sys, os
import numpy as np
from math import sqrt

CHANNEL_MEANS = (33.3184589445591/255,)
CHANNEL_STDS = (78.56747969985008/255,)


# In[2]:


from scipy.spatial.distance import pdist, cdist

def energy_distance(v, w):
    VV = np.mean(pdist(v, 'euclidean'))
    WW = np.mean(pdist(w, 'euclidean'))
    VW = np.mean(cdist(v, w, 'euclidean'))
    
    return 2 * VW - VV - WW


# In[3]:


from PIL import Image
from matplotlib import pyplot as plt

def display_sample(sample):
    img = sample.reshape((28, 28)) * 255.
    plt.imshow(Image.fromarray(img))
    plt.show()


# ## Compare all MNIST datasest

# In[4]:


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

# In[5]:


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


# In[6]:


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

fashion_energy = energy_distance(mnistTestX, fashionTestX)
print("Fashion Energy:", fashion_energy)


# In[7]:


def get_emnist_letters_test_data(emnist_letters_dir):
    """
    Return flattened and scaled EMNIST Letters test data as a numpy array.
    
    Saves/loads dataset from emnist_letters_dir.
    """
    print("Loading EMNIST Letters")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CHANNEL_MEANS, CHANNEL_STDS)
    ])
    
    dataset = datasets.EMNIST(emnist_letters_dir, "letters", train=False, download=True, transform=transform)
    dataset = np.array([a[0].numpy() for a in dataset])
    dataset = dataset.astype('float32')
    return dataset.reshape(dataset.shape[0], 784)

emnistTestX = get_emnist_letters_test_data('/media/tadenoud/DATADisk/datasets/emnist/')

emnist_energy = energy_distance(mnistTestX, emnistTestX)
print("EMNIST Letters Energy:", emnist_energy)


# In[8]:


def get_notmnist_test_data(notmnist_dir):
    """
    Return flattened and scaled NotMNIST test data as a numpy array.
    
    Loads dataset from notmnist_dir.
    """
    print("Loading NotMNIST")
    
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(CHANNEL_MEANS, CHANNEL_STDS)
    ])
    
    dataset = datasets.ImageFolder(notmnist_dir, transform=transform),
    dataset = np.array([a[0].numpy() for a in dataset[0]])
    dataset = dataset.astype('float32')
    return dataset.reshape(dataset.shape[0], 784)


notmnistTestX = get_notmnist_test_data('/media/tadenoud/DATADisk/datasets/notmnist/')

notmnist_energy = energy_distance(mnistTestX, notmnistTestX)
print("NotMNIST Energy:", notmnist_energy)


# In[9]:


def get_mnist_rot90_test_data(mnist_dir):
    """
    Return 90 degree rotated, flattened, and scaled MNIST test data as a numpy array.
    
    Loads dataset from notmnist_dir.
    """
    print("Loading MNIST rot90")
    
    transform = transforms.Compose([
        transforms.Lambda(lambda image: image.rotate(90)),
        transforms.ToTensor(),
        transforms.Normalize(CHANNEL_MEANS, CHANNEL_STDS)
    ])
    
    dataset = datasets.MNIST(mnist_dir, transform=transform, train=False, download=True)
    dataset = np.array([a[0].numpy() for a in dataset])
    dataset = dataset.astype('float32')
    return dataset.reshape(dataset.shape[0], 784)

mnistRot90TestX = get_mnist_rot90_test_data('/media/tadenoud/DATADisk/datasets/mnist/')

mnistrot90_energy = energy_distance(mnistTestX, mnistRot90TestX)

# display_sample(mnistRot90TestX[0])
# display_sample(mnistTestX[0])
print("MNIST rot90 Energy:", mnistrot90_energy)


# In[10]:


def get_gaussian_test_data():
    """Return flattened, and scaled Gaussian Noise test data as a numpy array."""
    print("Loading Gaussian Noise data")
    
    dataset = GaussianNoiseDataset((10000, 784), mean=0., std=1.)
    dataset = np.array([a for a, _ in iter(dataset)])
    return dataset.astype('float32')

gaussianTestX = get_gaussian_test_data()

gaussian_energy = energy_distance(mnistTestX, gaussianTestX)
print("Gaussian Energy:", gaussian_energy)


# In[11]:


import math

def get_uniform_test_data():
    """Return flattened, and scaled Uniform Noise test data as a numpy array."""
    print("Loading Uniform Noise data")
    
    dataset = UniformNoiseDataset((10000, 784), low=-math.sqrt(3), high=math.sqrt(3))
    dataset = np.array([a for a, _ in iter(dataset)])
    return dataset.astype('float32')


uniformTestX = get_uniform_test_data()

uniform_energy = energy_distance(mnistTestX, uniformTestX)
print("Uniform Energy:", uniform_energy)


# # MNIST 0-9 results

# In[12]:


import pandas as pd
from IPython.display import display

df = pd.DataFrame(columns=['energy'],
                  index=['fashion', 'letters', 'not_mnist', 'rot90', 'gaussian', 'uniform'])

df.loc['fashion'] = pd.Series({'energy': fashion_energy})
df.loc['letters'] = pd.Series({'energy': emnist_energy})
df.loc['not_mnist'] = pd.Series({'energy': notmnist_energy})
df.loc['rot90'] = pd.Series({'energy': mnistrot90_energy})
df.loc['gaussian'] = pd.Series({'energy': gaussian_energy})
df.loc['uniform'] = pd.Series({'energy': uniform_energy})

df = df.sort_values(by=['energy'])

display(df)


# In[13]:


df.to_pickle('../results/mnist10_energy.pkl')

