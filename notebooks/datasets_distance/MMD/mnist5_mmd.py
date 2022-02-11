
# coding: utf-8

# In[1]:


# Implementation from https://github.com/dougalsutherland/opt-mmd

import sys, os
import numpy as np
from math import sqrt

CHANNEL_MEANS = (33.430001959204674/255,)
CHANNEL_STDS = (78.86655405163765/255,)


# In[2]:


from sklearn.metrics.pairwise import euclidean_distances

def kernelwidth(X, Y):
    X = np.concatenate((X, Y), axis=0)
    res = euclidean_distances(X, X)
    return np.median(res)


# In[3]:


def rbf_mmd2(X, Y, sigma=0, biased=True):
    gamma = 1 / (2 * sigma**2)
    
    XX = np.dot(X, X.T)
    XY = np.dot(X, Y.T)
    YY = np.dot(Y, Y.T)
    
    X_sqnorms = np.diag(XX)
    Y_sqnorms = np.diag(YY)
    
    K_XY = np.exp(-gamma * (
        -2 * XY + X_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]))
    K_XX = np.exp(-gamma * (
        -2 * XX + X_sqnorms[:, np.newaxis] + X_sqnorms[np.newaxis, :]))
    K_YY = np.exp(-gamma * (
        -2 * YY + Y_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]))
    
    if biased:
        mmd2 = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
    else:
        m = K_XX.shape[0]
        n = K_YY.shape[0]

        mmd2 = ((K_XX.sum() - m) / (m * (m - 1))
              + (K_YY.sum() - n) / (n * (n - 1))
              - 2 * K_XY.mean())
    return mmd2


# In[4]:


from PIL import Image
from matplotlib import pyplot as plt

def display_sample(sample):
    img = sample.reshape((28, 28)) * 255.
    plt.imshow(Image.fromarray(img))
    plt.show()


# ## Compare all MNIST datasest

# In[5]:


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


# # MNIST 0-4

# In[8]:


CHANNEL_MEANS = (33.550922870635986/255,)
CHANNEL_STDS = (79.10186022520065/255,)

def get_mnist_images(mnist_dir):  
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(CHANNEL_MEANS, CHANNEL_STDS)
    ])
    
    dataset = datasets.ImageFolder(mnist_dir, transform=transform)
    dataset = np.array([a[0].numpy() for a in dataset])
    print(dataset.shape)
    dataset = dataset.astype('float32')
    return dataset.reshape(dataset.shape[0], 784)


mnist_test_0_4 = get_mnist_images('/media/tadenoud/DATADisk/datasets/mnist0_4/test')


# In[7]:


mnist_test_5_9 = get_mnist_images('/media/tadenoud/DATADisk/datasets/mnist5_9/test')

mnist_split_sigma = kernelwidth(mnist_test_0_4, mnist_test_5_9)
mnist_split_mmd = rbf_mmd2(mnist_test_0_4, mnist_test_5_9, sigma=mnist_split_sigma)
print("Split MNIST MMD:", mnist_split_mmd)


# In[8]:


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

fashion_sigma = kernelwidth(mnist_test_0_4, fashionTestX)
fashion_mmd = rbf_mmd2(mnist_test_0_4, fashionTestX, sigma=fashion_sigma)
print("Fashion MMD:", fashion_mmd)


# In[9]:


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

emnist_sigma = kernelwidth(mnist_test_0_4, emnistTestX)
emnist_mmd = rbf_mmd2(mnist_test_0_4, emnistTestX, sigma=emnist_sigma)
print("EMNIST Letters MMD:", emnist_mmd)


# In[10]:


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

notmnist_sigma = kernelwidth(mnist_test_0_4, notmnistTestX)
notmnist_mmd = rbf_mmd2(mnist_test_0_4, notmnistTestX, sigma=notmnist_sigma)
print("NotMNIST MMD:", notmnist_mmd)


# In[11]:


def get_mnist_0_4_rot90_test_data(mnist_dir):
    """
    Return 90 degree rotated, flattened, and scaled MNIST test data as a numpy array containing only digits 0-4.
    
    Loads dataset from notmnist_dir.
    """
    print("Loading MNIST 0-4 rot90")
    
    transform = transforms.Compose([
        transforms.Lambda(lambda image: image.rotate(90)),
        transforms.ToTensor(),
        transforms.Normalize(CHANNEL_MEANS, CHANNEL_STDS)
    ])
    
    dataset = DatasetSubset(datasets.MNIST(mnist_dir, transform=transform, train=False, download=True), 
                            [0,1,2,3,4], train=False)
    dataset = np.array([a[0].numpy() for a in dataset])
    dataset = dataset.astype('float32')
    return dataset.reshape(dataset.shape[0], 784)


mnistRot90TestX = get_mnist_0_4_rot90_test_data('/media/tadenoud/DATADisk/datasets/mnist/')

mnistrot90_sigma = kernelwidth(mnist_test_0_4, mnistRot90TestX)
mnistrot90_mmd = rbf_mmd2(mnist_test_0_4, mnistRot90TestX, sigma=mnistrot90_sigma)

# display_sample(mnistRot90TestX[0])
# display_sample(mnist_test_0_4[0])
print("MNIST rot90 MMD:", mnistrot90_mmd)


# In[12]:


def get_gaussian_test_data():
    """Return flattened, and scaled Gaussian Noise test data as a numpy array."""
    print("Loading Gaussian Noise data")
    
    dataset = GaussianNoiseDataset((10000, 784), mean=0., std=1.)
    dataset = np.array([a for a, _ in iter(dataset)])
    return dataset.astype('float32')

gaussianTestX = get_gaussian_test_data()

gaussian_sigma = kernelwidth(mnist_test_0_4, gaussianTestX)
gaussian_mmd = rbf_mmd2(mnist_test_0_4, gaussianTestX, sigma=gaussian_sigma)
print("Gaussian MMD:", gaussian_mmd)


# In[13]:


import math

def get_uniform_test_data():
    """Return flattened, and scaled Uniform Noise test data as a numpy array."""
    print("Loading Uniform Noise data")
    
    dataset = UniformNoiseDataset((10000, 784), low=-math.sqrt(3), high=math.sqrt(3))
    dataset = np.array([a for a, _ in iter(dataset)])
    return dataset.astype('float32')

uniformTestX = get_uniform_test_data()

uniform_sigma = kernelwidth(mnist_test_0_4, uniformTestX)
uniform_mmd = rbf_mmd2(mnist_test_0_4, uniformTestX, sigma=uniform_sigma)
print("Uniform MMD:", uniform_mmd)


# # MNIST 0-4  results

# In[14]:


import pandas as pd

df = pd.DataFrame(columns=['mmd', 'sigma'],
                  index=['5-9', 'fashion', 'letters', 'not_mnist', 'rot90', 'gaussian', 'uniform'])

df.loc['5-9'] = pd.Series({'mmd': mnist_split_mmd, 'sigma': mnist_split_sigma})
df.loc['fashion'] = pd.Series({'mmd': fashion_mmd, 'sigma': fashion_sigma})
df.loc['letters'] = pd.Series({'mmd': emnist_mmd, 'sigma': emnist_sigma})
df.loc['not_mnist'] = pd.Series({'mmd': notmnist_mmd, 'sigma': notmnist_sigma})
df.loc['rot90'] = pd.Series({'mmd': mnistrot90_mmd, 'sigma': mnistrot90_sigma})
df.loc['gaussian'] = pd.Series({'mmd': gaussian_mmd, 'sigma': gaussian_sigma})
df.loc['uniform'] = pd.Series({'mmd': uniform_mmd, 'sigma': uniform_sigma})

df = df.sort_values(by=['mmd'])

display(df)


# In[15]:


df.to_pickle('../results/mnist5_mmd.pkl')


# ## Calculate dataset means

# In[8]:


CHANNEL_MEANS = (33.430001959204674/255,)
CHANNEL_STDS = (78.86655405163765/255,)


def _get_mnist_images(mnist_dir):  
    """
    Return MNIST image data as a numpy array.
    Saves/loads dataset from mnist_dir.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
#         transforms.Normalize(CHANNEL_MEANS, CHANNEL_STDS)
    ])
    
    dataset = datasets.ImageFolder(mnist_dir, transform=transform)
    dataset = np.array([a[0].numpy() for a in dataset])
    return dataset.astype('float32')

data = _get_mnist_images('/media/tadenoud/DATADisk/datasets/mnist0_4/train')
print(data.shape)


# In[9]:


means = []
val = np.reshape(data[:,0,:,:], -1)
mean = np.mean(val)
print('mean (%d): %f' % (0, mean))
means.append(mean*255)

print()

stds = []

val = np.reshape(data[:,0,:,:], -1)
std = np.std(val)
print('std (%d): %f' % (0, std))
stds.append(std*255)

print()
print('CHANNEL_MEANS = ({}/255)'.format(*means))
print('CHANNEL_STDS = ({}/255)'.format(*stds))

