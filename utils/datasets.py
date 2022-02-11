from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np


class ImageDirDataset(Dataset):
    """Fetch image dataset from a directory.
    
    The directory where the images are located should contain only subdirectories with names
    that correspond to different image labels. Each of those subdirectories will themselves 
    contain images that can be opened by PIL.
    
    img_dir: string
        The path to the top level of the image dataset that matches the format in the class
        description.
        
    transform: function, optional
        Any pytorch transforms to perform on the image data when loaded.
    """
    def __init__(self, img_dir, transform=None):
        super(ImageDirDataset, self).__init__()
        self.img_dir = os.path.realpath(img_dir)
        self.transform = transform
                
        self.img_arr = np.array([])
        self.label_arr = np.array([])
        
        # Using top level directories as labels, get all image paths
        for label in os.listdir(self.img_dir):
            label_dir = os.path.join(self.img_dir, label)
            imgs = os.listdir(label_dir)
            
            img_paths = [os.path.join(label_dir, img) for img in imgs]
            labels = [label] * len(imgs)
            
            self.img_arr = np.append(self.img_arr, img_paths)
            self.label_arr = np.append(self.label_arr, labels)
            
        # Calculate data length
        self.data_len = self.label_arr.shape[0]
        
    def __len__(self):
        return self.data_len
    
    def __getitem__(self, idx):
        """Get image from the img_dir."""
        # Get image name
        single_image_name = self.img_arr[idx]
        # Open image
        img = Image.open(single_image_name)
        
        # Do transform operations
        if self.transform is not None:
            img = self.transform(img)
        
        # Get image label
        label = self.label_arr[idx]
        
        return (img, label)
    
    
from torch.utils.data import Dataset


class NoiseDataset(Dataset):
    """Abstract base class for noise datasets.
    Subclasses should re-implement __init__ and set self.data to a dataset with shape  of param shape.
    
    shape: tuple
        A tuple defining the shape you want the generated noise to be in channel last order.
        
    transform: function, optional
        Any pytorch transforms to perform on the image data when loaded.
    """
    def __init__(self, shape, transform=None):
        super(NoiseDataset, self).__init__()
        self.length = shape[0]
        self.shape = shape
        self.transform = transform
        
        # e.g. in subclasses
        # self.data = np.random.normal(0, 1, size=self.shape).astype("float32")
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        noise = self.data[idx]
        
        # Do transform operations
        if self.transform is not None:
            noise = self.transform(noise)

        label = 0
        return (noise, label)


class GaussianNoiseDataset(NoiseDataset):
    """Generate Gaussian noise images with mean and std.
             
    shape: tuple
        A tuple defining the shape you want the generated noise to be in channel last order.
        
    transform: function, optional
        Any pytorch transforms to perform on the image data when loaded.
    
    Example:
    gaussian_noise_data = GaussianNoiseDataset((10000, 28, 28, 1), transform=transforms.ToTensor())
    """
    def __init__(self, shape, mean=0, std=1, transform=None):
        super(GaussianNoiseDataset, self).__init__(shape, transform)
        self.mean = mean
        self.std = std
        self.data = np.random.normal(mean, std, size=shape).astype("float32")    
    
    
class UniformNoiseDataset(NoiseDataset):
    """Generate Uniform noise images between low and high, inclusive.

    shape: tuple
        A tuple defining the shape you want the generated noise to be in channel last order.
        
    transform: function, optional
        Any pytorch transforms to perform on the image data when loaded.
        
    Example:
    uniform_noise_data = UniformNoiseDataset((10000, 28, 28, 1), transform=transforms.ToTensor())
    """
    def __init__(self, shape, low=0, high=1, transform=None):
        super(UniformNoiseDataset, self).__init__(shape, transform)
        self.low = low
        self.high = high
        self.data = np.random.uniform(low, high, size=shape).astype("float32")
        
        
class DatasetSubset:
    """Create a new training dataset using some subset of class labels.
    
    dataset: instanceof(torch.utils.data.Dataset)
        The dataset you want to filter.
    
    inlier_labels: list or tuple:
        A list of the inlier data class labels you want to keep in
        the new dataset.
        
    transform: instanceof(torchvision.transforms)
        Any transforms you want to do to apply to the data.
        You may not need any transforms if they were already applied
        on dataset.
        
    train: bool
        Flag indicating if you are splitting the training or testing set.
    """
    def __init__(self, dataset, inlier_labels, transform=None, train=True):
        super(DatasetSubset, self).__init__()

        self.inlier_labels = inlier_labels
        self.dataset = dataset
        self.transform = transform

        if train:
            self.inlier_indices = [
                i for i, x in enumerate(self.dataset.train_labels) 
                if x in self.inlier_labels]
        else:
            self.inlier_indices = [
                i for i, x in enumerate(self.dataset.test_labels) 
                if x in self.inlier_labels]
        
        self.data = []
        self.labels = []
        
        for i in self.inlier_indices:
            x, y = self.dataset[i]
            self.data.append(x)
            self.labels.append(y)
            
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        
        if self.transform is not None:
            x = self.transform(x)

        return (x, y)