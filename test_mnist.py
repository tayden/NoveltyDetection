# coding: utf-8

import torch
from torchvision import datasets, transforms
import numpy as np
import math

from .utils.metrics import plot_roc
from .utils.metrics import get_summary_statistics
from .utils.datasets import GaussianNoiseDataset
from .utils.datasets import UniformNoiseDataset
from .utils import Progbar

torch.manual_seed(1)


def test_mnist(pred_function, batch_size=128, model_name="", plots=True,
                  mnist_dir='./datasets/mnist',
                  fashion_mnist_dir='./datasets/fashion_mnist',
                  emnist_dir='./datasets/emnist',
                  notmnist_dir='./datasets/notmnist'):
    """Test a novelty pred_function trained on MNIST using various datasets.

    - pred_function: np.array -> listof(Float)
        A decision function that consumes a batch of mnist images and returns a real value that indicates
        how "novel" a sample is. This value can be a scaled or unscaled reconstruction error value, distance from a
        decision function boundary, or similar.

    - batch_size: int (default 128)
        The batch size for processing samples.

    - model_name: Str (default "")
        The name of the model you are testing. Used in the title of output plots.

    - plots: Bool (default True)
        Flag to output graphs of ROC curve plots

    - mnist_dir: St (default "./datasets/mnist")
        The path to the MNIST directory on your computer. This data will be downloaded here if it doesn't exist.
        
    - fashion_mnist_dir: St (default "./datasets/fashion_mnist")
        The path to the FashionMNIST directory on your computer. This data will be downloaded here if it doesn't exist.
    
    - emnist_dir_dir: St (default "./datasets/emnist")
        The path to the EMNIST directory on your computer. This data will be downloaded here if it doesn't exist.
        
    - notmnist_dir: St (default "./datasets/notmnist/notMNIST_small")
        The path to the notMNIST_small directory on your computer. This data must be downloaded before the test is called.

    Returns summary statistics for the given pred_function as a Python dict. The dict contains FPR at 95% TPR, AUROC,
    AUPR_in, AUPR_out, and Error Rate metric values for each of fashion_mnist, emnist_letters, notmnist, mnist_rot90,
    gaussian noise, and uniform noise outlier datasets when compared to the MNIST test inlier dataset.

    """
    print("Running MNIST test suite")
    if model_name != "":
        model_name += " "

    inlier_data_loader = _get_mnist_test_loader(batch_size, mnist_dir)
    args = [inlier_data_loader, batch_size, model_name, plots]

    return {
        model_name: {
            "inlier_name": "MNIST",
            "outliers": {
                "fashion_mnist": _test_fashion_mnist_dataset(pred_function, fashion_mnist_dir, *args),
                "emnist_letters": _test_emnist_letters_dataset(pred_function, emnist_dir, *args),
                "notmnist": _test_notmnist_dataset(pred_function, notmnist_dir, *args),
                "mnist_rot90": _test_mnist_rot90_dataset(pred_function, mnist_dir, *args),
                "gaussian": _test_gaussian_noise_dataset(pred_function, *args),
                "uniform": _test_uniform_noise_dataset(pred_function, *args)
            }
        }
    }

def _get_mnist_test_loader(batch_size, mnist_dir):
    """Return a MNIST data loader depending on the value of the classes parameter.

    Data loader is batched to batch_size.
    """
    print("Loading inlier dataset")

    # Dataset transformation
    transform = transforms.ToTensor()

    # Load training and test sets
    kwargs = {"root": mnist_dir, "train": False, "transform": transform, "download": True}
    dataset = datasets.MNIST(**kwargs)

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


def _test_dataset(pred_function, inlier_data_loader, outlier_data_loader):
    num_inliers = len(inlier_data_loader.dataset.test_data)
    num_outliers = len(outlier_data_loader.dataset)
    
    progbar = Progbar(target=num_inliers + num_outliers)
    
    # Get predictions
    inlier_predictions = []
    for data, target in inlier_data_loader:
        inlier_predictions += list(pred_function(np.asarray(data)))
        progbar.add(len(data))

    outlier_predictions = []
    for data, target in outlier_data_loader:
        outlier_predictions += list(pred_function(np.asarray(data)))
        progbar.add(len(data))
        
    predictions = inlier_predictions + outlier_predictions
    labels = [1] * num_inliers + [0] * num_outliers
    
    return predictions, labels, get_summary_statistics(predictions, labels)


def _test_fashion_mnist_dataset(pred_function, fashion_mnist_dir, mnist_test_loader, batch_size, model_name, plots):
    print("Testing Fashion MNIST")
    
    transform = transforms.ToTensor()
    
    # Load the dataset
    fashion_mnist_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(fashion_mnist_dir, train=False, transform=transform, download=True),
        batch_size=batch_size, shuffle=False)

    predictions, labels, stats = _test_dataset(pred_function, mnist_test_loader, fashion_mnist_loader)

    if plots:
        plot_roc(predictions, labels, title=model_name + "Fashion MNIST ROC Curve")

    return stats


def _test_emnist_letters_dataset(pred_function, emnist_letters_dir, mnist_test_loader, batch_size, model_name, plots):
    print("Testing EMNIST Letters")
    
    transform = transforms.ToTensor()
    
    # Load the dataset
    emnist_letters_loader = torch.utils.data.DataLoader(
        datasets.EMNIST(emnist_letters_dir, "letters", train=False, transform=transform, download=True),
        batch_size=batch_size, shuffle=False)

    predictions, labels, stats = _test_dataset(pred_function, mnist_test_loader, emnist_letters_loader)

    if plots:
        plot_roc(predictions, labels, title=model_name + "EMNIST Letters ROC Curve")

    return stats


def _test_notmnist_dataset(pred_function, notmnist_dir, mnist_test_loader, batch_size, model_name, plots):
    print("Testing NotMNIST")
    
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    
    # Load the dataset
    notmnist_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(notmnist_dir, transform=transform),
        batch_size=batch_size, shuffle=False)

    predictions, labels, stats = _test_dataset(pred_function, mnist_test_loader, notmnist_loader)

    if plots:
        plot_roc(predictions, labels, title=model_name + "NotMNIST ROC Curve")

    return stats


def _test_mnist_rot90_dataset(pred_function, mnist_dir, mnist_test_loader, batch_size, model_name, plots):
    print("Testing MNIST (Rotated 90deg)")
    
    transform = transforms.Compose([
        transforms.Lambda(lambda image: image.rotate(90)),
        transforms.ToTensor()
    ])
    
    # Load the dataset
    mnist_rot_90_loader = torch.utils.data.DataLoader(
        datasets.MNIST(mnist_dir, train=False, transform=transform, download=True),
        batch_size=batch_size, shuffle=False)

    predictions, labels, stats = _test_dataset(pred_function, mnist_test_loader, mnist_rot_90_loader)

    if plots:
        plot_roc(predictions, labels, title=model_name + "MNIST (Rotated 90deg) ROC Curve")

    return stats


def _test_gaussian_noise_dataset(pred_function, mnist_test_loader, batch_size, model_name, plots):
    print("Testing Gaussian Noise")
    
    gaussian_transform = transforms.Compose([
        # TODO clip images to [0,1] domain
        transforms.ToTensor()
    ])
    
    # Load the dataset
    gaussian_loader = torch.utils.data.DataLoader(
        GaussianNoiseDataset((10000, 28, 28, 1), mean=0., std=1., transform=gaussian_transform),
        batch_size=batch_size, shuffle=True)

    predictions, labels, stats = _test_dataset(pred_function, mnist_test_loader, gaussian_loader)

    if plots:
        plot_roc(predictions, labels, title=model_name + "Gaussian Noise ROC Curve")

    return stats


def _test_uniform_noise_dataset(pred_function, mnist_test_loader, batch_size, model_name, plots):
    print("Testing Uniform Noise")
    
    uniform_transform = transforms.ToTensor()
    
    # Load the dataset
    uniform_loader = torch.utils.data.DataLoader(
        UniformNoiseDataset((10000, 28, 28, 1), low=-math.sqrt(3.), high=math.sqrt(3.), transform=uniform_transform),
        batch_size=batch_size, shuffle=True)

    predictions, labels, stats = _test_dataset(pred_function, mnist_test_loader, uniform_loader)

    if plots:
        plot_roc(predictions, labels, title=model_name + "Uniform Noise ROC Curve")

    return stats
