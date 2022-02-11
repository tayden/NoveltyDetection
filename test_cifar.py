# coding: utf-8

import torch
from torchvision import datasets, transforms
import numpy as np
import math

from utils.metrics import plot_roc
from utils.metrics import get_summary_statistics
from utils.datasets import GaussianNoiseDataset
from utils.datasets import UniformNoiseDataset
from utils import Progbar

torch.manual_seed(1)


def test_cifar_10(pred_function, batch_size=128, model_name="", plots=True,
                  cifar10_dir='./datasets/cifar10',
                  imagenet_dir='./datasets/imagenet/test',
                  lsun_dir='./datasets/lsun', isun_dir='./datasets/isun'):
    """Test a novelty pred_function trained on CIFAR10 using various datasets.

    - pred_function: np.array -> listof(Float)
        A decision function that consumes a batch of cifar images and returns a real value that indicates
        how "novel" a sample is. This value can be a scaled or unscaled reconstruction error value, distance from a
        decision function boundary, or similar.

    - batch_size: int (default 128)
        The batch size for processing samples.

    - model_name: Str (default "")
        The name of the model you are testing. Used in the title of output plots.

    - plots: Bool (default True)
        Flag to output graphs of ROC curve plots

    - cifar10_dir: St (default "./datasets/cifar10")
        The path to the cifar10 directory on your computer. This data will be downloaded here if it doesn't exist.
    
    - imagenet_dir: St (default "./datasets/imagenet/test")
        The path to the tiny-imagenet-200 test directory on your computer. Only the test data is required to be downloaded.

    - lsun_dir: St (default "./datasets/lsun")
        The path to the LSUN directory on your computer. Only the test data is required to be downloaded.

    - isun_dir: St (default "./datasets/isun")
        The path to the iSUN directory on your computer. Only the test data is required to be downloaded.


    Returns summary statistics for the given pred_function as a Python dict. The dict contains FPR at 95% TPR, AUROC,
    AUPR_in, AUPR_out, and Error Rate metric values for each of imagenet_crop, imagenet_resize, lsun_crop, lsun_resize,
    isun, gaussian noise, and uniform noise outlier datasets when compared to the CIFAR10 test inlier dataset.

    """
    print("Running CIFAR 10 test suite")
    if model_name != "":
        model_name += " "

    inlier_data_loader = _get_cifar_test_loader(10, batch_size, cifar10_dir)
    args = [inlier_data_loader, batch_size, model_name, plots]

    return {
        model_name: {
            "inlier_name": "CIFAR 10",
            "outliers": {
                "imagenet_crop": _test_imagenet_crop_dataset(pred_function, imagenet_dir, *args),
                "imagenet_resize": _test_imagenet_resize_dataset(pred_function, imagenet_dir, *args),
                "lsun_crop": _test_lsun_crop_dataset(pred_function, lsun_dir, *args),
                "lsun_resize": _test_lsun_resize_dataset(pred_function, lsun_dir, *args),
                "isun": _test_isun_dataset(pred_function, isun_dir, *args),
                "gaussian": _test_gaussian_noise_dataset(pred_function, *args),
                "uniform": _test_uniform_noise_dataset(pred_function, *args)
            }
        }
    }


def test_cifar_100(pred_function, batch_size=128, model_name="", plots=True,
                   cifar100_dir='./datasets/cifar100',
                   imagenet_dir='./datasets/imagenet/test',
                   lsun_dir='./datasets/lsun', isun_dir='./datasets/isun'):
    """Test a novelty pred_function trained on CIFAR100 using various datasets.

    - pred_function: np.array -> listof(Float)
        A decision function that consumes a batch of cifar images and returns a real value that indicates
        how "novel" a sample is. This value can be a scaled or unscaled reconstruction error value, distance from a
        decision function boundary, or similar.

    - batch_size: int (default 128)
        The batch size for processing samples.

    - model_name: Str (default "")
        The name of the model you are testing. Used in the title of output plots.

    - plots: Bool (default True)
        Flag to output graphs of ROC curve plots
        
    - cifar100_dir: St (default "./datasets/cifar100")
        The path to the cifar10 directory on your computer. This data will be downloaded here if it doesn't exist.

    - imagenet_dir: St (default "./datasets/imagenet/test")
        The path to the tiny-imagenet-200 test directory on your computer. Only the test data is required to be downloaded.

    - lsun_dir: St (default "./datasets/lsun")
        The path to the LSUN directory on your computer. Only the test data is required to be downloaded.

    - isun_dir: St (default "./datasets/isun")
        The path to the iSUN directory on your computer. Only the test data is required to be downloaded.


    Returns summary statistics for the given pred_function as a Python dict. The dict contains FPR at 95% TPR, AUROC,
    AUPR_in, AUPR_out, and Error Rate metric values for each of imagenet_crop, imagenet_resize, lsun_crop, lsun_resize,
    isun, gaussian noise, and uniform noise outlier datasets when compared to the CIFAR100 test inlier dataset.

    """
    print("Running CIFAR 100 test suite")
    
    if model_name != "":
        model_name += " "

    inlier_data_loader = _get_cifar_test_loader(100, batch_size, cifar100_dir)
    args = [inlier_data_loader, batch_size, model_name, plots]

    return {
        model_name: {
            "inlier_name": "CIFAR 100",
            "outliers": {
                "imagenet_crop": _test_imagenet_crop_dataset(pred_function, imagenet_dir, *args),
                "imagenet_resize": _test_imagenet_resize_dataset(pred_function, imagenet_dir, *args),
                "lsun_crop": _test_lsun_crop_dataset(pred_function, lsun_dir, *args),
                "lsun_resize": _test_lsun_resize_dataset(pred_function, lsun_dir, *args),
                "isun": _test_isun_dataset(pred_function, isun_dir, *args),
                "gaussian": _test_gaussian_noise_dataset(pred_function, *args),
                "uniform": _test_uniform_noise_dataset(pred_function, *args)
            }
        }
    }


def _get_cifar_test_loader(classes, batch_size, cifar_dir):
    """Return a CIFAR10 or CIFAR100 data loader depending on the value of the classes parameter.

    Data loader is batched to batch_size.
    """
    assert(classes == 10 or classes == 100)
    print("Loading inlier dataset")

    # Dataset transformation
    transform = transforms.ToTensor()

    # Load training and test sets
    kwargs = {"root": cifar_dir, "train": False, "transform": transform, "download": True}
    dataset = datasets.CIFAR10(**kwargs) if classes == 10 else datasets.CIFAR100(**kwargs)

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


def _test_imagenet_crop_dataset(pred_function, imagenet_dir, cifar_test_loader, batch_size, model_name, plots):
    print("Testing Tiny Imagenet (crop)")
    
    transform_crop = transforms.Compose([
        transforms.RandomCrop([32, 32]),
        transforms.ToTensor()
    ])

    # Load the dataset
    imagenet_crop_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(imagenet_dir, transform=transform_crop),
        batch_size=batch_size, shuffle=True)

    predictions, labels, stats = _test_dataset(pred_function, cifar_test_loader, imagenet_crop_loader)

    if plots:
        plot_roc(predictions, labels, title=model_name + "Imagenet (crop) ROC Curve")

    return stats


def _test_imagenet_resize_dataset(pred_function, imagenet_dir, cifar_test_loader, batch_size, model_name, plots):
    print("Testing Tiny Imagenet (resize)")
    
    transform_resize = transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.ToTensor()
    ])

    # Load the dataset
    imagenet_resize_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(imagenet_dir, transform=transform_resize),
        batch_size=batch_size, shuffle=True)

    predictions, labels, stats = _test_dataset(pred_function, cifar_test_loader, imagenet_resize_loader)

    if plots:
        plot_roc(predictions, labels, title=model_name + "Imagenet (resize) ROC Curve")

    return stats


def _test_lsun_crop_dataset(pred_function, lsun_dir, cifar_test_loader, batch_size, model_name, plots):
    print("Testing Tiny LSUN (crop)")
    
    transform_crop = transforms.Compose([
        transforms.RandomCrop([32, 32]),
        transforms.ToTensor()
    ])

    # Load the dataset
    lsun_crop_loader = torch.utils.data.DataLoader(
        datasets.LSUN(lsun_dir, classes='test', transform=transform_crop),
        batch_size=batch_size, shuffle=True)

    predictions, labels, stats = _test_dataset(pred_function, cifar_test_loader, lsun_crop_loader)

    if plots:
        plot_roc(predictions, labels, title=model_name + "LSUN (crop) ROC Curve")

    return stats


def _test_lsun_resize_dataset(pred_function, lsun_dir, cifar_test_loader, batch_size, model_name, plots):
    print("Testing Tiny LSUN (resize)")
    
    transform_resize = transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.ToTensor()
    ])

    # Load the dataset
    lsun_resize_loader = torch.utils.data.DataLoader(
        datasets.LSUN(lsun_dir, classes='test', transform=transform_resize),
        batch_size=batch_size, shuffle=True)

    predictions, labels, stats = _test_dataset(pred_function, cifar_test_loader, lsun_resize_loader)

    if plots:
        plot_roc(predictions, labels, title=model_name + "LSUN (resize) ROC Curve")

    return stats


def _test_isun_dataset(pred_function, isun_dir, cifar_test_loader, batch_size, model_name, plots):
    print("Testing Tiny iSUN")
    
    transform = transforms.ToTensor()

    # Load the dataset
    isun_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(isun_dir, transform=transform),
        batch_size=batch_size, shuffle=True)

    predictions, labels, stats = _test_dataset(pred_function, cifar_test_loader, isun_loader)

    if plots:
        plot_roc(predictions, labels, title=model_name + "iSUN ROC Curve")

    return stats


def _test_gaussian_noise_dataset(pred_function, cifar_test_loader, batch_size, model_name, plots):
    print("Testing Gaussian Noise")
    
    gaussian_transform = transforms.Compose([
        # TODO clip images to [0,1] domain
        transforms.ToTensor()
    ])

    # Load the dataset
    gaussian_loader = torch.utils.data.DataLoader(
        GaussianNoiseDataset((10000, 32, 32, 3), mean=0., std=1., transform=gaussian_transform),
        batch_size=batch_size, shuffle=True)

    predictions, labels, stats = _test_dataset(pred_function, cifar_test_loader, gaussian_loader)

    if plots:
        plot_roc(predictions, labels, title=model_name + "Gaussian Noise ROC Curve")

    return stats


def _test_uniform_noise_dataset(pred_function, cifar_test_loader, batch_size, model_name, plots):
    print("Testing Uniform Noise")
    
    uniform_transform = transforms.ToTensor()
    
    # Load the dataset
    uniform_loader = torch.utils.data.DataLoader(
        UniformNoiseDataset((10000, 32, 32, 3), low=math.sqrt(3), high=math.sqrt(3), transform=uniform_transform),
        batch_size=batch_size, shuffle=True)

    predictions, labels, stats = _test_dataset(pred_function, cifar_test_loader, uniform_loader)

    if plots:
        plot_roc(predictions, labels, title=model_name + "Uniform Noise ROC Curve")

    return stats
