#!/usr/bin/env python3

import torchvision as tv
import learn2learn as l2l

from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels, ConsecutiveLabels
from torchvision import transforms
import numpy as np
def cifarfs_tasksets(
    train_ways=5,
    train_samples=10,
    test_ways=5,
    test_samples=10,
    root='~/data',
    orig_label = False,
    data_augmentation=None,
    **kwargs,
):
    """Tasksets for CIFAR-FS benchmarks."""
    if data_augmentation is None:
        train_data_transform = tv.transforms.ToTensor()
        test_data_transform = tv.transforms.ToTensor()
    elif data_augmentation == 'lee2019':
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])

        train_data_transform = transforms.Compose([
                # lambda x: Image.fromarray(x),
                transforms.RandomCrop(32, padding=4),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                # lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize
            ])
        test_data_transform = transforms.Compose([
                # lambda x: Image.fromarray(x),
                transforms.ToTensor(),
                normalize
            ])
    elif data_augmentation == 'normalize':
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        train_data_transform = transforms.Compose([
                # lambda x: Image.fromarray(x),
                transforms.ToTensor(),
                normalize
            ])
        test_data_transform = transforms.Compose([
                # lambda x: Image.fromarray(x),
                transforms.ToTensor(),
                normalize
            ])
    else:
        raise ValueError()
    train_dataset = l2l.vision.datasets.CIFARFS(root=root,
                                                transform=train_data_transform,
                                                mode='train',
                                                download=True)
    valid_dataset = l2l.vision.datasets.CIFARFS(root=root,
                                                transform=test_data_transform,
                                                mode='validation',
                                                download=True)
    test_dataset = l2l.vision.datasets.CIFARFS(root=root,
                                               transform=test_data_transform,
                                               mode='test',
                                               download=True)
    train_dataset = l2l.data.MetaDataset(train_dataset)
    valid_dataset = l2l.data.MetaDataset(valid_dataset)
    test_dataset = l2l.data.MetaDataset(test_dataset)

    train_transforms = [
        NWays(train_dataset, train_ways),
        KShots(train_dataset, train_samples),
        LoadData(train_dataset),
        RemapLabels(train_dataset),
        ConsecutiveLabels(train_dataset),
    ]
    valid_transforms = [
        NWays(valid_dataset, test_ways),
        KShots(valid_dataset, test_samples),
        LoadData(valid_dataset),
        ConsecutiveLabels(valid_dataset),
        RemapLabels(valid_dataset),
    ]
    test_transforms = [
        NWays(test_dataset, test_ways),
        KShots(test_dataset, test_samples),
        LoadData(test_dataset),
        RemapLabels(test_dataset),
        ConsecutiveLabels(test_dataset),
    ]
    if orig_label:
        print()
        print('--------------------------------')
        print('No relabelling in training tasks.')
        print('--------------------------------')

        train_transforms = train_transforms[:3] # Remove the relabelling part

    _datasets = (train_dataset, valid_dataset, test_dataset)
    _transforms = (train_transforms, valid_transforms, test_transforms)
    return _datasets, _transforms
