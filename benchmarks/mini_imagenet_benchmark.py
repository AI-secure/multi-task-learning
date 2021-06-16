#!/usr/bin/env python3

import learn2learn as l2l

from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels, ConsecutiveLabels
from torchvision.transforms import (Compose, ToPILImage, ToTensor, RandomCrop, RandomHorizontalFlip,
                                    ColorJitter, Normalize)
from datasets import MiniImagenet
from torchvision import transforms
import numpy as np
import PIL
import torch
import os
def mini_imagenet_tasksets(
    train_ways=5,
    train_samples=10,
    test_ways=5,
    test_samples=10,
    root='~/data',
    data_augmentation=None,
    orig_label=False,
    trainval=False,
    **kwargs,
):
    """Tasksets for mini-ImageNet benchmarks."""
    if data_augmentation is None:
        train_data_transforms = None
        test_data_transforms = None
        print('===========================')
        print('No Data Transformation')

    elif data_augmentation == 'normalize':
        # TODO: this may need to permute dimensions
        train_data_transforms = Compose([
            lambda x: torch.from_numpy(x) / 255.0,
        ])
        test_data_transforms = train_data_transforms

    elif data_augmentation == 'lee2019':
        normalize = Normalize(
            mean=[120.39586422/255.0, 115.59361427/255.0, 104.54012653/255.0],
            std=[70.68188272/255.0, 68.27635443/255.0, 72.54505529/255.0],
        )
        train_data_transforms = Compose([
            lambda x: PIL.Image.fromarray(x.astype('uint8')),
            RandomCrop(84, padding=8),
            ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            RandomHorizontalFlip(),
            # lambda x: np.asarray(x),
            ToTensor(),
            normalize,
        ])
        test_data_transforms = Compose([
            lambda x: PIL.Image.fromarray(x.astype('uint8')),
            ToTensor(),
            normalize,
        ])
    else:
        raise('Invalid data_augmentation argument.')

    train_dataset = MiniImagenet(
        root=root,
        mode='train',
        transform=train_data_transforms,
        download=True,
    )
    valid_dataset = MiniImagenet(
        root=root,
        mode='validation',
        transform=test_data_transforms,
        download=True,
    )
    test_dataset = MiniImagenet(
        root=root,
        mode='test',
        transform=test_data_transforms,
        download=True,
    )

    if trainval:

        train_dataset = l2l.data.UnionMetaDataset([train_dataset, valid_dataset])
        valid_dataset = test_dataset

        print('-------------------------------')
        print('Merged Training and Validation sets for training. Use Test set for validation.')

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
        # Keep the original labels by removing the relabelling function
        print('\n--------------------------------')
        print('No relabelling in training tasks.')
        print('--------------------------------')
        train_transforms = train_transforms[:3] # Remove the relabelling part



    _datasets = (train_dataset, valid_dataset, test_dataset)
    _transforms = (train_transforms, valid_transforms, test_transforms)
    return _datasets, _transforms
