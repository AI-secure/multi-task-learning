#!/usr/bin/env python3

"""

**Description**

Some datasets commonly used in meta-learning vision tasks.
"""

# Original datasets
from learn2learn.vision.datasets.full_omniglot import FullOmniglot
from learn2learn.vision.datasets.vgg_flowers import VGGFlower102
from learn2learn.vision.datasets.fgvc_aircraft import FGVCAircraft
from learn2learn.vision.datasets.cu_birds200 import CUBirds200
from learn2learn.vision.datasets.describable_textures import DescribableTextures
from learn2learn.vision.datasets.quickdraw import Quickdraw
from learn2learn.vision.datasets.fgvc_fungi import FGVCFungi
from learn2learn.vision.datasets.cifarfs import CIFARFS
from learn2learn.vision.datasets.fc100 import FC100

# Modified datasets
from .mini_imagenet import MiniImagenet
from .tiered_imagenet import TieredImagenet

