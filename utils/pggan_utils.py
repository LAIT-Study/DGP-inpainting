from __future__ import print_function

import math
import os
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim.optimizer import Optimizer



def join_strings(base_string, strings):
    return base_string.join([item for item in strings if item])

def prepare_parser():
    usage = 'Parser for all scripts.'
    parser = ArgumentParser(description=usage)
    ### Bookkeping stuff ###  
    parser.add_argument(
        '--weights_root', type=str, default='weights',
        help='Default location to store weights (default: %(default)s)')
        
     ### Resume training stuff
    parser.add_argument(
        '--load_weights', type=str, default='',
        help='Suffix for which weights to load (e.g. best0, copy0) '
             '(default: %(default)s)')
    ### Model init stuff ###
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Random seed to use; affects both initialization and '
             ' dataloading. (default: %(default)s)')
    ### Log stuff ###
    parser.add_argument(
        '--no_tb', action='store_true', default=False,
        help='Do not use tensorboard? '
             '(default: %(default)s)')
    parser.add_argument(
        '--G_lr', type=float, default=1e-3,
        help='Learning rate to use for Generator (default: %(default)s)')
    parser.add_argument(
        '--D_lr', type=float, default=2e-4,
        help='Learning rate to use for Discriminator (default: %(default)s)')
    parser.add_argument(
        '--Z_lr_mult', type=float, default=50,
        help='Learning rate multiplication to use for Z (default: %(default)s)')
    parser.add_argument(
        '--G_B1', type=float, default=0.0,
        help='Beta1 to use for Generator (default: %(default)s)')
    parser.add_argument(
        '--D_B1', type=float, default=0.0,
        help='Beta1 to use for Discriminator (default: %(default)s)')
    parser.add_argument(
        '--G_B2', type=float, default=0.99,
        help='Beta2 to use for Generator (default: %(default)s)')
    parser.add_argument(
        '--D_B2', type=float, default=0.99,
        help='Beta2 to use for Discriminator (default: %(default)s)')
    parser.add_argument(
        '--n_classes', type=int, default=1000,
        help='Number of class conditions %(default)s)')
    return parser

class RandomCropLongEdge(object):
    """Crops the given PIL Image on the long edge with a random start point.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        size = (min(img.size), min(img.size))
        # Only step forward along this edge if it's the long edge
        i = (0 if size[0] == img.size[0] else np.random.randint(
            low=0, high=img.size[0] - size[0]))
        j = (0 if size[1] == img.size[1] else np.random.randint(
            low=0, high=img.size[1] - size[1]))
        return transforms.functional.crop(img, i, j, size[0], size[1])

    def __repr__(self):
        return self.__class__.__name__


# Utility file to seed rngs
def seed_rng(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

# Load a model's weights
def load_weights(G,
                 D,
                 weights_root,
                 name_suffix=None,
                 G_ema=None,
                 strict=False):
    def map_func(storage, location):
        return storage.cuda()

    if name_suffix:
        print('Loading %s weights from %s...' % (name_suffix, weights_root))
    else:
        print('Loading weights from %s...' % weights_root)
    if G is not None:
        G.load_state_dict(
            torch.load(
                'pretrained/250000_g.model',
                map_location=map_func),
            strict=strict)
    if D is not None:
        D.load_state_dict(
            torch.load(
                'pretrained/250000_d.model',
                map_location=map_func),
            strict=strict)
    if G_ema is not None:
        print('Loading ema generator...')
        G_ema.load_state_dict(
            torch.load(
                'pretrained/250000_g.model',
                map_location=map_func),
            strict=strict)
        print('Loading PGGAN Generator...')




