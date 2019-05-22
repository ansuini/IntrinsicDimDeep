# general imports
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchvision
from torchsummary import summary
from tqdm import tqdm
import pickle
from time import time
import numpy as np
np.seed = 1101
torch.manual_seed(0)

# path
import os
from os.path import join
cwd = os.getcwd()
parts = cwd.split('/scripts/mnist')
ROOT = parts[0]
os.chdir(ROOT)
import sys
sys.path.insert(0, ROOT)


# ID
from IDNN.intrinsic_dimension import estimate, block_analysis
from scipy.spatial.distance import pdist, squareform
import mnist_archs

# dataset constants
dataset='mnist'
mean_imgs = 0.1307
std_imgs = 0.3081
imgs_shape = (1,28,28)

# args
parser = argparse.ArgumentParser(description='Create MNIST data sample from training set for representation extraction')

parser.add_argument('--nsamples', default=2000, type=int,
                    metavar='N', help='data sample size (default: 2000)')

parser.add_argument('--save', default=0, type=int, metavar='save',
                    help='save dataset')


args = parser.parse_args()
nsamples = args.nsamples
save = args.save
print(args)

method = 'euclidean'
verbose = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if device.type == 'cuda' else {}

if save:
    RES = join(ROOT,'data', dataset, 'results')
    print('Results will be saved in {}'.format(RES))
else:
    print('Results will not be saved.')
    

loader = torch.utils.data.DataLoader(    
    datasets.MNIST(join(ROOT, 'data', dataset), train=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((mean_imgs,), (std_imgs,))              
            ])),
    batch_size=nsamples, shuffle=True, **kwargs)


inputs, labels = next(iter(loader))
   
print(labels)

# save sample
if save:
    sample = tuple([inputs,labels])
    torch.save(sample,join(RES,'sample_training.pt'))
