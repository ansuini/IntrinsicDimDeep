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
parser = argparse.ArgumentParser(description='Create MNIST (grad) data sample for representation extraction')

parser.add_argument('--dataset', metavar='DATASET',
                    choices=['mnist','mnist_grad'],
                    help=['original MNIST', 'MNIST with gradient'])

parser.add_argument('--train', default=0, type=int, metavar='FROM TRAINING SET',
                    help='training (0) or test (1) set')

parser.add_argument('--nsamples', default=2000, type=int,
                    metavar='N', help='data sample size (default: 2000)')

parser.add_argument('--save', default=0, type=int, metavar='save',
                    help='save dataset')


args = parser.parse_args()
train = args.train
nsamples = args.nsamples
dataset = args.dataset
save = args.save
print(args)

method = 'euclidean'
verbose = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if device.type == 'cuda' else {}

if save:
    RES = join(ROOT, 'data', dataset, 'results')
    print('Results will be saved in {}'.format(RES))
else:
    print('Results will not be saved.')
    

if dataset=='mnist':
    if train:
        loader = torch.utils.data.DataLoader(    
            datasets.MNIST(join(ROOT, 'data', dataset), train=False,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((mean_imgs,), (std_imgs,))              
                           ])),
            batch_size=nsamples, shuffle=False, **kwargs)
    else:
        loader = torch.utils.data.DataLoader(    
            datasets.MNIST(join(ROOT, 'data', dataset), train=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((mean_imgs,), (std_imgs,))              
                           ])),
            batch_size=nsamples, shuffle=False, **kwargs)
        
    imgs,labels= next(iter(loader))

else:
    
    # load dataset
    if train:
        data = torch.load(join(ROOT,'data', dataset,'MNIST','processed','training.pt'))
    else:
        data = torch.load(join(ROOT,'data', dataset,'MNIST','processed','test.pt'))
        

    # check min, max, compute mean, std, and ID
    imgs = data[0] 
    labels = data[1]
    
    maxi = imgs.max()
    mini = imgs.min()
    mean = imgs.mean()
    std =  imgs.std()
   
    perm = np.random.permutation(imgs.shape[0])[:nsamples]
    imgs = imgs[perm,:,:].view(nsamples,-1)
    labels = labels[perm]
    
    dist = squareform(pdist(imgs,method))
    dim = estimate(dist,verbose=verbose) 
    print('ID (before normalization): ' + str(dim[2]) ) 

    # normalization
    imgs = (imgs - mean)/std
    dist = squareform(pdist(imgs,method))
    dim = estimate(dist,verbose=verbose) 
    print('ID (after normalization): ' + str(dim[2]) ) 
    
    # reshape
    imgs = imgs.view(nsamples,1,28,28)
    
# save sample
if save:
    
    sample = tuple([imgs,labels])
    torch.save(sample,join(RES,'sample.pt'))
    
    # if mnist save also in mnist shuffled
    if dataset=='mnist':
        torch.save(sample, join(ROOT,'data', 'mnist_shuffled', 'results','sample.pt') )