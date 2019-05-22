from __future__ import print_function
import numpy as np
np.seed = 1101
from tqdm import tqdm
import torch
from torchvision import datasets, transforms
from torch.autograd import Variable

import argparse


import os
from os.path import join
cwd = os.getcwd()
parts = cwd.split('/scripts/mnist')
ROOT = parts[0]
os.chdir(ROOT)
import sys
sys.path.insert(0, ROOT)

RES = join(ROOT, 'data', 'mnist_grad', 'MNIST')

if not os.path.isdir(join(ROOT, 'data', 'mnist_grad')):
    print('Creating directory ' + join(ROOT, 'data', 'mnist_grad') )
    os.mkdir(join(ROOT, 'data', 'mnist_grad'))

if not os.path.isdir(RES):
    print('Creating directory ' + RES)
    os.mkdir(RES)
else:
    print('Results directory ' + RES)
    
if not os.path.isdir(join(RES,'processed')):
    print('Creating directory ' + join(RES,'processed') )
    os.mkdir(join(RES,'processed'))


import mnist_archs
from IDNN.intrinsic_dimension import estimate, block_analysis
from scipy.spatial.distance import pdist, squareform

# args
parser = argparse.ArgumentParser(description='MNIST with gradient Experiment')
parser.add_argument('--lambdavar', default=100, type=float, metavar='lambda',
                    help='strength of the perturbation')
parser.add_argument('--save', default=0, type=int, metavar='save',
                    help='save dataset')
parser.add_argument('--nsamples', default=2000, type=int, metavar='N',
                    help='number of samples to estimate ID')


args = parser.parse_args()
lambdavar = args.lambdavar
save = args.save
nsamples = args.nsamples
method = 'euclidean'
verbose = True


print('Lambda perturbation strength: {}'.format(lambdavar))

# load training and testdata 
train = torch.load(ROOT + '/data/mnist/MNIST/processed/training.pt')
test = torch.load(ROOT + '/data/mnist/MNIST/processed/test.pt')
    
# convert to float and normalize (the ID is invariant to this transformation)
train_data = train[0].float()/255.0
test_data  = test[0].float()/255.0

# sample of data
sample = test_data[:nsamples,:,:].view(nsamples,-1)

# preliminary checks: invariance with respect to rescaling
print('\n\nPreliminary check on a sample of shape: {}'.format(sample.shape))

dist = squareform(pdist(sample,method))
dim = estimate(dist,verbose=verbose)   
print('ID original data: {}'.format(dim[2]))

dist = squareform(pdist(sample*255,method))
dim = estimate(dist,verbose=verbose)   
print('ID original data rescaled (*255): {}'.format(dim[2]))

# construct and apply perturbation of the form [1,1,...,1] with strenght lambda
print('ID construct perturbed dataset...')
rns = np.random.uniform(0,lambdavar,sample.shape[0])
ones = torch.ones(784)
sample_grad = torch.zeros(sample.shape)
for l in range(sample.shape[0]):    
    sample_grad[l,:] = sample[l,:] + rns[l]*ones
print('Done.')
    
# check dimensionality reduction
print('\nCheck ID reduction')
dist = squareform(pdist(sample,method))
dim1 = estimate(dist,verbose=verbose)   
dist = squareform(pdist(sample_grad,method))
dim2 = estimate(dist,verbose=verbose) 
print('(scipy) --- original : ' 
          + str(dim1[2]) + '--- gradient : ' + str(dim2[2])  )    


print('\nCheck again the grad data, scaled to [0,1])')
minimum = sample_grad.min()
maximum = sample_grad.max()
sample_grad = (sample_grad - minimum)/(maximum - minimum)
dist = squareform(pdist(sample_grad,method))
dim = estimate(dist,verbose=verbose) 
print('(scipy) --- gradient (normalized to [0,1]): ' + str(dim[2])  )   



#-------------------------------- apply this to all the dataset

ones = torch.ones(784)
direction = ones
def stretch_dataset(lambdavar,direction,check=False):
    
    print('\n\n\nCreating mnist grad...')
    # load training and testdata 
    train = torch.load(join(ROOT,'data','mnist','MNIST','processed','training.pt'))
    test = torch.load(join(ROOT,'data','mnist','MNIST','processed','test.pt'))

    # convert to list
    train = list(train)
    test = list(test)
    
    # convert to float and normalize
    train_data = train[0].float()/255.0
    test_data  = test[0].float()/255.0
    
    # reshape
    train_data = train_data.view(train_data.shape[0],-1)
    test_data = test_data.view(test_data.shape[0],-1)
    
    # create perturbed dataset
    rns = np.random.uniform(0,lambdavar,train_data.shape[0])    
    train_data_grad = torch.zeros(train_data.shape) 
    for l in range(train_data.shape[0]):    
        train_data_grad[l,:] = train_data[l,:] + rns[l]*direction
        
    rns = np.random.uniform(0,lambdavar,test_data.shape[0])    
    test_data_grad = torch.zeros(test_data.shape) 
    for l in range(test_data.shape[0]):    
        test_data_grad[l,:] = test_data[l,:] + rns[l]*direction
        
    # normalize
    
    minimum = train_data_grad.min()
    maximum = train_data_grad.max()
    train_data_grad = (train_data_grad - minimum)/(maximum - minimum)
    
    minimum = test_data_grad.min()
    maximum = test_data_grad.max()
    test_data_grad = (test_data_grad - minimum)/(maximum - minimum)
    
    if check:
        
        perm = np.random.permutation(train_data.shape[0])[:nsamples]
        sample = train_data[perm,:].view(nsamples,-1)
        sample_grad = train_data_grad[perm,:].view(nsamples,-1)
        
        dist = squareform(pdist(sample,method))
        dim1 = estimate(dist,verbose=verbose) 
        
        dist = squareform(pdist(sample_grad,method))
        dim2 = estimate(dist,verbose=verbose) 
        
        print('Scipy (Training set) --- original : ' 
          + str(dim1[2]) + '--- gradient : ' + str(dim2[2])  ) 
        
        perm = np.random.permutation(test_data.shape[0])[:nsamples]
        sample = test_data[perm,:].view(nsamples,-1)
        sample_grad = test_data_grad[perm,:].view(nsamples,-1)
        
        dist = squareform(pdist(sample,method))
        dim1 = estimate(dist,verbose=verbose) 
        
        dist = squareform(pdist(sample_grad,method))
        dim2 = estimate(dist,verbose=verbose) 
        
        print('Scipy (Test set)  --- original : ' 
          + str(dim1[2]) + '--- gradient : ' + str(dim2[2])  )   
        
        
    # reshape back into the original form
    train_data_grad = train_data_grad.view(train_data_grad.shape[0],28,28)
    test_data_grad = test_data_grad.view(test_data_grad.shape[0],28,28)
    
    
    train[0] = train_data_grad
    test[0]  = test_data_grad
    train = tuple(train)
    test = tuple(test)
    
    return train,test
        
train, test = stretch_dataset(lambdavar,direction,check=True)

if save:
    print('Saving...')
    torch.save(train, join(ROOT,'data','mnist_grad','MNIST','processed','training.pt'))
    torch.save(test,  join(ROOT,'data','mnist_grad','MNIST','processed','test.pt'))
print('Done.')



# last check
print('\n\nOriginal data (training set)...')
train = torch.load(join(ROOT,'data','mnist','MNIST','processed','training.pt'))
test = torch.load(join(ROOT,'data','mnist','MNIST','processed','test.pt'))
train_data = train[0].float()/255.0
test_data  = test[0].float()/255.0
sample = test_data[:nsamples,:,:].view(nsamples,-1)
dist = squareform(pdist(sample,method))
dim = estimate(dist,verbose=verbose) 

print(train_data.max())
print(train_data.min())
print(train_data.mean())
print(train_data.std())
print('ID: ' + str(dim[2]) )   


print('\n\nPerturbed data (training set)...')
train = torch.load(join(ROOT,'data','mnist_grad','MNIST','processed','training.pt'))
test = torch.load(join(ROOT,'data','mnist_grad','MNIST','processed','test.pt'))
train_data = train[0].float()
test_data  = test[0].float()
sample = test_data[:nsamples,:,:].view(nsamples,-1)
dist = squareform(pdist(sample,method))
dim = estimate(dist,verbose=verbose) 

print(train_data.max())
print(train_data.min())
print(train_data.mean())
print(train_data.std())
print('ID: ' + str(dim[2]) ) 