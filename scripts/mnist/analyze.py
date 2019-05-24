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
import pickle
from time import time
from tqdm import tqdm
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


import mnist_archs
layers = ['input','conv1','pool1', 'conv2', 'pool2', 'd1','output']

from IDNN.intrinsic_dimension import estimate, block_analysis
from scipy.spatial.distance import pdist, squareform

# args
parser = argparse.ArgumentParser(description='MNIST Experiment')

parser.add_argument('--dataset', metavar='DATASET',
                    choices=['mnist','mnist_shuffled','mnist_grad'],
                    help=['original MNIST','shuffled MNIST','MNIST with gradient'])

parser.add_argument('--epochs', default=0, type=int, metavar='N',
                    help='number of total epochs to run (set it equal to the number of training epochs of your model)')

parser.add_argument('--fraction', default=0.9, type=float, metavar='fraction',
                    help='fraction of data resampling for error estimation')

parser.add_argument('--nres', default=50, type=int, metavar='N',
                    help='number of resamplings for error estimation')


args = parser.parse_args()
dataset = args.dataset
epochs = args.epochs
fraction = args.fraction
nres = args.nres

method = 'euclidean'
verbose=False

RES = join(ROOT, 'data', dataset, 'results')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': 2, 'pin_memory': True} if device.type == 'cuda' else {}

print(args)
print(kwargs)
print('Results will be saved in {}'.format(RES))

# load sample (the sample is the same for mnist and mnist_shuffled)

sample = torch.load(join(ROOT,'data', dataset, 'results', 'sample.pt') )
nsamples = sample[0].shape[0]


def computeID(r,epoch,nres,fraction):
    ID = []
    n = int(np.round(r.shape[0]*fraction))            
    dist = squareform(pdist(r,method))
    for i in range(nres):
        dist_s = dist
        perm = np.random.permutation(dist.shape[0])[0:n]
        dist_s = dist_s[perm,:]
        dist_s = dist_s[:,perm]
        ID.append(estimate(dist_s,verbose=verbose)[2])
    mean = np.mean(ID) 
    error = np.std(ID) 
    return mean,error

    
#ID at all layers

epoch = epochs-1
print('Extracting representations from all layers...')
# load model
model = torch.load(join(RES, 'model_' + str(epoch) + '.pt') )

# extract representations from the sample
out1, out2, out3, out4, out5, out6 = model.extract_all(sample[0].to(device),verbose=verbose)
out1 = out1.view(nsamples, -1).cpu().data
out2 = out2.view(nsamples, -1).cpu().data
out3 = out3.view(nsamples, -1).cpu().data
out4 = out4.view(nsamples, -1).cpu().data 
out5 = out5.view(nsamples, -1).cpu().data
out6 = out6.view(nsamples, -1).cpu().data 
  
torch.save(out1, join(RES, 'All_R1') )
torch.save(out2, join(RES, 'All_R2') )
torch.save(out3, join(RES, 'All_R3') )
torch.save(out4, join(RES, 'All_R4') )
torch.save(out5, join(RES, 'All_R5') )
torch.save(out6, join(RES, 'All_R6') )
print('Done.')

# ID of all layers
    
fname = join(RES, 'ID_all')
   
ID_all = []       
ID_all.append(computeID(sample[0].view(sample[0].shape[0],-1),epoch,nres,fraction))
# ID of all other layers
for j in range(1,len(layers)):
    r = torch.load(join(RES, 'All_R' + str(j)) ) 
    ID_all.append(computeID(r,epoch,nres,fraction)) 
ID_all = np.array(ID_all)
np.save(fname,ID_all)
print('Final result: {}'.format(ID_all[:,0]))
print('Done.')
    