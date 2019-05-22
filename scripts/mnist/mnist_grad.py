# general imports
from __future__ import print_function, division
import argparse
import math
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
from scipy.spatial.distance import pdist,squareform
method='euclidean'
verbose=True
import mnist_archs

# dataset constants

imgs_shape = (1,28,28)

# args
parser = argparse.ArgumentParser(description='MNIST (perturbed with gradients) Experiment')

parser.add_argument('--dataset', metavar='DATASET',
                    choices=['mnist_grad','mnist_one_pixel'],
                    help=['MNIST with gradient', 'MNIST with gradient only on one pixel'])
parser.add_argument('--epochs', default=3000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--bs', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--save', default=0, type=int, metavar='save',
                    help='save (0 no, 1 yes)')



args = parser.parse_args()
dataset = args.dataset
bs = args.bs
epochs = args.epochs
lr = args.lr
momentum = args.momentum 
save = args.save

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': 2, 'pin_memory': True} if device.type == 'cuda' else {}

RES = join(ROOT,'data', dataset, 'results')

if not os.path.isdir(RES):
    print('Creating directory ' + RES)
    os.mkdir(RES)
else:
    print('Results directory ' + RES)


# load dataset
train = torch.load(join(ROOT, 'data', dataset, 'MNIST', 'processed', 'training.pt'))
test = torch.load(join(ROOT, 'data', dataset, 'MNIST', 'processed', 'test.pt'))

# check min, max, compute mean, std, and ID
train_imgs = train[0]
test_imgs  = test[0]
train_labels = train[1]
test_labels = test[1]

maxi = train_imgs.max()
mini = train_imgs.min()
mean = train_imgs.mean()
std =  train_imgs.std()
print(train_imgs.max())
print(train_imgs.min())
print(train_imgs.mean())
print(train_imgs.std())

nsamples=2000
sample = test_imgs[:nsamples,:,:].view(nsamples,-1)
dist = squareform(pdist(sample,method))
dim = estimate(dist,verbose=verbose) 
print('ID: ' + str(dim[2]) ) 

# normalization
train_imgs = (train_imgs - mean)/std
test_imgs = (test_imgs - mean)/std

sample = test_imgs[:nsamples,:,:].view(nsamples,-1)
dist = squareform(pdist(sample,method))
dim = estimate(dist,verbose=verbose) 
print('ID (after normalization): ' + str(dim[2]) ) 

# permutation
perm = np.random.permutation(train_imgs.shape[0])
train_imgs = train_imgs[perm]
train_labels = train_labels[perm]

perm = np.random.permutation(test_imgs.shape[0])
test_imgs = test_imgs[perm]
test_labels = test_labels[perm]

# division in batches (arrays with indexes)
idx = np.arange(train_imgs.shape[0])
if idx.shape[0] % bs == 0:
    nbatches = int(idx.shape[0]/bs)
else:
    nbatches = int(idx.shape[0]/bs) + 1
chunks = np.array_split(idx, nbatches)
#print([len(c) for c in chunks])
print('N.of batches: {}'.format(nbatches))
  
print(args)
print(kwargs)
print('Results will be saved in {}'.format(RES))
               
# model instantiation                       
model = mnist_archs.Net()
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
criterion = F.nll_loss

print(summary(model,imgs_shape))

# evaluate model
def evaluate(model,imgs,cats):

    loss = 0.0
    correct = 0
    total = 0 
    
    idx = np.arange(imgs.shape[0])
    if idx.shape[0] % bs == 0:
        nbatches = int(idx.shape[0]/bs)
    else:
        nbatches = int(idx.shape[0]/bs) + 1
    chunks = np.array_split(idx, nbatches)
        
    with torch.no_grad():
        for c in chunks:
            inputs = imgs[c]
            inputs = inputs.view(len(c),1,28,28)
            labels = cats[c]
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
                           
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            loss += criterion(outputs, labels)    
            total += labels.size(0)
            
        acc = 100 * correct/total  
        loss = loss/total
        
    return loss,acc
        


# training and information collection
train_loss   = []
train_acc    = []
test_loss   = []
test_acc    = []

for epoch in tqdm(range(epochs)):
    
    model.train()
    for c in chunks:
        inputs = train_imgs[c]
        inputs = inputs.view(len(c),1,28,28)
        labels = train_labels[c]
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # save model
    if save:
        torch.save(model, join(RES, 'model_' + str(epoch) + '.pt'))
        
    
    # evaluate model on training and test data
    model.eval()
    loss, acc = evaluate(model,train_imgs, train_labels)
    train_loss.append(loss)
    train_acc.append(acc)
    
    loss, acc = evaluate(model,test_imgs, test_labels)
    test_loss.append(loss)
    test_acc.append(acc)

    # print statistics
    print('Test loss : %g --- Test acc : %g %%' % ( test_loss[-1], test_acc[-1] )) 
    
    
# save loss and accuracies
if save:
    tags = ['train_loss','test_loss', 'train_acc', 'test_acc' ]
    vals = [train_loss, test_loss, train_acc, test_acc]
    training_data = dict(zip(tags, vals))
    file = open(join(RES, 'training_data'),'wb')
    pickle.dump(training_data,file)
print('Done.')