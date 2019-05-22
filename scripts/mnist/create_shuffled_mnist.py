import torch
import numpy as np
np.seed = 1101


import os
from os.path import join
cwd = os.getcwd()
parts = cwd.split('/scripts/mnist')
ROOT = parts[0]
os.chdir(ROOT)
import sys
sys.path.insert(0, ROOT)


RES = join(ROOT, 'data', 'mnist_shuffled', 'MNIST')

if not os.path.isdir(join(ROOT, 'data', 'mnist_shuffled')):
    print('Creating directory ' + join(ROOT, 'data', 'mnist_shuffled') )
    os.mkdir(join(ROOT, 'data', 'mnist_shuffled'))

if not os.path.isdir(RES):
    print('Creating directory ' + RES)
    os.mkdir(RES)
else:
    print('Results directory ' + RES)
    
if not os.path.isdir(join(RES,'processed')):
    print('Creating directory ' + join(RES,'processed') )
    os.mkdir(join(RES,'processed'))


os.chdir(ROOT)

# load original data
train = torch.load(join(ROOT, 'data','mnist', 'MNIST','processed','training.pt') )
test  = torch.load(join(ROOT, 'data','mnist', 'MNIST','processed','test.pt') )

# separe images and labels
train = list(train)
test = list(test)
train_labels = train[1]
test_labels = test[1]

# permute training and test labels, reconstruct dataset in its original form
perm_train = np.random.permutation(train_labels.shape[0])
perm_test  = np.random.permutation(test_labels.shape[0])
train[1] = train_labels[perm_train,]
test[1] = train_labels[perm_test,]
train = tuple(train)
test  = tuple(test)

# save shuffled dataset
torch.save(train, join(RES,'processed','training.pt'))
torch.save(test, join(RES,'processed','test.pt'))