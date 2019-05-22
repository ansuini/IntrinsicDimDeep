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
from scipy.spatial.distance import pdist,squareform

import mnist_archs

# dataset constants

imgs_shape = (1,28,28)

# args
parser = argparse.ArgumentParser(description='MNIST Experiment')
parser.add_argument('--dataset', metavar='DATASET',
                    choices=['mnist','mnist_shuffled'],
                    help=['original MNIST','shuffled MNIST'])
parser.add_argument('--epochs', default=200, type=int, metavar='N',
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

RES = join(ROOT, 'data', dataset, 'results')

if not os.path.isdir(RES):
    print('Creating directory ' + RES)
    os.mkdir(RES)
else:
    print('Results directory ' + RES)



mean_imgs = 0.1307
std_imgs = 0.3081

print(args)
print(kwargs)

if save:
    print('Results will be saved in {}'.format(RES))

    
#loaders
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(join(ROOT, 'data', dataset), train=True, download=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((mean_imgs,), (std_imgs,))                                           
                   ])),
    batch_size=bs, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(    
    datasets.MNIST(join(ROOT, 'data', dataset), train=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((mean_imgs,), (std_imgs,))                       
                   ])),
    batch_size=bs, shuffle=True, **kwargs)
                     
# model instantiation                       
model = mnist_archs.Net()
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
criterion = F.nll_loss

print(summary(model,imgs_shape))



# evaluate model
def evaluate(model,loader):

    loss = 0.0
    correct = 0
    total = 0 
    
    with torch.no_grad():
        for data in loader:
            
            inputs, labels = data
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
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
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
    loss, acc = evaluate(model,train_loader)
    train_loss.append(loss)
    train_acc.append(acc)
    
    loss, acc = evaluate(model,test_loader)
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
