from __future__ import print_function, division
import numpy as np

import time
import argparse
import os
import os.path as path
from os import listdir 
from os.path import isfile, join
import copy
import pickle
from tqdm import tqdm
from collections import namedtuple
import subprocess

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
from torchsummary import summary
from torchvision.models import alexnet, vgg11, vgg13, vgg16, vgg19
from torchvision.models import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

# random generators init
torch.backends.cudnn.deterministic = True
torch.manual_seed(999)

# path
cwd = os.getcwd()
parts = cwd.split('/scripts/pretrained')
ROOT = parts[0]
os.chdir(ROOT)
import sys
sys.path.insert(0, ROOT)


from IDNN.intrinsic_dimension import estimate, block_analysis
from scipy.spatial.distance import pdist, squareform

archs = ['alexnet', 'vgg11', 'vgg13', 'vgg16','vgg19',
                    'vgg11_bn', 'vgg13_bn', 'vgg16_bn','vgg19_bn',
                    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

data_folder = join(ROOT, 'data', 'imagenet_training_mix')


parser = argparse.ArgumentParser(description='Extract representations in the last hidden layer in many architectures')

parser.add_argument('--arch', metavar='ARCH',
                    choices=archs,
                    help='model architecture: ' +
                        ' | '.join(archs) )

parser.add_argument('--nsamples', default=2000, type=int,
                    metavar='', help='number of samples (default: 2000)')

parser.add_argument('--bs', default=16, type=int,
                    metavar='', help='mini-batch size (default: 16)')

parser.add_argument('--divs', default=5, type=int,
                    metavar='', help='number of diverse samples (default: 5)')

parser.add_argument('--res', default=20, type=int,
                    metavar='', help='number of resamplings within each sample (default: 20)')

parser.add_argument('--trained', metavar='',
                    choices=['True','False'],
                    help='Trained or randomly initialized : ' +
                        ' | '.join(['True','False']) )

parser.add_argument('--save', default=0, type=int,
                    metavar='', help='save (0 no, 1 yes)')

args = parser.parse_args()
arch = args.arch
nsamples = args.nsamples
bs = args.bs
divs = args.divs
res = args.res
trained = args.trained
save = args.save

print(args)

if trained == 'True':
    print('\nnInstantiating pre-trained model')
    exec('model = ' + arch + '(pretrained=True)')
    results_folder = join(ROOT, 'data', 'pretrained', 'results', 'last_hidden')
else:
    print('\nnInstantiating randomly initialized model')
    exec('model = ' + arch + '(pretrained=False)')
    results_folder = join(ROOT, 'data', 'pretrained', 'results', 'last_hidden_untrained')
    
    
if not os.path.exists(join(ROOT, 'data', 'pretrained', 'results'):
    os.makedirs(join(ROOT, 'data', 'pretrained', 'results')
                
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

print('Results will be saved in {}'.format(results_folder))
                       
# remove last fully-connected layer
def getNewmodel(model):
    
    if 'AlexNet' in model.__class__.__name__:
        print('Apply AlexNet cut')
        new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
        model.classifier = new_classifier        
        
    if 'VGG' in model.__class__.__name__:
        print('Apply VGG cut')
        new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
        model.classifier = new_classifier
        
    if 'ResNet' in model.__class__.__name__:
        print('Apply ResNet cut')
        model = nn.Sequential(*(list(model.children())[:-1]))
        
    return model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


mean_imgs = [0.485, 0.456, 0.406]
std_imgs = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


# Data transformations (same as suggested by Soumith Chintala's script)
data_transforms =  transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])    
    

image_dataset = datasets.ImageFolder(data_folder, data_transforms)           
dataloader = torch.utils.data.DataLoader(image_dataset, 
                                         batch_size=bs, 
                                         shuffle=True, 
                                         num_workers=1)               
      

newmodel = getNewmodel(model)
del model
newmodel = newmodel.to(device)
newmodel.eval()

print('\n\nProcessing architecture ' + arch)
ID = []
for r in tqdm(range(divs)):    
    # Extract representation
    for i, data in enumerate(dataloader, 0):
        #print('{}/{}'.format(i*bs, nsamples))
        if i*bs > nsamples:
            break
        else:            
            inputs, _ = data  
            out = newmodel.forward(inputs.to(device)) 
            if i == 0:
                Out = out.view(inputs.shape[0], -1).cpu().data    
            else :               
                Out = torch.cat((Out, out.view(inputs.shape[0], -1).cpu().data),0) 
                Out = Out.detach()     
            del out
    
    # Compute ID
    print('Computing ID...')
    Out = Out.numpy().astype(np.float64)      
    nimgs = int(np.floor(nsamples*0.9))
    Id = []
    for r in tqdm(range(res)): 
        perm = np.random.permutation(Out.shape[0])[:nimgs]        
        dist = squareform(pdist(Out[perm,:]))
        try:
            est = estimate(dist,verbose=True) 
            est = [est[2],est[3]]
        except:
            est = []                             
        Id.append(est)
    Id = np.asarray(Id)
    ID.append(Id) 
    print('Done.')
    
ID = np.array(ID)


if save:
    np.save(join(results_folder, arch + '_ID_last_hidden.npy'), ID)