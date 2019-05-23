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

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
from torchsummary import summary
from torchvision.models import alexnet, vgg11, vgg13, vgg16, vgg19
from torchvision.models import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from scipy.stats import pearsonr

# random numbers https://discuss.pytorch.org/t/random-seed-initialization/7854/14
torch.backends.cudnn.deterministic = True
torch.manual_seed(999)

ROOT = '/home/ansuini/repos/intrinsic_dimension'
os.chdir(ROOT)

import sys
sys.path.append(ROOT)

from IDNN.intrinsic_dimension import estimate, block_analysis
from scipy.spatial.distance import pdist, squareform

archs = ['alexnet', 'vgg11', 'vgg13', 'vgg16','vgg19',
                    'vgg11_bn', 'vgg13_bn', 'vgg16_bn','vgg19_bn',
                    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

#data_folder = join(ROOT, 'data', 'ILSVRC2013_DET_test')
data_folder = join(ROOT, 'data', 'imagenet_training_sample_mix5')

parser = argparse.ArgumentParser(description='Extract representations in the last hidden layer in many architectures')

parser.add_argument('--arch', metavar='ARCH',
                    choices=archs,
                    help='model architecture: ' +
                        ' | '.join(archs) )

parser.add_argument('--nsamples', default=2000, type=int,
                    metavar='', help='number of samples (default: 2000)')

parser.add_argument('--bs', default=16, type=int,
                    metavar='', help='mini-batch size (default: 16)')

parser.add_argument('--trained', default=1, type=int,
                    metavar='', help='use pre-trained network (0 no, 1 yes)')
                    
                    
args = parser.parse_args()
arch = args.arch
nsamples = args.nsamples
bs = args.bs
trained = args.trained

print(args)

if trained == 1:
    print('Instantiating pre-trained model')
    exec('model = ' + arch + '(pretrained=True)')
    results_folder = join(ROOT, 'data', 'pretrained', 'last_hidden_pca_trained')
else:
    print('Instantiating randomly initialized model')
    exec('model = ' + arch + '(pretrained=False)')
    results_folder = join(ROOT, 'data', 'pretrained', 'last_hidden_pca_untrained')

# get pca dim
th = 0.9
def get_pca_dim(x, th):
    cs = np.cumsum(x)
    return np.argwhere(cs > th)[0][0]
                       
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


print('Processing architecture ' + arch)
ID = []
# Extract representation
for i, data in enumerate(dataloader, 0):
    print('{}/{}'.format(i*bs, nsamples))
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
np.save(join(results_folder, arch + '_out_last_hidden_layer_training_data.npy'), ID )

# PCA
scaler = StandardScaler()
scaler.fit(Out)
Outn = scaler.transform(Out)

pca = PCA()
pca.fit(Outn)
pickle.dump(pca, open(join(results_folder, arch + '_pca_training_data.pkl'),'wb' ) )

# Analysis of the elliptical dataset. This generates a dataset with the 
# same correlation structure of the original dataset

verbose=True
method='euclidean'

ID_original = []
ID_ellipse  = []
ID_PC = []
sizes = np.arange(0,100,5)*100
sizes = sizes[1:]
for nsamples in tqdm(sizes):
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
   
    # normal ID
    dist = squareform(pdist(Out,method))
    est = estimate(dist,verbose=verbose) 
    id_ori = est[2]
    ID_original.append(id_ori)
    
    # pca data
    pca = PCA()
    Out = StandardScaler().fit_transform(Out)
    pca.fit(Out)
    
    # the n.of eigenvalues should be the minimum between the n. of features
    # and the n. of data points
    neigs = len(pca.singular_values_)
    
    # id given by the pca : 90 % of variance
    id_pc = get_pca_dim(pca.explained_variance_ratio_,th)
    
    # check that the id_pc is lower than the number of features
    if id_pc > neigs:
        print('Error !')
        break
        
    ID_PC.append(id_pc)
    
    # generate data with the same correlation structure of the data and compute their id
    X_ellipse = np.zeros((Out.shape[0], np.min([neigs, Out.shape[0]] ) ) )
    for i in range(X_ellipse.shape[1]):
        X_ellipse[:,i] = np.sqrt(pca.singular_values_[i])*np.random.randn(Out.shape[0])
    
    dist = squareform(pdist(X_ellipse,method))
    est = estimate(dist,verbose=verbose) 
    id_ell = est[2]
    ID_ellipse.append(id_ell)
    
    print('nsamples : {}, ID : {}, ID_PC : {}, ID_ellipse : {}'.format(Out.shape[0], 
                                                                int(np.round(id_ori,3)),
                                                                    int(np.round(id_pc,3)), 
                                                                    int(np.round(id_ell,3)) ) )
    
tags = ['ID_original', 'ID_ellipse', 'ID_PC']
values = [ID_original, ID_ellipse, ID_PC]
IDs = dict(zip(tags,values))
pickle.dump(IDs,open(join(results_folder, arch + '_IDs_training_data.pkl'), 'wb') )