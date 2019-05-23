#-------------------------------------------------------------------------------------------
from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
import os.path as path
from os import listdir 
from os.path import isfile, join
import copy
import pickle
from tqdm import tqdm
import argparse
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchsummary import summary

import torchvision
from torchvision import datasets, models, transforms
from torchsummary import summary

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from scipy.stats import pearsonr

ROOT = '/home/ansuini/repos/intrinsic_dimension'
#ROOT = '/home/paperspace/repos/intrinsic_dimension'

os.chdir(ROOT)
sys.path.append(ROOT)

from IDNN.intrinsic_dimension import estimate, block_analysis
from scipy.spatial.distance import pdist, squareform

#-------------------------------------------------------------------------------------------

from torchvision.models import alexnet, vgg11, vgg13, vgg16, vgg19
from torchvision.models import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

#-------------------------------------------------------------------------------------------

archs = ['alexnet', 'vgg11', 'vgg13', 'vgg16','vgg19',
                    'vgg11_bn', 'vgg13_bn', 'vgg16_bn','vgg19_bn',
                    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

#-------------------------------------------------------------------------------------------

category_tags = ['n01882714','n02086240','n02087394','n02094433','n02100583','n02100735','n02279972', 'mix']
n_objects = len(category_tags) - 1
print('N.of single objects to evaluate : {}'.format(n_objects))

#-------------------------------------------------------------------------------------------

# random numbers https://discuss.pytorch.org/t/random-seed-initialization/7854/14
torch.backends.cudnn.deterministic = True
torch.manual_seed(999)

#-------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Hunchback shape computation')

parser.add_argument('--arch', metavar='ARCH',
                    choices=archs,
                    help='model architecture: ' +
                        ' | '.join(archs) )

parser.add_argument('--nsamples', default=500, type=int,
                    metavar='', help='number of samples (default: 500)')

parser.add_argument('--bs', default=16, type=int,
                    metavar='', help='mini-batch size (default: 16)')

parser.add_argument('--res', default=50, type=int,
                    metavar='', help='number of resamplings within each sample (default: 50)')

parser.add_argument('--trained', default=1, type=int,
                    metavar='', help='use trained 1, untrained 0')

parser.add_argument('--save', default=0, type=int,
                    metavar='', help='save (0 no, 1 yes)')


args = parser.parse_args()
arch = args.arch
nsamples = args.nsamples
bs = args.bs
res = args.res
trained = args.trained
print(args)

#-------------------------------------------------------------------------------------------

def getDepths(model):    
    count = 0    
    modules = []
    names = []
    depths = []    
    modules.append('input')
    names.append('input')
    depths.append(0)    
    
    for i,module in enumerate(model.features):       
        name = module.__class__.__name__
        if 'Conv2d' in name or 'Linear' in name:
            count += 1
        if 'MaxPool2d' in name:
            modules.append(module)
            depths.append(count)
            names.append('MaxPool2d')            
    for i,module in enumerate(model.classifier):
        name = module.__class__.__name__
        if 'Linear' in name:
            modules.append(module)    
            count += 1
            depths.append(count + 1)
            names.append('Linear')                       
    depths = np.array(depths)   
    return modules, names, depths

def getLayerDepth(layer):
    count = 0
    for m in layer:
        for c in m.children():
            name = c.__class__.__name__
            if 'Conv' in name:
                count += 1
    return count

def getResNetsDepths(model):    
    modules = []
    names = []
    depths = []  
    
    # input
    count = 0
    modules.append('input')
    names.append('input')
    depths.append(count)           
    # maxpooling
    count += 1
    modules.append(model.maxpool)
    names.append('maxpool')
    depths.append(count)     
    # 1 
    count += getLayerDepth(model.layer1)
    modules.append(model.layer1)
    names.append('layer1')
    depths.append(count)         
    # 2
    count += getLayerDepth(model.layer2)
    modules.append(model.layer2)
    names.append('layer2')
    depths.append(count)      
    # 3
    count += getLayerDepth(model.layer3)
    modules.append(model.layer3)
    names.append('layer3')
    depths.append(count)     
    # 4 
    count += getLayerDepth(model.layer4)
    modules.append(model.layer4)
    names.append('layer4')
    depths.append(count)      
    # average pooling
    count += 1
    modules.append(model.avgpool)
    names.append('avgpool')
    depths.append(count)     
    # output
    count += 1
    modules.append(model.fc)
    names.append('fc')
    depths.append(count)                      
    depths = np.array(depths)    
    return modules, names, depths

#-------------------------------------------------------------------------------------------

if trained == 1:
    print('Instantiating pre-trained model')
    exec('model = ' + arch + '(pretrained=True)')
    results_folder = join(ROOT, 'data', 'pretrained', 'hunchback_trained')
else:
    print('Instantiating randomly initialized model')
    exec('model = ' + arch + '(pretrained=False)')
    results_folder = join(ROOT, 'data', 'pretrained', 'hunchback_untrained')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

model.eval()
print('Training mode : {}'.format(model.training))

#-------------------------------------------------------------------------------------------

if 'resnet' in arch:
    modules, names, depths = getResNetsDepths(model)
else:
    modules, names, depths = getDepths(model)

#-------------------------------------------------------------------------------------------

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

#-------------------------------------------------------------------------------------------

th = 0.9
def get_pca_dim(x, th):
    cs = np.cumsum(x)
    return np.argwhere(cs > th)[0][0]

#-------------------------------------------------------------------------------------------

n_layers = len(modules)
nimgs = int(np.floor(nsamples*0.9))

ID        = np.zeros((n_objects + 1, n_layers))
IDerr     = np.zeros((n_objects + 1, n_layers))
PCA_DIM   = np.zeros((n_objects + 1, n_layers))
EVR = []
SV  = []
embdims = []

for i,tag in enumerate(category_tags):
    data_folder = join(ROOT, 'data', 'imagenet_training_sample', tag)
    image_dataset = datasets.ImageFolder(join(data_folder), data_transforms)           
    dataloader = torch.utils.data.DataLoader(image_dataset, 
                                             batch_size=bs, 
                                             shuffle=True, 
                                             num_workers=1)  

    sv = []
    evr = []
    
    for l,module in tqdm(enumerate(modules)):    
        for k, data in enumerate(dataloader, 0):
            if k*bs > nsamples:
                break
            else:  
                inputs, _ = data                          
                if module == 'input':                
                    hout = inputs                      
                else:            
                    hout = []
                    def hook(module, input, output):
                        hout.append(output)                
                    handle = module.register_forward_hook(hook)                            
                    out = model(inputs.to(device))
                    del out   
                    
                    if k==0:
                        print(len(hout))
                    
                    hout = hout[0] 
                    
                    if k==0:
                        print(hout.shape)
                    
                    handle.remove()

                if k == 0:
                    Out = hout.view(inputs.shape[0], -1).cpu().data    
                else :               
                    Out = torch.cat((Out, hout.view(inputs.shape[0], -1).cpu().data),0) 
                hout = hout.detach().cpu()
                del hout

        Out = Out.detach().cpu()  
        embdims.append(Out.shape[1])
                    
        # intrinsic dimension        
        Id = []
        for _ in range(res):        
            perm = np.random.permutation(Out.shape[0])[:nimgs]
            dist = squareform(pdist(Out[perm,:]),'euclidean')
            try:
                est = estimate(dist,verbose=True) 
                est = [est[2],est[3]]
            except:
                est = []
            Id.append(est)
        Id = np.asarray(Id)
        ID[i,l]    = np.mean(Id[:,0])
        IDerr[i,l] = np.std(Id[:,0])
                
        # PCA
        scaler = StandardScaler()
        scaler.fit(Out)
        Outn = scaler.transform(Out)

        pca = PCA()
        pca.fit(Outn)
        
        # compute PCA_DIM
        PCA_DIM[i,l] = get_pca_dim(pca.explained_variance_ratio_, th)

        sv.append(pca.singular_values_)
        evr.append(pca.explained_variance_ratio_)
        del Out
       
    SV.append(sv)
    EVR.append(evr)  
    
#-------------------------------------------------------------------------------------------

np.save(join(results_folder, arch + '_PCA_DIM.npy' ), PCA_DIM)
np.save(join(results_folder, arch + '_ID.npy' ), ID )
np.save(join(results_folder, arch + '_IDerr.npy' ), IDerr )
np.save(join(results_folder, arch + '_SV' ), SV )
np.save(join(results_folder, arch + '_EVR' ), EVR )
np.save(join(results_folder, arch + '_embdims' ), embdims)
np.save(join(results_folder, arch + '_depths' ), depths)
np.save(join(results_folder, arch + '_names' ), names)

#-------------------------------------------------------------------------------------------