from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import pickle
import argparse
from tqdm import tqdm


import os
from os.path import join
cwd = os.getcwd()
parts = cwd.split('/scripts/custom')
ROOT = parts[0]
os.chdir(ROOT)
import sys
sys.path.insert(0, ROOT)


from IDNN.intrinsic_dimension import estimate, block_analysis
from scipy.spatial.distance import pdist,squareform



# args
parser = argparse.ArgumentParser(description='Custom Dataset Experiment')

parser.add_argument('--extract', default=0, type=int, metavar='',
                    help='extract representations (0 no, 1 yes)')

parser.add_argument('--id_final', default=0, type=int, metavar='',
                    help='compute ID final state (0 no, 1 yes)')

parser.add_argument('--nres', default=20, type=int, metavar='N',
                    help='number of resamplings')

parser.add_argument('--do_block_analysis', default=0, type=int, metavar='',
                    help='do block analysis on last layers (0 no, 1 yes)')


args = parser.parse_args()
extract = args.extract
id_final = args.id_final
nres = args.nres
do_block_analysis = args.do_block_analysis


data_folder = join(ROOT, 'data', 'custom')
results_folder = join(ROOT, 'data', 'custom', 'results')
n_out_classes = 40


from vgg_mod import vgg16
data_transform = transforms.Compose([
        transforms.Resize( (224,224) , interpolation=2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
image_dataset = datasets.ImageFolder(os.path.join(data_folder, 'imgs'), data_transform)
dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=20, 
                                         shuffle=True, num_workers=1)

use_gpu = torch.cuda.is_available()


model = torch.load(join(results_folder, 'model_ft.pt') )
model.eval()
print(model.training)


if extract:
    verbose = False
    for i, data in tqdm(enumerate(dataloader, 0)): 
        inputs, _ = data                                              
        inputs = Variable(inputs.cuda())           
        out0,out1,out2,out3,out4,out5,out6,out7,out8 = model.extract_all(inputs, verbose)  
        if i == 0:
            Out0 = out0.view(inputs.shape[0], -1).cpu().data 
            Out1 = out1.view(inputs.shape[0], -1).cpu().data 
            Out2 = out2.view(inputs.shape[0], -1).cpu().data 
            Out3 = out3.view(inputs.shape[0], -1).cpu().data                 
            Out4 = out4.view(inputs.shape[0], -1).cpu().data  
            Out5 = out5.view(inputs.shape[0], -1).cpu().data 
            Out6 = out6.view(inputs.shape[0], -1).cpu().data 
            Out7 = out7.view(inputs.shape[0], -1).cpu().data                 
            Out8 = out8.view(inputs.shape[0], -1).cpu().data          
        else :    
            Out0 = torch.cat((Out0, out0.view(inputs.shape[0], -1).cpu().data),0)
            Out1 = torch.cat((Out1, out1.view(inputs.shape[0], -1).cpu().data),0) 
            Out2 = torch.cat((Out2, out2.view(inputs.shape[0], -1).cpu().data),0) 
            Out3 = torch.cat((Out3, out3.view(inputs.shape[0], -1).cpu().data),0)                 
            Out4 = torch.cat((Out4, out4.view(inputs.shape[0], -1).cpu().data),0) 
            Out5 = torch.cat((Out5, out5.view(inputs.shape[0], -1).cpu().data),0) 
            Out6 = torch.cat((Out6, out6.view(inputs.shape[0], -1).cpu().data),0) 
            Out7 = torch.cat((Out7, out7.view(inputs.shape[0], -1).cpu().data),0)                 
            Out8 = torch.cat((Out8, out8.view(inputs.shape[0], -1).cpu().data),0) 

    # save representations

    print(Out0.shape)
    print(Out1.shape)
    print(Out2.shape)
    print(Out3.shape)
    print(Out4.shape)
    print(Out5.shape)
    print(Out6.shape)
    print(Out7.shape)
    print(Out8.shape)


    torch.save(Out0, join(results_folder, 'Out0') )
    torch.save(Out1, join(results_folder, 'Out1') )
    torch.save(Out2, join(results_folder, 'Out2') )
    torch.save(Out3, join(results_folder, 'Out3') )
    torch.save(Out4, join(results_folder, 'Out4') )
    torch.save(Out5, join(results_folder, 'Out5') )
    torch.save(Out6, join(results_folder, 'Out6') )
    torch.save(Out7, join(results_folder, 'Out7') )
    torch.save(Out8, join(results_folder, 'Out8') )
    
    del Out0,Out1,Out2,Out3,Out4,Out5,Out6,Out7,Out8
    
    
# compute intrinsic dimension

if id_final:
    verbose = True
    method = 'euclidean'
    fraction = 0.9
    nres = 20
    fname = join(results_folder, 'ID')

    def computeID(r,nres,fraction):
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


    # ID of all layers
    fname = join(results_folder, 'ID_all')
    ID_all = []
    # ID of all other layers
    for j in tqdm(range(0,9)):         
        r = torch.load(join(results_folder, 'Out' + str(j) ) ) 
        ID_all.append(computeID(r,nres,fraction)) 
    ID_all = np.array(ID_all)
    np.save(fname,ID_all)
    print('Final result: {}'.format(ID_all[:,0]))
    print('Done.')

# compute block analysis
if do_block_analysis:
    fname = join(results_folder, 'BA')
    method = 'euclidean'
    BA=[]
    for j in range(5,9):         
        r = torch.load(join(results_folder, 'Out' + str(j))) 
        dist = squareform(pdist(r,method))
        ba = block_analysis(dist)         
        BA.append(ba)
    np.save(fname,BA)
    print('Done.')