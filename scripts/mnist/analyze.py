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
layers = ['input','pool1','pool2','d1','output']

from IDNN.intrinsic_dimension import estimate, block_analysis
from scipy.spatial.distance import pdist, squareform

# args
parser = argparse.ArgumentParser(description='MNIST Experiment')

parser.add_argument('--dataset', metavar='DATASET',
                    choices=['mnist','mnist_shuffled','mnist_grad'],
                    help=['original MNIST','shuffled MNIST','MNIST with gradient'])

parser.add_argument('--epochs', default=0, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--extract', default=0, type=int, metavar='',
                    help='extract representations (0 no, 1 yes)')

parser.add_argument('--step', default=1, type=int, metavar='N',
                    help='step (epochs to jump in the sequence)')

parser.add_argument('--id_evolution', default=0, type=int, metavar='',
                    help='compute time evolution of ID in checkpoint layers')

parser.add_argument('--id_final', default=0, type=int, metavar='',
                    help='compute ID in checkpoint layers at the end of training')

parser.add_argument('--id_final_all_layers', default=0, type=int, metavar='',
                    help='compute ID in all layers at end of training')

parser.add_argument('--id_final_all_layers_training_set', default=0, type=int, metavar='',
                    help='compute ID in all layers at end of training')


parser.add_argument('--do_block_analysis', default=0, type=int, metavar='',
                    help='block analysis in checkpoint layers at end of training')

parser.add_argument('--fraction', default=0.9, type=float, metavar='fraction',
                    help='fraction of data resampling for error estimation')

parser.add_argument('--nres', default=50, type=int, metavar='N',
                    help='number of resamplings for error estimation')



args = parser.parse_args()
dataset = args.dataset
epochs = args.epochs
extract = args.extract
fraction = args.fraction
nres = args.nres
step = args.step
id_evolution = args.id_evolution
id_final = args.id_final
id_final_all_layers = args.id_final_all_layers
id_final_all_layers_training_set = args.id_final_all_layers_training_set
do_block_analysis = args.do_block_analysis


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
#sample_training = torch.load(join(ROOT,'data', dataset, 'results', 'sample_training.pt') )
nsamples = sample[0].shape[0]
#nsamples_training = sample_training[0].shape[0]

# sample of epochs to extract data
eps = np.arange(0,epochs,step)

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


# extract representations
if extract:
    print('Extracting representations...')
    for epoch in tqdm(eps):
        # load model
        model = torch.load(join(RES, 'model_' + str(epoch) + '.pt') )

        # extract representations from the sample of test data
        out1, out2, out3, out4 = model.extract(sample[0].to(device),verbose=verbose)
        out1 = out1.view(nsamples, -1).cpu().data
        out2 = out2.view(nsamples, -1).cpu().data
        out3 = out3.view(nsamples, -1).cpu().data
        out4 = out4.view(nsamples, -1).cpu().data 
        torch.save(out1, join(RES, 'R1_' + str(epoch) ) )
        torch.save(out2, join(RES, 'R2_' + str(epoch) ) )
        torch.save(out3, join(RES, 'R3_' + str(epoch) ) )
        torch.save(out4, join(RES, 'R4_' + str(epoch) ) )
    print('Done.')
    
    
# compute the intrinsic dimension evolution
if id_evolution:
    print('Computing intrinsic dimension evolution...')
    fname = join(RES, 'ID_evolution.p')
    ID = []  
    count = 0
    tot = 0
    for i in tqdm(eps):  
        Id = []
        for j in range(1,len(layers)):  
            tot += 1
            r = torch.load(join(RES, 'R' + str(j) + '_' + str(i) ) ) 
            dist = squareform(pdist(r,method))            
            try:                                
                est = estimate(dist,verbose=verbose) 
                est = [est[2],est[3]]
            except:
                print('Warning ! Losing data.')
                count += 1
                est = []                             
            Id.append(est)           
        ID.append(Id)
    print('Data loss : {}'.format(count/tot))

    ID = np.array(ID)
    tags = ['layers','ID']
    vals = [layers, ID]
    evolution = dict(zip(tags,vals))
    pickle.dump(evolution, open( fname, "wb"))   
    print('Done.')

    
# intrinsic dimension at the end of training
if id_final:
    verbose = True
    epoch = epochs-step
    fname = join(RES, 'ID')
    

    # ID of the input
    ID = []
    ID.append(computeID(sample[0].view(sample[0].shape[0],-1),epoch,nres,fraction))
    # ID of all other layers
    for j in range(1,len(layers)):         
        r = torch.load(join(RES, 'R' + str(j) + '_' + str(epoch) ) ) 
        ID.append(computeID(r,epoch,nres,fraction)) 
    ID = np.array(ID)
    np.save(fname,ID)
    print('Final result: {}'.format(ID[:,0]))
    print('Done.')
    

#ID at all layers
if id_final_all_layers:
    epoch = epochs-step
    print('Extracting representations from all layers...')
    # load model
    model = torch.load(join(RES, 'model_' + str(epoch) + '.pt') )

    # extract representations from the sample of test data
    out1, out2, out3, out4, out5, out6 = model.extract_all(sample[0].to(device),verbose=verbose)
    out1 = out1.view(nsamples, -1).cpu().data
    out2 = out2.view(nsamples, -1).cpu().data
    out3 = out3.view(nsamples, -1).cpu().data
    out4 = out4.view(nsamples, -1).cpu().data 
    out5 = out5.view(nsamples, -1).cpu().data
    out6 = out6.view(nsamples, -1).cpu().data 
    torch.save(out1, join(RES, 'All_R1_' + str(epoch) ) )
    torch.save(out2, join(RES, 'All_R2_' + str(epoch) ) )
    torch.save(out3, join(RES, 'All_R3_' + str(epoch) ) )
    torch.save(out4, join(RES, 'All_R4_' + str(epoch) ) )
    torch.save(out5, join(RES, 'All_R5_' + str(epoch) ) )
    torch.save(out6, join(RES, 'All_R6_' + str(epoch) ) )
    print('Done.')

    # ID of all layers
    fname = join(RES, 'ID_all')
    ID_all = []
    ID_all.append(computeID(sample[0].view(sample[0].shape[0],-1),epoch,nres,fraction))
    # ID of all other layers
    for j in range(1,len(layers)+2):         
        r = torch.load(join(RES, 'All_R' + str(j) + '_' + str(epoch) ) ) 
        ID_all.append(computeID(r,epoch,nres,fraction)) 
    ID_all = np.array(ID_all)
    np.save(fname,ID_all)
    print('Final result: {}'.format(ID_all[:,0]))
    print('Done.')
    
#ID at all layers (training set)
if id_final_all_layers_training_set:
    epoch = epochs-step
    print('Extracting representations from all layers...')
    # load model
    model = torch.load(join(RES, 'model_' + str(epoch) + '.pt') )

    # extract representations from the sample of training data
    out1, out2, out3, out4, out5, out6 = model.extract_all(sample_training[0].to(device),verbose=verbose)
    out1 = out1.view(nsamples_training, -1).cpu().data
    out2 = out2.view(nsamples_training, -1).cpu().data
    out3 = out3.view(nsamples_training, -1).cpu().data
    out4 = out4.view(nsamples_training, -1).cpu().data 
    out5 = out5.view(nsamples_training, -1).cpu().data
    out6 = out6.view(nsamples_training, -1).cpu().data 
    torch.save(out1, join(RES, 'All_R1_training_' + str(epoch) ) )
    torch.save(out2, join(RES, 'All_R2_training_' + str(epoch) ) )
    torch.save(out3, join(RES, 'All_R3_training_' + str(epoch) ) )
    torch.save(out4, join(RES, 'All_R4_training_' + str(epoch) ) )
    torch.save(out5, join(RES, 'All_R5_training_' + str(epoch) ) )
    torch.save(out6, join(RES, 'All_R6_training_' + str(epoch) ) )
    print('Done.')

    # ID of all layers
    fname = join(RES, 'ID_all_training_set')
    ID_all = []
    ID_all.append(computeID(sample[0].view(sample[0].shape[0],-1),epoch,nres,fraction))
    # ID of all other layers
    for j in range(1,len(layers)+2):         
        r = torch.load(join(RES, 'All_R' + str(j) + '_training_' + str(epoch) ) ) 
        ID_all.append(computeID(r,epoch,nres,fraction)) 
    ID_all = np.array(ID_all)
    np.save(fname,ID_all)
    print('Final result: {}'.format(ID_all[:,0]))
    print('Done.')
    

# compute block analysis
if do_block_analysis:
    epoch = epochs-step
    fname = join(RES, 'BA')
    method = 'euclidean'
    BA=[]
    for j in range(1,len(layers)):         
        r = torch.load(join(RES, 'R' + str(j) + '_' + str(epoch) ) ) 
        dist = squareform(pdist(r,method))
        BA.append(block_analysis(dist))       
    np.save(fname,BA)
    print('Done.')