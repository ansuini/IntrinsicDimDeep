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


import os
from os.path import join
cwd = os.getcwd()
parts = cwd.split('/scripts/custom')
ROOT = parts[0]
os.chdir(ROOT)
import sys
sys.path.insert(0, ROOT)


data_folder = join(ROOT,'data', 'custom')
results_folder = join(ROOT, 'data', 'custom', 'results')

if not os.path.isdir(results_folder):
    print('Creating directory ' + results_folder)
    os.mkdir(results_folder)
else:
    print('Results directory ' + results_folder)

n_out_classes = 40

from vgg_mod import vgg16

# Data transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize( (224,224) , interpolation=2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize( (224,224) , interpolation=2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'imgs': transforms.Compose([
        transforms.Resize( (224,224) , interpolation=2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    
}


bs = 6
image_datasets = {x: datasets.ImageFolder(os.path.join(data_folder, x), data_transforms[x])
                  for x in ['train', 'test', 'imgs']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=bs, shuffle=True, num_workers=4)
                  for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes
print(class_names)

extractiondataloader = torch.utils.data.DataLoader(image_datasets['imgs'], 
                                                   batch_size=20, shuffle=True, num_workers=4)


use_gpu = torch.cuda.is_available()


nsamples = 1440 # n samples for intrinsic dimensionality computations
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    
    train_loss = []
    train_acc  = []
    test_loss = []
    test_acc  = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and test phase
        for phase in ['train', 'test']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data
                
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
          
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)                      
                running_corrects += torch.sum(preds == labels).item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)                                                
            else:
                test_loss.append(epoch_loss)
                test_acc.append(epoch_acc) 
                
               
            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()
        
        # extract representations on the whole dataset           
        for i, data in enumerate(extractiondataloader, 0):      
        
            if bs*(i+1) > nsamples :
                break
            inputs, _ = data                
            inputs = Variable(inputs.cuda())      
            out1,out2,out3,out4 = model.extract(inputs, 'False')  
            if i == 0:    
                Out1 = out1.view(inputs.shape[0], -1).cpu().data 
                Out2 = out2.view(inputs.shape[0], -1).cpu().data 
                Out3 = out3.view(inputs.shape[0], -1).cpu().data                 
                Out4 = out4.view(inputs.shape[0], -1).cpu().data          
            else :     
                Out1 = torch.cat((Out1, out1.view(inputs.shape[0], -1).cpu().data),0) 
                Out2 = torch.cat((Out2, out2.view(inputs.shape[0], -1).cpu().data),0) 
                Out3 = torch.cat((Out3, out3.view(inputs.shape[0], -1).cpu().data),0)                 
                Out4 = torch.cat((Out4, out4.view(inputs.shape[0], -1).cpu().data),0)        
                
        # save representations       
        torch.save(Out1, join(results_folder, 'Out5_' + str(epoch) ) ) 
        torch.save(Out2, join(results_folder, 'Out6_' + str(epoch) ) ) 
        torch.save(Out3, join(results_folder, 'Out7_' + str(epoch) ) ) 
        torch.save(Out4, join(results_folder, 'Out8_' + str(epoch) ) )   
        
        # save model        
        torch.save(model, join(results_folder, 'model_' + str(epoch) + '.pt') )
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    
    # save training data
    tags = ['train_loss','test_loss', 'train_acc', 'test_acc']
    vals = [train_loss,test_loss,train_acc,test_acc]
    training_data = dict(zip(tags, vals))
    file = open(join(results_folder, "training_data"),'wb')
    pickle.dump(training_data,file)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model




model_ft = vgg16(pretrained=True)
for param in model_ft.parameters():
    param.requires_grad = False
    
# Parameters of newly constructed modules have requires_grad=True by default
model_ft.features._modules['28'] = nn.Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

num_ftrs = model_ft.classifier._modules['0'].in_features
model_ft.classifier._modules['0'] = nn.Linear(num_ftrs, 4096)

num_ftrs = model_ft.classifier._modules['3'].in_features
model_ft.classifier._modules['3'] = nn.Linear(num_ftrs, 4096)

num_ftrs = model_ft.classifier._modules['6'].in_features
model_ft.classifier._modules['6'] = nn.Linear(num_ftrs, n_out_classes)


if use_gpu:
    model_ft = model_ft.cuda()

criterion = nn.CrossEntropyLoss()

# Only trainable parameters will be optimized
#optimizer_ft = torch.optim.SGD(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=1e-3)

# Syntax to train layers differentially (with different learning rates)

optimizer_ft = torch.optim.SGD([
                {'params': model_ft.features._modules['28'].parameters(), 'lr' : 1e-4},
                {'params': model_ft.classifier.parameters(), 'lr': 1e-3}
            ], lr=1e-2, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)



model_ft = train_model(model_ft, criterion, optimizer_ft,
                         exp_lr_scheduler, num_epochs=15)


# save best model
torch.save(model_ft, join(results_folder, 'model_ft.pt') )
