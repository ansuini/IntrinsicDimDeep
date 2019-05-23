#!/bin/bash

declare -a arr=('alexnet' 'vgg11' 'vgg13' 'vgg16' 'vgg19' 'vgg11_bn' 'vgg13_bn' 'vgg16_bn' 'vgg19_bn' 'resnet18' 'resnet34' 'resnet50' 'resnet101' 'resnet152')

declare -a arr_untrained=('vgg16')


nsamples=2000
bs=16

# pretrained networks
trained=1
for i in "${arr[@]}"
do
   echo "$i"
   python last_hidden_pca.py --arch $i --nsamples $nsamples --bs $bs --trained $trained
done


# randomly initialized networks
trained=0
for i in "${arr[@]}"
do
   echo "$i"
   python last_hidden_pca.py --arch $i --nsamples $nsamples --bs $bs --trained $trained
done