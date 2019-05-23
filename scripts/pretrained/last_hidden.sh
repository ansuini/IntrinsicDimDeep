#!/bin/bash

declare -a arr=('alexnet' 'vgg11' 'vgg13' 'vgg16' 'vgg19' 'vgg11_bn' 'vgg13_bn' 'vgg16_bn' 'vgg19_bn' 'resnet18' 'resnet34' 'resnet50' 'resnet101' 'resnet152')

nsamples=2000
bs=16
divs=5
res=20
save=1

# pretrained networks
trained=True
for i in "${arr[@]}"
do
   echo "$i"
   python last_hidden.py --arch $i --nsamples $nsamples --bs $bs --divs $divs --res $res --trained $trained --save $save
done





