#!/bin/bash

# declare an array variable

declare -a arr=('alexnet' 'vgg11' 'vgg13' 'vgg16' 'vgg19' 'vgg11_bn' 'vgg13_bn' 'vgg16_bn' 'vgg19_bn' 'resnet18' 'resnet34' 'resnet50' 'resnet101' 'resnet152')

declare -a arr_untrained=('vgg16')

nsamples=500
bs=16
res=5
save=1

trained=1
for i in "${arr[@]}":
do
   echo "$i"
   python hunchback.py --arch $i --nsamples $nsamples --bs $bs --res $res --trained $trained --save $save
done


trained=0
for i in "${arr_untrained[@]}"
do
   echo "$i"
   python hunchback.py --arch $i --nsamples $nsamples --bs $bs --res $res --trained $trained --save $save
done
