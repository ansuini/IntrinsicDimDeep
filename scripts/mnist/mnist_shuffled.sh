#!/bin/bash

epochs=500
bs=64
lr=0.0005
momentum=0.9
save=1

python mnist.py --dataset mnist_shuffled --epochs $epochs --bs $bs --lr $lr --momentum $momentum --save $save