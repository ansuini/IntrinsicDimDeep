#!/bin/bash

dataset=mnist_grad
epochs=5000
bs=64
lr=0.0001
momentum=0.9
save=1

python mnist_grad.py --dataset $dataset --epochs $epochs --bs $bs --lr $lr --momentum $momentum --save $save