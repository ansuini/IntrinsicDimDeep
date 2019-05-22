#!/bin/bash


epochs=200
bs=64
lr=0.00004
momentum=0.0
save=1

python mnist.py --dataset mnist --epochs $epochs --bs $bs --lr $lr --momentum $momentum --save $save
